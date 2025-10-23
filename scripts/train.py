import os
from loguru import logger
logger.info(
    "Environment Variables:\n" +
    "\n".join(f"{k}={v}" for k, v in os.environ.items())
)
import torch
import torch.optim as optim
import hydra
import math
from omegaconf import DictConfig

from utils.wandb import init_wandb
from utils.seed import seed_everything
from utils.config_handler import get_outputs_path, save_git_diff

from utils.viz import plot_sample_img_grid

from model.diffusion import UNet
from schedulers.ddim import DDIM
from training.train_diffusion import train_img_diffusion_model
from training.train_regression import train_regression_model
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from utils.config_handler import remove_weight_prefixes
from utils.dataloader import get_dataset
from omegaconf import OmegaConf



def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr_factor=0.1):
    """
    Cosine annealing schedule with warmup and a minimum learning rate defined as a fraction of the initial LR.
    
    Args:
        optimizer: PyTorch optimizer.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.
        num_cycles (float): Number of cosine cycles. Default is 0.5 (cosine decay).
        min_lr_factor (float): Minimum learning rate as a fraction of initial LR (e.g., 0.1).
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

# Main
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(config: DictConfig):
    outputs_dir = get_outputs_path()
    logger.info(f"Outputs directory: {outputs_dir}")
    logger.add(f"{outputs_dir}/train.log")
    logger.debug("Running with config:\n" + OmegaConf.to_yaml(config))
    
    model_path = config['train']['checkpoint_path']
    if model_path is None or model_path == "":
        model_path = os.path.join(outputs_dir, "model.pth")
    
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        logger.info(f"Distributed training initialized on device {device} with rank {ddp_rank}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
    
    if master_process:
        init_wandb(config, outputs_dir, **config['wandb'])
        if config['save_git_diff']:
            save_git_diff(outputs_dir)
    else:
        logger.remove()
    seed = config['train']['seed']
    seed_everything(seed)
    
    
    torch.set_float32_matmul_precision(config['float32_matmul_precision'])
    dataset = get_dataset(config)
    
    if master_process:
        import math
        max_samples = min(len(dataset), 25)
        grid_size = int(math.sqrt(max_samples))
        plot_sample_img_grid(dataset, grid_size=grid_size, save_path=os.path.join(outputs_dir, "sample_grid.pdf"))

    val_fraction = config['dataset']['val_fraction']

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))

    batch_size = config['train']['batch_size']
    num_workers = config['train']['num_workers']

    if ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Fix for DataLoader worker bus error
    import multiprocessing as mp
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # Use safer DataLoader settings to avoid bus errors
    safe_num_workers = min(num_workers, mp.cpu_count() // 2) if num_workers > 0 else 0
    if safe_num_workers != num_workers:
        print(f"Warning: Reduced num_workers from {num_workers} to {safe_num_workers} for stability")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # only shuffle if not using DistributedSampler
        num_workers=safe_num_workers,
        persistent_workers=config['train'].get('persistent_workers', True) if safe_num_workers > 0 else False,
        pin_memory=True if safe_num_workers > 0 else False,
        multiprocessing_context='spawn' if safe_num_workers > 0 else None,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=safe_num_workers,
        persistent_workers=config['train'].get('persistent_workers', True) if safe_num_workers > 0 else False,
        pin_memory=True if safe_num_workers > 0 else False,
        multiprocessing_context='spawn' if safe_num_workers > 0 else None,
    )

    model = UNet(
        device=device,
        **config['model']
        )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    if config['compile_model']:
        model = torch.compile(model)

    # Load checkpoint before DDP
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        state_dict = checkpoint["state_dict"]
        state_dict = remove_weight_prefixes(state_dict)
        model.load_state_dict(state_dict)
        if master_process:
            logger.info(f"Loaded model from {model_path}")

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[ddp_local_rank],
        )
        
    if config['train']['optimizer_config']['type'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **config['train']['optimizer_config']['params'])
    elif config['train']['optimizer_config']['type'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **config['train']['optimizer_config']['params'])
    else:
        raise ValueError(f"Unsupported optimizer: {config['train']['optimizer_config']['type']}")
    
    num_training_steps = config['train']['num_epochs'] * len(train_dataloader) // config['train']['grad_accum_steps']
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_training_steps=num_training_steps, **config['train']['lr_scheduler_config'])

    # Detect training mode from config
    training_mode = config.get('training_mode', 'diffusion')  # Default to diffusion for backward compatibility
    
    # Alternative detection: if scheduler has diffusion_steps == 1, assume regression
    if training_mode == 'diffusion' and config['scheduler'].get('diffusion_steps', 100) == 1:
        training_mode = 'regression'
        logger.info("Detected regression mode from scheduler config (diffusion_steps=1)")
    
    logger.info(f"Training mode: {training_mode}")
    
    if training_mode == 'regression':
        # Regression training - no scheduler needed
        train_regression_model(
            config,
            model,
            train_dataloader,
            val_dataloader,
            dataset,
            optimizer,
            lr_scheduler,
            outputs_dir,
            ddp=ddp,
            master_process=master_process,
            ddp_world_size=ddp_world_size,
            device=device,
            **config['train'])
    else:
        # Diffusion training
        scheduler = DDIM(device=device, **config['scheduler'])
        train_img_diffusion_model(
            config,
            model,
            scheduler,
            train_dataloader,
            val_dataloader,
            dataset,
            optimizer,
            lr_scheduler,
            outputs_dir,
            ddp=ddp,
            master_process=master_process,
            ddp_world_size=ddp_world_size,
            device=device,
            **config['train'])

    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    main()