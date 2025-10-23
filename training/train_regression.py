import torch
import torch.nn as nn
from model.diffusion import UNet
from loguru import logger
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from utils.viz import plot_condition_vs_images_grid, plot_training_visualization_grid
import os
import time
import torch.distributed as dist
from utils.metrics import squareness_metric, alignment_metric
from utils.viz import tensor_to_binary_mask

dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

def validate_regression_model(
    config,
    model,
    val_dataloader,
    dataset,
    device="cuda",
    ddp=False,
    master_process=True,
    autocast_dtype='bfloat16',
    loss_config={},
    n_samples_for_metrics=1000,
    seed=42,
    **kwargs,
):
    model.eval()
    val_loss = 0.0
    num_batches = 0
    quality_scores = []
    alignment_scores = []
    total_metric_samples = 0

    with torch.no_grad():
        val_iter = tqdm(val_dataloader, desc="Validation", leave=False) if master_process else val_dataloader

        for condition_imgs, target_imgs in val_iter:
            condition_imgs = condition_imgs.to(device, non_blocking=True)
            target_imgs = target_imgs.to(device, non_blocking=True)

            # Use t=0 for regression (no noise)
            t = torch.zeros(condition_imgs.size(0), 1, device=device)

            with torch.autocast(device_type=device, dtype=dtype_map[autocast_dtype]):
                pred_imgs = model(torch.zeros_like(target_imgs), t, condition_imgs)
                loss = calc_regression_loss(
                    pred_imgs,
                    target_imgs,
                    dataset=dataset,
                    **loss_config,
                )

            val_loss += loss.item()
            num_batches += 1
            if master_process:
                val_iter.set_postfix(val_loss=loss.item())

            # Metrics collection (only up to n_samples_for_metrics)
            remaining = n_samples_for_metrics - total_metric_samples
            if remaining <= 0:
                continue

            to_sample = min(remaining, condition_imgs.size(0))
            pred_sample = pred_imgs[:to_sample]
            
            for condition_img, pred_img in zip(condition_imgs[:to_sample], pred_sample):
                pred_img_converted = dataset.to_image(pred_img)  # Convert to image format
                pred_binary = tensor_to_binary_mask(pred_img_converted)
                condition_binary = tensor_to_binary_mask(condition_img)

                quality = squareness_metric(pred_binary, **config['metrics']['quality'])
                alignment = alignment_metric(pred_binary, condition_binary, **config['metrics']['alignment'])

                quality_scores.append(quality)
                alignment_scores.append(alignment)

            total_metric_samples += to_sample

    # Convert losses and metrics to tensors for DDP reduction
    val_loss_tensor = torch.tensor(val_loss, device=device)
    num_batches_tensor = torch.tensor(num_batches, device=device)
    quality_tensor = torch.tensor(quality_scores, device=device)
    alignment_tensor = torch.tensor(alignment_scores, device=device)
    metric_count_tensor = torch.tensor(len(quality_scores), device=device)

    if ddp:
        # Reduce losses
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        # Reduce metrics
        dist.all_reduce(metric_count_tensor, op=dist.ReduceOp.SUM)

        # Pad to max length for reduction
        max_len = max(quality_tensor.numel(), alignment_tensor.numel())
        pad = lambda x: torch.cat([x, torch.zeros(max_len - x.numel(), device=device)])
        quality_tensor = pad(quality_tensor)
        alignment_tensor = pad(alignment_tensor)

        dist.all_reduce(quality_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(alignment_tensor, op=dist.ReduceOp.SUM)

        avg_quality = (quality_tensor.sum() / metric_count_tensor).item()
        avg_alignment = (alignment_tensor.sum() / metric_count_tensor).item()
    else:
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        avg_alignment = sum(alignment_scores) / len(alignment_scores) if alignment_scores else 0.0

    avg_val_loss = (val_loss_tensor / num_batches_tensor).item()
    return {
        "val_loss": avg_val_loss,
        "avg_quality": avg_quality,
        "avg_alignment": avg_alignment,
        "n_samples": total_metric_samples,
    }

def sample_regression(
    model,
    dataloader,
    dataset,
    n_samples: int,
    seed: int,
    log_to_wandb: bool,
    wandb_title: str = 'Validation Samples',
    **kwargs
):
    model.eval()
    results = []
    
    logged_images = []
    for sample_idx in range(n_samples):
        # Assume each sample is (condition_img, gt_target_img)
        condition_img, gt_target_img = dataloader.dataset[sample_idx]

        # Add batch dim and move to device
        condition = condition_img.unsqueeze(0).to(model.device)      # [1, C, H, W]
        gt_target_img = gt_target_img.unsqueeze(0).to(model.device)  # [1, C, H, W]
        gt_target_img = dataset.to_image(gt_target_img)  # Convert to image format
        
        with torch.no_grad():
            # Use t=0 for regression
            t = torch.zeros(1, 1, device=model.device)
            pred_img = model(torch.zeros_like(gt_target_img), t, condition)
            pred_img = dataset.to_image(pred_img)  # Convert to image format

        results.append(pred_img)

        fig_or_path = plot_condition_vs_images_grid(
            condition_img,
            gt=gt_target_img,
            predictions=[pred_img],
            x_axis=["Prediction"],
            draw_condition_on_pred=True,
            title=f"Sample={sample_idx}"
        )
        if log_to_wandb:
            logged_images.append(wandb.Image(fig_or_path, caption=f"Sample={sample_idx}"))

    if log_to_wandb:
        wandb.log({wandb_title: logged_images})
    return results

def calc_regression_loss(
    pred_imgs,
    target_imgs,
    dataset,
    loss_func,
    piecewise_loss,
    **kwargs
):
    # Select PyTorch loss function
    if loss_func == "l2" or loss_func == "mse":
        criterion = nn.MSELoss(reduction='none')
    elif loss_func == "l1":
        criterion = nn.L1Loss(reduction='none')
    elif loss_func == "bce":
        # For binary cross entropy, convert to [0,1] range
        pred_imgs = torch.sigmoid(pred_imgs)
        target_imgs = (target_imgs + 1) / 2  # Convert from [-1,1] to [0,1]
        criterion = nn.BCELoss(reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_func}")

    # Compute per-element loss
    loss_per_elem = criterion(pred_imgs, target_imgs)
            
    if not piecewise_loss:
        return loss_per_elem.mean()

    # Piecewise: mask target vs background
    target_imgs_converted = dataset.to_image(target_imgs)
    target_mask = (target_imgs_converted == -1)

    loss_target = loss_per_elem[target_mask].mean()
    loss_background = loss_per_elem[~target_mask].mean()

    return 0.5 * loss_target + 0.5 * loss_background

def train_regression_model(
    config,
    model: UNet,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    dataset,
    optimizer,
    lr_scheduler,
    outputs_dir,
    grad_accum_steps=1,
    num_epochs=20,
    validation_config={},
    sampling_config={},
    train_viz_config={},
    loss_config={},
    checkpoint_every_n_epochs=10,
    log_every_n_steps=100,
    autocast_dtype='bfloat16',
    clip_grad_norm=False,
    ddp=False,
    master_process=True,
    ddp_world_size=1,
    clean_prev_checkpoints=True,
    device=None,
    **kwargs,
):
    train_step = 0
    dataloader_iterator = iter(train_dataloader)  # Manual reset avoids caching entire epoch like itertools.cycle

    for epoch in range(num_epochs):
        model.train()
        if isinstance(train_dataloader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        epoch_loss = 0.0
        num_steps_per_epoch = len(train_dataloader) // grad_accum_steps
      
        for step in range(num_steps_per_epoch):
            t0 = time.time()
            optimizer.zero_grad()
            step_loss = 0.0
            
            for micro_step in range(grad_accum_steps):
                try:
                    condition_imgs, target_imgs = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(train_dataloader)
                    condition_imgs, target_imgs = next(dataloader_iterator)
                condition_imgs = condition_imgs.to(device)
                target_imgs = target_imgs.to(device)

                # Use t=0 for regression (no diffusion process)
                t = torch.zeros(condition_imgs.size(0), 1, device=device)

                with torch.autocast(device_type=device, dtype=dtype_map[autocast_dtype]):
                    # Feed zero noise (regression mode) 
                    pred_imgs = model(torch.zeros_like(target_imgs), t, condition_imgs)
                    
                    loss = calc_regression_loss(
                        pred_imgs,
                        target_imgs,
                        dataset=dataset,
                        **loss_config,
                    )

                loss = loss / grad_accum_steps
                if ddp:
                    model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)  # Only sync gradients on the last micro-step
                loss.backward()
                step_loss += loss.detach()
                
            if ddp:
                dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
            grad_norm = None
            if clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
            optimizer.step()
            lr_scheduler.step()

            epoch_loss += step_loss.item()
            if master_process and (train_step + 1) % log_every_n_steps == 0:
                wandb.log(
                    {
                        "batch_loss": step_loss.item(),
                        "grad_norm": grad_norm,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                    },
                    step=train_step
                )

            if (train_step + 1) % train_viz_config['plot_every_n_steps'] == 0:
                image_pixel_space = dataset.to_image(target_imgs)
                train_fig = plot_training_visualization_grid(
                    condition_imgs,
                    target_imgs,
                    target_imgs,  # posterior same as target for regression
                    image_pixel_space,
                    pred_imgs.detach(),  # Use predictions as "eps" for visualization
                    pred_imgs.detach(),  # Use predictions as "x_t" for visualization  
                    t,
                    **train_viz_config,
                )
                if master_process:
                    wandb.log({
                        "Training Viz": wandb.Image(train_fig, caption=f"Train step {train_step}"),
                        }, step=train_step)
            train_step += 1
            torch.cuda.synchronize()  # Ensure all operations are complete before measuring time
            t1 = time.time()
            dt = (t1 - t0)
            imgs_per_sec = grad_accum_steps * condition_imgs.size(0) * ddp_world_size / dt
            if master_process:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}, lr: {lr_scheduler.get_last_lr()[0]:.6f}, Grad Norm: {grad_norm:.4f}, Time: {dt*1000:.2f} ms, Images/sec: {imgs_per_sec:.2f}")
                
        avg_loss = epoch_loss / len(train_dataloader)
        if master_process:
            wandb.log({"epoch": epoch + 1, "epoch_loss": avg_loss}, step=train_step)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % validation_config['validate_every_n_epochs'] == 0:
            logger.info(f"Validating model at epoch {epoch + 1}")
            res = validate_regression_model(
                config,
                model,
                val_dataloader,
                dataset,
                device=device,
                ddp=ddp,
                master_process=master_process,
                autocast_dtype=autocast_dtype,
                loss_config=loss_config,
                **validation_config,
            )
            avg_val_loss = res['val_loss']
            if master_process:
                wandb.log(res, step=train_step)
            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
            
        if master_process and (epoch + 1) % sampling_config['sample_every_n_epochs'] == 0:
            logger.info(f"Sampling images for epoch {epoch + 1}")
            sample_regression(
                model,
                val_dataloader,
                dataset,
                wandb_title="Validation Samples",
                **sampling_config
            )
            
            sample_regression(
                model,
                train_dataloader,
                dataset,
                wandb_title="Training Samples",
                **sampling_config
            )
            
        if master_process and (epoch + 1) % checkpoint_every_n_epochs == 0:
            model.eval()
            logger.info(f"Saving model checkpoint for epoch {epoch + 1}")
            checkpoints_dir = os.path.join(outputs_dir, "checkpoints")
            os.makedirs(checkpoints_dir, exist_ok=True)
            model_save_path = os.path.join(checkpoints_dir, f"checkpoint_{epoch + 1}.pth")
            torch.save({
                "state_dict": model.state_dict(),
                "config": config
            }, model_save_path)
            logger.info(f"Model saved to {model_save_path}")
            
            if clean_prev_checkpoints:
                # Remove older checkpoints
                for filename in os.listdir(checkpoints_dir):
                    if filename.startswith("checkpoint_") and filename != f"checkpoint_{epoch + 1}.pth":
                        os.remove(os.path.join(checkpoints_dir, filename))
                        logger.info(f"Removed old checkpoint: {filename}")
