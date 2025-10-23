import os
import wandb
from omegaconf import OmegaConf
def init_wandb(
    config,
    outputs_dir,
    project,
    entity,
    mode,
    api_key=None):
    """
    Initialize Weights & Biases for tracking experiments.

    Args:
        num_epochs (int): Number of epochs for training.
        p_null (float): Probability of null condition.
        lambda_eps (float): Weight for epsilon loss.
        optimizer (torch.optim.Optimizer): Optimizer used in training.
    """
    if "WANDB_API_KEY" not in os.environ and api_key is not None and api_key != "":
        os.environ["WANDB_API_KEY"] = api_key

    config = OmegaConf.to_container(config, resolve=True)
    dir_name = os.path.basename(outputs_dir)
    wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        name=dir_name,
        config=config
    )
    