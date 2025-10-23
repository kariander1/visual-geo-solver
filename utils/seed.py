import torch
import numpy as np

def seed_everything(seed: int = 42):
    """
    Seed everything for reproducibility.
    
    Args:
        seed
    """    
    torch.manual_seed(seed)
    np.random.seed(seed)
    