import os
from loguru import logger
from omegaconf import OmegaConf
logger.info(
    "Environment Variables:\n" +
    "\n".join(f"{k}={v}" for k, v in os.environ.items())
)
import time

import torch
import cv2
import numpy as np
from math import ceil

from utils.seed import seed_everything
from data.Curves import CurveImageDataset
from utils.metrics import squareness_metric, alignment_metric, compute_alignment_score_from_box
from utils.viz import tensor_to_binary_mask
from utils.config_handler import remove_weight_prefixes
from utils.dataloader import get_dataset

from model.diffusion import UNet
from schedulers.ddim import DDIM
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from tqdm import tqdm
seed = 44
n_samples = 64
guidance_scale = 1
conds_to_generate = 64
show_gt = True
seed_everything(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_path = 'model/checkpoints/checkpoint_1000.pth' # W/distance maps
# checkpoint_path = 'model/checkpoints/checkpoint_640.pth'
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
config = checkpoint['config']

# Model
model = UNet(
    device=device,
    **config['model']
    )

state_dict = checkpoint["state_dict"]
state_dict = remove_weight_prefixes(state_dict)
model.load_state_dict(state_dict)
model.eval()

# Scheduler
scheduler = DDIM(device=device,
                    **config['scheduler'],)

# Dataset
dataset = get_dataset(config)

curve_imgs = []
gt_imgs = []
logger.info("Generating terminals...")
for i in tqdm(range(conds_to_generate)):
    curve_img, gt_img = dataset.generate_sample()
    curve_imgs.append(curve_img)
    gt_imgs.append(gt_img)

condition = torch.stack(curve_imgs, dim=0).to(device)

# Time how long it takes to sample
logger.info("Starting sampling...")
start_time = time.time()
pred_square_img = scheduler.sample(
    model,
    n_samples=n_samples,
    condition=condition,
    guidance_scale=guidance_scale,
    seed=seed,
    return_dict=True,
)['x_0'] # B X 1 X H X W

end_time = time.time()
logger.info(f"Sampling took {end_time - start_time:.2f} seconds for {n_samples} samples.")


scale_factor = 2

# Background
height, width = (dataset.image_size, dataset.image_size)
background_rgb = np.ones((height, width, 3), dtype=np.uint8) * 255
blue_overlay = np.full_like(background_rgb, (173, 216, 230), dtype=np.uint8)
alpha_bg = 0.3
alpha_red = 0.65  # stronger red
background_rgb = cv2.addWeighted(background_rgb, 1 - alpha_bg, blue_overlay, alpha_bg, 0)

# Grid dims: if show_gt -> double the columns (GT|Pred for each sample)
num_samples = pred_square_img.shape[0]
grid_cols = int(n_samples**0.5)  # e.g., 8 when n_samples=64
grid_rows = ceil(num_samples / grid_cols)
pair_factor = 2 if show_gt else 1

fig, axs = plt.subplots(
    grid_rows, grid_cols * pair_factor,
    figsize=(4 * grid_cols * pair_factor, 4 * grid_rows)
)
axs = np.atleast_2d(axs)

# --- Column headers only on the first row ---
if show_gt:
    for col in range(grid_cols):
        axs[0, col * 2].set_title("GT", fontsize=14, pad=10)
        axs[0, col * 2 + 1].set_title("Pred", fontsize=14, pad=10)
else:
    for col in range(grid_cols):
        axs[0, col].set_title("Pred", fontsize=14, pad=10)
        
for idx in range(num_samples):
    row = idx // grid_cols
    base_col = (idx % grid_cols) * pair_factor
    cond_idx = idx // (n_samples // conds_to_generate)

    curve_binary = tensor_to_binary_mask(curve_imgs[cond_idx])
    gt_binary = tensor_to_binary_mask(dataset.to_image(gt_imgs[cond_idx]))

    # --- GT panel (left) ---
    if show_gt:
        ax_gt = axs[row, base_col]
        overlay_gt = background_rgb.copy()

        # draw GT square in black
        overlay_gt[gt_binary > 0] = [0, 0, 255]
        # draw curve in black
        overlay_gt[curve_binary > 0] = [0, 0, 0]

        overlay_gt_upsampled = cv2.resize(
            overlay_gt, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
        )
        ax_gt.imshow(overlay_gt_upsampled)
        ax_gt.axis("off")


    # --- Pred panel (right or only) ---
    ax_pred = axs[row, base_col + (1 if show_gt else 0)]
    pred = dataset.to_image(pred_square_img[idx])
    square_binary = tensor_to_binary_mask(pred)

    overlay = background_rgb.copy().astype(np.float32)
    red_layer = np.zeros_like(overlay, dtype=np.float32)
    red_layer[square_binary > 0] = [255, 0, 0]
    overlay[square_binary > 0] = (
        alpha_red * red_layer[square_binary > 0] +
        (1 - alpha_red) * overlay[square_binary > 0]
    )
    overlay[curve_binary > 0] = [0, 0, 0]

    overlay = overlay.astype(np.uint8)
    overlay_upsampled = cv2.resize(
        overlay, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
    )
    ax_pred.imshow(overlay_upsampled)
    ax_pred.axis("off")

# Hide unused axes
total_cells = grid_rows * grid_cols * pair_factor
start = (num_samples * pair_factor) if show_gt else num_samples
for flat_idx in range(start, total_cells):
    r = flat_idx // (grid_cols * pair_factor)
    c = flat_idx %  (grid_cols * pair_factor)
    axs[r, c].axis("off")

plt.tight_layout()
plt.savefig('grid_steiner_overlay.png', dpi=600)
plt.show()
logger.info("Sampled images saved to 'grid_steiner_overlay.png'.")