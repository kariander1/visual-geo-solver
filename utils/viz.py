import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from matplotlib.patches import Rectangle
import math
import matplotlib.pyplot as plt
from typing import List, Union
from typing import Optional
from shapely.geometry import LineString
from loguru import logger

def reconstruct_square_corners(square_tensor: torch.Tensor):
    """
    Reconstruct the 4 corners of a square from its parameters: [cx, cy, side, angle]
    """
    cx, cy, side, angle = square_tensor.tolist()
    half = side / 2
    corners = np.array([
        [-half, -half],
        [-half, half],
        [half,  half],
        [half, -half]
    ])
    rot_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    rotated = corners @ rot_matrix.T + np.array([cx, cy])
    return rotated

def to_pixel_coords(coords, image_size):
    coords = (coords + 1) / 2 * (image_size - 1)
    return coords.round().astype(np.int32)



def draw_square(
    img,
    square_tensor,
    image_size,
    thickness=3,
    fill_color=0,
    fill_shape=False
):
    if square_tensor.shape != (4,2):
        square_corners = reconstruct_square_corners(square_tensor)
    else:
        square_corners = square_tensor.numpy()
    square_px = to_pixel_coords(square_corners, image_size)

    if fill_shape:
        # Draw filled polygon (interior)
        cv2.fillPoly(img, [square_px], color=fill_color)
    else:
        # Draw just the outline
        cv2.polylines(img, [square_px], isClosed=True, color=fill_color, thickness=thickness)
        
def draw_ellipse(
    img,
    ellipse_tensor,
    image_size,
    thickness=3,
    fill_color=0):
    cx, cy, a, b, angle = ellipse_tensor.tolist()
    center_px = to_pixel_coords(np.array([cx, cy]), image_size)
    axes_px = (int(a * (image_size - 1) / 2), int(b * (image_size - 1) / 2))
    angle_deg = np.degrees(angle)
    cv2.ellipse(img, center=tuple(center_px), axes=axes_px, angle=angle_deg,
                startAngle=0, endAngle=360, color=fill_color, thickness=thickness)




def plot_training_visualization_grid(
    curve_imgs: torch.Tensor,
    square_imgs: torch.Tensor,
    posterior: torch.Tensor,
    image_pixel_space: torch.Tensor,
    eps: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    num_samples_to_plot: int = 5,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    curve_alpha: float = 0.7,
    **kwargs
):
    """
    Efficient: Visualize selected training samples without converting entire batch.
    """
    B = curve_imgs.size(0)
    num_samples_to_plot = min(num_samples_to_plot, B)

    titles = ["Data", "Condition", "Input", "x₀", "Image (Pixel Space)", "Epsilon", "x_t"]

    fig, axes = plt.subplots(
        nrows=num_samples_to_plot,
        ncols=len(titles),
        figsize=(len(titles) * 2.5, num_samples_to_plot * 2.5),
    )

    if num_samples_to_plot == 1:
        axes = axes[np.newaxis, :]  # Ensure 2D shape

    for row in range(num_samples_to_plot):
        # Get individual tensors
        curve = curve_imgs[row]
        square = square_imgs[row]
        x0 = posterior[row]
        noise = eps[row]
        xt = x_t[row]
        image_pixel = image_pixel_space[row]

        # Convert to uint8 images
        curve_np = tensor_to_np_img(curve)
        square_np = tensor_to_np_img(square)
        x0_np = tensor_to_np_img(x0)
        image_np = tensor_to_np_img(image_pixel)
        eps_np = tensor_to_np_img(noise)
        xt_np = tensor_to_np_img(xt)

        # Combined overlays
        data_img = combine_curve_and_square_images(square_np, curve_np, curve_alpha)
        red_curve_img = combine_curve_and_square_images(np.ones_like(square_np) * 255, curve_np, curve_alpha=1.0)

        # Collect for display
        images = [data_img, red_curve_img, square_np, x0_np, image_np, eps_np, xt_np]

        for col, img in enumerate(images):
            ax = axes[row, col]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col], fontsize=12)

        # Add timestep annotation
        timestep = t[row].item() if t.ndim == 1 else t[row, 0].item()
        axes[row, 0].text(5, 10, f"t = {timestep}", color="black", fontsize=10, ha='left', va='top')

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
        logger.info(f"Saved training viz to {save_path}")
        plt.close(fig)
        return save_path
    else:
        plt.close(fig)
        return fig

def combine_curve_and_square_images(
    square_np: np.ndarray,
    curve_np: np.ndarray,
    curve_alpha: float = 0.7,
) -> np.ndarray:
    """
    Combine square and curve images (single or batch) into RGB blended images.

    Args:
        square_np (np.ndarray): [H, W] or [B, H, W] grayscale square image(s).
        curve_np (np.ndarray): [H, W] or [B, H, W] curve mask(s) (black = curve).
        curve_alpha (float): Alpha blending factor for the curve overlay.

    Returns:
        np.ndarray: Blended RGB image(s), shape [H, W, 3] or [B, H, W, 3].
    """
    single_image = False
    if square_np.ndim == 2:
        square_np = square_np[None, ...]  # → [1, H, W]
        curve_np = curve_np[None, ...]
        single_image = True
    elif square_np.ndim != 3 or curve_np.ndim != 3:
        raise ValueError("Inputs must be [H, W] or [B, H, W] shaped arrays.")

    B, H, W = square_np.shape

    # Convert grayscale to RGB
    square_rgb = np.stack([square_np] * 3, axis=-1).astype(np.uint8)  # [B, H, W, 3]

    # Create red overlay where curve == 0
    overlay = square_rgb.copy()
    red_color = np.array([255, 0, 0], dtype=np.uint8)
    mask = curve_np == 0  # [B, H, W]
    overlay[mask] = red_color

    # Alpha blend
    blended = ((1 - curve_alpha) * square_rgb + curve_alpha * overlay).astype(np.uint8)

    return blended[0] if single_image else blended


def tensor_to_np_img(tensor_img: torch.Tensor) -> np.ndarray:
    """
    Vectorized: Convert tensor(s) to uint8 NumPy image(s), scaled to [0, 255].

    Supports:
    - [C, H, W] → [H, W] or [H, W, 3]
    - [B, C, H, W] → [B, H, W] or [B, H, W, 3]
    """
    tensor_img = tensor_img.detach().cpu()

    # Handle non-batched case
    if tensor_img.ndim == 3:
        tensor_img = tensor_img.unsqueeze(0)  # → [1, C, H, W]

    B, C, H, W = tensor_img.shape

    # Flatten per-sample for min/max: [B, C*H*W]
    flat = tensor_img.view(B, -1)
    min_vals = flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_vals = flat.max(dim=1)[0].view(B, 1, 1, 1)

    denom = (max_vals - min_vals).clamp(min=1e-8)
    norm = (tensor_img - min_vals) / denom  # → [0, 1]
    scaled = (norm * 255).clamp(0, 255).to(torch.uint8)

    if C == 1:
        out = scaled[:, 0]  # → [B, H, W]
    elif C == 3:
        out = scaled.permute(0, 2, 3, 1)  # → [B, H, W, 3]
    else:
        raise ValueError(f"Unsupported channel size: {C}")

    out_np = out.numpy()
    return out_np[0] if out_np.shape[0] == 1 else out_np


def plot_sample_img_grid(dataset, grid_size=5, save_path=None, curve_alpha=0.7):
    """
    Plot a grid of combined square and curve images using OpenCV blending.
    
    Assumes dataset[i] returns: (curve_img, square_img) as [1, H, W] tensors.
    Curve will be overlaid as red on top of the grayscale square image.
    """


    fig, axes = plt.subplots(grid_size, grid_size*2, figsize=(grid_size*2, grid_size))
    image_size = dataset.image_size

    for i in range(grid_size):
        for j in range(0,grid_size*2,2):
            idx = i * grid_size + j//2
            curve_img, square_img = dataset[idx]

            # Convert tensors to [0, 255] NumPy arrays
            square_pixel = dataset.to_image(square_img)
            square_pixel_np = tensor_to_np_img(square_pixel)
            square_np = tensor_to_np_img(square_img)
            curve_np = tensor_to_np_img(curve_img)

            combined_img = combine_curve_and_square_images(
                square_np,
                curve_np,
                curve_alpha
            )

            combined_img_pixel = combine_curve_and_square_images(
                square_pixel_np,
                curve_np,
                curve_alpha
            )
            
            ax = axes[i, j]
            ax.imshow(combined_img)
            rect = Rectangle((0, 0), image_size - 1, image_size - 1,
                             linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            ax.axis("off")

            ax = axes[i, j+1]
            ax.imshow(combined_img_pixel)
            rect = Rectangle((0, 0), image_size - 1, image_size - 1,
                             linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            ax.axis("off")
            
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Grid saved to {save_path}")
    else:
        plt.show(block=True)
    plt.close()
    
def plot_condition_vs_images_grid(
    condition_img: torch.Tensor,
    predictions: List[torch.Tensor],
    x_axis: List[float],
    save_path: str = None,
    draw_condition_on_pred: bool = False,
    title: str = None,
    column_prefix: str = "CFG=",
    gt=None
):
    """
    Plot a condition image above a grid of generated images with column and row headers.

    Args:
        condition_img (Tensor): (1, C, H, W) or (C, H, W) tensor
        predictions (List[Tensor]): list of tensors of shape (n_seeds, C, H, W), one per guidance scale
        guidance_scale (List[float]): list of guidance scales (should match len(predictions))
        save_path (str): optional, path to save the image
        title (str): optional figure title
    """

    if condition_img.dim() == 4:
        condition_img = condition_img[0]

    condition_img_np = tensor_to_np_img(condition_img)
    condition_img_np_red = combine_curve_and_square_images(
        np.ones_like(condition_img_np) * 255,  # Red overlay
        condition_img_np,
        curve_alpha=1.0
    )
    if gt is not None and gt.dim() == 4:
        gt = gt[0]
        
    gt_img_np = tensor_to_np_img(gt) if gt is not None else None
    gt_img_np = combine_curve_and_square_images(
        gt_img_np,  # Red overlay
        condition_img_np,
    ) if gt_img_np is not None else None
    n_rows = predictions[0].shape[0]
    n_cols = len(predictions)
    fig_height = 1 + n_rows  # 1 row for condition
    fig_width = n_cols + 1   # +1 for row headers

    fig, axes = plt.subplots(fig_height, fig_width, figsize=(fig_width * 2, fig_height * 2))
    axes = np.array(axes).reshape(fig_height, fig_width)

    # --- Top row: Column headers with guidance scale ---
    for col in range(n_cols):
        ax = axes[0, col + 1]  # +1 because col 0 is for row headers
        ax.imshow(condition_img_np_red)
        ax.set_title(f"{column_prefix} {str(x_axis[col])}", fontsize=10)
        ax.axis("off")

    if gt is not None:
        # Add ground truth image in the first column of the top row
        axes[0, 0].imshow(gt_img_np, cmap='gray')
        axes[0, 0].set_title("Ground Truth", fontsize=10)
        axes[0, 0].axis("off")
    # Empty corner cell
    axes[0, 0].axis("off")

    # --- Main grid with row headers ---
    for row in range(n_rows):
        # Row header (leftmost column)
        ax = axes[row + 1, 0]
        ax.text(0.5, 0.5, f"Seed {row}", ha="center", va="center", fontsize=10)
        ax.axis("off")

        for col in range(n_cols):
            ax = axes[row + 1, col + 1]
            # ax.imshow(to_np_img(predictions[col][row]), cmap='gray', vmin=0, vmax=1)
            pred_img = tensor_to_np_img(predictions[col][row])
            if draw_condition_on_pred:
                combined_img = combine_curve_and_square_images(
                    pred_img,
                    condition_img_np
                )
                ax.imshow(combined_img, cmap='gray', vmin=0, vmax=1)
            else:
                ax.imshow(pred_img, cmap='gray', vmin=0, vmax=1)
            ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
        logger.info(f"Saved image grid to: {save_path}")
        plt.close(fig)
        return save_path
    else:
        plt.close(fig)
        return fig

def is_self_intersecting(x, y):
    curve = LineString(np.column_stack([x, y]))
    return not curve.is_simple


def tensor_to_binary_mask(tensor_img, threshold=127):
    img_np = tensor_img.detach().float().cpu().numpy()
    if img_np.ndim == 3:
        img_np = img_np[0]
    elif img_np.ndim == 4:
        img_np = img_np[0, 0]  # for batch shape [1, 1, H, W]

    img_min, img_max = img_np.min(), img_np.max()
    if img_max == img_min:
        img_uint8 = np.zeros_like(img_np, dtype=np.uint8)
    else:
        img_scaled = (img_np - img_min) / (img_max - img_min)
        img_uint8 = (img_scaled * 255).astype(np.uint8)
    _, binary = cv2.threshold(img_uint8, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary