
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so absolute imports like `utils.*` work
# when running this file directly (e.g., `uv run scripts/evaluate_steiner.py`).
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import matplotlib.patches as patches

import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.dataloader import get_dataset
from data.Curves import CurveImageDataset
import cv2
import torch
import numpy as np

from utils.config_handler import remove_weight_prefixes
from scipy.ndimage import binary_erosion

from model.diffusion import UNet
from schedulers.ddim import DDIM
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from torch.utils.data import DataLoader, random_split


def _to_numpy01(img):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    return (img < 0)   # threshold at 0


def get_mask_edge(square_binary: np.ndarray, thickness: int = 1) -> np.ndarray:
    sb = square_binary.astype(bool)
    eroded = binary_erosion(sb)
    base_edge = sb & (~eroded)          # 1-px edge
    dilated = binary_dilation(base_edge, iterations=thickness-1)
    return dilated


def _to_numpy255(img):
    return (_to_numpy01(img) * 255).astype(np.uint8)

def compute_alignment_score_from_box(box, square_mask, dist_transform, decay_scale=7.5):
    """
    Computes alignment score between box corners and curve using distance transform.

    Args:
        box (np.ndarray): 4x2 array of box corners (float or int)
        square_mask (np.ndarray): binary mask from which to find closest real pixels
        dist_transform (np.ndarray): precomputed distance transform from curve mask
        decay_scale (float): decay factor for exponential decay

    Returns:
        float: alignment score (higher is better)
        list: list of distances used for scoring
        list: list of (x, y) points sampled in the mask
    """
    if not np.any(square_mask):
        return 0.0, [], []

    # Extract foreground mask points
    mask_pts = np.column_stack(np.where(square_mask > 0))[:, [1, 0]]  # (x, y) format

    distances = []
    sampled_points = []
    for corner in box:
        dists = np.linalg.norm(mask_pts - corner, axis=1)
        closest = mask_pts[np.argmin(dists)]
        x, y = int(closest[0]), int(closest[1])
        distances.append(dist_transform[y, x])
        sampled_points.append((x, y))

    avg_dist = np.mean(distances)
    # score = float(np.exp(-avg_dist / decay_scale))
    score = -avg_dist

    return score, distances, sampled_points, avg_dist

import numpy as np
import cv2
import argparse



def squareness_metric_4_polygon_fit(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)

    # Try polygon approximation first
    epsilon = 0.02 * cv2.arcLength(cnt, True)  # tolerance = 2% of perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        # Found a quadrilateral
        box = approx.reshape(-1, 2)
        rect_area = cv2.contourArea(box)
    else:
        # Fallback to minAreaRect
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            return 0.0
        box = cv2.boxPoints(rect)
        rect_area = w * h

    # Compute overlap: area of mask vs area of polygon
    mask_area = cv2.contourArea(cnt)
    ratio = mask_area / rect_area if rect_area > 0 else 0

    # Aspect ratio penalty
    if len(approx) == 4:
        # For quadrilateral: compute side lengths
        edges = [np.linalg.norm(box[i] - box[(i+1)%4]) for i in range(4)]
        aspect = max(edges) / min(edges) if min(edges) > 0 else np.inf
    else:
        # Fallback: use rect sides
        (w, h) = rect[1]
        aspect = max(w, h) / min(w, h) if min(w, h) > 0 else np.inf

    penalty = np.exp(-abs(aspect - 1) * 2)

    return float(ratio * penalty)


def squareness_metric_moments(mask: np.ndarray) -> float:
    """
    Compute a rotation- and translation-invariant squareness score
    based on the second-order central image moments.

    Args:
        mask (np.ndarray): Binary mask (nonzero pixels = shape)

    Returns:
        float: Squareness score in [0,1], where 1 = perfect square/circle.
    """
    # Compute raw image moments
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:  # empty mask
        return 0.0

    # Normalize second-order central moments -> covariance of shape
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    cov = np.array([[mu20, mu11], [mu11, mu02]])

    # Eigenvalues = variances along principal axes
    eigvals, _ = np.linalg.eigh(cov)
    if np.min(eigvals) <= 0:
        return 0.0

    ratio = eigvals.max() / eigvals.min()

    # Squareness penalty: 1 if ratio=1, decays as ratio moves away from 1
    score = np.exp(-2 * abs(ratio - 1))

    return float(score)

def squareness_metric(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    
    cnt = max(contours, key=cv2.contourArea)

    # Compute min area rectangle (rotation invariant)
    rect = cv2.minAreaRect(cnt)
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return 0.0
    
    area = cv2.contourArea(cnt)
    rect_area = w * h
    ratio = area / rect_area
    
    # Aspect ratio penalty
    aspect = max(w, h) / min(w, h)
    penalty = np.exp(-abs(aspect - 1)*2)
    
    return float(ratio * penalty)

    
def snap_square_by_rigid_transform(
    square_mask, curve_mask,
    angle_range=(-15, 15), angle_step=0.5,
    trans_step=2, trans_range=5,
    lambda_reg: float = 0.0,
):
    """
    Snap the square to the curve using rotation + translation (rigid transform),
    with a single penalization parameter lambda_reg for larger transforms:
        penalty = lambda_reg * ( ||t||^2 + (L * theta)^2 )
    where theta is in radians and L is half the square's diagonal (in pixels).
    """
    square_mask = _to_numpy255(square_mask)
    curve_mask  = _to_numpy255(curve_mask)
    h, w = square_mask.shape
    if not np.any(square_mask):
        return square_mask.copy()

    # --- Distance transform from the curve (fixed) ---
    dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

    # --- Compute square center & characteristic length L from original mask ---
    ys, xs = np.where(square_mask > 0)
    center = np.mean(np.stack([xs, ys], axis=1), axis=0)  # (cx, cy)

    # Get the square's side estimate from min-area rect of the original mask
    contours, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt0 = max(contours, key=cv2.contourArea)
        rect0 = cv2.minAreaRect(cnt0)       # ((cx, cy), (w_side, h_side), angle)
        (w_side, h_side) = rect0[1]
        s = 0.5 * (w_side + h_side)         # average side length estimate
        # Half the diagonal:
        L = (s * np.sqrt(2)) / 2.0 if s > 0 else 1.0
    else:
        # Fallback: infer s from area (assuming roughly square)
        area = float(np.count_nonzero(square_mask))
        s = np.sqrt(area) if area > 0 else 1.0
        L = (s * np.sqrt(2)) / 2.0

    best_score = -np.inf
    best_mask = square_mask.copy()

    num_steps = int(np.floor((angle_range[1] - angle_range[0]) / angle_step)) + 1
    for angle_deg in np.linspace(angle_range[0], angle_range[1], num_steps):
        theta = np.deg2rad(angle_deg)  # radians for the rotation term
        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle_deg, 1.0)

        for dx in range(-trans_range, trans_range + 1, trans_step):
            for dy in range(-trans_range, trans_range + 1, trans_step):
                # Compose rotation + translation
                M = rot_mat.copy()
                M[0, 2] += dx
                M[1, 2] += dy

                transformed_mask = cv2.warpAffine(square_mask, M, (w, h), flags=cv2.INTER_NEAREST)

                contours_t, _ = cv2.findContours(transformed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours_t:
                    continue
                cnt = max(contours_t, key=cv2.contourArea)
                epsilon = 0.02 * cv2.arcLength(cnt, True)  # tolerance factor (2% of perimeter)
                approx = cv2.approxPolyDP(cnt, epsilon, True)

                if len(approx) == 4:
                    box = approx.reshape(-1, 2)
                else:
                    # fallback: still use rectangle if not 4-sided
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)

                # Alignment score (your existing logic)
                align_score, _, _, _ = compute_alignment_score_from_box(box, transformed_mask, dist_transform)
                quality_score = squareness_metric(transformed_mask)
                # --- Single-parameter transform penalty (option #2) ---
                trans_norm_sq = float(dx*dx + dy*dy)          # ||t||^2 in pixels^2
                rot_term_sq   = float((L * theta) * (L * theta))  # (L*theta)^2
                penalty = lambda_reg * (trans_norm_sq + rot_term_sq)

                total_score = align_score - penalty

                if total_score > best_score:
                    best_score = total_score
                    best_mask = transformed_mask.copy()

    return best_mask, best_score


def get_box(square_mask: np.ndarray) -> np.ndarray:
    contours_t, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_t:
        None
    cnt = max(contours_t, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)  # tolerance factor (2% of perimeter)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        box = approx.reshape(-1, 2)
    else:
        # fallback: still use rectangle if not 4-sided
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
    return box

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate square generation model')
    parser.add_argument('--checkpoint', type=str,
                        default='model/checkpoints_curves/checkpoint_520.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers for data loading')
    parser.add_argument('--no-snap', action='store_true', help='Disable snapping squares to curves')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    w_snap = not args.no_snap
    seed = args.seed
    image_size = 128
    output_dir = args.output_dir

    batch_size = args.batch_size
    num_workers = args.num_workers

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    # n_squares_per_curve = 3
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint
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


    dataset = get_dataset(config)

    val_fraction = config['dataset']['val_fraction']

    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


    import pandas as pd

    results = []

    for curve_imgs, square_gt_imgs in tqdm(val_dataloader):

        pred_square_img = scheduler.sample(
            model,
            n_samples=len(square_gt_imgs),
            condition=curve_imgs,
            guidance_scale=1,
            seed=seed,
            return_dict=True,
        )['x_0']  # B X 1 X H X W

        for idx, (sq_gt, sq, crv) in enumerate(zip(list(square_gt_imgs), list(pred_square_img), list(curve_imgs))):

            if w_snap:
                square_mask, _ = snap_square_by_rigid_transform(sq, crv)
            else:
                square_mask = _to_numpy255(sq)
            square_gt_mask = _to_numpy255(sq_gt)
            curve_mask  = _to_numpy255(crv)
            dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

            box    = get_box(square_mask)
            box_gt = get_box(square_gt_mask)
            if box is None or box_gt is None:
                continue

            # Distances
            _, _, _, avg_dist    = compute_alignment_score_from_box(box, square_mask, dist_transform)
            _, _, _, avg_dist_gt = compute_alignment_score_from_box(box_gt, square_gt_mask, dist_transform)

            # Prediction metrics
            quality_score_rect    = squareness_metric(square_mask)
            quality_score_moments = squareness_metric_moments(square_mask)
            quality_score_polygon = squareness_metric_4_polygon_fit(square_mask)

            # GT metrics
            quality_score_gt_rect    = squareness_metric(square_gt_mask)
            quality_score_gt_moments = squareness_metric_moments(square_gt_mask)
            quality_score_gt_polygon = squareness_metric_4_polygon_fit(square_gt_mask)

            # Collect all results per sample
            results.append({
                "sample_id": idx,
                "avg_dist_pred": avg_dist,
                "avg_dist_gt": avg_dist_gt,
                "squareness_rect_pred": quality_score_rect,
                "squareness_moments_pred": quality_score_moments,
                "squareness_polygon_pred": quality_score_polygon,
                "squareness_rect_gt": quality_score_gt_rect,
                "squareness_moments_gt": quality_score_gt_moments,
                "squareness_polygon_gt": quality_score_gt_polygon,
            })

    # Save all per-sample results into CSV
    df = pd.DataFrame(results)
    out_filename = "eval_squares_snapped_results.csv" if w_snap else "eval_squares_results.csv"
    out_path = os.path.join(output_dir, out_filename)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} samples to {out_path}")
    print(df.describe())
