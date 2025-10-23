
import numpy as np
import cv2
from typing import List
import os
import cv2
import numpy as np
from pathlib import Path

def squareness_metric(mask: np.ndarray, type='squareness') -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)  # center, (w, h), angle
    (w, h) = rect[1]
    if w == 0 or h == 0:
        return 0.0

    box = cv2.boxPoints(rect).astype(np.int32)  # Convert to int for drawing

    # Create a filled mask of the rotated rectangle
    rect_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(rect_mask, box, 255)

    # Compute IoU
    intersection = np.logical_and(mask > 0, rect_mask > 0).sum()
    union = np.logical_or(mask > 0, rect_mask > 0).sum()
    if union == 0:
        return 0.0
    iou = intersection / union

    # Penalize for deviation from square (aspect ratio)
    aspect_ratio = max(w, h) / min(w, h)
    square_penalty = np.exp(-abs(aspect_ratio - 1) * 2)  # lower if not square

    rectangleness = iou
    squareness = rectangleness * square_penalty
    if type == 'squareness':
        return float(squareness)
    elif type == 'rectangleness':
        return float(rectangleness)
    else:
        raise ValueError(f"Unknown squareness type: {type}")



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
    score = float(np.exp(-avg_dist / decay_scale))
    # score = -avg_dist
    return score, distances, sampled_points, avg_dist


def alignment_metric(square_mask, curve_mask, decay_scale=7.5, debug_path=None, idx=None):
    """
    Computes alignment score using the distance transform, where distances are sampled from the closest
    real pixel in the predicted mask (not directly from the rotated rectangle corners).
    
    Saves an annotated debug image if requested.

    Args:
        square_mask (np.ndarray): predicted binary mask (uint8)
        curve_mask (np.ndarray): ground-truth binary mask (uint8)
        debug_path (str or Path): directory to save debug images
        idx (int): sample index (used for filenames)

    Returns:
        float: alignment score
    """
    dist_transform = cv2.distanceTransform(255 - curve_mask, cv2.DIST_L2, 5)

    contours, _ = cv2.findContours(square_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    # For each corner, find the closest real pixel in the square mask
    score, distances, sampled_points, _ = compute_alignment_score_from_box(
        box, square_mask, dist_transform, decay_scale
    )


    # Save debug images if requested
    if debug_path is not None and idx is not None:
        debug_dir = Path(debug_path)
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Distance transform visualization
        dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        dist_vis_color = cv2.applyColorMap(dist_vis, cv2.COLORMAP_JET)
        overlay = dist_vis_color.copy()

        # Draw original rectangle
        cv2.polylines(overlay, [box], isClosed=True, color=(0, 255, 0), thickness=2)

        # Annotate snapped real pixels and their distances
        for i, ((x, y), d) in enumerate(zip(sampled_points, distances)):
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)  # Yellow dot
            cv2.putText(
                overlay,
                f"{d:.1f}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA
            )

        cv2.imwrite(str(debug_dir / f"alignment_debug_{idx}.png"), overlay)

    return score
