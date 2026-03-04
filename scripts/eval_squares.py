
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so absolute imports work
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, random_split

from utils.dataloader import get_dataset
from utils.config_handler import remove_weight_prefixes
from model.diffusion import UNet

# Reuse geometry helpers from eval_squares
from scripts.eval_squares import (
    _to_numpy01,
    _to_numpy255,
    get_box,
    squareness_metric,
    snap_square_by_rigid_transform,
    compute_alignment_score_from_box,
    _largest_component_pts,
    _min_area_rect,
    _box_points,
)


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def parametric_to_pixel(pts, image_size):
    """Convert normalised [-1, 1] coordinates to pixel coordinates."""
    return pts * (image_size // 2) + image_size // 2


# ---------------------------------------------------------------------------
# True parametric helpers (no discretisation)
# ---------------------------------------------------------------------------

def point_to_polyline_distance(point, polyline):
    """Minimum distance from a point to a closed polyline.

    Args:
        point: (2,) array.
        polyline: (N, 2) array — vertices of a closed polyline.

    Returns:
        float: minimum Euclidean distance.
    """
    p1 = polyline                          # (N, 2)
    p2 = np.roll(polyline, -1, axis=0)     # (N, 2)

    d = p2 - p1                            # segment vectors
    f = point - p1                          # vectors from p1 to query

    dd = np.sum(d * d, axis=1)             # |segment|²
    t = np.sum(f * d, axis=1) / (dd + 1e-12)
    t = np.clip(t, 0, 1)

    closest = p1 + t[:, None] * d
    dists = np.linalg.norm(point - closest, axis=1)
    return float(np.min(dists))


def alignment_parametric(corners, curve_pts):
    """Mean min-distance from 4 corners to a continuous curve polyline.

    Both inputs should be in the same coordinate space (either normalised
    or pixel — doesn't matter as long as they match).

    Returns:
        float: mean distance in the given coordinate units.
    """
    dists = [point_to_polyline_distance(c, curve_pts) for c in corners]
    return float(np.mean(dists))


import cv2


def squareness_continuous(corners):
    """Squareness metric from the paper, computed on float corner coordinates.

    Q(S) = (area / (w*h)) * exp(-2 * |max(w,h)/min(w,h) - 1|)

    Args:
        corners: (4, 2) float array.

    Returns:
        float: squareness score in [0, 1].  1 = perfect square.
    """
    # Area via shoelace
    x, y = corners[:, 0], corners[:, 1]
    A = 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))
    # Min-area bounding rect
    rect = cv2.minAreaRect(corners.astype(np.float32))
    w, h = rect[1]
    if w * h == 0 or min(w, h) == 0:
        return 0.0
    fill = A / (w * h)
    aspect = max(w, h) / min(w, h)
    return float(fill * np.exp(-2 * abs(aspect - 1)))


# ---------------------------------------------------------------------------
# Bilinear DT helpers (semi-parametric — still uses rasterised DT)
# ---------------------------------------------------------------------------

def _bilinear_interp(img, x, y):
    """Bilinear interpolation of *img* at float coordinates (x, y)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    H, W = img.shape
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H - 1)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    dx = x - x0
    dy = y - y0
    return (img[y0, x0] * (1 - dx) * (1 - dy) +
            img[y0, x1] * dx * (1 - dy) +
            img[y1, x0] * (1 - dx) * dy +
            img[y1, x1] * dx * dy)


def alignment_bilinear_dt(corners, dist_transform):
    """Mean bilinear-interpolated DT value at 4 float corners."""
    dists = _bilinear_interp(dist_transform, corners[:, 0], corners[:, 1])
    return float(np.mean(dists))


def alignment_pixel(corners, dist_transform):
    """Integer-rounded DT lookup at 4 corners (pixel-space alignment)."""
    H, W = dist_transform.shape
    ix = np.clip(np.rint(corners[:, 0]).astype(int), 0, W - 1)
    iy = np.clip(np.rint(corners[:, 1]).astype(int), 0, H - 1)
    return float(dist_transform[iy, ix].mean())


# ---------------------------------------------------------------------------
# Parametric snap (operates on float corners against the continuous curve)
# ---------------------------------------------------------------------------

def snap_parametric(corners, curve_pts_px, n_candidates=40000,
                    angle_range=(-15, 15), trans_range=12, rng=None):
    """Parametric snap: search over random rigid transforms.

    Score = –mean(nearest-vertex distance) for each candidate.
    Uses KDTree for fast nearest-vertex lookup to rank candidates,
    then returns exact polyline alignment for the best one.

    Args:
        corners: (4, 2) float array — corner positions in pixel coords.
        curve_pts_px: (N, 2) float array — curve polyline in pixel coords.
        n_candidates: number of random (angle, dx, dy) to try.
        angle_range: (min_deg, max_deg).
        trans_range: max absolute translation in pixels.
        rng: numpy Generator.

    Returns:
        best_corners: (4, 2) float array.
        best_score: float (negative mean distance; higher = better).
    """
    from scipy.spatial import cKDTree

    if rng is None:
        rng = np.random.default_rng()

    cx, cy = corners.mean(axis=0)

    # Include identity as first candidate
    N_total = n_candidates + 1
    angles_deg = np.empty(N_total)
    dxs = np.empty(N_total)
    dys = np.empty(N_total)
    angles_deg[0], dxs[0], dys[0] = 0.0, 0.0, 0.0
    angles_deg[1:] = rng.uniform(angle_range[0], angle_range[1], size=n_candidates)
    dxs[1:] = rng.uniform(-trans_range, trans_range, size=n_candidates)
    dys[1:] = rng.uniform(-trans_range, trans_range, size=n_candidates)

    thetas = np.deg2rad(angles_deg)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    rel = corners - np.array([cx, cy])  # (4, 2)

    rot_x = cos_t[:, None] * rel[None, :, 0] + sin_t[:, None] * rel[None, :, 1]
    rot_y = -sin_t[:, None] * rel[None, :, 0] + cos_t[:, None] * rel[None, :, 1]

    cand_x = rot_x + (cx + dxs)[:, None]  # (N_total, 4)
    cand_y = rot_y + (cy + dys)[:, None]

    # KDTree for fast nearest-vertex scoring
    tree = cKDTree(curve_pts_px)

    # Score all candidates: sum of nearest-vertex distances for 4 corners
    all_pts = np.stack([cand_x.ravel(), cand_y.ravel()], axis=1)  # (N_total*4, 2)
    dists, _ = tree.query(all_pts)
    dists = dists.reshape(N_total, 4)  # (N_total, 4)
    scores = -dists.mean(axis=1)  # negative mean distance

    best_idx = int(np.argmax(scores))
    best_corners = np.stack([cand_x[best_idx], cand_y[best_idx]], axis=1)
    return best_corners, float(scores[best_idx])


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_box_on_ax(ax, corners, color='lime', label=None):
    from matplotlib.patches import Polygon as MplPoly
    poly = MplPoly(corners, closed=True, fill=False,
                   edgecolor=color, linewidth=1.5, linestyle='-')
    ax.add_patch(poly)
    ax.plot(corners[:, 0], corners[:, 1], 'o', color=color, markersize=4)
    if label:
        ax.text(corners[0, 0], corners[0, 1] - 4, label,
                fontsize=6, color=color, ha='left', va='bottom')


def make_visualization(vis_records, output_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_samples = len(vis_records)
    modes = ['GT', 'GT (raster)', 'Pred raw', 'Snap pixel', 'Snap param.']
    n_modes = len(modes)

    fig, axes = plt.subplots(n_samples, n_modes,
                             figsize=(3.2 * n_modes, 3.2 * n_samples),
                             squeeze=False)

    mode_keys = ['gt', 'gt_raster', 'pred_raw', 'snap_pixel', 'snap_param']
    colors = ['cyan', 'yellow', 'lime', 'orange', 'magenta']

    for row, rec in enumerate(vis_records):
        curve = rec['curve_mask']
        for col, (mkey, title, clr) in enumerate(zip(mode_keys, modes, colors)):
            ax = axes[row, col]
            ax.imshow(curve, cmap='gray', vmin=0, vmax=255)
            entry = rec.get(mkey)
            if entry is not None and entry['box'] is not None:
                _draw_box_on_ax(ax, entry['box'], color=clr)
                txt = (f"Sq(cont)={entry['sq_cont']:.3f}\n"
                       f"Al(px)={entry['al_px']:.2f}\n"
                       f"Al(param)={entry['al_param']:.4f}")
                ax.text(2, 124, txt, fontsize=5, color='white',
                        va='bottom', family='monospace',
                        bbox=dict(facecolor='black', alpha=0.6, pad=1))
            if row == 0:
                ax.set_title(title, fontsize=9)
            ax.axis('off')

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved visualization -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Continuous-space (parametric) evaluation for inscribed squares')
    p.add_argument('--checkpoint', type=str,
                   default='model/checkpoints_curves/checkpoint_520.pth',
                   help='Path to model checkpoint')
    p.add_argument('--output-dir', type=str, default='evaluation_results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--sampling-steps', type=int, default=25)
    p.add_argument('--sampling-schedule', type=str,
                   default='first_mid_last:20,3,2')
    p.add_argument('--gt-only', action='store_true',
                   help='Skip model inference; evaluate GT only (sanity check)')
    p.add_argument('--n-snap-candidates', type=int, default=40000)
    p.add_argument('--n-vis', type=int, default=8)
    p.add_argument('--dataset-path', type=str, default=None,
                   help='Override dataset path from checkpoint config')
    p.add_argument('--n-eval', type=int, default=None,
                   help='Max number of val samples to evaluate (default: all)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.sampling_steps is not None and args.sampling_steps <= 0:
        args.sampling_steps = None
    gt_only = args.gt_only
    seed = args.seed
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # ------------------------------------------------------------------
    # Load checkpoint & config
    # ------------------------------------------------------------------
    from omegaconf import OmegaConf
    checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = checkpoint['config']
    # Allow adding new keys to the config struct
    OmegaConf.set_struct(config, False)

    dataset_path = _Path(str(config['dataset']['dataset_path']))
    if args.dataset_path:
        dataset_path = _Path(args.dataset_path)
        config['dataset']['dataset_path'] = args.dataset_path
    actual_count = len(list(dataset_path.glob("curve_*.png")))
    if actual_count > 0 and actual_count != config['dataset']['num_samples']:
        print(f"Overriding num_samples: {config['dataset']['num_samples']} -> {actual_count}")
        config['dataset']['num_samples'] = actual_count

    image_size = config['dataset']['image_size']

    # ------------------------------------------------------------------
    # Check for parametric data
    # ------------------------------------------------------------------
    has_parametric = (dataset_path / "curve_pts_0.npy").exists()
    if has_parametric:
        print("Parametric data (.npy) found — using true continuous metrics.")
        config['dataset']['return_parametric'] = True
    else:
        print("WARNING: No parametric .npy files found. "
              "Falling back to raster-based evaluation.\n"
              "Regenerate dataset with latest code to get true parametric metrics.")

    # ------------------------------------------------------------------
    # Model + scheduler
    # ------------------------------------------------------------------
    if not gt_only:
        model = UNet(device=device, **config['model'])
        state_dict = remove_weight_prefixes(checkpoint['state_dict'])
        model.load_state_dict(state_dict)
        model.eval()

        scheduler_type = config.get('scheduler_type', 'ddim')
        if scheduler_type == 'flow_matching':
            from schedulers.flow_matching import FlowMatching
            scheduler = FlowMatching(device=device, **config['scheduler'])
        else:
            from schedulers.ddim import DDIM
            scheduler = DDIM(device=device, **config['scheduler'])

    # ------------------------------------------------------------------
    # Dataset / dataloader (validation split)
    # ------------------------------------------------------------------
    dataset = get_dataset(config)
    val_fraction = config['dataset']['val_fraction']
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed))

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------
    import pandas as pd

    results = []
    vis_records = []
    sample_idx = 0
    n_eval = args.n_eval  # None means all

    for batch in tqdm(val_dataloader, desc="Evaluating"):

        if n_eval is not None and sample_idx >= n_eval:
            break

        if has_parametric:
            curve_imgs, square_gt_imgs, curve_pts_batch, sq_corners_batch = batch
        else:
            curve_imgs, square_gt_imgs = batch
            curve_pts_batch = None
            sq_corners_batch = None

        if gt_only:
            pred_square_img = square_gt_imgs
        else:
            pred_square_img = scheduler.sample(
                model,
                n_samples=len(square_gt_imgs),
                condition=curve_imgs,
                guidance_scale=1,
                seed=seed,
                return_dict=True,
                sampling_steps=args.sampling_steps,
                sampling_schedule=args.sampling_schedule,
            )['x_0']

        B = len(curve_imgs)
        for i in range(B):
            crv = curve_imgs[i]
            sq_gt = square_gt_imgs[i]
            sq_pred = pred_square_img[i]

            curve_mask = _to_numpy255(crv)
            gt_mask = _to_numpy255(sq_gt)
            pred_mask = _to_numpy255(sq_pred)
            dist_transform = distance_transform_edt(curve_mask == 0)

            # --- Parametric GT data (if available) ---
            if has_parametric:
                # Convert from normalised [-1,1] to pixel coords
                gt_corners_param = parametric_to_pixel(
                    sq_corners_batch[i].numpy(), image_size)
                curve_pts_px = parametric_to_pixel(
                    curve_pts_batch[i].numpy(), image_size)
            else:
                gt_corners_param = None
                curve_pts_px = None

            # --- Extract corners from raster ---
            box_gt_raster = get_box(gt_mask)
            box_pred_raw = get_box(pred_mask)

            if box_pred_raw is None:
                sample_idx += 1
                continue

            # Use parametric GT corners when available, else raster
            box_gt = gt_corners_param if gt_corners_param is not None else box_gt_raster
            if box_gt is None:
                sample_idx += 1
                continue

            # --- Pixel snap (existing raster method) ---
            snapped_mask, _ = snap_square_by_rigid_transform(sq_pred, crv)
            box_snap_pixel = get_box(snapped_mask)

            # --- Parametric snap ---
            if curve_pts_px is not None:
                box_snap_param, _ = snap_parametric(
                    box_pred_raw, curve_pts_px,
                    n_candidates=args.n_snap_candidates, rng=rng)
            else:
                # Fallback: use DT-based snap
                box_snap_param = box_pred_raw  # no snap

            # --- Compute metrics for each mode ---
            row = dict(sample_id=sample_idx)

            for label, box in [('gt', box_gt),
                                ('gt_raster', box_gt_raster),
                                ('pred_raw', box_pred_raw),
                                ('snap_pixel', box_snap_pixel),
                                ('snap_param', box_snap_param)]:
                if box is None:
                    for suffix in ('_sq_px', '_al_px', '_sq_cont', '_al_param'):
                        row[f'{label}{suffix}'] = np.nan
                    continue

                # Pixel-space squareness from raster
                if label in ('snap_param', 'pred_raw'):
                    sq_px = squareness_metric(pred_mask)  # rigid preserves shape
                elif label == 'snap_pixel':
                    sq_px = squareness_metric(snapped_mask)
                elif label in ('gt', 'gt_raster'):
                    sq_px = squareness_metric(gt_mask)

                # Pixel-space alignment (integer DT lookup)
                al_px = alignment_pixel(box, dist_transform)

                # Parametric alignment (point-to-polyline on continuous curve)
                if curve_pts_px is not None:
                    al_param = alignment_parametric(box, curve_pts_px)
                else:
                    al_param = alignment_bilinear_dt(box, dist_transform)

                # Continuous squareness (same formula as paper, on float corners)
                sq_cont = squareness_continuous(box)

                row[f'{label}_sq_px'] = sq_px
                row[f'{label}_al_px'] = al_px
                row[f'{label}_sq_cont'] = sq_cont
                row[f'{label}_al_param'] = al_param

            results.append(row)

            # Visualisation data
            if len(vis_records) < args.n_vis:
                vis_entry = dict(curve_mask=curve_mask)
                for label, box in [('gt', box_gt),
                                    ('gt_raster', box_gt_raster),
                                    ('pred_raw', box_pred_raw),
                                    ('snap_pixel', box_snap_pixel),
                                    ('snap_param', box_snap_param)]:
                    if box is None:
                        vis_entry[label] = None
                    else:
                        vis_entry[label] = dict(
                            box=box,
                            sq_cont=row[f'{label}_sq_cont'],
                            al_px=row[f'{label}_al_px'],
                            al_param=row[f'{label}_al_param'],
                        )
                vis_records.append(vis_entry)

            sample_idx += 1
            if n_eval is not None and sample_idx >= n_eval:
                break
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    tag = "gt_only" if gt_only else "full"
    csv_path = os.path.join(output_dir, f"eval_continuous_{tag}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} samples -> {csv_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY  (mean over all valid samples)")
    print("=" * 78)

    modes = [('gt', 'GT'),
             ('gt_raster', 'GT (raster)'),
             ('pred_raw', 'Pred raw'),
             ('snap_pixel', 'Snap pixel'),
             ('snap_param', 'Snap param.')]

    header = (f"{'Mode':<16} {'Sq(px)':>8} {'Sq(cont)':>9} "
              f"{'Al(px)':>8} {'Al(param)':>10}")
    print(header)
    print("-" * len(header))
    for key, label in modes:
        sq_px = df[f'{key}_sq_px'].mean()
        sq_ct = df[f'{key}_sq_cont'].mean()
        al_px = df[f'{key}_al_px'].mean()
        al_pr = df[f'{key}_al_param'].mean()
        print(f"{label:<16} {sq_px:>8.3f} {sq_ct:>9.3f} "
              f"{al_px:>8.2f} {al_pr:>10.4f}")

    # Combined rebuttal table
    print("\n" + "=" * 78)
    print("REBUTTAL TABLE")
    print("=" * 78)
    header2 = f"{'':>16} {'Squareness':>12} {'Alignment':>12}"
    print(header2)
    print("-" * len(header2))

    # Pixel-space block
    print("Pixel-space")
    for key, label, sq_col, al_col in [
        ('gt',        '  GT',        'sq_px', 'al_px'),
        ('pred_raw',  '  Pred',      'sq_px', 'al_px'),
        ('snap_pixel','  + snap',    'sq_px', 'al_px'),
    ]:
        sq = df[f'{key}_{sq_col}'].mean()
        al = df[f'{key}_{al_col}'].mean()
        print(f"{label:<16} {sq:>12.3f} {al:>12.2f}")

    print("-" * len(header2))

    # Parametric-space block
    print("Parametric-space")
    for key, label, sq_col, al_col in [
        ('gt',         '  GT',          'sq_cont', 'al_param'),
        ('gt_raster',  '  GT (raster)', 'sq_cont', 'al_param'),
        ('pred_raw',   '  Pred',        'sq_cont', 'al_param'),
        ('snap_param', '  + snap',      'sq_cont', 'al_param'),
    ]:
        sq = df[f'{key}_{sq_col}'].mean()
        al = df[f'{key}_{al_col}'].mean()
        print(f"{label:<16} {sq:>12.3f} {al:>12.4f}")

    # LaTeX version
    print("\n% LaTeX version:")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"& Squareness $\uparrow$ & Alignment $\downarrow$ \\")
    print(r"\midrule")
    print(r"\multicolumn{3}{l}{\textit{Pixel-space evaluation}} \\")
    for key, label, sq_col, al_col in [
        ('gt',        'GT',     'sq_px', 'al_px'),
        ('pred_raw',  'Pred',   'sq_px', 'al_px'),
        ('snap_pixel','+ snap', 'sq_px', 'al_px'),
    ]:
        sq = df[f'{key}_{sq_col}'].mean()
        al = df[f'{key}_{al_col}'].mean()
        print(f"{label} & {sq:.3f} & {al:.2f} \\\\")
    print(r"\midrule")
    print(r"\multicolumn{3}{l}{\textit{Parametric-space evaluation}} \\")
    for key, label, sq_col, al_col in [
        ('gt',         'GT',          'sq_cont', 'al_param'),
        ('gt_raster',  'GT (raster)', 'sq_cont', 'al_param'),
        ('pred_raw',   'Pred',        'sq_cont', 'al_param'),
        ('snap_param', '+ snap',      'sq_cont', 'al_param'),
    ]:
        sq = df[f'{key}_{sq_col}'].mean()
        al = df[f'{key}_{al_col}'].mean()
        print(f"{label} & {sq:.3f} & {al:.2f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    if vis_records:
        viz_path = os.path.join(output_dir, f"eval_continuous_viz_{tag}.png")
        make_visualization(vis_records, viz_path)


if __name__ == '__main__':
    main()
