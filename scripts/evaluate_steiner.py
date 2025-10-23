import json
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so absolute imports like `utils.*` work
# when running this file directly (e.g., `uv run scripts/evaluate_steiner.py`).
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing import Pool, get_context
import multiprocessing as mp
from functools import partial
from itertools import combinations
from collections import defaultdict

import torch
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay, QhullError

from utils.seed import seed_everything
from utils.config_handler import remove_weight_prefixes
from data.GeoSteinerDataset import GeoSteinerDataset
from model.diffusion import UNet
from schedulers.ddim import DDIM
from scripts.extract_steiner_graph import SteinerGraphExtractor


def score_steiner_sample_worker(args):
    """Worker function for graph analysis and scoring of generated samples."""
    sample_idx, sample_tensor, instance, image_size, graph_extractor_config = args

    # Create a new graph extractor for this worker
    graph_extractor = SteinerGraphExtractor(**graph_extractor_config)

    # Convert tensor to discrete image (ensure it's on CPU)
    if sample_tensor.dim() == 3:
        sample_tensor = sample_tensor[0]
    # Ensure tensor is detached and on CPU
    if hasattr(sample_tensor, 'detach'):
        tensor_np = sample_tensor.detach().cpu().numpy()
    else:
        # Already a numpy array or CPU tensor
        tensor_np = sample_tensor.numpy() if hasattr(sample_tensor, 'numpy') else sample_tensor

    # Direct classification from [-1, 1] to 3 classes with robust thresholds
    v = tensor_np
    vertex_mask = v <= -0.5
    edge_mask = v >= 0.5

    # Assemble discrete image
    discrete_img = np.ones(tensor_np.shape, dtype=np.uint8) * 127
    discrete_img[edge_mask] = 255
    discrete_img[vertex_mask] = 0

    # Ensure correct size
    if discrete_img.shape != (image_size, image_size):
        discrete_img = cv2.resize(discrete_img, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    try:
        # Get terminal pixel coordinates
        denom = image_size - 1
        terminal_pixels = []
        for tx, ty in instance['terminal_points']:
            px = float(np.clip(tx * denom, 0.0, float(denom)))
            py = float(np.clip(ty * denom, 0.0, float(denom)))
            terminal_pixels.append((px, py))

        # Extract graph
        vertices, edges = graph_extractor.extract_graph(
            discrete_img,
            reference_points=terminal_pixels
        )

        if len(vertices) == 0 or len(edges) == 0:
            # Return failed result with all metrics as None/False
            return {
                'sample_idx': sample_idx,
                'score': -1.0,
                'vertices': [],
                'edges': [],
                'predicted_weight': None,
                'optimality_ratio': None,
                'is_valid_tree': False,
                'is_connected': False,
                'covers_all_terminals': False,
                'extraction_success': False,
                'error_message': "No vertices or edges extracted"
            }

        # Calculate metrics
        predicted_weight, is_valid_tree, is_connected, covers_terminals = _calculate_metrics_worker(
            vertices, edges, instance, image_size
        )

        if not covers_terminals or predicted_weight is None:
            score = -2.0  # Doesn't cover terminals gets very low score
            optimality_ratio = None
        else:
            # Use optimality ratio as the primary scoring metric
            optimal_weight = instance['total_length']
            if optimal_weight > 0 and predicted_weight > 0:
                optimality_ratio = predicted_weight / optimal_weight
                score = 2.0 - optimality_ratio  # Higher score for lower ratios

                # Bonus for valid tree structure
                if is_valid_tree:
                    score += 0.5
                if is_connected:
                    score += 0.5

                score = max(score, 0.0)
            else:
                score = max(0.0, 1.0 - predicted_weight)
                optimality_ratio = None

        return {
            'sample_idx': sample_idx,
            'score': score,
            'vertices': vertices,
            'edges': edges,
            'predicted_weight': predicted_weight,
            'optimality_ratio': optimality_ratio,
            'is_valid_tree': is_valid_tree,
            'is_connected': is_connected,
            'covers_all_terminals': covers_terminals,
            'extraction_success': True,
            'error_message': None
        }

    except Exception as e:
        # Return failed result with error message
        return {
            'sample_idx': sample_idx,
            'score': -3.0,
            'vertices': [],
            'edges': [],
            'predicted_weight': None,
            'optimality_ratio': None,
            'is_valid_tree': False,
            'is_connected': False,
            'covers_all_terminals': False,
            'extraction_success': False,
            'error_message': str(e)
        }


def _calculate_metrics_worker(vertices, edges, instance, image_size):
    """Worker function version of _calculate_metrics."""
    if not vertices:
        return None, False, False, False

    # Create NetworkX graph
    G = nx.Graph()
    for i, (x, y) in enumerate(vertices):
        G.add_node(i, pos=(x, y))
    G.add_edges_from(edges)

    # Check connectivity and tree structure
    is_connected = nx.is_connected(G) if len(vertices) > 1 else True
    is_valid_tree = is_connected and len(edges) == len(vertices) - 1

    # Calculate weight
    if not vertices or not edges:
        predicted_weight = 0.0
    else:
        denom = (image_size - 1)
        normalized_vertices = [(x / denom, y / denom) for x, y in vertices]

        total_weight = 0.0
        for i, j in edges:
            x1, y1 = normalized_vertices[i]
            x2, y2 = normalized_vertices[j]
            weight = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_weight += weight
        predicted_weight = total_weight

    # Check terminal coverage
    denom = (image_size - 1)
    normalized_vertices = [(x / denom, y / denom) for x, y in vertices]
    terminal_points = instance['terminal_points']

    covered_terminals = 0
    threshold = 0.03
    for tx, ty in terminal_points:
        for vx, vy in normalized_vertices:
            dist = np.sqrt((tx - vx)**2 + (ty - vy)**2)
            if dist <= threshold:
                covered_terminals += 1
                break

    covers_terminals = (covered_terminals == len(terminal_points))

    return predicted_weight, is_valid_tree, is_connected, covers_terminals


def _orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Signed area (scaled by 2) of triangle pqr. Positive if p->q->r is CCW."""
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


def _build_edge_map(triangles: List[Tuple[int, int, int]]):
    """Map undirected edges to the triangle indices that contain them."""
    edge_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, tri in enumerate(triangles):
        for u, v in combinations(tri, 2):
            if u == v:
                continue
            edge = (u, v) if u < v else (v, u)
            edge_map[edge].append(idx)
    return edge_map


def _ensure_ccw(triangle: Tuple[int, int, int], points: np.ndarray) -> Tuple[int, int, int]:
    """Return the triangle with vertices ordered counter-clockwise."""
    a, b, c = triangle
    if _orientation(points[a], points[b], points[c]) < 0:
        return (a, c, b)
    return triangle


def _apply_random_edge_flips(
    triangles: List[Tuple[int, int, int]],
    points: np.ndarray,
    rng: np.random.Generator,
    num_flips: int,
) -> List[Tuple[int, int, int]]:
    """Perform random legal diagonal flips to obtain a random triangulation."""
    if len(triangles) < 2:
        return triangles

    for _ in range(num_flips):
        edge_map = _build_edge_map(triangles)
        interior_edges = [edge for edge, tri_idxs in edge_map.items() if len(tri_idxs) == 2]
        if not interior_edges:
            break

        edge = interior_edges[rng.integers(len(interior_edges))]
        tri_indices = edge_map[edge]
        if len(tri_indices) != 2:
            continue

        t_idx, u_idx = tri_indices
        tri_t = triangles[t_idx]
        tri_u = triangles[u_idx]

        shared = set(tri_t).intersection(tri_u)
        if len(shared) != 2:
            continue

        a, b = tuple(shared)
        unique_t = [v for v in tri_t if v not in shared]
        unique_u = [v for v in tri_u if v not in shared]
        if len(unique_t) != 1 or len(unique_u) != 1:
            continue

        c = unique_t[0]
        d = unique_u[0]

        pa, pb, pc, pd = points[a], points[b], points[c], points[d]

        orient_ac = _orientation(pa, pb, pc)
        orient_ad = _orientation(pa, pb, pd)
        orient_ca = _orientation(pc, pd, pa)
        orient_cb = _orientation(pc, pd, pb)

        if (
            abs(orient_ac) < 1e-12
            or abs(orient_ad) < 1e-12
            or abs(orient_ca) < 1e-12
            or abs(orient_cb) < 1e-12
        ):
            continue  # Degenerate quadrilateral

        if orient_ac * orient_ad >= 0 or orient_ca * orient_cb >= 0:
            continue  # Not a convex quadrilateral

        new_edge = (c, d) if c < d else (d, c)
        if new_edge in edge_map:
            # Diagonal already present, skip to avoid multi-edges
            continue

        new_tri_t = _ensure_ccw((c, a, d), points)
        new_tri_u = _ensure_ccw((c, d, b), points)

        triangles[t_idx] = new_tri_t
        triangles[u_idx] = new_tri_u

    return triangles


def _generate_random_planar_graph(
    points: np.ndarray,
    rng: np.random.Generator,
) -> nx.Graph:
    """Generate a planar graph via Delaunay triangulation and random flips."""
    n_points = len(points)
    G = nx.Graph()
    G.add_nodes_from(range(n_points))

    if n_points <= 1:
        return G

    if n_points == 2:
        dist = float(np.linalg.norm(points[0] - points[1]))
        G.add_edge(0, 1, weight=dist)
        return G

    delaunay = None
    try:
        delaunay = Delaunay(points)
        triangles = [tuple(int(v) for v in tri) for tri in delaunay.simplices]
    except QhullError:
        # Fallback: complete graph ensures connectivity for degenerate input
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = float(np.linalg.norm(points[i] - points[j]))
                G.add_edge(i, j, weight=dist)
        return G

    # Apply random flips to move away from the deterministic Delaunay triangulation
    num_flips = max(n_points * 5, 10)
    triangles = _apply_random_edge_flips(triangles, points, rng, num_flips)

    for tri in triangles:
        for u, v in combinations(tri, 2):
            if u == v:
                continue
            if not G.has_edge(u, v):
                dist = float(np.linalg.norm(points[u] - points[v]))
                G.add_edge(u, v, weight=dist)

    # Ensure hull edges are present
    if delaunay is not None:
        for u, v in delaunay.convex_hull:
            if not G.has_edge(u, v):
                dist = float(np.linalg.norm(points[u] - points[v]))
                G.add_edge(int(u), int(v), weight=dist)

    return G


def _build_complete_graph(points: np.ndarray) -> nx.Graph:
    """Construct the complete graph with Euclidean edge weights."""
    n_points = len(points)
    G = nx.Graph()
    for i in range(n_points):
        G.add_node(i)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dist = float(np.linalg.norm(points[i] - points[j]))
            G.add_edge(i, j, weight=dist)
    return G


def _sum_tree_weight(tree: nx.Graph) -> float:
    """Sum weighted edge lengths for a spanning tree graph."""
    weight = 0.0
    for _, _, data in tree.edges(data=True):
        weight += float(data.get('weight', 0.0))
    return weight


def _compute_planar_and_mst_baselines(
    terminal_points: List[Tuple[float, float]],
    optimal_weight: float,
    seed: Optional[int]
) -> Dict[str, Dict[str, Optional[object]]]:
    """Compute random planar and MST baselines for comparison."""
    baseline = {
        'random_tree': {'weight': None, 'ratio': None, 'edges': None},
        'planar_mst': {'weight': None, 'ratio': None, 'edges': None},
        'complete_mst': {'weight': None, 'ratio': None, 'edges': None},
    }

    if optimal_weight is None or optimal_weight <= 0:
        return baseline

    points = np.asarray(terminal_points, dtype=float)
    if len(points) == 0:
        return baseline

    rng = np.random.default_rng(seed)
    planar_graph = _generate_random_planar_graph(points, rng)

    if planar_graph.number_of_edges() > 0:
        # Random spanning tree baseline
        tree = nx.random_spanning_tree(planar_graph, seed=rng)
        random_tree = nx.Graph()
        random_tree.add_nodes_from(tree.nodes(data=True))
        for u, v in tree.edges():
            dist = float(np.linalg.norm(points[u] - points[v]))
            random_tree.add_edge(u, v, weight=dist)
        random_tree_weight = _sum_tree_weight(random_tree)
        random_tree_ratio = random_tree_weight / optimal_weight if optimal_weight > 0 else None
        baseline['random_tree'] = {
            'weight': random_tree_weight,
            'ratio': random_tree_ratio,
            'edges': list(random_tree.edges()),
        }

        # MST of the random planar graph
        if nx.is_connected(planar_graph):
            planar_mst = nx.minimum_spanning_tree(planar_graph, weight='weight')
            planar_mst_weight = _sum_tree_weight(planar_mst)
            planar_mst_ratio = planar_mst_weight / optimal_weight if optimal_weight > 0 else None
            baseline['planar_mst'] = {
                'weight': planar_mst_weight,
                'ratio': planar_mst_ratio,
                'edges': list(planar_mst.edges()),
            }

    # MST of the complete graph over terminals
    if len(points) >= 2:
        complete_graph = _build_complete_graph(points)
        complete_mst = nx.minimum_spanning_tree(complete_graph, weight='weight')
        complete_mst_weight = _sum_tree_weight(complete_mst)
        complete_mst_ratio = complete_mst_weight / optimal_weight if optimal_weight > 0 else None
        baseline['complete_mst'] = {
            'weight': complete_mst_weight,
            'ratio': complete_mst_ratio,
            'edges': list(complete_mst.edges()),
        }

    return baseline


@dataclass
class InstanceResult:
    """Results for a single Steiner tree instance."""
    instance_id: int
    num_terminals: int
    optimal_weight: float
    predicted_weight: Optional[float]
    optimality_ratio: Optional[float]
    random_tree_weight: Optional[float]
    random_tree_ratio: Optional[float]
    planar_mst_weight: Optional[float]
    planar_mst_ratio: Optional[float]
    complete_mst_weight: Optional[float]
    complete_mst_ratio: Optional[float]
    is_valid_tree: bool
    is_connected: bool
    covers_all_terminals: bool
    num_extracted_vertices: int
    num_extracted_edges: int
    extraction_success: bool
    inference_time: float
    extraction_time: float
    error_message: Optional[str] = None


class SteinerEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        steiner_data_path: str = "data/steiner_data",
        output_dir: str = "evaluation_results",
        device: str = "auto",
        debug: bool = False,
        seed: int = 42,
        use_regression: bool = False,
        best_of_n: int = 1,
        num_workers: int = 10,
        inference_steps: Optional[int] = None
    ):
        """
        Initialize the Steiner tree evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            steiner_data_path: Path to steiner data directory
            output_dir: Output directory for results
            device: Device to use ('cuda', 'cpu', or 'auto')
            debug: Enable debug mode with visualizations
            seed: Random seed for reproducibility
            use_regression: Use direct regression inference instead of diffusion sampling
            best_of_n: For diffusion models, generate n samples and select the best one
            num_workers: Number of worker processes for best-of-N scoring (default: 10)
            inference_steps: Number of denoising steps for diffusion sampling (None = use config default)
        """
        self.checkpoint_path = checkpoint_path
        self.steiner_data_path = Path(steiner_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug = debug
        self.use_regression = use_regression
        self.best_of_n = best_of_n
        self.num_workers = min(num_workers, mp.cpu_count())
        self.inference_steps = inference_steps
        self.seed = seed
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        seed_everything(seed)
        
        # Track debug images for grid compilation
        self.debug_images = [] if debug else None
        # Track raw comparison images (GT vs Generated)
        self.comparison_images = [] if debug else None
        
        # Initialize components
        self._load_model()
        self._init_graph_extractor()
        
    def _load_model(self):
        """Load the trained diffusion model."""
        print(f"Loading model checkpoint: {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, weights_only=False, map_location=self.device)
        self.config = checkpoint['config']
        
        # Initialize model
        self.model = UNet(device=self.device, **self.config['model'])
        state_dict = remove_weight_prefixes(checkpoint["state_dict"])
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Initialize scheduler (only for diffusion models)
        if not self.use_regression:
            scheduler_config = self.config['scheduler'].copy()
            # Override inference steps if specified by user
            if self.inference_steps is not None:
                scheduler_config['diffusion_steps'] = self.inference_steps
            self.scheduler = DDIM(device=self.device, **scheduler_config)
        else:
            self.scheduler = None
        
        # Initialize GeoSteinerDataset for the test instances
        # Use the same distance transform setting as training
        self.use_distance_transform = self.config['dataset'].get('use_distance_transform', False)
        self.dataset = GeoSteinerDataset(
            data_dir=str(self.steiner_data_path),
            split='val',  # Use validation split as test data
            train_split=0.0,  # Same split ratio as training
            image_size=self.config['dataset']['image_size'],
            # Match generation defaults: nodes should have radius 2
            node_radius=self.config['dataset'].get('node_radius', 2),
            edge_width=self.config['dataset'].get('edge_width', 2),
            use_distance_transform=self.use_distance_transform,
            seed=42  # Same seed for consistent splits
        )
        self.image_size = self.config['dataset']['image_size']
        print(f"Model loaded successfully on {self.device}")

        # Log multiprocessing configuration
        print(f"Best-of-{self.best_of_n} sampling enabled with {self.num_workers} worker processes")
        if not self.use_regression:
            # Get the actual inference steps from scheduler config
            actual_steps = self.scheduler.diffusion_steps if hasattr(self.scheduler, 'diffusion_steps') else 'unknown'
            override_msg = f" (overridden from config)" if self.inference_steps is not None else ""
            print(f"Diffusion sampling with {actual_steps} inference steps{override_msg}")
        
        
    def _init_graph_extractor(self):
        """Initialize the graph extraction component."""
        self.graph_extractor = SteinerGraphExtractor(
            vertex_radius=self.config['dataset'].get('node_radius', 2),
            edge_width=self.config['dataset'].get('edge_width', 2),
            coverage_threshold=0.9,
            proximity_threshold=4.0,
            debug=self.debug  # Enable debug if overall debug is enabled
        )
        
    def run_evaluation(self) -> List[InstanceResult]:
        """Run evaluation on all test instances."""
        print(f"Starting evaluation on {len(self.dataset)} instances...")
        
        results = []
        failed_count = 0
        
        for instance_idx in tqdm(range(len(self.dataset)), desc="Evaluating instances"):
            try:
                result = self._evaluate_instance(instance_idx)
                results.append(result)
                
                if not result.extraction_success:
                    failed_count += 1
                    
            except Exception as e:
                instance = self.dataset.get_sample_info(instance_idx)
                print(f"Failed to evaluate instance {instance['instance_id']}: {e}")
                failed_count += 1
                results.append(InstanceResult(
                    instance_id=instance['instance_id'],
                    num_terminals=instance['num_terminals'],
                    optimal_weight=instance['total_length'],
                    predicted_weight=None,
                    optimality_ratio=None,
                    random_tree_weight=None,
                    random_tree_ratio=None,
                    planar_mst_weight=None,
                    planar_mst_ratio=None,
                    complete_mst_weight=None,
                    complete_mst_ratio=None,
                    is_valid_tree=False,
                    is_connected=False,
                    covers_all_terminals=False,
                    num_extracted_vertices=0,
                    num_extracted_edges=0,
                    extraction_success=False,
                    inference_time=0,
                    extraction_time=0,
                    error_message=str(e)
                ))
        
        print(f"Evaluation completed. {failed_count} instances failed.")
        return results
    
    def _evaluate_instance(self, instance_idx: int) -> InstanceResult:
        """Evaluate a single instance using dataset."""
        # Get condition and target from dataset
        condition, target = self.dataset[instance_idx]
        condition = condition.unsqueeze(0).to(self.device)  # Add batch dimension: (1, 1, 128, 128)
        
        # Get instance metadata for metrics using dataset's correct indexing
        instance = self.dataset.get_sample_info(instance_idx)
        instance_id = instance['instance_id']
        optimal_weight = instance['total_length']
        
        # Run inference - always generate all samples first
        start_time = time.time()

        with torch.no_grad():
            if self.use_regression:
                # Regression: generate samples directly
                all_samples = []
                for _ in range(self.best_of_n):
                    target_shape = (1, 1, self.image_size, self.image_size)
                    zero_noise = torch.zeros(target_shape, device=self.device)
                    t = torch.zeros(1, 1, device=self.device)
                    pred_tensor = self.model(zero_noise, t, condition)
                    all_samples.append(pred_tensor)
                all_samples = torch.cat(all_samples, dim=0)  # [n, 1, 128, 128]
            else:
                # Diffusion: generate samples using scheduler
                condition_expanded = condition.expand(self.best_of_n, -1, -1, -1)
                pred_result = self.scheduler.sample(
                    self.model,
                    n_samples=self.best_of_n,
                    condition=condition_expanded,
                    guidance_scale=1,
                    seed=42,
                    return_dict=True,
                )
                all_samples = pred_result['x_0']  # [n, 1, 128, 128]


        inference_time = time.time() - start_time

        # Now evaluate all samples in parallel
        graph_extractor_config = {
            'vertex_radius': self.config['dataset'].get('node_radius', 2),
            'edge_width': self.config['dataset'].get('edge_width', 2),
            'coverage_threshold': 0.9,
            'proximity_threshold': 4.0,
            'debug': False
        }

        worker_args = []
        for sample_idx in range(self.best_of_n):
            sample_cpu = all_samples[sample_idx, 0].cpu()  # [H, W] tensor on CPU
            worker_args.append((
                sample_idx,
                sample_cpu,
                instance,
                self.image_size,
                graph_extractor_config
            ))

        # Score all samples in parallel
        with get_context('spawn').Pool(self.num_workers) as pool:
            results = pool.map(score_steiner_sample_worker, worker_args)

        # Find the best result
        best_result = max(results, key=lambda x: x['score'])
        best_idx = best_result['sample_idx']
        best_score = best_result['score']

        pred_img_tensor = all_samples[best_idx:best_idx+1]
        worker_result = best_result

        if self.debug and instance_idx < 3:
            print(f"Best-of-{self.best_of_n} selection: best score = {best_score:.4f} (sample {best_idx})")
        
        # Use metrics from worker (always available now)
        vertices = worker_result['vertices']
        edges = worker_result['edges']
        predicted_weight = worker_result['predicted_weight']
        optimality_ratio = worker_result['optimality_ratio']
        is_valid_tree = worker_result['is_valid_tree']
        is_connected = worker_result['is_connected']
        covers_terminals = worker_result['covers_all_terminals']
        extraction_success = worker_result['extraction_success']
        error_message = worker_result['error_message']
        extraction_time = 0.0  # Already included in inference time for multiprocessing

        random_seed = (self.seed or 0) + int(instance_id)
        baseline_results = _compute_planar_and_mst_baselines(
            instance['terminal_points'],
            optimal_weight,
            seed=random_seed
        )

        random_baseline = baseline_results['random_tree']
        planar_mst_baseline = baseline_results['planar_mst']
        complete_mst_baseline = baseline_results['complete_mst']

        random_tree_weight = random_baseline['weight']
        random_tree_ratio = random_baseline['ratio']

        planar_mst_weight = planar_mst_baseline['weight']
        planar_mst_ratio = planar_mst_baseline['ratio']

        complete_mst_weight = complete_mst_baseline['weight']
        complete_mst_ratio = complete_mst_baseline['ratio']

        # Convert prediction to discrete for debug visualization
        pred_discrete = self._convert_tensor_to_discrete(pred_img_tensor[0, 0])
        
        # Save debug visualization if requested
        if self.debug:
            # Convert tensors/images for visualization
            condition_np = ((condition[0, 0].cpu().numpy() + 1.0) * 127.5).astype(np.uint8)
            target_discrete = self._convert_tensor_to_discrete(target.squeeze(0))
            self._save_debug_visualization(
                instance_id,
                condition_np,
                target_discrete,
                pred_discrete,
                vertices,
                edges,
                instance,
                random_baseline,
                planar_mst_baseline,
                complete_mst_baseline
            )
            # Also save the raw predicted discrete image for debugging
            # cv2.imwrite(str(self.output_dir / f"pred_128x128_{instance_id:04d}.png"), pred_discrete)


            # Store raw comparison images for grid
            if self.comparison_images is not None:
                self.comparison_images.append((instance_id, target_discrete, pred_discrete, instance))
        
        return InstanceResult(
            instance_id=instance_id,
            num_terminals=instance['num_terminals'],
            optimal_weight=optimal_weight,
            predicted_weight=predicted_weight,
            optimality_ratio=optimality_ratio,
            random_tree_weight=random_tree_weight,
            random_tree_ratio=random_tree_ratio,
            planar_mst_weight=planar_mst_weight,
            planar_mst_ratio=planar_mst_ratio,
            complete_mst_weight=complete_mst_weight,
            complete_mst_ratio=complete_mst_ratio,
            is_valid_tree=is_valid_tree,
            is_connected=is_connected,
            covers_all_terminals=covers_terminals,
            num_extracted_vertices=len(vertices),
            num_extracted_edges=len(edges),
            extraction_success=extraction_success,
            inference_time=inference_time,
            extraction_time=extraction_time,
            error_message=error_message
        )
    
    
    def _calculate_graph_weight(self, vertices: List[Tuple[float, float]], edges: List[Tuple[int, int]]) -> float:
        """Calculate total weight of graph in normalized coordinates."""
        if not vertices or not edges:
            return 0.0
            
        # Convert pixel coordinates to normalized coordinates [0,1]
        # Convert pixel coordinates to normalized coordinates [0,1] consistent with data
        denom = (self.image_size - 1)
        normalized_vertices = [(x / denom, y / denom) for x, y in vertices]
        
        # Calculate total edge weight in normalized coordinates
        total_weight = 0.0
        for i, j in edges:
            x1, y1 = normalized_vertices[i]
            x2, y2 = normalized_vertices[j]
            weight = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_weight += weight
            
        return total_weight
    
    def _calculate_metrics(
        self,
        vertices: List[Tuple[float, float]],
        edges: List[Tuple[int, int]],
        instance: Dict
    ) -> Tuple[Optional[float], bool, bool, bool]:
        """Calculate performance metrics for extracted graph."""
        if not vertices:
            return None, False, False, False
        
        # Create NetworkX graph
        G = nx.Graph()
        for i, (x, y) in enumerate(vertices):
            G.add_node(i, pos=(x, y))
        G.add_edges_from(edges)
        
        # Check if it's a valid tree (connected and |E| = |V| - 1)
        is_connected = nx.is_connected(G) if len(vertices) > 1 else True
        is_valid_tree = is_connected and len(edges) == len(vertices) - 1
        
        # Calculate weight using centralized function
        predicted_weight = self._calculate_graph_weight(vertices, edges)
        
        # Convert pixel coordinates to normalized for terminal coverage check
        # Use (image_size - 1) to invert the dataset's pixelization
        denom = (self.image_size - 1)
        normalized_vertices = [(x / denom, y / denom) for x, y in vertices]
        
        # Check if all terminals are covered (within reasonable distance)
        terminal_points = instance['terminal_points']
        covers_terminals = self._check_terminal_coverage(normalized_vertices, terminal_points)
        
        return predicted_weight, is_valid_tree, is_connected, covers_terminals
    
    def _check_terminal_coverage(
        self, 
        vertices: List[Tuple[float, float]], 
        terminals: List[List[float]],
        threshold: float = 0.03
    ) -> bool:
        """Check if all terminals are covered by vertices within threshold distance.
        
        Args:
            vertices: Extracted vertices in normalized coordinates [0,1]
            terminals: Terminal points in normalized coordinates [0,1]  
            threshold: Maximum distance for a terminal to be considered covered
            
        Returns:
            True if all terminals are covered, False otherwise
        """
        covered_terminals = 0
        for tx, ty in terminals:
            covered = False
            min_dist = float('inf')
            for vx, vy in vertices:
                dist = np.sqrt((tx - vx)**2 + (ty - vy)**2)
                min_dist = min(min_dist, dist)
                if dist <= threshold:
                    covered = True
                    break
            if covered:
                covered_terminals += 1
                
        # Log coverage for debugging
        coverage_rate = covered_terminals / len(terminals)
        if coverage_rate < 1.0:
            print(f"Warning: Only {covered_terminals}/{len(terminals)} terminals covered ({coverage_rate:.1%})")
            
        return coverage_rate == 1.0

    def _get_terminal_pixel_coords(self, terminals: List[List[float]]) -> List[Tuple[float, float]]:
        """Convert normalized terminal coordinates to pixel space (float for precision)."""
        if not terminals:
            return []

        denom = self.image_size - 1
        coords = []
        for tx, ty in terminals:
            px = float(np.clip(tx * denom, 0.0, float(denom)))
            py = float(np.clip(ty * denom, 0.0, float(denom)))
            coords.append((px, py))

        return coords
    
    def _convert_tensor_to_discrete(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a model output tensor to a discrete 3-value image expected by the extractor.

        Representation:
        - 0   = vertices (black)
        - 127 = background (gray)
        - 255 = edges (white)
        """
        # Ensure 2D numpy array
        if tensor.dim() == 3:
            # Allow tensors coming as (1, H, W)
            tensor = tensor[0]
        tensor_np = tensor.detach().cpu().numpy()

        H, W = tensor_np.shape

        # Direct classification from [-1, 1] to 3 classes with robust thresholds
        v = tensor_np
        vertex_mask = v <= -0.5
        edge_mask = v >= 0.5
        # Background is the rest

        # Assemble discrete image
        out = np.ones((H, W), dtype=np.uint8) * 127
        out[edge_mask] = 255
        out[vertex_mask] = 0

        # Ensure correct size
        if out.shape != (self.image_size, self.image_size):
            out = cv2.resize(out, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return out
    
    
    def _save_debug_visualization(
        self,
        instance_id: int,
        points_img: np.ndarray,
        target_img: np.ndarray,
        pred_img: np.ndarray,
        vertices: List[Tuple[float, float]],
        edges: List[Tuple[int, int]],
        instance: Dict,
        random_baseline: Dict[str, Optional[object]],
        planar_mst_baseline: Dict[str, Optional[object]],
        complete_mst_baseline: Dict[str, Optional[object]]
    ):
        """Save debug visualization for an instance with baseline comparisons."""
        random_tree_weight = random_baseline.get('weight') if random_baseline else None
        random_tree_ratio = random_baseline.get('ratio') if random_baseline else None
        random_tree_edges = random_baseline.get('edges') if random_baseline else None

        planar_mst_weight = planar_mst_baseline.get('weight') if planar_mst_baseline else None
        planar_mst_ratio = planar_mst_baseline.get('ratio') if planar_mst_baseline else None
        planar_mst_edges = planar_mst_baseline.get('edges') if planar_mst_baseline else None

        complete_mst_weight = complete_mst_baseline.get('weight') if complete_mst_baseline else None
        complete_mst_ratio = complete_mst_baseline.get('ratio') if complete_mst_baseline else None
        complete_mst_edges = complete_mst_baseline.get('edges') if complete_mst_baseline else None

        has_random_tree = all(val is not None for val in (random_tree_weight, random_tree_ratio, random_tree_edges))
        has_planar_mst = all(val is not None for val in (planar_mst_weight, planar_mst_ratio, planar_mst_edges))
        has_complete_mst = all(val is not None for val in (complete_mst_weight, complete_mst_ratio, complete_mst_edges))

        n_cols = 3
        if has_random_tree:
            n_cols += 1
        if has_planar_mst:
            n_cols += 1
        if has_complete_mst:
            n_cols += 1

        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axes_list = list(axes) if isinstance(axes, np.ndarray) else [axes]
        axis_idx = 0

        # Original points
        axes_list[axis_idx].imshow(points_img, cmap='gray')
        axes_list[axis_idx].set_title('Input (Terminal Points)')
        axes_list[axis_idx].axis('off')
        axis_idx += 1

        # Display ground truth solution from dataset target
        axes_list[axis_idx].imshow(target_img, cmap='gray')
        axes_list[axis_idx].set_title(f'Ground Truth Optimal\n(Weight: {instance["total_length"]:.3f})')
        axes_list[axis_idx].axis('off')
        axis_idx += 1

        # Extracted graph overlaid on predicted solution
        predicted_weight = self._calculate_graph_weight(vertices, edges)
        optimality_ratio = (predicted_weight / instance["total_length"]) if predicted_weight > 0 else float('inf')

        if vertices:
            _, is_valid_tree, is_connected, covers_terminals = self._calculate_metrics(vertices, edges, instance)
        else:
            is_connected, is_valid_tree, covers_terminals = False, False, False

        axes_list[axis_idx].imshow(pred_img, cmap='gray')
        for i, (x, y) in enumerate(vertices):
            axes_list[axis_idx].plot(x, y, 'ro', markersize=6)
            axes_list[axis_idx].text(x + 2, y + 2, str(i), color='red', fontsize=6)

        for i, j in edges:
            x1, y1 = vertices[i]
            x2, y2 = vertices[j]
            axes_list[axis_idx].plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)

        conn_str = "✓" if is_connected else "✗"
        tree_str = "✓" if is_valid_tree else "✗"
        term_str = "✓" if covers_terminals else "✗"

        if random_tree_weight is not None and random_tree_ratio is not None:
            random_info = f'Random Tree: {random_tree_weight:.3f} ({random_tree_ratio:.3f}x)'
        else:
            random_info = 'Random Tree: N/A'

        axes_list[axis_idx].set_title(
            f'Extracted Graph ({len(vertices)}V, {len(edges)}E)\n'
            f'Weight: {predicted_weight:.3f}, Ratio: {optimality_ratio:.3f}\n'
            f'{random_info}\n'
            f'Connected:{conn_str} Tree:{tree_str} Terminals:{term_str}'
        )
        axes_list[axis_idx].axis('off')
        axis_idx += 1

        terminal_pixels = self._get_terminal_pixel_coords(instance['terminal_points'])

        if has_random_tree:
            ax_random = axes_list[axis_idx]
            ax_random.imshow(points_img, cmap='gray')
            for u, v in random_tree_edges:
                x1, y1 = terminal_pixels[u]
                x2, y2 = terminal_pixels[v]
                ax_random.plot([x1, x2], [y1, y2], color='lime', linewidth=2, alpha=0.8)

            for idx, (x, y) in enumerate(terminal_pixels):
                ax_random.plot(x, y, 'ro', markersize=6)
                ax_random.text(x + 2, y + 2, str(idx), color='red', fontsize=6)

            ax_random.set_title(
                f'Random Planar Tree\nWeight: {random_tree_weight:.3f}, Ratio: {random_tree_ratio:.3f}'
            )
            ax_random.axis('off')
            axis_idx += 1

        if has_planar_mst:
            ax_planar = axes_list[axis_idx]
            ax_planar.imshow(points_img, cmap='gray')
            for u, v in planar_mst_edges:
                x1, y1 = terminal_pixels[u]
                x2, y2 = terminal_pixels[v]
                ax_planar.plot([x1, x2], [y1, y2], color='orange', linewidth=2, alpha=0.8)

            for idx, (x, y) in enumerate(terminal_pixels):
                ax_planar.plot(x, y, 'ro', markersize=6)
                ax_planar.text(x + 2, y + 2, str(idx), color='red', fontsize=6)

            ax_planar.set_title(
                f'Planar Graph MST\nWeight: {planar_mst_weight:.3f}, Ratio: {planar_mst_ratio:.3f}'
            )
            ax_planar.axis('off')
            axis_idx += 1

        if has_complete_mst:
            ax_complete = axes_list[axis_idx]
            ax_complete.imshow(points_img, cmap='gray')
            for u, v in complete_mst_edges:
                x1, y1 = terminal_pixels[u]
                x2, y2 = terminal_pixels[v]
                ax_complete.plot([x1, x2], [y1, y2], color='cyan', linewidth=2, alpha=0.8)

            for idx, (x, y) in enumerate(terminal_pixels):
                ax_complete.plot(x, y, 'ro', markersize=6)
                ax_complete.text(x + 2, y + 2, str(idx), color='red', fontsize=6)

            ax_complete.set_title(
                f'Complete Graph MST\nWeight: {complete_mst_weight:.3f}, Ratio: {complete_mst_ratio:.3f}'
            )
            ax_complete.axis('off')
            axis_idx += 1

        for leftover_axis in axes_list[axis_idx:]:
            leftover_axis.axis('off')

        debug_path = self.output_dir / f"debug_instance_{instance_id:04d}.png"
        plt.tight_layout()
        plt.savefig(debug_path, dpi=150)

        if self.debug_images is not None:
            import cv2
            debug_img = cv2.imread(str(debug_path))
            if debug_img is not None:
                self.debug_images.append((instance_id, debug_img))

        plt.close()


    def generate_reports(self, results: List[InstanceResult]):
        """Generate comprehensive evaluation reports."""
        print("Generating evaluation reports...")
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                'instance_id': result.instance_id,
                'num_terminals': result.num_terminals,
                'optimal_weight': result.optimal_weight,
                'predicted_weight': result.predicted_weight,
                'optimality_ratio': result.optimality_ratio,
                'random_tree_weight': result.random_tree_weight,
                'random_tree_ratio': result.random_tree_ratio,
                'planar_mst_weight': result.planar_mst_weight,
                'planar_mst_ratio': result.planar_mst_ratio,
                'complete_mst_weight': result.complete_mst_weight,
                'complete_mst_ratio': result.complete_mst_ratio,
                'is_valid_tree': result.is_valid_tree,
                'is_connected': result.is_connected,
                'covers_all_terminals': result.covers_all_terminals,
                'num_extracted_vertices': result.num_extracted_vertices,
                'num_extracted_edges': result.num_extracted_edges,
                'extraction_success': result.extraction_success,
                'inference_time': result.inference_time,
                'extraction_time': result.extraction_time,
                'error_message': result.error_message
            })
        
        # Save JSON report
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Generate summary statistics
        successful_results = [r for r in results if r.extraction_success and r.predicted_weight is not None]
        
        summary = {
            'total_instances': len(results),
            'successful_extractions': len(successful_results),
            'extraction_success_rate': len(successful_results) / len(results),
            'valid_trees': sum(1 for r in successful_results if r.is_valid_tree),
            'connected_graphs': sum(1 for r in successful_results if r.is_connected),
            'terminal_coverage': sum(1 for r in successful_results if r.covers_all_terminals),
        }

        random_ratios_all = [r.random_tree_ratio for r in results if r.random_tree_ratio is not None]
        if random_ratios_all:
            summary.update({
                'mean_random_tree_ratio': float(np.mean(random_ratios_all)),
                'std_random_tree_ratio': float(np.std(random_ratios_all)),
                'min_random_tree_ratio': float(np.min(random_ratios_all)),
                'max_random_tree_ratio': float(np.max(random_ratios_all))
            })

        planar_mst_ratios_all = [r.planar_mst_ratio for r in results if r.planar_mst_ratio is not None]
        if planar_mst_ratios_all:
            summary.update({
                'mean_planar_mst_ratio': float(np.mean(planar_mst_ratios_all)),
                'std_planar_mst_ratio': float(np.std(planar_mst_ratios_all)),
                'min_planar_mst_ratio': float(np.min(planar_mst_ratios_all)),
                'max_planar_mst_ratio': float(np.max(planar_mst_ratios_all))
            })

        complete_mst_ratios_all = [r.complete_mst_ratio for r in results if r.complete_mst_ratio is not None]
        if complete_mst_ratios_all:
            summary.update({
                'mean_complete_mst_ratio': float(np.mean(complete_mst_ratios_all)),
                'std_complete_mst_ratio': float(np.std(complete_mst_ratios_all)),
                'min_complete_mst_ratio': float(np.min(complete_mst_ratios_all)),
                'max_complete_mst_ratio': float(np.max(complete_mst_ratios_all))
            })

        if successful_results:
            # Only consider fully valid solutions for optimality metrics
            valid_solutions = [r for r in successful_results
                             if (r.optimality_ratio is not None and
                                 r.is_valid_tree and
                                 r.is_connected and
                                 r.covers_all_terminals)]
            ratios = [r.optimality_ratio for r in valid_solutions]
            if ratios:
                summary.update({
                    'mean_optimality_ratio': np.mean(ratios),
                    'median_optimality_ratio': np.median(ratios),
                    'std_optimality_ratio': np.std(ratios),
                    'min_optimality_ratio': np.min(ratios),
                    'max_optimality_ratio': np.max(ratios),
                    'better_than_optimal_count': sum(1 for r in ratios if r < 1.0),
                    'valid_solutions_count': len(valid_solutions),
                    'mean_inference_time': np.mean([r.inference_time for r in successful_results]),
                    'mean_extraction_time': np.mean([r.extraction_time for r in successful_results])
                })
        
        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Total instances: {summary['total_instances']}")
        print(f"Successful extractions: {summary['successful_extractions']} ({summary['extraction_success_rate']:.1%})")
        if successful_results:
            print(f"Valid trees: {summary['valid_trees']} ({summary['valid_trees']/len(successful_results):.1%})")
            print(f"Connected graphs: {summary['connected_graphs']} ({summary['connected_graphs']/len(successful_results):.1%})")
            print(f"Terminal coverage: {summary['terminal_coverage']} ({summary['terminal_coverage']/len(successful_results):.1%})")
            
            if 'mean_optimality_ratio' in summary:
                print("\nOptimality Ratio Statistics (valid solutions only):")
                print(f"  Valid solutions used: {summary['valid_solutions_count']}/{len(successful_results)}")
                print(f"  Mean: {summary['mean_optimality_ratio']:.3f}")
                print(f"  Median: {summary['median_optimality_ratio']:.3f}")
                print(f"  Std: {summary['std_optimality_ratio']:.3f}")
                print(f"  Range: [{summary['min_optimality_ratio']:.3f}, {summary['max_optimality_ratio']:.3f}]")
                print(f"  Better than optimal: {summary['better_than_optimal_count']}")
                
                print("\nTiming:")
                print(f"  Mean inference time: {summary['mean_inference_time']:.3f}s")
                print(f"  Mean extraction time: {summary['mean_extraction_time']:.3f}s")

        if 'mean_random_tree_ratio' in summary:
            print("\nRandom Planar Tree Baseline (all instances with valid optimal weight):")
            print(f"  Mean ratio: {summary['mean_random_tree_ratio']:.3f}")
            print(f"  Std: {summary['std_random_tree_ratio']:.3f}")
            print(f"  Range: [{summary['min_random_tree_ratio']:.3f}, {summary['max_random_tree_ratio']:.3f}]")

        if 'mean_planar_mst_ratio' in summary:
            print("\nPlanar Graph MST Baseline:")
            print(f"  Mean ratio: {summary['mean_planar_mst_ratio']:.3f}")
            print(f"  Std: {summary['std_planar_mst_ratio']:.3f}")
            print(f"  Range: [{summary['min_planar_mst_ratio']:.3f}, {summary['max_planar_mst_ratio']:.3f}]")

        if 'mean_complete_mst_ratio' in summary:
            print("\nComplete Graph MST Baseline:")
            print(f"  Mean ratio: {summary['mean_complete_mst_ratio']:.3f}")
            print(f"  Std: {summary['std_complete_mst_ratio']:.3f}")
            print(f"  Range: [{summary['min_complete_mst_ratio']:.3f}, {summary['max_complete_mst_ratio']:.3f}]")

        print(f"\nResults saved to: {self.output_dir}")
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        # Generate grid visualization if debug images were collected and not too many
        if self.debug_images and len(self.debug_images) <= 100:
            self._generate_debug_grid()

        # Generate GT vs Generated comparison grid
        if self.comparison_images:
            self._generate_comparison_grid()
    
    def _generate_visualizations(self, results: List[InstanceResult]):
        """Generate visualization plots for the results."""
        successful_results = [r for r in results if r.extraction_success and r.predicted_weight is not None]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Optimality ratio histogram (only valid trees that cover all terminals)
        ratios = [r.optimality_ratio for r in successful_results 
                 if r.optimality_ratio is not None and r.is_valid_tree and r.covers_all_terminals]
        
        if ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(ratios, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=1.0, color='red', linestyle='--', label='Optimal (ratio=1.0)')
            plt.xlabel('Optimality Ratio (Predicted/Optimal)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Optimality Ratios')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "optimality_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Scatter plot: optimality vs number of terminals (only valid trees that cover all terminals)
            terminals = [r.num_terminals for r in successful_results 
                        if r.optimality_ratio is not None and r.is_valid_tree and r.covers_all_terminals]
            plt.figure(figsize=(10, 6))
            plt.scatter(terminals, ratios, alpha=0.6)
            plt.axhline(y=1.0, color='red', linestyle='--', label='Optimal')
            plt.xlabel('Number of Terminals')
            plt.ylabel('Optimality Ratio')
            plt.title('Optimality Ratio vs Problem Complexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "optimality_vs_complexity.png", dpi=150, bbox_inches='tight')
            plt.close()

    def _generate_debug_grid(self):
        """Generate a single grid image containing all debug visualizations with borders."""
        if not self.debug_images:
            return
            
        print(f"Generating debug grid with {len(self.debug_images)} images...")
        
        # Sort images by instance ID
        self.debug_images.sort(key=lambda x: x[0])
        
        # Get the first image to determine dimensions
        first_img = self.debug_images[0][1]
        img_height, img_width = first_img.shape[:2]
        
        # Calculate grid dimensions (try to make it roughly square)
        n_images = len(self.debug_images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        
        # Border thickness
        border_thickness = 2
        
        # Calculate total grid dimensions
        total_width = cols * img_width + (cols + 1) * border_thickness
        total_height = rows * img_height + (rows + 1) * border_thickness
        
        # Create the grid image (white background)
        grid_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place each debug image in the grid (resize if needed to ensure consistent tile size)
        for idx, (instance_id, debug_img) in enumerate(self.debug_images):
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            y_start = row * (img_height + border_thickness) + border_thickness
            y_end = y_start + img_height
            x_start = col * (img_width + border_thickness) + border_thickness
            x_end = x_start + img_width
            
            # Resize image if it doesn't match the expected tile size
            if debug_img.shape[0] != img_height or debug_img.shape[1] != img_width:
                debug_img = cv2.resize(debug_img, (img_width, img_height), interpolation=cv2.INTER_AREA)

            # Place the image
            grid_img[y_start:y_end, x_start:x_end] = debug_img


            # Add black borders (already handled by the white background and positioning)
            # The borders are created by the white space and we'll add black lines
            
        # Add black border lines
        # Horizontal lines
        for i in range(rows + 1):
            y = i * (img_height + border_thickness)
            grid_img[y:y+border_thickness, :] = 0  # Black
            
        # Vertical lines  
        for j in range(cols + 1):
            x = j * (img_width + border_thickness)
            grid_img[:, x:x+border_thickness] = 0  # Black
        
        # Save the grid image
        grid_path = self.output_dir / "debug_grid_all_instances.png"
        cv2.imwrite(str(grid_path), grid_img)
        print(f"Debug grid saved to: {grid_path}")
        print(f"Grid dimensions: {rows}x{cols} ({total_height}x{total_width} pixels)")

    def _generate_comparison_grid(self):
        """Generate comparison grid as PDF with vector text overlays."""
        if not self.comparison_images:
            return

        print(f"Generating PDF comparison grid with {len(self.comparison_images)} triplets (GT|Generated|Diff)...")

        # Import matplotlib with PDF backend
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.patches as patches

        # Sort by instance ID
        self.comparison_images.sort(key=lambda x: x[0])

        # Fixed 4 samples per row
        cols = 4
        n_pairs = len(self.comparison_images)
        rows = int(np.ceil(n_pairs / cols))

        # Get image dimensions from first pair
        first_gt = self.comparison_images[0][1]  # target_discrete
        img_h, img_w = first_gt.shape

        # Calculate figure size (each cell is GT|Generated|Diff side by side)
        cell_width_inches = 6  # Each GT|Generated|Diff triplet takes 6 inches
        cell_height_inches = 2  # Height per row

        fig_width = cols * cell_width_inches
        fig_height = rows * cell_height_inches

        # Create figure with tight grid
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height),
                                gridspec_kw={'wspace': 0, 'hspace': 0})

        # Handle case where we have only one row or column
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # Process each GT|Generated pair
        for idx, (instance_id, gt_img, pred_img, instance_info) in enumerate(self.comparison_images):
            row = idx // cols
            col = idx % cols

            # Skip if we're beyond our grid
            if row >= rows:
                break

            ax = axes[row, col]

            # Create diff image (signed difference with colormap)
            diff_raw = gt_img.astype(np.float32) - pred_img.astype(np.float32)  # Range: [-255, 255]
            diff_normalized = (diff_raw + 255) / 510  # Normalize to [0, 1]
            diff_colored = (plt.cm.bwr(diff_normalized)[:, :, :3] * 255).astype(np.uint8)  # Apply colormap

            # Convert grayscale images to RGB for consistent stacking
            gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
            pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)

            # Create side-by-side comparison image (GT|Generated|Diff)
            combined_rgb = np.hstack([gt_rgb, pred_rgb, diff_colored])

            # Add terminal points as blue dots on GT and Generated sides only
            terminal_points = instance_info['terminal_points']
            for tx, ty in terminal_points:
                # GT side (left)
                px_gt = int(tx * (img_w - 1))
                py_gt = int(ty * (img_h - 1))
                cv2.circle(combined_rgb, (px_gt, py_gt), 2, (255, 0, 0), -1)  # Blue in BGR

                # Generated side (middle)
                px_pred = int(tx * (img_w - 1)) + img_w
                py_pred = int(ty * (img_h - 1))
                cv2.circle(combined_rgb, (px_pred, py_pred), 2, (255, 0, 0), -1)  # Blue in BGR

                # Diff side (right) - no overlay to keep diff clean

            # Display the combined image
            ax.imshow(combined_rgb)
            ax.set_xlim(0, combined_rgb.shape[1])
            ax.set_ylim(combined_rgb.shape[0], 0)  # Flip Y axis for image coordinates
            ax.set_aspect('equal')

            # Keep spines for borders but remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            # Add high-resolution vector text number at lower left (no background)
            ax.text(5, combined_rgb.shape[0] - 5, str(idx),
                   fontsize=12, fontweight='bold', color='black',
                   verticalalignment='bottom', horizontalalignment='left')

        # Hide unused subplots and ensure borders are visible
        for idx in range(len(self.comparison_images), rows * cols):
            row = idx // cols
            col = idx % cols
            if row < rows and col < cols:
                axes[row, col].axis('off')

        # Ensure all used axes have visible black borders
        for idx in range(len(self.comparison_images)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            # Set black borders
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)  # Thicker for visibility
                spine.set_visible(True)

        # Adjust layout (tight spacing for grid effect)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

        # Save as PDF only
        pdf_path = self.output_dir / "comparison_grid_gt_vs_generated.pdf"
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

        plt.close()

        print(f"Comparison grid saved as PDF: {pdf_path}")
        print(f"Grid dimensions: {rows}x{cols} triplets")


def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    # This must be done before any other multiprocessing operations
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Start method already set, ignore
        pass

    parser = argparse.ArgumentParser(description="Evaluate Steiner tree solver model")
    parser.add_argument(
        "--checkpoint", 
        default="model/checkpoints/checkpoint_1000.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--steiner-data-path",
        default="data/steiner_data",
        help="Path to steiner data directory"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with visualizations"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of instances to evaluate (for testing)"
    )
    parser.add_argument(
        "--use-regression",
        action="store_true",
        help="Use direct regression inference instead of diffusion sampling"
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="For diffusion models, generate N samples and select the best one (default: 1)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of worker processes for best-of-N scoring (default: 10, auto-limited by CPU count)"
    )
    parser.add_argument(
        "--inference-steps",
        type=int,
        default=None,
        help="Number of denoising steps for diffusion sampling (default: use config value, ignored for regression)"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Skip evaluation, load results from existing CSV and regenerate reports only"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.best_of_n < 1:
        print("Error: --best-of-n must be >= 1")
        return 1

    if args.num_workers < 1:
        print("Error: --num-workers must be >= 1")
        return 1

    if args.inference_steps is not None and args.inference_steps < 1:
        print("Error: --inference-steps must be >= 1")
        return 1

    if args.use_regression and args.best_of_n > 1:
        print("Warning: --best-of-n is ignored for regression models (only applies to diffusion)")

    # Handle report-only mode
    if args.report_only:
        # Check if results.csv exists in output directory
        output_dir = Path(args.output_dir)
        csv_path = output_dir / "results.csv"

        if not csv_path.exists():
            print(f"Error: No results.csv found at {csv_path}")
            print("Run evaluation first without --report-only flag")
            return 1

        # Create minimal evaluator for report generation (no model loading)
        output_dir.mkdir(exist_ok=True)

        # Create a minimal evaluator instance just for report generation
        class ReportOnlyEvaluator:
            def __init__(self, output_dir):
                self.output_dir = Path(output_dir)
                self.debug_images = None  # No debug images in report-only mode
                self.comparison_images = None  # No comparison images

            def load_results_from_csv(self, csv_path=None):
                """Load results from existing CSV file and convert to InstanceResult objects."""
                if csv_path is None:
                    csv_path = self.output_dir / "results.csv"

                print(f"Loading results from: {csv_path}")

                if not Path(csv_path).exists():
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")

                df = pd.read_csv(csv_path)
                results = []
                has_random_cols = 'random_tree_weight' in df.columns and 'random_tree_ratio' in df.columns
                has_planar_cols = 'planar_mst_weight' in df.columns and 'planar_mst_ratio' in df.columns
                has_complete_cols = 'complete_mst_weight' in df.columns and 'complete_mst_ratio' in df.columns

                for _, row in df.iterrows():
                    random_weight = None
                    random_ratio = None
                    if has_random_cols:
                        random_weight = row['random_tree_weight'] if pd.notna(row['random_tree_weight']) else None
                        random_ratio = row['random_tree_ratio'] if pd.notna(row['random_tree_ratio']) else None

                    planar_weight = None
                    planar_ratio = None
                    if has_planar_cols:
                        planar_weight = row['planar_mst_weight'] if pd.notna(row['planar_mst_weight']) else None
                        planar_ratio = row['planar_mst_ratio'] if pd.notna(row['planar_mst_ratio']) else None

                    complete_weight = None
                    complete_ratio = None
                    if has_complete_cols:
                        complete_weight = row['complete_mst_weight'] if pd.notna(row['complete_mst_weight']) else None
                        complete_ratio = row['complete_mst_ratio'] if pd.notna(row['complete_mst_ratio']) else None

                    result = InstanceResult(
                        instance_id=int(row['instance_id']),
                        num_terminals=int(row['num_terminals']),
                        optimal_weight=float(row['optimal_weight']),
                        predicted_weight=row['predicted_weight'] if pd.notna(row['predicted_weight']) else None,
                        optimality_ratio=row['optimality_ratio'] if pd.notna(row['optimality_ratio']) else None,
                        random_tree_weight=random_weight,
                        random_tree_ratio=random_ratio,
                        planar_mst_weight=planar_weight,
                        planar_mst_ratio=planar_ratio,
                        complete_mst_weight=complete_weight,
                        complete_mst_ratio=complete_ratio,
                        is_valid_tree=bool(row['is_valid_tree']),
                        is_connected=bool(row['is_connected']),
                        covers_all_terminals=bool(row['covers_all_terminals']),
                        num_extracted_vertices=int(row['num_extracted_vertices']),
                        num_extracted_edges=int(row['num_extracted_edges']),
                        extraction_success=bool(row['extraction_success']),
                        inference_time=float(row['inference_time']),
                        extraction_time=float(row['extraction_time']),
                        error_message=row['error_message'] if pd.notna(row['error_message']) else None
                    )
                    results.append(result)

                print(f"Loaded {len(results)} results from CSV")
                return results

        evaluator = ReportOnlyEvaluator(args.output_dir)

        # Copy the generate_reports method from SteinerEvaluator
        evaluator.generate_reports = SteinerEvaluator.generate_reports.__get__(evaluator)
        evaluator._generate_visualizations = SteinerEvaluator._generate_visualizations.__get__(evaluator)
        evaluator._generate_debug_grid = SteinerEvaluator._generate_debug_grid.__get__(evaluator)
        evaluator._generate_comparison_grid = SteinerEvaluator._generate_comparison_grid.__get__(evaluator)

        # Load results and generate reports
        results = evaluator.load_results_from_csv()
        evaluator.generate_reports(results)

        print("Reports regenerated successfully!")
        return 0

    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return 1

    # Verify steiner data exists
    steiner_data_path = Path(args.steiner_data_path)
    if not steiner_data_path.exists():
        print(f"Error: Steiner data not found at {steiner_data_path}")
        return 1

    # Run evaluation
    evaluator = SteinerEvaluator(
        checkpoint_path=args.checkpoint,
        steiner_data_path=args.steiner_data_path,
        output_dir=args.output_dir,
        device=args.device,
        debug=args.debug,
        seed=args.seed,
        use_regression=args.use_regression,
        best_of_n=args.best_of_n,
        num_workers=args.num_workers,
        inference_steps=args.inference_steps
    )

    # Limit instances if requested
    if args.limit:
        # Create a limited dataset by modifying the dataset indices
        limited_indices = evaluator.dataset.indices[:args.limit]
        evaluator.dataset.indices = limited_indices
        print(f"Limited evaluation to {len(evaluator.dataset)} instances")

    results = evaluator.run_evaluation()
    evaluator.generate_reports(results)
    
    print("Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
