
import sys
from pathlib import Path as _Path

# Ensure project root is on sys.path so absolute imports like `utils.*` work
# when running this file directly (e.g., `uv run scripts/evaluate_steiner.py`).
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import json
import argparse
import math
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
import cv2
import matplotlib.pyplot as plt

from utils.seed import seed_everything
from utils.config_handler import remove_weight_prefixes
from data.PolygonDataset import PolygonDataset
from model.diffusion import UNet
from schedulers.ddim import DDIM
from scripts.extract_polygon_graph import PolygonGraphExtractor
from polygon_generation.random_polygonization import generate_random_polygon_variations


@dataclass
class PolygonInstanceResult:
    """Results for a single polygonization instance."""
    instance_id: int
    num_points: int
    optimal_area: float
    optimal_perimeter: float
    predicted_area: Optional[float]
    predicted_perimeter: Optional[float]
    area_ratio: Optional[float]
    perimeter_ratio: Optional[float]
    compactness_optimal: Optional[float]
    compactness_predicted: Optional[float]
    is_valid_polygon: bool
    num_extracted_vertices: int
    point_coverage_rate: float
    extraction_success: bool
    inference_time: float
    extraction_time: float
    matches_ground_truth: bool
    error_message: Optional[str] = None


class PolygonizationEvaluator:
    def __init__(
        self,
        checkpoint_path: str,
        polygon_data_path: str = "data/polygon_data",
        output_dir: str = "evaluation_results",
        device: str = "auto",
        debug: bool = False,
        seed: int = 42,
        use_regression: bool = False,
        best_of_n: int = 1,
        generate_comparison_grid: bool = True,
        use_whole_dataset: bool = False,
        use_random_baseline: bool = False
    ):
        """
        Initialize the polygonization evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            polygon_data_path: Path to polygon data directory
            output_dir: Output directory for results
            device: Device to use ('cuda', 'cpu', or 'auto')
            debug: Enable debug mode with visualizations
            seed: Random seed for reproducibility
            use_regression: Use direct regression inference instead of diffusion sampling
            best_of_n: For diffusion models, generate n samples and select the best one
            use_whole_dataset: If True, use entire dataset instead of just validation split
            use_random_baseline: If True, generate random valid polygons instead of using model
        """
        self.checkpoint_path = checkpoint_path
        self.polygon_data_path = Path(polygon_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.debug = debug
        self.use_regression = use_regression
        self.best_of_n = best_of_n
        self.use_whole_dataset = use_whole_dataset
        self.use_random_baseline = use_random_baseline
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        seed_everything(seed)
        
        # Track debug images for grid compilation
        self.debug_images = [] if debug else None
        # Track raw comparison images (GT vs Generated)
        self.comparison_images = [] if generate_comparison_grid else None

        # Define color schemes for different samples (same as trajectory visualizer)
        self.color_schemes = ["lightblue", "green", "orange", "pink", "yellow", "cyan"]

        # Initialize components
        if not self.use_random_baseline:
            self._load_model()
        else:
            self._init_dataset()
        self._init_polygon_extractor()

    def _init_dataset(self):
        """Initialize dataset for random baseline mode."""
        # Use default config values for dataset initialization
        self.fill_polygon = False
        self.image_size = 128

        # Determine dataset split parameters
        if self.use_whole_dataset:
            # Use entire dataset by putting everything in validation
            split = 'val'
            train_split = 0.0  # Put everything in validation split
        else:
            # Use validation split as test data (default behavior)
            split = 'val'
            train_split = 0.8

        self.dataset = PolygonDataset(
            data_dir=str(self.polygon_data_path),
            split=split,
            train_split=train_split,
            image_size=self.image_size,
            node_radius=2,
            edge_width=2,
            fill_polygon=self.fill_polygon,
            seed=42  # Same seed for consistent splits
        )
        print(f"Dataset initialized for random baseline mode")

    def _load_model(self):
        """Load the trained diffusion model."""
        if self.use_random_baseline:
            print("Using random baseline - skipping model loading")
            return
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
            self.scheduler = DDIM(device=self.device, **self.config['scheduler'])
        else:
            self.scheduler = None
        
        # Initialize PolygonDataset for the test instances
        self.fill_polygon = self.config['dataset'].get('fill_polygon', False)

        # Determine dataset split parameters
        if self.use_whole_dataset:
            # Use entire dataset by putting everything in validation
            split = 'val'
            train_split = 0.0  # Put everything in validation split
        else:
            # Use validation split as test data (default behavior)
            split = 'val'
            train_split = 0.8

        self.dataset = PolygonDataset(
            data_dir=str(self.polygon_data_path),
            split=split,
            train_split=train_split,
            image_size=self.config['dataset']['image_size'],
            node_radius=self.config['dataset'].get('node_radius', 2),
            edge_width=self.config['dataset'].get('edge_width', 2),
            fill_polygon=self.fill_polygon,
            seed=42  # Same seed for consistent splits
        )
        self.image_size = self.config['dataset']['image_size']
        print(f"Model loaded successfully on {self.device}")
        
    def _init_polygon_extractor(self):
        """Initialize the polygon extraction component."""
        self.polygon_extractor = PolygonGraphExtractor(
            debug=self.debug  # Enable debug if overall debug is enabled
        )
        
    def run_evaluation(self, limit: Optional[int] = None) -> List[PolygonInstanceResult]:
        """Run evaluation on all test instances."""
        total_instances = min(len(self.dataset), limit) if limit else len(self.dataset)
        print(f"Starting evaluation on {total_instances} instances...")
        
        results = []
        failed_count = 0
        
        for instance_idx in tqdm(range(total_instances), desc="Evaluating instances"):
            try:
                result = self._evaluate_instance(instance_idx)
                results.append(result)
                
                if not result.extraction_success:
                    failed_count += 1
                    
            except Exception as e:
                instance = self.dataset.get_sample_info(instance_idx)
                print(f"Failed to evaluate instance {instance['instance_id']}: {e}")
                failed_count += 1
                results.append(PolygonInstanceResult(
                    instance_id=instance['instance_id'],
                    num_points=instance['num_points'],
                    optimal_area=instance.get('polygon_area', 0.0),
                    optimal_perimeter=instance.get('polygon_perimeter', 0.0),
                    predicted_area=None,
                    predicted_perimeter=None,
                    area_ratio=None,
                    perimeter_ratio=None,
                    compactness_optimal=None,
                    compactness_predicted=None,
                    is_valid_polygon=False,
                    num_extracted_vertices=0,
                    point_coverage_rate=0.0,
                    extraction_success=False,
                    inference_time=0,
                    extraction_time=0,
                    matches_ground_truth=False,
                    error_message=str(e)
                ))
        
        print(f"Evaluation completed. {failed_count} instances failed.")
        return results
    
    def _evaluate_instance(self, instance_idx: int) -> PolygonInstanceResult:
        """Evaluate a single instance using dataset."""
        # Get condition and target from dataset
        condition, target = self.dataset[instance_idx]
        condition = condition.unsqueeze(0).to(self.device)  # Add batch dimension: (1, 1, 128, 128)
        
        # Get instance metadata for metrics using dataset's correct indexing
        instance = self.dataset.get_sample_info(instance_idx)
        instance_id = instance['instance_id']
        
        # Run inference or generate random polygon
        start_time = time.time()

        if self.use_random_baseline:
            # Generate random polygon baseline
            gt_points = instance.get('points', [])
            if isinstance(gt_points[0], list):
                gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
            else:
                gt_points = [(float(p[0]), float(p[1])) for p in gt_points]

            # Generate a single random polygon variation
            random_polygons = generate_random_polygon_variations(
                gt_points,
                num_variations=1,
                seed=42 + instance_idx,  # Unique seed per instance
                min_area_ratio=0.01,  # Allow small polygons
                use_cpp=False  # Use Python implementation for consistency
            )

            if random_polygons:
                random_polygon = random_polygons[0]
                order = random_polygon['order']

                # Create a synthetic image representation (not used but needed for consistency)
                pred_img_tensor = torch.zeros((1, 1, self.image_size, self.image_size))

                if self.debug and instance_idx < 3:
                    print(f"Random polygon generated: order={order}, area={random_polygon['area']:.4f}")
            else:
                # Fallback: create empty prediction
                pred_img_tensor = torch.zeros((1, 1, self.image_size, self.image_size))
                order = list(range(len(gt_points)))  # Simple order as fallback

        else:
            # Model-based inference
            with torch.no_grad():
                if self.use_regression:
                    # Direct regression inference (no diffusion process)
                    target_shape = (1, 1, self.image_size, self.image_size)  # Same shape as condition
                    zero_noise = torch.zeros(target_shape, device=self.device)
                    t = torch.zeros(1, 1, device=self.device)  # t=0 for regression
                    pred_img_tensor = self.model(zero_noise, t, condition)
                else:
                    # Diffusion sampling with best-of-n selection
                    if self.best_of_n > 1:
                        # Generate all samples in a single batch for efficiency
                        # Expand condition to match batch size
                        condition_expanded = condition.expand(self.best_of_n, -1, -1, -1)  # [n, 1, 128, 128]

                        pred_result = self.scheduler.sample(
                            self.model,
                            n_samples=self.best_of_n,
                            condition=condition_expanded,
                            guidance_scale=1,
                            seed=42,
                            return_dict=True,
                        )
                        all_samples = pred_result['x_0']  # [n, 1, 128, 128]

                        # Score each sample and select the best one
                        best_pred = None
                        best_score = -float('inf')
                        best_idx = 0

                        for sample_idx in range(self.best_of_n):
                            sample_tensor = all_samples[sample_idx:sample_idx+1]  # Keep batch dimension

                            # Convert to binary for quality evaluation
                            sample_binary = self._convert_tensor_to_binary(sample_tensor[0, 0])

                            # Score this sample (use area-based metric for polygons)
                            score = self._score_polygon_sample(sample_binary, instance)

                            if score > best_score:
                                best_score = score
                                best_pred = sample_tensor
                                best_idx = sample_idx

                        pred_img_tensor = best_pred

                        if self.debug and instance_idx < 3:
                            print(f"Best-of-{self.best_of_n} selection: best score = {best_score:.4f} (sample {best_idx})")
                    else:
                        # Single sample (original behavior)
                        pred_result = self.scheduler.sample(
                            self.model,
                            n_samples=1,
                            condition=condition,
                            guidance_scale=1,
                            seed=42,
                            return_dict=True,
                        )
                        pred_img_tensor = pred_result['x_0']
        inference_time = time.time() - start_time
        
        # Convert prediction to new format: background=0, edges=127, interior=-127
        pred_img_np = pred_img_tensor[0, 0].cpu().numpy()  # Shape: (128, 128)
        # Map [-1, 1] to [-127, 127]: +1->127 (edges), -1->-127 (interior), 0->0 (background)
        # Keep as float32 to preserve gradations, only convert to int16 for extraction
        pred_binary = (pred_img_np * 127).astype(np.float32)
        
        # Ensure the image is 128x128 for polygon extraction
        if pred_binary.shape != (self.image_size, self.image_size):
            pred_binary = cv2.resize(pred_binary, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # Pass to polygon extractor (expects new format: bg=0, edges=127, interior=-127)
        pred_binary_for_extraction = pred_binary
        
        if self.debug and instance_idx < 3:  # Debug first few instances
            print(f"\\nInstance {instance_id} binary conversion:")
            print(f"  pred_img_tensor range: [{pred_img_tensor.min():.3f}, {pred_img_tensor.max():.3f}]")
            print(f"  pred_binary unique values: {np.unique(pred_binary)}")
            print(f"  pred_binary shape: {pred_binary.shape}")
            
            # Check a few pixels around GT points
            gt_points = instance.get('points', [])[:2]  # First 2 points
            for i, (x, y) in enumerate(gt_points):
                px = int(x * (self.image_size - 1))
                py = int(y * (self.image_size - 1))
                val = pred_binary[py, px] if 0 <= px < self.image_size and 0 <= py < self.image_size else "OOB"
                print(f"  GT point {i} ({x:.3f},{y:.3f}) -> pixel ({px},{py}): value={val}")
        
        # Extract polygon structure
        start_time = time.time()
        try:
            if self.use_random_baseline:
                # For random baseline, use the generated order directly
                gt_points = instance.get('points', [])
                if isinstance(gt_points[0], list):
                    gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
                else:
                    gt_points = [(float(p[0]), float(p[1])) for p in gt_points]

                # Convert polygon order to vertices in pixel coordinates
                vertices = []
                for point_idx in order:
                    if point_idx < len(gt_points):
                        x, y = gt_points[point_idx]
                        px = int(x * (self.image_size - 1))
                        py = int(y * (self.image_size - 1))
                        vertices.append((px, py))

                # Calculate area and perimeter from the polygon
                if len(vertices) >= 3:
                    # Calculate area using shoelace formula
                    n = len(vertices)
                    area_sum = 0
                    for i in range(n):
                        j = (i + 1) % n
                        area_sum += vertices[i][0] * vertices[j][1]
                        area_sum -= vertices[j][0] * vertices[i][1]
                    pred_area = abs(area_sum) / 2.0

                    # Convert to normalized coordinates for area scaling
                    pixel_area = pred_area
                    normalized_area = pixel_area / ((self.image_size - 1) ** 2)
                    pred_area = normalized_area

                    # Calculate perimeter
                    perimeter = 0
                    for i in range(n):
                        j = (i + 1) % n
                        dx = vertices[j][0] - vertices[i][0]
                        dy = vertices[j][1] - vertices[i][1]
                        perimeter += math.sqrt(dx*dx + dy*dy)

                    # Normalize perimeter
                    pred_perimeter = perimeter / (self.image_size - 1)

                    # Generate edges list for visualization
                    edges = [(i, (i + 1) % len(order)) for i in range(len(order))]
                else:
                    pred_area = 0.0
                    pred_perimeter = 0.0
                    edges = []

                extraction_success = True
                error_message = None
            else:
                # Model-based: extract from image using existing method
                gt_points = instance.get('points', [])
                if isinstance(gt_points[0], list):
                    gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
                else:
                    gt_points = [(float(p[0]), float(p[1])) for p in gt_points]

                vertices, pred_area, pred_perimeter, edges = self.polygon_extractor.extract_polygon_from_points(
                    pred_binary_for_extraction, gt_points
                )
                extraction_success = True
                error_message = None
        except Exception as e:
            vertices, pred_area, pred_perimeter, edges = [], 0.0, 0.0, []
            extraction_success = False
            error_message = str(e)
        extraction_time = time.time() - start_time
        
        # Calculate metrics
        if extraction_success and len(vertices) >= 3:
            # Calculate performance metrics
            optimal_area = instance.get('polygon_area', 0.0)
            optimal_perimeter = instance.get('polygon_perimeter', 0.0)
            
            # Area and perimeter ratios
            area_ratio = pred_area / optimal_area if optimal_area > 0 else None
            perimeter_ratio = pred_perimeter / optimal_perimeter if optimal_perimeter > 0 else None
            
            # Compactness scores (4π·area/perimeter²)
            compactness_optimal = self._calculate_compactness(optimal_area, optimal_perimeter)
            compactness_predicted = self._calculate_compactness(pred_area, pred_perimeter)
            
            # Check point coverage
            point_coverage_rate = self._check_point_coverage(vertices, instance)

            # Check if extracted polygon matches ground truth (simplified: area ratio ≈ 1)
            matches_ground_truth = area_ratio is not None and abs(area_ratio - 1.0) < 0.001  # Within 0.1% of optimal

            is_valid_polygon = len(vertices) >= 3 and self._is_valid_polygon(vertices)
            
        else:
            optimal_area = instance.get('polygon_area', 0.0)
            optimal_perimeter = instance.get('polygon_perimeter', 0.0)
            area_ratio = None
            perimeter_ratio = None
            compactness_optimal = self._calculate_compactness(optimal_area, optimal_perimeter)
            compactness_predicted = None
            point_coverage_rate = 0.0
            matches_ground_truth = False
            is_valid_polygon = False
        
        # Save debug visualization if requested
        # Convert condition to same format as target/prediction for consistent visualization
        condition_np = (condition[0, 0].cpu().numpy() * 127).astype(np.int16)

        # Convert target to new format for visualization
        target_np = target.squeeze(0).cpu().numpy()  # Shape: (128, 128)
        # Map [-1, 1] to [-127, 127]: +1->127 (edges), -1->-127 (interior), 0->0 (background)
        target_viz = (target_np * 127).astype(np.int16)

        if self.debug:
            self._save_debug_visualization(
                instance_id, condition_np, target_viz, pred_binary, vertices, edges, instance, pred_area
            )

        # Store raw comparison images for grid (convert to discrete with color cycling)
        if self.comparison_images is not None:
            # Cycle through color schemes for each sample
            color_scheme = self.color_schemes[len(self.comparison_images) % len(self.color_schemes)]

            target_discrete = self._convert_tensor_to_discrete(target.squeeze(0), color_scheme)
            pred_discrete = self._convert_tensor_to_discrete(pred_img_tensor[0, 0], color_scheme)
            self.comparison_images.append((instance_id, target_discrete, pred_discrete, instance))
        
        return PolygonInstanceResult(
            instance_id=instance_id,
            num_points=instance['num_points'],
            optimal_area=optimal_area,
            optimal_perimeter=optimal_perimeter,
            predicted_area=pred_area,
            predicted_perimeter=pred_perimeter,
            area_ratio=area_ratio,
            perimeter_ratio=perimeter_ratio,
            compactness_optimal=compactness_optimal,
            compactness_predicted=compactness_predicted,
            is_valid_polygon=is_valid_polygon,
            num_extracted_vertices=len(vertices),
            point_coverage_rate=point_coverage_rate,
            extraction_success=extraction_success,
            inference_time=inference_time,
            extraction_time=extraction_time,
            matches_ground_truth=matches_ground_truth,
            error_message=error_message
        )
    
    def _calculate_compactness(self, area: float, perimeter: float) -> Optional[float]:
        """Calculate compactness score (4π·area/perimeter²)."""
        if area > 0 and perimeter > 0:
            return 4 * np.pi * area / (perimeter ** 2)
        return None
    
    def _is_valid_polygon(self, vertices: List[Tuple[int, int]]) -> bool:
        """Check if the extracted vertices form a valid polygon."""
        if len(vertices) < 3:
            return False
        
        # Additional validity checks could be added here
        # For now, just check minimum vertex count
        return True
    
    def _check_point_coverage(
        self, 
        vertices: List[Tuple[int, int]], 
        instance: Dict,
        threshold: float = 15.0  # pixel distance threshold
    ) -> float:
        """Check how well extracted polygon vertices cover the input points.
        
        Args:
            vertices: Extracted vertices in pixel coordinates
            instance: Instance data containing original points
            threshold: Maximum distance for a point to be considered covered
            
        Returns:
            Coverage rate (0.0 to 1.0)
        """
        if not vertices:
            return 0.0
            
        # Get original points in pixel coordinates
        original_points = np.array(instance['points'])
        original_pixel_coords = (original_points * (self.image_size - 1)).astype(int)
        
        covered_points = 0
        for orig_x, orig_y in original_pixel_coords:
            # Find minimum distance to any extracted vertex
            min_dist = float('inf')
            for vert_x, vert_y in vertices:
                dist = np.sqrt((orig_x - vert_x)**2 + (orig_y - vert_y)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist <= threshold:
                covered_points += 1
        
        return covered_points / len(original_pixel_coords)


    def _convert_tensor_to_binary(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert model output tensor to new format: bg=0, edges=127, interior=-127."""
        tensor_np = tensor.cpu().numpy()  # Shape: (128, 128)
        # Map [-1, 1] to [-127, 127]: +1->127 (edges), -1->-127 (interior), 0->0 (background)
        # Keep as float32 to preserve continuous values and avoid binarization
        binary_img = (tensor_np * 127).astype(np.float32)

        # Ensure the image is 128x128
        if binary_img.shape != (self.image_size, self.image_size):
            binary_img = cv2.resize(binary_img, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return binary_img

    def _convert_tensor_to_discrete(self, tensor: torch.Tensor, color_scheme: str = "default") -> np.ndarray:
        """Convert a model output tensor to a discrete 3-value image for visualization.

        Representation:
        - 0   = interior (black)
        - 127 = background (gray)
        - 255 = edges (white)

        Args:
            tensor: Model output tensor
            color_scheme: Color scheme to use ("default" for grayscale, or specific color name)
        """
        # Ensure 2D numpy array
        if tensor.dim() == 3:
            tensor = tensor[0]
        tensor_np = tensor.detach().cpu().numpy()

        H, W = tensor_np.shape

        # Direct classification from [-1, 1] to 3 classes with robust thresholds
        v = tensor_np
        interior_mask = v <= -0.5
        edge_mask = v >= 0.5
        # Background is the rest

        if color_scheme == "default":
            # Assemble discrete image: grayscale
            out = np.ones((H, W), dtype=np.uint8) * 127  # Start with background (gray)
            out[edge_mask] = 255  # Edges are white
            out[interior_mask] = 0  # Interior is black
        else:
            # Create RGB image with colored edges/interior
            out = np.ones((H, W, 3), dtype=np.uint8) * 127

            # Define color schemes (BGR format for OpenCV)
            color_schemes = {
                "lightblue": {"edge": [255, 255, 255], "interior": [230, 216, 173]},
                "green": {"edge": [255, 255, 255], "interior": [50, 200, 50]},
                "orange": {"edge": [255, 255, 255], "interior": [0, 130, 200]},
                "pink": {"edge": [255, 255, 255], "interior": [203, 192, 255]},
                "yellow": {"edge": [255, 255, 255], "interior": [0, 200, 200]},
                "cyan": {"edge": [255, 255, 255], "interior": [200, 200, 0]},
            }

            colors = color_schemes.get(color_scheme, color_schemes["lightblue"])
            out[edge_mask] = colors["edge"]
            out[interior_mask] = colors["interior"]

        # Ensure correct size
        if out.shape[:2] != (self.image_size, self.image_size):
            if len(out.shape) == 3:  # Color image
                out = cv2.resize(out, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            else:  # Grayscale image
                out = cv2.resize(out, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return out
    
    def _score_polygon_sample(self, binary_img: np.ndarray, instance: Dict) -> float:
        """Score a polygon sample for best-of-n selection."""
        try:
            # Get GT points from instance data
            gt_points = instance.get('points', [])
            if isinstance(gt_points[0], list):
                gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
            else:
                gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
            
            # Extract polygon using the same method as evaluation
            vertices, pred_area, pred_perimeter, edges = self.polygon_extractor.extract_polygon_from_points(
                binary_img, gt_points
            )
            
            if len(vertices) < 3 or pred_area <= 0:
                return -1.0  # Invalid polygon gets low score
            
            # Use area ratio as the primary scoring metric (closer to optimal = higher score)
            optimal_area = instance.get('polygon_area', 0.0)
            if optimal_area > 0:
                area_ratio = pred_area / optimal_area
                # Score favors ratios close to 1.0, penalizes deviations
                score = 1.0 - abs(1.0 - area_ratio)
                return max(score, 0.0)  # Ensure non-negative
            else:
                # Fallback: just use predicted area (larger is better for max area problem)
                return pred_area
                
        except Exception as e:
            # Failed extraction gets very low score
            return -2.0
    
    def _save_debug_visualization(
        self,
        instance_id: int,
        points_img: np.ndarray,
        target_img: np.ndarray,
        pred_img: np.ndarray,
        vertices: List[Tuple[int, int]],
        edges: List[Tuple[int, int]],
        instance: Dict,
        pred_area: float
    ):
        """Save debug visualization for an instance."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Input points (new format: bg=0, edges=127, interior=-127)
        points_viz = points_img.copy().astype(float)
        # Map: -127→64 (dark gray), 0→127 (mid gray), 127→255 (white)
        points_viz = np.clip(points_viz + 127, 0, 254).astype(np.uint8)
        axes[0].imshow(points_viz, cmap='gray')
        axes[0].set_title('Input (Points)')
        axes[0].axis('off')

        # Ground truth polygon from dataset target (new format: bg=0, edges=127, interior=-127)
        target_viz = target_img.copy().astype(float)
        # Map: -127→64 (dark gray), 0→127 (mid gray), 127→255 (white)
        target_viz = np.clip(target_viz + 127, 0, 254).astype(np.uint8)
        axes[1].imshow(target_viz, cmap='gray')
        optimal_area = instance.get('polygon_area', 0.0)
        axes[1].set_title(f'Ground Truth Optimal\\n(Area: {optimal_area:.3f})')
        axes[1].axis('off')

        # Use the area already calculated from float coordinates during extraction
        area_ratio = (pred_area / optimal_area) if optimal_area > 0 and pred_area > 0 else float('inf')

        # Visualize prediction (new format: bg=0, edges=127, interior=-127)
        pred_viz = pred_img.copy().astype(float)
        # Map: -127→0 (black), 0→127 (mid gray), 127→254 (white)
        pred_viz = np.clip(pred_viz + 127, 0, 254).astype(np.uint8)
        axes[2].imshow(pred_viz, cmap='gray')
        
        # Show GT points and detected edges
        gt_points = instance.get('points', [])
        if gt_points:
            # Convert GT points to pixel coordinates for visualization
            image_size = self.image_size
            gt_pixel_coords = []
            for x, y in gt_points:
                px = int(x * (image_size - 1))
                py = int(y * (image_size - 1))
                gt_pixel_coords.append((px, py))
            
            # Draw GT points
            gt_x = [p[0] for p in gt_pixel_coords]
            gt_y = [p[1] for p in gt_pixel_coords]
            axes[2].scatter(gt_x, gt_y, c='red', s=40, zorder=5, marker='o', label='GT Points')
            
            # Draw detected edges between GT points
            for i, j in edges:
                if i < len(gt_pixel_coords) and j < len(gt_pixel_coords):
                    x1, y1 = gt_pixel_coords[i]
                    x2, y2 = gt_pixel_coords[j]
                    axes[2].plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.7)
            
            # Draw final polygon if vertices exist
            if vertices and len(vertices) >= 3:
                vertex_x = [v[0] for v in vertices]
                vertex_y = [v[1] for v in vertices]
                # Close the polygon
                vertex_x_closed = vertex_x + [vertex_x[0]]
                vertex_y_closed = vertex_y + [vertex_y[0]]
                axes[2].plot(vertex_x_closed, vertex_y_closed, 'b-', linewidth=3, alpha=0.8, label='Final Polygon')
        
        # Create validation status string
        valid_str = "✓" if len(vertices) >= 3 else "✗"
        coverage = self._check_point_coverage(vertices, instance)
        
        axes[2].set_title(f'Detected Structure ({len(vertices)}V, {len(edges)}E)\\n'
                         f'Area: {pred_area:.3f}, Ratio: {area_ratio:.3f}\\n'
                         f'Valid: {valid_str}, Coverage: {coverage:.1%}')
        axes[2].axis('off')
        axes[2].legend()
        
        debug_path = self.output_dir / f"debug_instance_{instance_id:04d}.png"
        plt.tight_layout()
        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
        
        # Store debug image for grid compilation if debug mode is enabled
        if self.debug_images is not None:
            # Read the saved debug image and store it
            debug_img = cv2.imread(str(debug_path))
            if debug_img is not None:
                self.debug_images.append((instance_id, debug_img))
        
        plt.close()
    
    def generate_reports(self, results: List[PolygonInstanceResult]):
        """Generate comprehensive evaluation reports."""
        print("Generating evaluation reports...")
        
        # Save detailed results as JSON
        results_data = []
        for result in results:
            results_data.append({
                'instance_id': result.instance_id,
                'num_points': result.num_points,
                'optimal_area': result.optimal_area,
                'optimal_perimeter': result.optimal_perimeter,
                'predicted_area': result.predicted_area,
                'predicted_perimeter': result.predicted_perimeter,
                'area_ratio': result.area_ratio,
                'perimeter_ratio': result.perimeter_ratio,
                'compactness_optimal': result.compactness_optimal,
                'compactness_predicted': result.compactness_predicted,
                'is_valid_polygon': result.is_valid_polygon,
                'num_extracted_vertices': result.num_extracted_vertices,
                'point_coverage_rate': result.point_coverage_rate,
                'extraction_success': result.extraction_success,
                'inference_time': result.inference_time,
                'extraction_time': result.extraction_time,
                'matches_ground_truth': result.matches_ground_truth,
                'error_message': result.error_message
            })
        
        # Save JSON report
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "results.csv", index=False)
        
        # Generate summary statistics
        successful_results = [r for r in results if r.extraction_success and r.predicted_area is not None]
        
        summary = {
            'total_instances': len(results),
            'successful_extractions': len(successful_results),
            'extraction_success_rate': len(successful_results) / len(results) if results else 0,
            'valid_polygons': sum(1 for r in successful_results if r.is_valid_polygon),
            'gt_matches': sum(1 for r in successful_results if r.matches_ground_truth),
            'gt_match_rate': sum(1 for r in successful_results if r.matches_ground_truth) / len(successful_results) if successful_results else 0,
            'mean_point_coverage': np.mean([r.point_coverage_rate for r in successful_results]) if successful_results else 0,
        }
        
        if successful_results:
            # Cap ratios at 1.0 (treat anything above 1.0 as 1.0)
            area_ratios = [min(r.area_ratio, 1.0) for r in successful_results if r.area_ratio is not None and r.is_valid_polygon]
            perimeter_ratios = [min(r.perimeter_ratio, 1.0) for r in successful_results if r.perimeter_ratio is not None and r.is_valid_polygon]
            compactness_scores = [r.compactness_predicted for r in successful_results if r.compactness_predicted is not None and r.is_valid_polygon]
            
            if area_ratios:
                summary.update({
                    'mean_area_ratio': np.mean(area_ratios),
                    'median_area_ratio': np.median(area_ratios),
                    'std_area_ratio': np.std(area_ratios),
                    'min_area_ratio': np.min(area_ratios),
                    'max_area_ratio': np.max(area_ratios),
                })
                
            if perimeter_ratios:
                summary.update({
                    'mean_perimeter_ratio': np.mean(perimeter_ratios),
                    'median_perimeter_ratio': np.median(perimeter_ratios),
                })
                
            if compactness_scores:
                summary.update({
                    'mean_compactness_predicted': np.mean(compactness_scores),
                    'median_compactness_predicted': np.median(compactness_scores),
                })
                
            summary.update({
                'mean_inference_time': np.mean([r.inference_time for r in successful_results]),
                'mean_extraction_time': np.mean([r.extraction_time for r in successful_results])
            })
        
        # Save summary
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary to console
        print("\\n" + "="*50)
        print("POLYGONIZATION EVALUATION SUMMARY")
        print("="*50)
        print(f"Total instances: {summary['total_instances']}")
        print(f"Successful extractions: {summary['successful_extractions']} ({summary['extraction_success_rate']:.1%})")
        
        if successful_results:
            print(f"Valid polygons: {summary['valid_polygons']} ({summary['valid_polygons']/len(successful_results):.1%})")
            print(f"Ground truth matches: {summary['gt_matches']} ({summary['gt_match_rate']:.1%})")
            print(f"Mean point coverage: {summary['mean_point_coverage']:.1%}")
            
            if 'mean_area_ratio' in summary:
                print(f"\\nArea Ratio Statistics:")
                print(f"  Mean: {summary['mean_area_ratio']:.3f}")
                print(f"  Median: {summary['median_area_ratio']:.3f}")
                print(f"  Std: {summary['std_area_ratio']:.3f}")
                print(f"  Range: [{summary['min_area_ratio']:.3f}, {summary['max_area_ratio']:.3f}]")
                
            if 'mean_perimeter_ratio' in summary:
                print(f"\\nPerimeter Ratio Statistics:")
                print(f"  Mean: {summary['mean_perimeter_ratio']:.3f}")
                print(f"  Median: {summary['median_perimeter_ratio']:.3f}")
                
            if 'mean_compactness_predicted' in summary:
                print(f"\\nCompactness Statistics:")
                print(f"  Mean: {summary['mean_compactness_predicted']:.3f}")
                print(f"  Median: {summary['median_compactness_predicted']:.3f}")
                
            print(f"\\nTiming:")
            print(f"  Mean inference time: {summary['mean_inference_time']:.3f}s")
            print(f"  Mean extraction time: {summary['mean_extraction_time']:.3f}s")
        
        print(f"\\nResults saved to: {self.output_dir}")
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        # Generate grid visualization if debug images were collected
        if self.debug_images:
            self._generate_debug_grid()

        # Generate GT vs Generated comparison grid
        if self.comparison_images:
            self._generate_comparison_grid()
    
    def _generate_visualizations(self, results: List[PolygonInstanceResult]):
        """Generate visualization plots for the results."""
        successful_results = [r for r in results if r.extraction_success and r.predicted_area is not None]
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # Area ratio histogram (only valid polygons) - cap at 1.0
        area_ratios = [min(r.area_ratio, 1.0) for r in successful_results
                      if r.area_ratio is not None and r.is_valid_polygon]

        if area_ratios:
            plt.figure(figsize=(10, 6))
            plt.hist(area_ratios, bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(x=1.0, color='red', linestyle='--', label='Optimal (ratio=1.0)')
            plt.xlabel('Area Ratio (Predicted/Optimal)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Area Ratios')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "area_ratio_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Scatter plot: area ratio vs number of points (only valid polygons)
            num_points = [r.num_points for r in successful_results
                         if r.area_ratio is not None and r.is_valid_polygon]
            # Re-cap area_ratios for the scatter plot (using the same capped values)
            capped_area_ratios = [min(r.area_ratio, 1.0) for r in successful_results
                                 if r.area_ratio is not None and r.is_valid_polygon]
            plt.figure(figsize=(10, 6))
            plt.scatter(num_points, capped_area_ratios, alpha=0.6)
            plt.axhline(y=1.0, color='red', linestyle='--', label='Optimal')
            plt.xlabel('Number of Points')
            plt.ylabel('Area Ratio')
            plt.title('Area Ratio vs Problem Complexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / "area_ratio_vs_complexity.png", dpi=150, bbox_inches='tight')
            plt.close()

    def _generate_debug_grid(self):
        """Generate a single grid image containing all debug visualizations with borders."""
        if not self.debug_images:
            return
            
        print(f"Generating debug grid with {len(self.debug_images)} images...")
        
        # Sort images by instance ID
        self.debug_images.sort(key=lambda x: x[0])
        
        # Standardize all images to the same size
        # Find the maximum dimensions
        max_height = max(img.shape[0] for _, img in self.debug_images)
        max_width = max(img.shape[1] for _, img in self.debug_images)
        
        # Resize all images to the same dimensions
        standardized_images = []
        for instance_id, debug_img in self.debug_images:
            # Resize to max dimensions
            if debug_img.shape[0] != max_height or debug_img.shape[1] != max_width:
                resized_img = cv2.resize(debug_img, (max_width, max_height))
            else:
                resized_img = debug_img
            standardized_images.append((instance_id, resized_img))
        
        # Use standardized dimensions
        img_height, img_width = max_height, max_width
        
        # Calculate grid dimensions (try to make it roughly square)
        n_images = len(standardized_images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        
        # Border thickness
        border_thickness = 2
        
        # Calculate total grid dimensions
        total_width = cols * img_width + (cols + 1) * border_thickness
        total_height = rows * img_height + (rows + 1) * border_thickness
        
        # Create the grid image (white background)
        grid_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255
        
        # Place each debug image in the grid
        for idx, (instance_id, debug_img) in enumerate(standardized_images):
            row = idx // cols
            col = idx % cols
            
            # Calculate position
            y_start = row * (img_height + border_thickness) + border_thickness
            y_end = y_start + img_height
            x_start = col * (img_width + border_thickness) + border_thickness
            x_end = x_start + img_width
            
            # Place the image
            grid_img[y_start:y_end, x_start:x_end] = debug_img
        
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
        # self.comparison_images.sort(key=lambda x: x[0])

        # Fixed 4 samples per row
        cols = 4
        n_pairs = len(self.comparison_images)
        rows = int(np.ceil(n_pairs / cols))

        # Get image dimensions from first pair
        first_gt = self.comparison_images[0][1]  # target_discrete
        img_h, img_w = first_gt.shape[:2]  # Handle both 2D and 3D arrays

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
            # Convert to grayscale for diff calculation if images are color
            if len(gt_img.shape) == 3:  # Color image
                gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
            else:
                gt_gray = gt_img

            if len(pred_img.shape) == 3:  # Color image
                pred_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
            else:
                pred_gray = pred_img

            diff_raw = gt_gray.astype(np.float32) - pred_gray.astype(np.float32)  # Range: [-255, 255]
            diff_normalized = (diff_raw + 255) / 510  # Normalize to [0, 1]
            diff_colored = (plt.cm.bwr(diff_normalized)[:, :, :3] * 255).astype(np.uint8)  # Apply colormap

            # Convert images to RGB for consistent stacking (handle both grayscale and color)
            if len(gt_img.shape) == 2:  # Grayscale
                gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_GRAY2RGB)
            else:  # Already RGB
                gt_rgb = gt_img

            if len(pred_img.shape) == 2:  # Grayscale
                pred_rgb = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2RGB)
            else:  # Already RGB
                pred_rgb = pred_img

            # Create side-by-side comparison image (GT|Generated|Diff)
            combined_rgb = np.hstack([gt_rgb, pred_rgb, diff_colored])

            # Add polygon points as red dots on GT and Generated sides
            polygon_points = instance_info.get('points', [])
            for px, py in polygon_points:
                # GT side (left)
                px_gt = int(px * (img_w - 1))
                py_gt = int(py * (img_h - 1))
                cv2.circle(combined_rgb, (px_gt, py_gt), 2, (255, 0, 0), -1)  # Red

                # Generated side (middle)
                px_pred = int(px * (img_w - 1)) + img_w
                py_pred = int(py * (img_h - 1))
                cv2.circle(combined_rgb, (px_pred, py_pred), 2, (255, 0, 0), -1)  # Red

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
    parser = argparse.ArgumentParser(description="Evaluate polygonization solver model")
    parser.add_argument(
        "--checkpoint", 
        default="model/checkpoints/checkpoint_1000.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--polygon-data-path",
        default="data/polygon_data",
        help="Path to polygon data directory"
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
        "--no-comparison-grid",
        action="store_true",
        help="Disable generation of comparison grid visualization"
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
        "--from-csv",
        type=str,
        default=None,
        help="Generate reports from existing CSV file (skips evaluation)"
    )
    parser.add_argument(
        "--use-whole-dataset",
        action="store_true",
        help="Use entire dataset instead of just validation split"
    )
    parser.add_argument(
        "--use-random-baseline",
        action="store_true",
        help="Generate random valid polygons instead of using model for baseline comparison"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if args.best_of_n < 1:
        print("Error: --best-of-n must be >= 1")
        return 1
    
    if args.use_regression and args.best_of_n > 1:
        print("Warning: --best-of-n is ignored for regression models (only applies to diffusion)")
    
    # Check if we're generating reports from existing CSV
    if args.from_csv:
        # Load results from CSV
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            print(f"Error: CSV file not found at {csv_path}")
            return 1

        print(f"Loading results from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Convert DataFrame back to list of PolygonInstanceResult objects
        results = []
        for _, row in df.iterrows():
            results.append(PolygonInstanceResult(
                instance_id=int(row['instance_id']),
                num_points=int(row['num_points']),
                optimal_area=float(row['optimal_area']),
                optimal_perimeter=float(row['optimal_perimeter']),
                predicted_area=float(row['predicted_area']) if pd.notna(row['predicted_area']) else None,
                predicted_perimeter=float(row['predicted_perimeter']) if pd.notna(row['predicted_perimeter']) else None,
                area_ratio=float(row['area_ratio']) if pd.notna(row['area_ratio']) else None,
                perimeter_ratio=float(row['perimeter_ratio']) if pd.notna(row['perimeter_ratio']) else None,
                compactness_optimal=float(row['compactness_optimal']) if pd.notna(row['compactness_optimal']) else None,
                compactness_predicted=float(row['compactness_predicted']) if pd.notna(row['compactness_predicted']) else None,
                is_valid_polygon=bool(row['is_valid_polygon']),
                num_extracted_vertices=int(row['num_extracted_vertices']),
                point_coverage_rate=float(row['point_coverage_rate']),
                extraction_success=bool(row['extraction_success']),
                inference_time=float(row['inference_time']),
                extraction_time=float(row['extraction_time']),
                matches_ground_truth=bool(row['matches_ground_truth']),
                error_message=str(row['error_message']) if pd.notna(row['error_message']) else None
            ))

        # Create a minimal evaluator just for report generation
        # We need to create output directory but don't need model loading
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # Create a dummy evaluator with just output_dir set
        evaluator = type('DummyEvaluator', (), {
            'output_dir': output_dir,
            'debug_images': None,
            'comparison_images': None
        })()

        # Generate reports using the loaded results
        # We need to use the generate_reports method, so we'll need to patch it
        import types
        evaluator.generate_reports = types.MethodType(PolygonizationEvaluator.generate_reports, evaluator)
        evaluator._generate_visualizations = types.MethodType(PolygonizationEvaluator._generate_visualizations, evaluator)
        evaluator._generate_debug_grid = types.MethodType(PolygonizationEvaluator._generate_debug_grid, evaluator)
        evaluator._generate_comparison_grid = types.MethodType(PolygonizationEvaluator._generate_comparison_grid, evaluator)

        evaluator.generate_reports(results)
    else:
        # Normal evaluation flow
        # Verify checkpoint exists (skip if using random baseline)
        if not args.use_random_baseline and not Path(args.checkpoint).exists():
            print(f"Error: Checkpoint not found at {args.checkpoint}")
            return 1

        # Verify polygon data exists
        polygon_data_path = Path(args.polygon_data_path)
        if not polygon_data_path.exists():
            print(f"Error: Polygon data not found at {polygon_data_path}")
            return 1

        # Run evaluation
        evaluator = PolygonizationEvaluator(
            checkpoint_path=args.checkpoint,
            polygon_data_path=args.polygon_data_path,
            output_dir=args.output_dir,
            device=args.device,
            debug=args.debug,
            seed=args.seed,
            use_regression=args.use_regression,
            best_of_n=args.best_of_n,
            generate_comparison_grid=not args.no_comparison_grid,
            use_whole_dataset=args.use_whole_dataset,
            use_random_baseline=args.use_random_baseline
        )

        results = evaluator.run_evaluation(limit=args.limit)
        evaluator.generate_reports(results)
    
    print("Evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())