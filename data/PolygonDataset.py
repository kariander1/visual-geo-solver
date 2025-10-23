import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
from scipy.ndimage import distance_transform_edt


class PolygonDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated polygon instances.
    Reads JSON files and converts to the format expected by the network:
    - condition: binary image of all points (polygon vertices)
    - target: binary image of complete polygon
    """
    
    def __init__(
        self,
        data_dir: str = None,
        polygon_data_path: str = None,
        num_samples: Optional[int] = None,
        image_size: int = 128,
        node_radius: int = 7,
        edge_width: int = 2,
        use_distance_transform: bool = False,
        fill_polygon: bool = False,
        split: str = 'train',
        train_split: float = 0.8,
        seed: int = 42,
        **kwargs: Any
    ):
        """
        Args:
            data_dir: Path to directory containing instances/ subdirectory with JSON files
            polygon_data_path: Alternative name for data_dir (for compatibility)
            num_samples: Maximum number of samples to load (None = all)
            image_size: Size of output images
            node_radius: Radius of nodes in pixels
            edge_width: Width of edges in pixels
            use_distance_transform: Whether to apply distance transform to target
            fill_polygon: Whether to fill polygon interior (default True)
            split: 'train' or 'val' split
            train_split: Fraction of data to use for training
            seed: Random seed for reproducible splits
        """
        # Support both parameter names for compatibility
        if polygon_data_path is not None and data_dir is None:
            data_dir = polygon_data_path
        elif data_dir is None and polygon_data_path is None:
            raise ValueError("Either data_dir or polygon_data_path must be provided")
            
        self.data_dir = Path(data_dir)
        self.instances_dir = self.data_dir / "instances"
        self.image_size = image_size
        self.node_radius = node_radius
        self.edge_width = edge_width
        self.use_distance_transform = use_distance_transform
        self.fill_polygon = fill_polygon
        self.max_dist = 2**0.5 * image_size
        self.eps = 1  # in pixels
        
        # Define inverse transform (to go back to binary image)
        # Using instance methods instead of lambda to avoid pickling issues
        
        # Detect data format and load instances
        ndjson_path = self.data_dir / "instances.ndjson"
        if ndjson_path.exists():
            print(f"Detected NDJSON format: {ndjson_path}")
            self.instances = self._load_ndjson(ndjson_path)
            self.use_ndjson = True
        else:
            print(f"Using separate JSON files from: {self.instances_dir}")
            self.instance_files = sorted(self.instances_dir.glob("instance_*.json"))
            if not self.instance_files:
                raise ValueError(f"No instance files found in {self.instances_dir}")
            self.use_ndjson = False
        
        # Determine dataset size
        if self.use_ndjson:
            dataset_size = len(self.instances)
        else:
            dataset_size = len(self.instance_files)
        
        # Limit samples if specified
        if num_samples is not None:
            dataset_size = min(dataset_size, num_samples)
        
        # Split data
        np.random.seed(seed)
        indices = np.random.permutation(dataset_size)
        split_idx = int(len(indices) * train_split)
        
        if split == 'train':
            self.indices = indices[:split_idx]
        elif split == 'val':
            self.indices = indices[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"Loaded {len(self.indices)} {split} instances from {self.data_dir} ({'NDJSON' if self.use_ndjson else 'separate files'})")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _load_ndjson(self, ndjson_path: Path) -> List[Dict[str, Any]]:
        """Load instances from NDJSON file."""
        instances = []
        try:
            with open(ndjson_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        instance = json.loads(line)
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line {line_num} in {ndjson_path}: {e}")
                        continue
            print(f"Loaded {len(instances)} instances from NDJSON file")
            return instances
        except Exception as e:
            raise ValueError(f"Failed to load NDJSON file {ndjson_path}: {e}")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            condition: (1, H, W) tensor with polygon vertices, values in [-1, 1]
            target: (1, H, W) tensor with complete polygon, values in [-1, 1]
        """
        # Load instance data based on format
        if self.use_ndjson:
            data = self.instances[self.indices[idx]]
        else:
            instance_file = self.instance_files[self.indices[idx]]
            with open(instance_file, 'r') as f:
                data = json.load(f)
        
        # Generate condition image (all points)
        points = np.array(data['points'])
        condition_img = self._draw_nodes(points)
        
        # Generate target image (complete polygon)
        polygon_order = data['polygon_order']
        target_img = self._draw_polygon(points, polygon_order)
        
        # Convert to tensors and normalize to [-1, 1]
        condition = self._img_to_tensor(condition_img)
        target = self._img_to_tensor(target_img)
        
        # Apply distance transform if requested
        if self.use_distance_transform:
            target = self._apply_distance_transform(target)
        
        return condition, target
    
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single sample on-demand. Returns (condition, target) tensors."""
        # Pick a random instance
        idx = np.random.randint(0, len(self.indices))
        return self.__getitem__(idx)
    
    def _draw_nodes(self, positions: np.ndarray) -> np.ndarray:
        """Draw nodes as circles on a white background."""
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 127
        
        # Scale positions from [0, 1] to pixel coordinates
        pixel_coords = (positions * (self.image_size - 1)).astype(int)
        
        for center in pixel_coords:
            cv2.circle(img, tuple(center), self.node_radius, color=0, thickness=-1)
        
        return img
    
    def _draw_polygon(self, positions: np.ndarray, polygon_order: List[int]) -> np.ndarray:
        """Draw complete polygon with nodes and edges in specified order."""
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 127
        
        # Scale positions from [0, 1] to pixel coordinates
        pixel_coords = (positions * (self.image_size - 1)).astype(int)
        
        # Draw polygon
        if len(polygon_order) >= 3:
            # Create polygon points in order
            polygon_points = []
            for idx in polygon_order:
                if idx < len(pixel_coords):
                    polygon_points.append(pixel_coords[idx])
            
            if len(polygon_points) >= 3:
                polygon_points = np.array(polygon_points, dtype=np.int32)
                
                # Fill polygon interior if requested
                if self.fill_polygon:
                    cv2.fillPoly(img, [polygon_points], color=0)
                
                # Draw polygon edges
                cv2.polylines(img, [polygon_points], isClosed=True, color=255, thickness=self.edge_width)
        
        # Draw nodes
        # for center in pixel_coords:
        #     cv2.circle(img, tuple(center), self.node_radius, color=0, thickness=-1)
        
        return img
    
    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor."""
        # Normalize from [0, 255] to [-1, 1]
        # Use tensor creation that's more compatible with NumPy versions
        tensor = (torch.from_numpy(img).float() - 127.0) / 127.0
        return tensor.unsqueeze(0)  # Add channel dimension
    
    def _apply_distance_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply distance transform to binary image."""
        # Convert to numpy for distance transform
        img = tensor.squeeze(0).numpy()
        
        # Binary mask (True where background)
        binary_mask = (img != -1)
        
        # Compute distance transform
        distance_map = distance_transform_edt(binary_mask)
        
        # Normalize to [-1, 1]
        # Use tensor creation that's more compatible with NumPy versions
        distance_tensor = torch.tensor(distance_map, dtype=torch.float32)
        distance_tensor = (distance_tensor / self.max_dist) * 2 - 1
        
        return distance_tensor.unsqueeze(0)
    
    def _distance_to_image(self, distance_map):
        """Convert distance transform back to binary image."""
        image = torch.ones_like(distance_map)
        mask = ((distance_map + 1) / 2 * self.max_dist <= self.eps)
        image[mask] = -1
        return image

    def to_image(self, x):
        """Convert tensor to image format. Uses distance transform inverse if enabled."""
        if self.use_distance_transform:
            return self._distance_to_image(x)
        else:
            return x
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        if self.use_ndjson:
            return self.instances[self.indices[idx]]
        else:
            instance_file = self.instance_files[self.indices[idx]]
            with open(instance_file, 'r') as f:
                return json.load(f)


class PolygonDatasetDynamic(PolygonDataset):
    """
    Dynamic version that can generate new samples on-the-fly using polygon solver.
    Falls back to pre-generated data if available.
    """
    
    def __init__(
        self,
        data_dir: str,
        polygon_solver_path: Optional[str] = None,
        generate_on_demand: bool = False,
        **kwargs
    ):
        """
        Args:
            data_dir: Path to pre-generated data
            polygon_solver_path: Path to polygon solver (for dynamic generation)
            generate_on_demand: Whether to generate new samples if requested index exceeds dataset
        """
        super().__init__(data_dir, **kwargs)
        
        self.generate_on_demand = generate_on_demand
        if generate_on_demand and polygon_solver_path:
            # Import polygon generator if needed for dynamic generation
            try:
                from polygon_generation.generate_polygons_data import PolygonDataGenerator
                self.generator = PolygonDataGenerator(polygon_solver_path, data_dir)
            except ImportError:
                print("Warning: Could not import polygon generator for dynamic generation")
                self.generator = None
        else:
            self.generator = None
    
    def __getitem__(self, idx: int):
        """Get item, potentially generating new data if needed."""
        if idx < len(self.indices):
            return super().__getitem__(idx)
        elif self.generate_on_demand and self.generator:
            # Generate new sample on the fly
            num_points = np.random.randint(3, 11)  # Random between 3-10 points
            points = self.generator.generate_random_points(num_points)
            polygon_order, area, _ = self.generator.solve_max_area_polygonalization(points)
            
            # Convert to images
            condition_img = self._draw_nodes(points)
            target_img = self._draw_polygon(points, polygon_order)
            
            condition = self._img_to_tensor(condition_img)
            target = self._img_to_tensor(target_img)
            
            if self.use_distance_transform:
                target = self._apply_distance_transform(target)
            
            return condition, target
        else:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")


if __name__ == "__main__":
    # Test the dataset
    import matplotlib.pyplot as plt
    
    # Load dataset
    dataset = PolygonDataset(
        data_dir="data/polygon_data",
        num_samples=10,
        use_distance_transform=False,
        fill_polygon=False,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        condition, target = dataset[i]
        
        # Convert tensors to numpy for visualization
        condition_np = ((condition.squeeze(0).numpy() + 1) * 127.5).astype(np.uint8)
        target_np = ((target.squeeze(0).numpy() + 1) * 127.5).astype(np.uint8)
        
        axes[0, i].imshow(condition_np, cmap='gray')
        axes[0, i].set_title(f"Points {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_np, cmap='gray')
        axes[1, i].set_title(f"Polygon {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("polygon_dataset_samples.png")
    plt.show()
    
    print("\nSample info for first instance:")
    info = dataset.get_sample_info(0)
    print(f"  Points: {info['num_points']}")
    print(f"  Polygon area: {info['polygon_area']:.4f}")
    print(f"  Solver used: {info['solver_used']}")