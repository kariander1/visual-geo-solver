import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import json
from scipy.ndimage import distance_transform_edt


class GeoSteinerDataset(Dataset):
    """
    PyTorch Dataset for loading pre-generated GeoSteiner instances.
    Reads JSON files and converts to the format expected by the network:
    - condition: binary image of terminal points
    - target: binary image of complete Steiner tree
    """
    
    def __init__(
        self,
        data_dir: str = None,
        steiner_data_path: str = None,
        num_samples: Optional[int] = None,
        image_size: int = 128,
        node_radius: int = 2,
        edge_width: int = 2,
        use_distance_transform: bool = False,
        split: str = 'train',
        train_split: float = 0.8,
        seed: int = 42,
        **kwargs: Any
    ):
        """
        Args:
            data_dir: Path to directory containing instances/ subdirectory with JSON files
            steiner_data_path: Alternative name for data_dir (for compatibility)
            num_samples: Maximum number of samples to load (None = all)
            image_size: Size of output images
            node_radius: Radius of nodes in pixels
            edge_width: Width of edges in pixels
            use_distance_transform: Whether to apply distance transform to target
            split: 'train' or 'val' split
            train_split: Fraction of data to use for training
            seed: Random seed for reproducible splits
        """
        # Support both parameter names for compatibility
        if steiner_data_path is not None and data_dir is None:
            data_dir = steiner_data_path
        elif data_dir is None and steiner_data_path is None:
            raise ValueError("Either data_dir or steiner_data_path must be provided")
            
        self.data_dir = Path(data_dir)
        self.instances_dir = self.data_dir / "instances"
        self.image_size = image_size
        self.node_radius = node_radius
        self.edge_width = edge_width
        self.use_distance_transform = use_distance_transform
        self.max_dist = 2**0.5 * image_size
        self.eps = 1  # in pixels
        
        # Define inverse transform (to go back to binary image)
        # Note: Using method instead of lambda to support multiprocessing pickle
        
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            condition: (1, H, W) tensor with terminal points, values in [-1, 1]
            target: (1, H, W) tensor with complete Steiner tree, values in [-1, 1]
        """
        # Load instance data based on format
        if self.use_ndjson:
            data = self.instances[self.indices[idx]]
        else:
            instance_file = self.instance_files[self.indices[idx]]
            with open(instance_file, 'r') as f:
                data = json.load(f)
        
        # Generate condition image (terminal points only)
        terminal_points = np.array(data['terminal_points'])
        condition_img = self._draw_nodes(terminal_points)
        
        # Generate target image (complete Steiner tree)
        steiner_points = np.array(data['steiner_points']) if data['steiner_points'] else np.empty((0, 2))
        all_points = np.vstack([terminal_points, steiner_points]) if len(steiner_points) > 0 else terminal_points
        edges = data['edges']
        target_img = self._draw_steiner_tree(all_points, edges)
        
        # Convert to tensors and normalize to [-1, 1]
        condition = self._img_to_tensor(condition_img)
        target = self._img_to_tensor(target_img)
        
        # Apply distance transform if requested
        if self.use_distance_transform:
            target = self._apply_distance_transform(target)
        
        return condition, target
    
    def to_image(self, tensor):
        """Convert tensor back to binary image format for visualization."""
        if self.use_distance_transform:
            return self._distance_to_image(tensor)
        else:
            return tensor
    
    def generate_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a single sample on-demand. Returns (condition, target) tensors."""
        # Pick a random instance
        idx = np.random.randint(0, len(self.indices))
        return self.__getitem__(idx)
    
    def _draw_nodes(self, positions: np.ndarray) -> np.ndarray:
        """Draw nodes as small circles. Background=127, vertices=0."""
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 127
        
        # Scale positions from [0, 1] to pixel coordinates
        pixel_coords = (positions * (self.image_size - 1)).astype(int)
        
        for center in pixel_coords:
            cv2.circle(img, tuple(center), self.node_radius, color=0, thickness=-1)
        
        return img
    
    def _draw_steiner_tree(self, positions: np.ndarray, edges: List[List[int]]) -> np.ndarray:
        """Draw complete Steiner tree. Background=127, edges=255, vertices=0."""
        img = np.ones((self.image_size, self.image_size), dtype=np.uint8) * 127
        
        # Scale positions from [0, 1] to pixel coordinates
        pixel_coords = (positions * (self.image_size - 1)).astype(int)
        
        # Draw edges first (value 255 = +1 after normalization)
        for edge in edges:
            if len(edge) == 2:
                u, v = edge
                if u < len(pixel_coords) and v < len(pixel_coords):
                    pt1 = tuple(pixel_coords[u])
                    pt2 = tuple(pixel_coords[v])
                    cv2.line(img, pt1, pt2, 255, self.edge_width)
        
        # Draw nodes on top (value 0 = -1 after normalization)
        for center in pixel_coords:
            cv2.circle(img, tuple(center), self.node_radius, color=0, thickness=-1)
        
        return img
    
    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Convert numpy image to normalized tensor."""
        # Normalize: 127->0 (background), 0->-1 (vertices), 255->+1 (edges)
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
        distance_tensor = torch.from_numpy(distance_map).float()
        distance_tensor = (distance_tensor / self.max_dist) * 2 - 1
        
        return distance_tensor.unsqueeze(0)
    
    def _distance_to_image(self, distance_map):
        """Convert distance transform back to binary image."""
        image = torch.ones_like(distance_map)
        mask = ((distance_map + 1) / 2 * self.max_dist <= self.eps)
        image[mask] = -1
        return image
    
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
    
    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a sample."""
        if self.use_ndjson:
            return self.instances[self.indices[idx]]
        else:
            instance_file = self.instance_files[self.indices[idx]]
            with open(instance_file, 'r') as f:
                return json.load(f)


class GeoSteinerDatasetDynamic(GeoSteinerDataset):
    """
    Dynamic version that can generate new samples on-the-fly using GeoSteiner.
    Falls back to pre-generated data if available.
    """
    
    def __init__(
        self,
        data_dir: str,
        geosteiner_path: Optional[str] = None,
        generate_on_demand: bool = False,
        **kwargs
    ):
        """
        Args:
            data_dir: Path to pre-generated data
            geosteiner_path: Path to GeoSteiner installation (for dynamic generation)
            generate_on_demand: Whether to generate new samples if requested index exceeds dataset
        """
        super().__init__(data_dir, **kwargs)
        
        self.generate_on_demand = generate_on_demand
        if generate_on_demand and geosteiner_path:
            from steiner_generation.generate_steiner_data import SteinerDataGenerator
            self.generator = SteinerDataGenerator(geosteiner_path, data_dir)
        else:
            self.generator = None
    
    def __getitem__(self, idx: int):
        """Get item, potentially generating new data if needed."""
        if idx < len(self.indices):
            return super().__getitem__(idx)
        elif self.generate_on_demand and self.generator:
            # Generate new sample on the fly
            num_points = np.random.randint(3, 8)  # Random between 3-7 terminals
            points = self.generator.generate_random_points(num_points)
            steiner_points, _, _ = self.generator.solve_steiner_tree(points)
            edges, _ = self.generator.extract_graph_structure(points)
            
            # Convert to images
            condition_img = self._draw_nodes(points)
            all_points = np.vstack([points, steiner_points]) if len(steiner_points) > 0 else points
            target_img = self._draw_steiner_tree(all_points, edges)
            
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
    dataset = GeoSteinerDataset(
        data_dir="data/steiner_data",
        num_samples=10,
        use_distance_transform=False,
        split='train'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Visualize a few samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    
    for i in range(4):
        condition, target = dataset[i]
        
        # Convert tensors to numpy for visualization
        condition_np = (condition.squeeze(0).numpy() * 127.0 + 127.0).astype(np.uint8)
        target_np = (target.squeeze(0).numpy() * 127.0 + 127.0).astype(np.uint8)
        
        axes[0, i].imshow(condition_np, cmap='gray')
        axes[0, i].set_title(f"Terminals {i+1}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(target_np, cmap='gray')
        axes[1, i].set_title(f"Steiner Tree {i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig("geosteiner_dataset_samples.png")
    plt.show()
    
    print("\nSample info for first instance:")
    info = dataset.get_sample_info(0)
    print(f"  Terminals: {info['num_terminals']}")
    print(f"  Steiner points: {info['num_steiner_points']}")
    print(f"  Total length: {info['total_length']:.4f}")