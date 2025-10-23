#!/usr/bin/env python3
"""
Script to extract Steiner tree graph structure from binary images.

This script takes a binary image representing a Steiner tree and extracts:
1. Vertex positions (nodes in the graph)
2. Edge connections between vertices
3. Graph structure as adjacency list or edge list

The binary image format:
- Black pixels (0) represent graph elements (nodes + edges)  
- White pixels (255) represent background
- Image size: 128x128 (configurable)
- Node radius: ~7 pixels
- Edge width: ~2 pixels
"""

import numpy as np
import cv2
import networkx as nx
from typing import List, Tuple, Optional, Dict, Set
from scipy.ndimage import label, distance_transform_edt, maximum_filter
from skimage.morphology import disk, binary_dilation, binary_erosion
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import argparse


class SteinerGraphExtractor:
    def __init__(
        self, 
        vertex_radius: int = 2,
        edge_width: int = 2,
        coverage_threshold: float = 0.7,
        proximity_threshold: float = 2.0,
        debug: bool = False
    ):
        """
        Initialize the Steiner tree extractor.
        
        Args:
            edge_width: Expected width of edges in pixels  
            coverage_threshold: Minimum fraction of line pixels that must be foreground for edge detection
            proximity_threshold: Maximum distance from line to vertex to reject edge (avoids edges through vertices)
            debug: Whether to show debug visualizations
        """
        self.vertex_radius = vertex_radius
        self.edge_width = edge_width
        self.coverage_threshold = coverage_threshold
        self.proximity_threshold = proximity_threshold
        self.debug = debug
        
    def extract_graph(
        self,
        binary_img: np.ndarray,
        reference_points: Optional[List[Tuple[float, float]]] = None
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """
        Extract graph structure from binary image.
        
        Args:
            binary_img: Binary image (0=black/graph, 255=white/background)
            
        Returns:
            Tuple of (vertices, edges) where:
            - vertices: List of (x, y) coordinates
            - edges: List of (vertex_idx1, vertex_idx2) tuples
        """
        if self.debug:
            print(f"Input image shape: {binary_img.shape}")
            print(f"Unique values: {np.unique(binary_img)}")
            
        # Handle new 3-value format: 127=background, 0=vertices, 255=edges
        # Create binary masks for vertices and edges
        vertex_mask = (binary_img == 0).astype(np.uint8)  # vertices are 0
        edge_mask = (binary_img == 255).astype(np.uint8)  # edges are 255
        
        # For vertex detection, we only need vertex pixels
        vertex_binary = vertex_mask
        
        # Step 1: Detect vertices (nodes) from vertex mask
        vertices = self._detect_vertices(vertex_binary)

        if reference_points:
            vertices = self._snap_vertices_to_reference(vertices, reference_points)
        
        # Step 2: Detect edges by checking line coverage in edge mask
        edges = self._detect_edges(edge_mask, vertices)
        edges = self._prune_edges_with_close_targets(vertices, edges)

        return vertices, edges
    
    def _detect_vertices(self, binary_img: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect vertex positions using morphological operations and connected components.
        
        Strategy:
        1. Use morphological opening to separate nodes from edges
        2. Find connected components that match node size criteria
        3. Calculate centroids of these components
        """
        # Create structuring element for opening (removes thin edges, keeps thick nodes)
        kernel_size = max(1, self.vertex_radius - 1)
        kernel = disk(kernel_size, strict_radius=False)
        
        # For the new representation, vertices are already isolated as value 1 in the mask
        vertex_regions = binary_img.copy()
        
        if self.debug:
            plt.figure(figsize=(12, 4))
            plt.subplot(131)
            plt.imshow(binary_img, cmap='gray')
            plt.title('Vertex Mask (0=vertices)')
            plt.subplot(132)
            plt.imshow(vertex_regions, cmap='gray')
            plt.title('Vertex Regions')
        
        # Find connected components of vertex regions
        labeled, num_features = label(vertex_regions)
        
        # First collect all vertex pixels and group them
        vertex_pixels = []
        approx_area = max(np.pi * max(1.0, float(self.vertex_radius)) ** 2, 1.0)
        for region in regionprops(labeled):
            vertex_pixels.extend(self._find_region_centers(region, approx_area))
        
        # Group nearby pixels into vertices (within vertex radius)
        vertices = []
        used = set()
        merge_threshold = max(1.0, self.vertex_radius * 0.6)
        for i, (x1, y1) in enumerate(vertex_pixels):
            if i in used:
                continue
            # Start a new vertex group
            group_x, group_y, count = x1, y1, 1
            used.add(i)
            
            # Find nearby pixels to group with this one
            for j, (x2, y2) in enumerate(vertex_pixels[i+1:], i+1):
                if j in used:
                    continue
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if distance <= merge_threshold:
                    group_x += x2
                    group_y += y2
                    count += 1
                    used.add(j)
            
            # Use average position of the group
            vertices.append((int(round(group_x / count)), int(round(group_y / count))))
        
        if self.debug:
            plt.subplot(133)
            plt.imshow(binary_img, cmap='gray')
            for i, (x, y) in enumerate(vertices):
                plt.plot(x, y, 'ro', markersize=8)
                plt.text(x+3, y+3, str(i), color='red', fontsize=8)
            plt.title(f'Detected Vertices ({len(vertices)})')
            plt.tight_layout()
            plt.savefig('debug_vertex_detection.png', dpi=150, bbox_inches='tight')
            print("Vertex detection plot saved to: debug_vertex_detection.png")
            plt.close()
            
        return vertices


    def _find_region_centers(self, region, approx_area: float) -> List[Tuple[int, int]]:
        """Find one or more vertex centers within a connected region."""
        region_mask = region.image.astype(bool)
        if region_mask.size == 0:
            return []

        coords = np.column_stack(np.nonzero(region_mask)).astype(float)
        coords[:, 0] += region.bbox[0]  # y positions
        coords[:, 1] += region.bbox[1]  # x positions

        if len(coords) == 0:
            return []

        dist = distance_transform_edt(region_mask)
        expected_vertices = max(1, int(round(region.area / approx_area)))
        expected_vertices = min(expected_vertices, len(coords))

        footprint_size = max(3, int(np.ceil(self.vertex_radius * 1.25)) * 2 + 1)
        footprint = np.ones((footprint_size, footprint_size), dtype=bool)
        local_max = (dist == maximum_filter(dist, footprint=footprint))

        threshold = max(0.5, self.vertex_radius / 2.0)
        candidate_coords = np.argwhere(local_max & (dist >= threshold))

        if candidate_coords.size == 0:
            candidate_coords = np.array([np.unravel_index(np.argmax(dist), dist.shape)])

        candidate_values = dist[candidate_coords[:, 0], candidate_coords[:, 1]]
        order = np.argsort(-candidate_values)

        if expected_vertices <= 1:
            center = coords.mean(axis=0)
            return [(int(round(center[1])), int(round(center[0])))]

        points = coords[:, ::-1]  # (x, y)

        initial_centers = []
        for idx in order[:expected_vertices]:
            y, x = candidate_coords[idx]
            initial_centers.append([x + region.bbox[1], y + region.bbox[0]])

        if len(initial_centers) < expected_vertices:
            supplement = points[np.linspace(0, len(points) - 1, expected_vertices, dtype=int)]
            for candidate in supplement:
                if len(initial_centers) >= expected_vertices:
                    break
                initial_centers.append(candidate.tolist())

        initial_centers = np.array(initial_centers, dtype=float)

        centers = self._run_kmeans(points, expected_vertices, initial_centers)

        return [(int(round(cx)), int(round(cy))) for cx, cy in centers]


    def _run_kmeans(
        self,
        points: np.ndarray,
        k: int,
        initial_centers: Optional[np.ndarray] = None,
        max_iters: int = 25
    ) -> np.ndarray:
        """Simple k-means clustering for 2D points."""
        if len(points) == 0:
            return np.zeros((0, 2))

        if k >= len(points):
            return points.copy()

        if initial_centers is not None and len(initial_centers) == k:
            centers = initial_centers.astype(float)
        else:
            indices = np.linspace(0, len(points) - 1, k, dtype=int)
            centers = points[indices].astype(float)

        for _ in range(max_iters):
            distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = centers.copy()
            for idx in range(k):
                cluster = points[labels == idx]
                if len(cluster) == 0:
                    new_centers[idx] = points[np.random.randint(len(points))]
                else:
                    new_centers[idx] = cluster.mean(axis=0)

            if np.allclose(new_centers, centers):
                centers = new_centers
                break

            centers = new_centers

        return centers


    def _snap_vertices_to_reference(
        self,
        vertices: List[Tuple[int, int]],
        reference_points: List[Tuple[float, float]],
        snap_threshold: Optional[float] = None
    ) -> List[Tuple[float, float]]:
        """Snap detected vertices to reference points (e.g., terminals) when close enough."""
        if not vertices or not reference_points:
            return [(float(x), float(y)) for x, y in vertices]

        if snap_threshold is None:
            snap_threshold = max(2.0, float(self.vertex_radius) * 2.5)

        snapped_vertices = [(float(x), float(y)) for x, y in vertices]
        available_indices = set(range(len(vertices)))

        for rx, ry in reference_points:
            best_idx = None
            best_dist = float('inf')

            for idx in available_indices:
                vx, vy = vertices[idx]
                dist = float(np.hypot(vx - rx, vy - ry))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist <= snap_threshold:
                snapped_vertices[best_idx] = (rx, ry)
                available_indices.remove(best_idx)

        return snapped_vertices


    def _detect_edges(self, binary_img: np.ndarray, vertices: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Detect edges by checking if straight lines between vertices are covered by foreground pixels.
        
        Much simpler and more robust approach:
        1. For each pair of vertices, draw a line between them
        2. Check if most pixels along that line are foreground (black/0)
        3. If coverage is above threshold, consider it an edge
        """
        if len(vertices) <= 1:
            return []
            
        edges = []
        
        # Check all pairs of vertices
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if (self._line_covered_by_foreground(vertices[i], vertices[j], binary_img, self.coverage_threshold) and
                    not self._line_passes_through_vertex(vertices[i], vertices[j], vertices, i, j)):
                    edges.append((i, j))
        
        if self.debug:
            print(f"Detected {len(edges)} edges: {edges}")
            
            plt.figure(figsize=(8, 8))
            plt.imshow(binary_img, cmap='gray')
            
            # Draw vertices
            for i, (x, y) in enumerate(vertices):
                plt.plot(x, y, 'ro', markersize=10)
                plt.text(x+3, y+3, str(i), color='red', fontsize=12, weight='bold')
                
            # Draw edges
            for i, j in edges:
                x1, y1 = vertices[i]
                x2, y2 = vertices[j]
                plt.plot([x1, x2], [y1, y2], 'b-', linewidth=2, alpha=0.7)
                
            plt.title(f'Extracted Graph: {len(vertices)} vertices, {len(edges)} edges')
            plt.axis('equal')
            plt.savefig('debug_extracted_graph.png', dpi=150, bbox_inches='tight')
            print("Extracted graph plot saved to: debug_extracted_graph.png")
            plt.close()
            
        return edges

    def _prune_edges_with_close_targets(
        self,
        vertices: List[Tuple[int, int]],
        edges: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """Remove duplicate edges from a vertex to multiple nearby vertices, keeping the shorter."""
        if len(edges) < 2:
            return edges

        close_threshold = max(self.vertex_radius * 2.0, self.edge_width * 2.5)
        indices_by_vertex: Dict[int, List[Tuple[int, int]]] = {}
        for idx, (u, v) in enumerate(edges):
            indices_by_vertex.setdefault(u, []).append((idx, v))
            indices_by_vertex.setdefault(v, []).append((idx, u))

        to_remove: Set[int] = set()

        for base, neighbor_entries in indices_by_vertex.items():
            if len(neighbor_entries) < 2:
                continue

            for i in range(len(neighbor_entries)):
                idx_i, neighbor_i = neighbor_entries[i]
                if idx_i in to_remove:
                    continue

                for j in range(i + 1, len(neighbor_entries)):
                    idx_j, neighbor_j = neighbor_entries[j]

                    if idx_j in to_remove or neighbor_i == neighbor_j:
                        continue

                    dist_neighbors = np.hypot(
                        vertices[neighbor_i][0] - vertices[neighbor_j][0],
                        vertices[neighbor_i][1] - vertices[neighbor_j][1]
                    )

                    if dist_neighbors > close_threshold:
                        continue

                    length_i = np.hypot(
                        vertices[base][0] - vertices[neighbor_i][0],
                        vertices[base][1] - vertices[neighbor_i][1]
                    )
                    length_j = np.hypot(
                        vertices[base][0] - vertices[neighbor_j][0],
                        vertices[base][1] - vertices[neighbor_j][1]
                    )

                    if length_i <= length_j:
                        to_remove.add(idx_j)
                    else:
                        to_remove.add(idx_i)
                        break  # idx_i removed; no need to compare further

        if not to_remove:
            return edges

        pruned_edges = [edge for idx, edge in enumerate(edges) if idx not in to_remove]
        return pruned_edges
    
    def _line_covered_by_foreground(
        self, 
        v1: Tuple[int, int], 
        v2: Tuple[int, int], 
        binary_img: np.ndarray,
        coverage_threshold: float = 0.7
    ) -> bool:
        """
        Check if the line between two vertices is mostly covered by foreground pixels.
        
        Args:
            v1, v2: Vertex coordinates (x, y)
            binary_img: Binary image where foreground pixels > 0
            coverage_threshold: Minimum fraction of line that must be foreground
            
        Returns:
            True if line is sufficiently covered by foreground pixels
        """
        x1, y1 = v1
        x2, y2 = v2

        # All pixels along the centerline
        line_pixels = self._get_line_pixels(x1, y1, x2, y2)
        n = len(line_pixels)
        if n == 0:
            return False

        # Skip endpoints where vertex discs overlap edges
        skip = max(1, int(self.vertex_radius))
        start = min(skip, n)
        end = max(start, n - skip)
        segment = line_pixels[start:end]
        if len(segment) <= self.proximity_threshold:
            # If the edge is extremely short (within vertices), accept
            return True

        # Sample a small square patch around each segment pixel to account for edge width and gaps
        # r = max(1, int(np.ceil(self.edge_width / 2)))
        r = 1
        H, W = binary_img.shape

        covered = 0
        for x, y in segment:
            if 0 <= x < W and 0 <= y < H:
                x1p = max(0, x - r)
                y1p = max(0, y - r)
                x2p = min(W, x + r + 1)
                y2p = min(H, y + r + 1)
                if np.any(binary_img[y1p:y2p, x1p:x2p] > 0):
                    covered += 1

        coverage = covered / len(segment)

        if self.debug:
            print(f"Line ({x1},{y1}) -> ({x2},{y2}): {covered}/{len(segment)} = {coverage:.2f} (skip={skip}, r={r})")

        return coverage >= coverage_threshold
    
    def _get_line_pixels(self, x1: float, y1: float, x2: float, y2: float) -> List[Tuple[int, int]]:
        """
        Get all pixel coordinates along a line using Bresenham's algorithm.
        Converts float inputs to integers for pixel-based algorithm.
        """
        # Convert float coordinates to integers for Bresenham's algorithm
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        pixels = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        x, y = x1, y1
        
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        
        error = dx - dy
        
        while True:
            pixels.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * error
            
            if e2 > -dy:
                error -= dy
                x += x_inc
                
            if e2 < dx:
                error += dx
                y += y_inc
        
        return pixels
    
    def _line_passes_through_vertex(
        self, 
        v1: Tuple[int, int], 
        v2: Tuple[int, int], 
        all_vertices: List[Tuple[int, int]], 
        v1_idx: int, 
        v2_idx: int
    ) -> bool:
        """
        Check if the line between v1 and v2 passes through or very close to any other vertex.
        
        Args:
            v1, v2: Start and end vertices
            all_vertices: List of all vertices  
            v1_idx, v2_idx: Indices of v1 and v2 to exclude from checking
            
        Returns:
            True if line passes close to any other vertex
        """
        x1, y1 = v1
        x2, y2 = v2
        
        # Check all other vertices
        for k, (x3, y3) in enumerate(all_vertices):
            if k == v1_idx or k == v2_idx:
                continue  # Skip the endpoints

            dist_v1 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
            dist_v2 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

            # Skip points that are effectively part of the endpoints (vertices clustered together)
            endpoint_buffer = max(self.vertex_radius * 2.0, self.edge_width * 2.5)
            if dist_v1 <= endpoint_buffer or dist_v2 <= endpoint_buffer:
                continue

            # Calculate distance from point (x3, y3) to line segment (x1,y1)-(x2,y2)
            distance = self._point_to_line_distance(x1, y1, x2, y2, x3, y3)
            
            if distance <= self.proximity_threshold:
                if self.debug:
                    print(f"Line ({x1},{y1})->({x2},{y2}) passes through vertex {k} ({x3},{y3}) at distance {distance:.2f}")
                return True
                
        return False
    
    def _point_to_line_distance(self, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int) -> float:
        """
        Calculate the shortest distance from point (x3, y3) to line segment (x1, y1)-(x2, y2).
        """
        # Vector from line start to end
        dx = x2 - x1
        dy = y2 - y1
        
        # If the line segment has zero length, return distance to the point
        if dx == 0 and dy == 0:
            return np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
        
        # Parameter t represents position along the line segment
        # t = 0 at (x1, y1), t = 1 at (x2, y2)
        t = max(0, min(1, ((x3 - x1) * dx + (y3 - y1) * dy) / (dx * dx + dy * dy)))
        
        # Closest point on the line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Distance from point to closest point on line segment
        return np.sqrt((x3 - closest_x)**2 + (y3 - closest_y)**2)


def load_binary_image(image_path: str) -> np.ndarray:
    """Load image - expects 3-value format: 127=background, 0=vertices, 255=edges."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Keep original 3-value format: 127=background, 0=vertices, 255=edges
    return img


def save_graph_to_networkx(vertices: List[Tuple[int, int]], edges: List[Tuple[int, int]]) -> nx.Graph:
    """Convert extracted graph to NetworkX format."""
    G = nx.Graph()
    
    # Add vertices with position attributes
    for i, (x, y) in enumerate(vertices):
        G.add_node(i, pos=(x, y))
    
    # Add edges
    G.add_edges_from(edges)
    
    return G


def visualize_extracted_graph(
    binary_img: np.ndarray, 
    vertices: List[Tuple[int, int]], 
    edges: List[Tuple[int, int]],
    save_path: Optional[str] = None
):
    """Visualize the extracted graph overlaid on the original image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_img, cmap='gray')
    
    # Draw vertices
    for i, (x, y) in enumerate(vertices):
        plt.plot(x, y, 'ro', markersize=12)
        plt.text(x+5, y+5, str(i), color='red', fontsize=12, weight='bold')
    
    # Draw edges
    for i, j in edges:
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=3, alpha=0.8)
    
    plt.title(f'Extracted Steiner Tree: {len(vertices)} vertices, {len(edges)} edges')
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Extract Steiner tree graph from binary image")
    parser.add_argument("image_path", help="Path to binary image file")
    parser.add_argument("--output", "-o", help="Output path for visualization")
    parser.add_argument("--debug", "-d", action="store_true", help="Show debug visualizations")
    parser.add_argument("--edge-width", type=int, default=2, help="Expected edge width in pixels")
    parser.add_argument("--coverage-threshold", type=float, default=0.7, help="Minimum line coverage for edge detection (0.0-1.0)")
    parser.add_argument("--proximity-threshold", type=float, default=3.0, help="Maximum distance from line to vertex to reject edge (pixels)")
    
    args = parser.parse_args()
    
    # Load image
    try:
        binary_img = load_binary_image(args.image_path)
        print(f"Loaded image: {args.image_path}, shape: {binary_img.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Extract graph
    extractor = SteinerGraphExtractor(
        edge_width=args.edge_width,
        coverage_threshold=args.coverage_threshold,
        proximity_threshold=args.proximity_threshold,
        debug=args.debug
    )
    
    vertices, edges = extractor.extract_graph(binary_img)
    
    print(f"\nExtracted Graph:")
    print(f"Vertices ({len(vertices)}): {vertices}")
    print(f"Edges ({len(edges)}): {edges}")
    
    # Calculate total edge weight (sum of Euclidean distances)
    total_weight = 0.0
    edge_weights = []
    for i, j in edges:
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        weight = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        edge_weights.append(weight)
        total_weight += weight
    
    print(f"\nEdge Weights:")
    for k, (i, j) in enumerate(edges):
        print(f"  Edge {i}-{j}: {edge_weights[k]:.2f}")
    print(f"Total Edge Weight: {total_weight:.2f}")
    
    # Create NetworkX graph
    G = save_graph_to_networkx(vertices, edges)
    print(f"\nNetworkX Graph Info:")
    print(f"Nodes: {list(G.nodes(data=True))}")
    print(f"Edges: {list(G.edges())}")
    
    # Visualize
    output_path = args.output if args.output else None
    visualize_extracted_graph(binary_img, vertices, edges, output_path)


if __name__ == "__main__":
    main()
