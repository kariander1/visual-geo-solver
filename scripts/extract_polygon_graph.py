#!/usr/bin/env python3
"""
Script to extract polygon graph structure from binary images using ground truth points.

This script takes a binary image representing a polygon and ground truth point positions,
then determines which edges exist by checking line coverage in the binary image.

Approach:
1. Use ground truth points as vertex positions (converted to pixel coordinates)
2. For each possible edge in the complete graph, check if it exists in the solution
3. An edge exists if at least 90% of the line pixels are covered in the binary image
4. Reconstruct the polygon from the detected edges

The binary image format:
- Black pixels (0) represent polygon elements (vertices + edges + fill)  
- White pixels (255) represent background
- Image size: 128x128 (configurable)
- Node radius: ~7 pixels
- Edge width: ~2 pixels
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import argparse
import itertools
from shapely.geometry import Polygon as ShapelyPolygon, LineString
import networkx as nx


class PolygonGraphExtractor:
    def __init__(
        self, 
        coverage_threshold: float = 0.9,
        edge_width: int = 2,
        debug: bool = False
    ):
        """
        Initialize the polygon extractor.
        
        Args:
            coverage_threshold: Minimum fraction of line pixels that must be covered for edge detection (0.9 = 90%)
            edge_width: Expected width of edges in pixels (used for line sampling)
            debug: Whether to show debug visualizations
        """
        self.coverage_threshold = coverage_threshold
        self.edge_width = edge_width
        self.debug = debug
        
    def extract_polygon_from_points(
        self,
        binary_img: np.ndarray,
        gt_points: List[Tuple[float, float]]
    ) -> Tuple[List[Tuple[int, int]], float, float, List[Tuple[int, int]]]:
        """
        Extract polygon structure from binary image using ground truth points.

        Args:
            binary_img: Image with format: background=0, edges=127, interior=-127 (or normalized: bg=0, edges=1, interior=-1)
            gt_points: Ground truth points in normalized coordinates [0,1]

        Returns:
            Tuple of (vertices, area, perimeter, edges) where:
            - vertices: List of (x, y) pixel coordinates of vertices in polygon
            - area: Polygon area in normalized coordinates [0,1]
            - perimeter: Polygon perimeter in normalized coordinates [0,1]
            - edges: List of (i, j) tuples indicating edges between vertices
        """
        if self.debug:
            print(f"Input image shape: {binary_img.shape}")
            print(f"Unique values: {np.unique(binary_img)}")
            print(f"Value range: [{binary_img.min():.3f}, {binary_img.max():.3f}]")
            print(f"Ground truth points: {len(gt_points)}")
            for i, (x, y) in enumerate(gt_points):
                print(f"  GT Point {i}: ({x:.3f}, {y:.3f})")

        # New format: background=0, edges=127, interior=-127
        # Use threshold for robustness instead of exact match
        edge_mask = (binary_img > 64)  # Values closer to 127 than to 0 or -127

        if self.debug:
            edge_pixels = np.sum(edge_mask)
            interior_pixels = np.sum(binary_img < -64)  # Values closer to -127
            background_pixels = np.sum(np.abs(binary_img) <= 64)  # Values closer to 0
            print(f"  Edge pixels (>64): {edge_pixels}")
            print(f"  Interior pixels (<-64): {interior_pixels}")
            print(f"  Background pixels (|value|<=64): {background_pixels}")

        # For edge detection, we only care about the edge pixels
        binary_mask = edge_mask
        
        # Convert GT points to pixel coordinates
        image_size = binary_img.shape[0]
        pixel_vertices = []
        for i, (x, y) in enumerate(gt_points):
            px = int(x * (image_size - 1))
            py = int(y * (image_size - 1))
            pixel_vertices.append((px, py))
            if self.debug:
                print(f"  Pixel {i}: ({px}, {py})")
        
        if self.debug:
            print(f"Binary mask: {binary_mask.sum()} foreground pixels out of {binary_mask.size} total")
            print(f"Coverage threshold: {self.coverage_threshold}")
            
        # Detect edges using line coverage
        edges = self._detect_edges_by_coverage(binary_mask, pixel_vertices)
        
        # Check for self-intersecting edges using original GT coordinates
        has_intersections = self._check_for_intersecting_edges(gt_points, edges)
        
        # Order vertices to form a polygon
        if has_intersections:
            # Return empty result - invalid polygonization
            if self.debug:
                print("Polygonization is invalid due to self-intersecting edges")
            return [], 0.0, 0.0, edges
        
        ordered_vertices, vertex_indices = self._order_vertices_for_polygon(pixel_vertices, edges)
        
        # Calculate area and perimeter using original GT coordinates
        if len(ordered_vertices) >= 3 and vertex_indices:
            # Use original GT coordinates for accurate area calculation
            ordered_gt_coords = [gt_points[i] for i in vertex_indices]
            area, perimeter = self._calculate_polygon_metrics_from_coords(ordered_gt_coords)
        else:
            area, perimeter = 0.0, 0.0
            
        if self.debug:
            self._debug_visualize_extraction(binary_img, pixel_vertices, ordered_vertices, edges)
            
        return ordered_vertices, area, perimeter, edges
    
    def _detect_edges_by_coverage(
        self, 
        binary_mask: np.ndarray, 
        vertices: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Detect edges by checking line coverage between all pairs of vertices.
        
        Args:
            binary_mask: Boolean mask where True = polygon pixels
            vertices: List of (x, y) pixel coordinates
            
        Returns:
            List of (i, j) tuples indicating edges between vertex i and vertex j
        """
        edges = []
        n_vertices = len(vertices)
        
        # Check all possible edges in the complete graph
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if self._is_edge_covered(binary_mask, vertices[i], vertices[j]):
                    edges.append((i, j))
                    
        if self.debug:
            print(f"Found {len(edges)} edges from {n_vertices} vertices")
            
        return edges
    
    def _check_for_intersecting_edges(
        self, 
        vertices: List[Tuple[float, float]], 
        edges: List[Tuple[int, int]]
    ) -> bool:
        """
        Check if any edges intersect with each other (except at shared vertices).
        
        Args:
            vertices: List of (x, y) normalized coordinates [0,1]
            edges: List of (i, j) tuples indicating edges between vertex i and vertex j
            
        Returns:
            True if any edges intersect in their interiors
        """
        if len(edges) <= 1:
            return False
            
        # Check all pairs of edges for intersections
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                if self._edges_intersect(vertices, edges[i], edges[j]):
                    if self.debug:
                        v1_1, v1_2 = edges[i]
                        v2_1, v2_2 = edges[j]
                        print(f"Edge intersection found: {v1_1}-{v1_2} intersects with {v2_1}-{v2_2}")
                    return True
        
        return False
    
    def _edges_intersect(
        self,
        vertices: List[Tuple[float, float]], 
        edge1: Tuple[int, int], 
        edge2: Tuple[int, int]
    ) -> bool:
        """
        Check if two edges intersect (except at shared vertices).
        
        Args:
            vertices: List of vertex coordinates (normalized [0,1])
            edge1: (i, j) tuple for first edge
            edge2: (k, l) tuple for second edge
            
        Returns:
            True if edges intersect in their interiors
        """
        i, j = edge1
        k, l = edge2
        
        # Skip if edges share a vertex
        if i == k or i == l or j == k or j == l:
            return False
            
        # Get coordinates
        x1, y1 = vertices[i]
        x2, y2 = vertices[j]
        x3, y3 = vertices[k]
        x4, y4 = vertices[l]
        
        # Use line intersection formula
        # Line 1: (x1,y1) to (x2,y2)
        # Line 2: (x3,y3) to (x4,y4)
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        # Lines are parallel (lenient threshold for angles up to ~3 degrees)
        if abs(denom) < 0.05:
            return False
            
        # Calculate intersection parameters
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        # Check if intersection occurs within both line segments
        return 0 < t < 1 and 0 < u < 1
    
    def _is_edge_covered(
        self,
        binary_mask: np.ndarray,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> bool:
        """
        Check if an edge between two points is covered in the binary mask.

        More lenient approach similar to Steiner extractor:
        - Skip endpoints to avoid vertex disc overlap
        - Use small patch around each pixel to account for edge width
        """
        x1, y1 = point1
        x2, y2 = point2

        # Get all pixel coordinates along the line using Bresenham's algorithm
        line_points = self._get_line_points(x1, y1, x2, y2)
        n = len(line_points)

        if n == 0:
            return False

        # Skip endpoints where vertex discs might overlap edges
        # Use a more conservative skip distance for polygons
        skip = max(1, self.edge_width)
        start = min(skip, n)
        end = max(start, n - skip)
        segment = line_points[start:end]

        if len(segment) <= 2:
            # If the edge is extremely short (within vertices), accept it
            return True

        # Sample a small patch around each segment pixel to account for edge width and gaps
        r = 1  # Patch radius - check immediate neighbors
        H, W = binary_mask.shape

        covered_count = 0
        for x, y in segment:
            if 0 <= x < W and 0 <= y < H:
                # Check a small patch around the pixel for more tolerance
                x1p = max(0, x - r)
                y1p = max(0, y - r)
                x2p = min(W, x + r + 1)
                y2p = min(H, y + r + 1)

                # If any pixel in the patch is covered, count it
                if np.any(binary_mask[y1p:y2p, x1p:x2p]):
                    covered_count += 1

        coverage_ratio = covered_count / len(segment)
        is_edge = coverage_ratio >= self.coverage_threshold

        # if self.debug:
        #     status = "✓" if is_edge else "✗"
        #     print(f"Edge ({x1},{y1})->({x2},{y2}): {coverage_ratio:.2%} covered ({covered_count}/{len(segment)}) skip={skip} r={r} {status}")

        return is_edge
    
    
    def _get_original_value(self, x: int, y: int) -> str:
        """Helper to get original pixel value for debugging."""
        try:
            # This is a hack - we don't have access to original image here
            # But we can infer from the binary mask
            return "estimated"
        except:
            return "unknown"
    
    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """
        Get all pixel coordinates along a line using Bresenham's line algorithm.
        """
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            points.append((x, y))
            
            if x == x2 and y == y2:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
                
        return points
    
    def _order_vertices_for_polygon(
        self, 
        vertices: List[Tuple[int, int]], 
        edges: List[Tuple[int, int]]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        """
        Order vertices to form a polygon by finding a cycle in the graph.
        
        Returns:
            Tuple of (ordered_vertices, vertex_indices) where vertex_indices 
            contains the indices of vertices in the cycle order
        """
        if len(vertices) < 3 or len(edges) == 0:
            return [], []
            
        # Create graph from edges
        G = nx.Graph()
        G.add_nodes_from(range(len(vertices)))
        G.add_edges_from(edges)
        
        # Find any cycle in the graph using simple DFS
        cycle = self._find_simple_cycle(G)
        if cycle:
            if self.debug:
                print(f"Found cycle: {cycle}")
            # Convert vertex indices to actual coordinates and return both
            ordered_vertices = [vertices[i] for i in cycle]
            return ordered_vertices, cycle
        
        if self.debug:
            print("No cycle found")
        return [], []
    
    def _find_simple_cycle(self, G: nx.Graph) -> Optional[List[int]]:
        """
        Find a Hamiltonian cycle (visiting all vertices) using simple DFS.
        """
        n = G.number_of_nodes()
        if n < 3:
            return None
        
        def dfs(path: List[int], visited: set) -> Optional[List[int]]:
            if len(path) == n:
                # Check if we can return to start to complete the cycle
                if path[0] in G.neighbors(path[-1]):
                    return path
                return None
            
            current = path[-1]
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    path.append(neighbor)
                    visited.add(neighbor)
                    result = dfs(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.remove(neighbor)
            
            return None
        
        # Try starting from each vertex
        for start_vertex in G.nodes():
            result = dfs([start_vertex], {start_vertex})
            if result is not None:
                return result
        
        return None
                
    def _order_vertices_counterclockwise(self, vertices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Order vertices in counterclockwise direction."""
        if len(vertices) < 3:
            return vertices
            
        # Find centroid
        cx = sum(x for x, y in vertices) / len(vertices)
        cy = sum(y for x, y in vertices) / len(vertices)
        
        # Sort by angle from centroid
        def angle_from_centroid(vertex):
            x, y = vertex
            return np.arctan2(y - cy, x - cx)
            
        return sorted(vertices, key=angle_from_centroid)
    
    def _calculate_polygon_metrics(self, vertices: List[Tuple[int, int]], image_size: int) -> Tuple[float, float]:
        """
        Calculate polygon area and perimeter in normalized coordinates.
        
        Args:
            vertices: List of (x, y) pixel coordinates
            image_size: Image size for normalization
            
        Returns:
            Tuple of (area, perimeter) in normalized coordinates [0,1]
        """
        if len(vertices) < 3:
            return 0.0, 0.0
        
        # Convert to normalized coordinates [0, 1]
        # Use (image_size - 1) to match the forward conversion: px = int(x * (image_size - 1))
        normalized_vertices = [(x / (image_size - 1), y / (image_size - 1)) for x, y in vertices]
        
        try:
            # Create shapely polygon for accurate calculations
            polygon = ShapelyPolygon(normalized_vertices)
            
            # Handle invalid polygons
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Fix self-intersections
            
            area = float(polygon.area)
            perimeter = float(polygon.length)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating polygon metrics: {e}")
            # Fallback to simple calculation
            area = self._calculate_area_shoelace(normalized_vertices)
            perimeter = self._calculate_perimeter_euclidean(normalized_vertices)
        
        return area, perimeter
    
    def _calculate_polygon_metrics_from_coords(self, coords: List[Tuple[float, float]]) -> Tuple[float, float]:
        """
        Calculate polygon area and perimeter directly from normalized coordinates.
        
        Args:
            coords: List of (x, y) normalized coordinates [0,1]
            
        Returns:
            Tuple of (area, perimeter) in normalized coordinates [0,1]
        """
        if len(coords) < 3:
            return 0.0, 0.0
        
        try:
            # Create shapely polygon directly from normalized coordinates
            polygon = ShapelyPolygon(coords)
            
            # Handle invalid polygons
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # Fix self-intersections
            
            area = float(polygon.area)
            perimeter = float(polygon.length)
            
        except Exception as e:
            if self.debug:
                print(f"Error calculating polygon metrics from coords: {e}")
            # Fallback to simple calculation
            area = self._calculate_area_shoelace(coords)
            perimeter = self._calculate_perimeter_euclidean(coords)
        
        return area, perimeter
    
    def _calculate_area_shoelace(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using shoelace formula."""
        if len(vertices) < 3:
            return 0.0
            
        n = len(vertices)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
            
        return abs(area) / 2.0
    
    def _calculate_perimeter_euclidean(self, vertices: List[Tuple[float, float]]) -> float:
        """Calculate polygon perimeter using Euclidean distances."""
        if len(vertices) < 2:
            return 0.0
            
        perimeter = 0.0
        n = len(vertices)
        
        for i in range(n):
            j = (i + 1) % n
            dx = vertices[j][0] - vertices[i][0]
            dy = vertices[j][1] - vertices[i][1]
            perimeter += np.sqrt(dx*dx + dy*dy)
            
        return perimeter
    
    def _debug_visualize_extraction(
        self, 
        binary_img: np.ndarray, 
        pixel_vertices: List[Tuple[int, int]], 
        ordered_vertices: List[Tuple[int, int]], 
        edges: List[Tuple[int, int]]
    ):
        """Create debug visualization showing extraction results."""
        if not self.debug:
            return
            
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original binary image
        axes[0].imshow(binary_img, cmap='gray')
        axes[0].set_title('Binary Input')
        axes[0].axis('off')
        
        # GT vertices and detected edges
        axes[1].imshow(binary_img, cmap='gray')
        if pixel_vertices:
            vertex_x = [v[0] for v in pixel_vertices]
            vertex_y = [v[1] for v in pixel_vertices]
            axes[1].scatter(vertex_x, vertex_y, c='red', s=50, zorder=5, label=f'{len(pixel_vertices)} GT points')
            
            # Draw detected edges
            for i, j in edges:
                x1, y1 = pixel_vertices[i]
                x2, y2 = pixel_vertices[j]
                axes[1].plot([x1, x2], [y1, y2], 'g-', linewidth=2, alpha=0.7)
                
        axes[1].set_title(f'Detected Edges ({len(edges)} edges)')
        axes[1].legend()
        axes[1].axis('off')
        
        # Final ordered polygon
        axes[2].imshow(binary_img, cmap='gray')
        if ordered_vertices:
            vertex_x = [v[0] for v in ordered_vertices]
            vertex_y = [v[1] for v in ordered_vertices]
            axes[2].scatter(vertex_x, vertex_y, c='red', s=50, zorder=5, label=f'{len(ordered_vertices)} vertices')
            
            # Draw polygon edges
            if len(ordered_vertices) >= 3:
                # Close the polygon
                vertex_x_closed = vertex_x + [vertex_x[0]]
                vertex_y_closed = vertex_y + [vertex_y[0]]
                axes[2].plot(vertex_x_closed, vertex_y_closed, 'b-', linewidth=2, alpha=0.7, label='Final Polygon')
                
        axes[2].set_title('Final Ordered Polygon')
        axes[2].legend()
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    

def main():
    parser = argparse.ArgumentParser(description="Extract polygon from binary image using ground truth points")
    parser.add_argument("image_path", help="Path to binary image file")
    parser.add_argument("--points", nargs='+', type=float, help="GT points as x1 y1 x2 y2 ... (normalized coordinates)")
    parser.add_argument("--json-path", help="Path to JSON file with GT points")
    parser.add_argument("--debug", action="store_true", help="Show debug visualizations")
    parser.add_argument("--coverage-threshold", type=float, default=0.9, help="Coverage threshold for edge detection")
    parser.add_argument("--edge-width", type=int, default=2, help="Expected edge width")
    
    args = parser.parse_args()
    
    # Load image
    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    # Get GT points
    gt_points = []
    if args.json_path:
        # Load from JSON file
        import json
        with open(args.json_path, 'r') as f:
            data = json.load(f)
        gt_points = data.get('points', [])
        gt_points = [(float(p[0]), float(p[1])) for p in gt_points]
    elif args.points:
        # Parse from command line
        if len(args.points) % 2 != 0:
            print("Error: Points must be provided as pairs (x y)")
            return
        gt_points = [(args.points[i], args.points[i+1]) for i in range(0, len(args.points), 2)]
    else:
        print("Error: Must provide either --points or --json-path")
        return
    
    print(f"Using {len(gt_points)} ground truth points")
    
    # Extract polygon
    extractor = PolygonGraphExtractor(
        coverage_threshold=args.coverage_threshold,
        edge_width=args.edge_width,
        debug=args.debug
    )
    
    vertices, area, perimeter, edges = extractor.extract_polygon_from_points(img, gt_points)
    
    print(f"\\nExtracted polygon:")
    print(f"  Vertices: {len(vertices)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Area: {area:.6f}")
    print(f"  Perimeter: {perimeter:.6f}")
    
    if vertices:
        print(f"  Vertex coordinates (pixel):")
        for i, (x, y) in enumerate(vertices):
            print(f"    {i}: ({x}, {y})")
    
    if edges:
        print(f"  Detected edges:")
        for i, (v1, v2) in enumerate(edges):
            print(f"    {i}: {v1} - {v2}")
    
    # Calculate compactness (isoperimetric ratio)
    if area > 0 and perimeter > 0:
        compactness = 4 * np.pi * area / (perimeter ** 2)
        print(f"  Compactness: {compactness:.6f} (1.0 = perfect circle)")
        
    # Show success rate
    total_possible_edges = len(gt_points) * (len(gt_points) - 1) // 2
    print(f"\\nEdge detection rate: {len(edges)}/{total_possible_edges} ({len(edges)/total_possible_edges:.1%})")


if __name__ == "__main__":
    main()