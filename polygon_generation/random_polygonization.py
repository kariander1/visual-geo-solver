#!/usr/bin/env python3
"""
random_polygonization.py

Generate multiple random valid simple polygons for a given set of points.
Uses various strategies to create diverse polygon variations.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Set
import math
import random
import numpy as np
from .max_area_polygonization import segments_properly_intersect, polygon_area

Point = Tuple[float, float]
EPS = 1e-12


def is_simple_polygon(points: List[Point], order: List[int]) -> bool:
    """Check if a polygon defined by order is simple (non-self-intersecting)."""
    if len(order) < 3:
        return False
    
    n = len(order)
    for i in range(n):
        edge1_start = points[order[i]]
        edge1_end = points[order[(i + 1) % n]]
        
        for j in range(i + 2, n):
            if (i == 0 and j == n - 1):
                continue
                
            edge2_start = points[order[j]]
            edge2_end = points[order[(j + 1) % n]]
            
            if segments_properly_intersect(edge1_start, edge1_end, edge2_start, edge2_end):
                return False
    
    return True


def compute_polygon_metrics(points: List[Point], order: List[int]) -> dict:
    """Compute various metrics for a polygon."""
    if len(order) < 3:
        return {"area": 0, "perimeter": 0, "compactness": 0}
    
    area = polygon_area(points, order)
    
    perimeter = 0
    n = len(order)
    for i in range(n):
        p1 = points[order[i]]
        p2 = points[order[(i + 1) % n]]
        perimeter += math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    compactness = 4 * math.pi * area / (perimeter**2) if perimeter > 0 else 0
    
    return {
        "area": area,
        "perimeter": perimeter,
        "compactness": compactness
    }


class RandomPolygonGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
    
    def random_permutation_strategy(self, points: List[Point], max_attempts: int = 1000) -> Optional[List[int]]:
        """Generate random permutations and return the first valid simple polygon."""
        n = len(points)
        if n < 3:
            return None
        
        for _ in range(max_attempts):
            order = list(range(n))
            self.rng.shuffle(order)
            
            if is_simple_polygon(points, order):
                return order
        
        return None
    
    def angle_based_strategy(self, points: List[Point], perturbation: float = 0.3) -> Optional[List[int]]:
        """Sort points by angle with random perturbation."""
        if len(points) < 3:
            return None
        
        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)
        
        def angle_with_perturbation(i: int) -> float:
            p = points[i]
            base_angle = math.atan2(p[1] - center_y, p[0] - center_x)
            noise = self.rng.uniform(-perturbation, perturbation)
            return base_angle + noise
        
        order = sorted(range(len(points)), key=angle_with_perturbation)
        
        if is_simple_polygon(points, order):
            return order
        
        return None
    
    def incremental_construction_strategy(self, points: List[Point], max_attempts: int = 100) -> Optional[List[int]]:
        """Build polygon incrementally by adding points that don't create intersections."""
        if len(points) < 3:
            return None
        
        for attempt in range(max_attempts):
            remaining = list(range(len(points)))
            order = []
            
            start_idx = self.rng.choice(remaining)
            order.append(start_idx)
            remaining.remove(start_idx)
            
            for step in range(len(points) - 1):
                if not remaining:
                    break
                
                valid_candidates = []
                
                for candidate_idx in remaining:
                    test_order = order + [candidate_idx]
                    
                    if len(test_order) < 3 or is_simple_polygon(points, test_order):
                        valid_candidates.append(candidate_idx)
                
                if not valid_candidates:
                    break
                
                next_idx = self.rng.choice(valid_candidates)
                order.append(next_idx)
                remaining.remove(next_idx)
            
            if len(order) == len(points) and is_simple_polygon(points, order):
                return order
        
        return None
    
    def convex_hull_perturbation_strategy(self, points: List[Point]) -> Optional[List[int]]:
        """Start with convex hull and try to insert interior points."""
        from scipy.spatial import ConvexHull
        
        if len(points) < 3:
            return None
        
        try:
            points_array = np.array(points)
            hull = ConvexHull(points_array)
            hull_order = hull.vertices.tolist()
            
            if len(hull_order) == len(points):
                return hull_order
            
            order = hull_order[:]
            interior_points = [i for i in range(len(points)) if i not in hull_order]
            self.rng.shuffle(interior_points)
            
            for interior_idx in interior_points:
                best_position = None
                
                for insert_pos in range(len(order)):
                    test_order = order[:insert_pos] + [interior_idx] + order[insert_pos:]
                    if is_simple_polygon(points, test_order):
                        best_position = insert_pos
                        break
                
                if best_position is not None:
                    order.insert(best_position, interior_idx)
                else:
                    order.append(interior_idx)
            
            if is_simple_polygon(points, order):
                return order
            
        except Exception:
            pass
        
        return None
    
    def generate_random_polygons(self, points: List[Point], num_variations: int = 5, 
                                min_area_ratio: float = 0.1, max_attempts_per_strategy: int = 100) -> List[dict]:
        """Generate multiple random valid polygons using different strategies."""
        if len(points) < 3:
            return []
        
        strategies = [
            ("random_permutation", self.random_permutation_strategy)
        ]
        
        polygons = []
        seen_normalized_orders = set()
        
        def normalize_order(order):
            """Normalize polygon order to detect true duplicates (same polygon, different starting point)."""
            if not order:
                return tuple()
            # Find minimum index and rotate to start from there
            min_idx = order.index(min(order))
            normalized = order[min_idx:] + order[:min_idx]
            # Also check reverse direction
            reversed_order = normalized[:1] + normalized[1:][::-1]
            # Return the lexicographically smaller one
            return tuple(min(normalized, reversed_order))
        
        # Try each strategy multiple times to get diversity
        total_attempts = 0
        max_total_attempts = num_variations * max_attempts_per_strategy
        
        while len(polygons) < num_variations and total_attempts < max_total_attempts:
            total_attempts += 1
            
            # Cycle through strategies but with some randomization
            strategy_idx = (total_attempts - 1) % len(strategies)
            if self.rng.random() < 0.3:  # 30% chance to pick random strategy
                strategy_idx = self.rng.randint(0, len(strategies) - 1)
            
            strategy_name, strategy_func = strategies[strategy_idx]
            
            try:
                # Add some randomization to strategies
                if strategy_name == "random_permutation":
                    order = strategy_func(points, max_attempts=max_attempts_per_strategy)
                elif strategy_name == "angle_based":
                    # Vary the perturbation amount
                    perturbation = self.rng.uniform(0.1, 0.8)
                    old_func = strategy_func
                    strategy_func = lambda pts: self.angle_based_strategy(pts, perturbation)
                    order = strategy_func(points)
                elif strategy_name == "incremental_construction":
                    order = strategy_func(points, max_attempts=max_attempts_per_strategy)
                else:  # convex_hull_perturbation
                    order = strategy_func(points)
                
                if order is None:
                    continue
                
                # Use better duplicate detection
                normalized_order = normalize_order(order)
                if normalized_order in seen_normalized_orders:
                    continue
                
                metrics = compute_polygon_metrics(points, order)
                
                # More lenient area filter
                if metrics["area"] < min_area_ratio * 0.1:
                    continue
                
                polygon_data = {
                    "order": order,
                    "area": metrics["area"],
                    "perimeter": metrics["perimeter"],
                    "compactness": metrics["compactness"],
                    "strategy": strategy_name
                }
                
                polygons.append(polygon_data)
                seen_normalized_orders.add(normalized_order)
                
            except Exception as e:
                continue
        
        # If we still don't have enough, be more aggressive with random permutations
        while len(polygons) < num_variations and total_attempts < max_total_attempts * 2:
            total_attempts += 1
            try:
                order = self.random_permutation_strategy(points, max_attempts=200)
                if order is None:
                    continue
                    
                normalized_order = normalize_order(order)
                if normalized_order in seen_normalized_orders:
                    continue
                
                metrics = compute_polygon_metrics(points, order)
                if metrics["area"] < min_area_ratio * 0.05:  # Even more lenient
                    continue
                
                polygon_data = {
                    "order": order,
                    "area": metrics["area"],
                    "perimeter": metrics["perimeter"],
                    "compactness": metrics["compactness"],
                    "strategy": "random_permutation_aggressive"
                }
                
                polygons.append(polygon_data)
                seen_normalized_orders.add(normalized_order)
                
            except Exception:
                continue
        
        # Sort by area (largest first) and return requested number
        polygons.sort(key=lambda x: -x["area"])
        return polygons[:num_variations]


def generate_random_polygon_variations(points: List[Point], num_variations: int = 5, 
                                     seed: Optional[int] = None, min_area_ratio: float = 0.1,
                                     use_cpp: bool = True) -> List[dict]:
    """
    Generate multiple random valid simple polygons for a given set of points.
    
    Args:
        points: List of 2D points
        num_variations: Number of polygon variations to generate
        seed: Random seed for reproducibility
        min_area_ratio: Minimum area ratio to filter degenerate polygons
        use_cpp: Use C++ implementation for better performance
    
    Returns:
        List of polygon dictionaries with order, area, perimeter, compactness, and strategy
    """
    
    # Try C++ implementation first if available
    if use_cpp:
        try:
            import max_area_polygon_cpp
            import numpy as np
            
            # Convert points to numpy array
            points_array = np.array(points, dtype=np.float64)
            
            # Use C++ implementation
            variations = max_area_polygon_cpp.generate_random_polygon_variations(
                points_array,
                seed=seed if seed is not None else 0,
                num_variations=num_variations,
                min_area_ratio=min_area_ratio,
                max_attempts_per_variation=10000000
            )
            
            if variations:
                # print(f"✓ Using high-performance C++ random polygon generator")
                return variations
            else:
                print(f"⚠ C++ generator produced no results, falling back to Python")
        except ImportError:
            print(f"⚠ C++ random polygon generator not available, using Python")
        except Exception as e:
            print(f"⚠ C++ random polygon generator failed ({e}), falling back to Python")
    
    # Fall back to Python implementation
    generator = RandomPolygonGenerator(seed)
    return generator.generate_random_polygons(points, num_variations, min_area_ratio)


def main():
    """Test the random polygon generation."""
    test_points = [
        (0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9), (0.5, 0.5)
    ]
    
    print(f"Testing random polygon generation with {len(test_points)} points")
    
    variations = generate_random_polygon_variations(test_points, num_variations=10, seed=42)
    
    print(f"Generated {len(variations)} polygon variations:")
    for i, poly in enumerate(variations):
        print(f"  {i+1}: Order {poly['order']}, Area {poly['area']:.4f}, "
              f"Strategy: {poly['strategy']}")


if __name__ == "__main__":
    main()