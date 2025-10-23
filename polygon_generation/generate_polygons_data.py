#!/usr/bin/env python3
"""
Generate random point sets and their corresponding maximum area polygonalizations.
Creates textual data and visualizations for each instance.
Optimized version with multiprocessing support and C++ acceleration.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import threading

# Import our polygon solvers
try:
    from max_area_polygonization import max_area_polygon
    from max_area_polygon_ilp import solve_max_area_polygon_ilp
    HAS_ILP = True
except ImportError as e:
    print(f"Warning: Could not import ILP solver: {e}")
    from max_area_polygonization import max_area_polygon
    HAS_ILP = False

# Import random polygon generator
try:
    from random_polygonization import generate_random_polygon_variations
    HAS_RANDOM = True
except ImportError as e:
    print(f"Warning: Could not import random polygon generator: {e}")
    HAS_RANDOM = False

# Try to import the C++ solver
try:
    import max_area_polygon_cpp
    HAS_CPP = True
    # print("✓ Using high-performance C++ solver")
except ImportError:
    HAS_CPP = False
    print("⚠ C++ solver not available, using Python solver")

# Set matplotlib parameters for performance
plt.rcParams.update({
    'figure.max_open_warning': 0,  # Disable warning about too many figures
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'black',
    'font.size': 8,
})


class NDJSONWriter:
    """Thread-safe NDJSON writer for concurrent polygon generation."""
    
    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.lock = threading.Lock()
        self.file_handle = None
        
    def __enter__(self):
        self.file_handle = open(self.output_path, 'w')
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
    
    def write_instance(self, instance_data):
        """Write a single instance as one line of JSON."""
        with self.lock:
            json_line = json.dumps(instance_data, separators=(',', ':'))
            self.file_handle.write(json_line + '\n')
            self.file_handle.flush()

class PolygonDataGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        
        # Ensure output directories exist
        (self.output_dir / "instances").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "solutions").mkdir(parents=True, exist_ok=True)
        if not os.getenv('SKIP_IMAGES'):
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
    
    def generate_random_points(self, num_points, seed=None, min_dist=0.15, 
                              node_radius=2, edge_width=2, image_size=128, use_cpp=True):
        """Generate random points with padding and minimum distance constraints."""
        if use_cpp and HAS_CPP:
            # Use high-performance C++ point generation
            seed_val = seed if seed is not None else 0
            try:
                points = max_area_polygon_cpp.generate_random_points(
                    num_points=num_points,
                    seed=seed_val,
                    min_dist=min_dist,
                    node_radius=node_radius,
                    edge_width=edge_width,
                    image_size=image_size,
                    max_attempts=10000000
                )
                return points
            except Exception as e:
                print(f"Warning: C++ point generation failed, falling back to Python: {e}")
        
        # Fall back to Python implementation (simplified)
        pad = (node_radius + edge_width) / image_size
        rng = np.random.RandomState(seed) if seed is not None else np.random
        
        points = []
        for _ in range(1000):  # max attempts
            if len(points) >= num_points:
                break
            candidate = rng.rand(2) * (1 - 2 * pad) + pad
            if all(np.linalg.norm(candidate - np.array(p)) > min_dist for p in points):
                points.append(candidate)
        
        if len(points) < num_points:
            raise ValueError(f"Could not place {num_points} points")
        
        return np.array(points)
    
    def solve_max_area_polygonalization(self, points, use_ilp=False, use_cpp=True, time_limit=30):
        """Solve maximum area polygonalization using appropriate solver."""
        points_list = [(float(x), float(y)) for x, y in points]
        
        try:
            if use_ilp and HAS_ILP and len(points) >= 8:
                order, area = solve_max_area_polygon_ilp(
                    points_list, time_limit=time_limit, solver_name="CBC", msg=False
                )
                solver_used = "ILP"
            elif use_cpp and HAS_CPP:
                import numpy as np
                points_array = np.array(points_list)
                order, area = max_area_polygon_cpp.max_area_polygon(
                    points_array, time_limit=time_limit, precompute_cross=True
                )
                solver_used = "C++"
            else:
                order, area = max_area_polygon(
                    points_list, time_limit=time_limit, precompute_cross=True
                )
                solver_used = "Python"
            
            if not order or area <= 0:
                raise RuntimeError("No valid polygonalization found")
                
            return order, area, solver_used
            
        except Exception as e:
            raise RuntimeError(f"Polygonalization solving failed: {e}")
    
    def generate_random_polygons(self, points, num_variations=3, seed=None, min_area_ratio=0.1):
        """Generate multiple random polygon variations for the same point set."""
        if not HAS_RANDOM:
            raise RuntimeError("Random polygon generator not available")
        
        points_list = [(float(x), float(y)) for x, y in points]
        
        try:
            variations = generate_random_polygon_variations(
                points_list, 
                num_variations=num_variations,
                seed=seed,
                min_area_ratio=min_area_ratio
            )
            
            if not variations:
                raise RuntimeError("No valid random polygons found")
            
            return variations
            
        except Exception as e:
            raise RuntimeError(f"Random polygon generation failed: {e}")
    
    def create_visualization(self, points, polygons, instance_id, polygon_mode="max_area"):
        """Create optimized visualization with minimal overhead."""
        if os.getenv('SKIP_IMAGES'):
            return None, None
            
        try:
            if polygon_mode == "random" and isinstance(polygons, list):
                # Multiple polygon variations
                num_polygons = min(len(polygons), 4)  # Limit to 4 variations
                fig_width = 4 * num_polygons
                plt.figure(figsize=(fig_width, 4), dpi=100)
                
                for i, poly_data in enumerate(polygons[:num_polygons]):
                    plt.subplot(1, num_polygons, i + 1)
                    plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
                    
                    order = poly_data['order']
                    if order and len(order) >= 3:
                        polygon_points = [points[j] for j in order]
                        polygon_points.append(polygon_points[0])
                        
                        xs = [p[0] for p in polygon_points]
                        ys = [p[1] for p in polygon_points]
                        plt.plot(xs, ys, 'g-', alpha=0.7, linewidth=1.5)
                        plt.fill(xs, ys, 'green', alpha=0.1)
                    
                    plt.title(f"Var {i+1}\\nA={poly_data['area']:.3f}", fontsize=8)
                    plt.axis('equal')
                    plt.xlim(-0.05, 1.05)
                    plt.ylim(-0.05, 1.05)
            elif polygon_mode == "single":
                # Single polygon visualization (for individual random polygon files)
                order = polygons if isinstance(polygons, list) and all(isinstance(x, int) for x in polygons) else None
                plt.figure(figsize=(8, 4), dpi=100)
                
                # Left subplot: points only
                plt.subplot(1, 2, 1)
                plt.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=5)
                plt.title(f"Points ({len(points)})", fontsize=10)
                plt.axis('equal')
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                
                # Right subplot: solution
                plt.subplot(1, 2, 2)
                plt.scatter(points[:, 0], points[:, 1], c='red', s=20, zorder=5)
                
                if order and len(order) >= 3:
                    polygon_points = [points[i] for i in order]
                    polygon_points.append(polygon_points[0])
                    
                    xs = [p[0] for p in polygon_points]
                    ys = [p[1] for p in polygon_points]
                    plt.plot(xs, ys, 'g-', alpha=0.7, linewidth=1.5)
                    plt.fill(xs, ys, 'green', alpha=0.1)
                
                plt.title(f"Random Polygon", fontsize=10)
                plt.axis('equal')
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
            else:
                # Single polygon (max_area mode)
                order = polygons if isinstance(polygons, list) and all(isinstance(x, int) for x in polygons) else None
                plt.figure(figsize=(8, 4), dpi=100)
                
                # Left subplot: points only
                plt.subplot(1, 2, 1)
                plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
                plt.title(f"Points ({len(points)})", fontsize=10)
                plt.axis('equal')
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
                
                # Right subplot: solution
                plt.subplot(1, 2, 2)
                plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
                
                if order and len(order) >= 3:
                    polygon_points = [points[i] for i in order]
                    polygon_points.append(polygon_points[0])
                    
                    xs = [p[0] for p in polygon_points]
                    ys = [p[1] for p in polygon_points]
                    plt.plot(xs, ys, 'g-', alpha=0.7, linewidth=1.5)
                    plt.fill(xs, ys, 'green', alpha=0.1)
                
                plt.title(f"Solution", fontsize=10)
                plt.axis('equal')
                plt.xlim(-0.05, 1.05)
                plt.ylim(-0.05, 1.05)
            
            # Save with minimal options
            solution_img_path = self.output_dir / "images" / f"instance_{instance_id:04d}.png"
            plt.savefig(solution_img_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()  # Important: close figure to free memory
            
            return None, str(solution_img_path)
            
        except Exception as e:
            plt.close()  # Make sure to close on error too
            return None, None


def generate_single_instance(args):
    """Optimized single instance generation."""
    instance_id, num_points, output_dir, use_ilp, use_cpp, time_limit, polygon_mode, polygons_per_set, min_area_ratio, output_format, ndjson_writer = args
    
    try:
        generator = PolygonDataGenerator(output_dir)
        
        # Generate points
        points = generator.generate_random_points(num_points, seed=instance_id, use_cpp=use_cpp)
        
        if polygon_mode == "random":
            # Generate random polygon variations
            polygons = generator.generate_random_polygons(
                points, num_variations=polygons_per_set, seed=instance_id, min_area_ratio=min_area_ratio
            )
            
            # Save each variation as a separate JSON file (matching max_area format)
            saved_files = []
            for var_idx, polygon_data in enumerate(polygons):
                # Calculate the global instance ID (spread across all variations)
                global_instance_id = (instance_id - 1) * polygons_per_set + var_idx + 1
                
                # Create individual visualization
                points_img, solution_img = generator.create_visualization(
                    points, polygon_data['order'], global_instance_id, polygon_mode="single"
                )
                
                # Prepare data structure matching max_area format
                instance_data = {
                    "instance_id": global_instance_id,
                    "num_points": len(points),
                    "points": points.tolist(),
                    "polygon_mode": "random",
                    "polygon_order": polygon_data['order'],
                    "polygon_area": polygon_data['area'],
                    "solver_used": polygon_data['strategy'],
                    "solution_image": solution_img,
                    # Additional fields specific to random polygons
                    "polygon_perimeter": polygon_data['perimeter'],
                    "polygon_compactness": polygon_data['compactness']
                }
                
                # Save data based on output format
                if output_format == "ndjson":
                    # For NDJSON format, we'll collect data and write later
                    pass
                else:
                    # Save individual file (default behavior)
                    instance_file = Path(output_dir) / "instances" / f"instance_{global_instance_id:04d}.json"
                    with open(instance_file, 'w') as f:
                        json.dump(instance_data, f, indent=None, separators=(',', ':'))  # Compact JSON
                
                saved_files.append({
                    "global_id": global_instance_id,
                    "variation_idx": var_idx,
                    "area": polygon_data['area'],
                    "strategy": polygon_data['strategy'],
                    "instance_data": instance_data if output_format == "ndjson" else None
                })
            
            return {
                "id": instance_id,
                "num_points": len(points),
                "num_variations": len(polygons),
                "polygon_mode": "random",
                "saved_files": saved_files,
                "status": "success"
            }
            
        else:
            # Generate maximum area polygon (original behavior)
            order, area, solver_used = generator.solve_max_area_polygonalization(
                points, use_ilp=use_ilp, use_cpp=use_cpp, time_limit=time_limit
            )
            
            # Create visualization
            points_img, solution_img = generator.create_visualization(points, order, instance_id)
            
            # Minimal data structure
            instance_data = {
                "instance_id": instance_id,
                "num_points": len(points),
                "points": points.tolist(),
                "polygon_mode": "max_area",
                "polygon_order": order,
                "polygon_area": area,
                "solver_used": solver_used,
                "solution_image": solution_img
            }
            
            # Save data based on output format
            if output_format == "ndjson":
                # For NDJSON format, we'll collect data and write later
                pass
            else:
                # Save individual file (default behavior)
                instance_file = Path(output_dir) / "instances" / f"instance_{instance_id:04d}.json"
                with open(instance_file, 'w') as f:
                    json.dump(instance_data, f, indent=None, separators=(',', ':'))  # Compact JSON
            
            return {
                "id": instance_id,
                "num_points": len(points),
                "polygon_area": area,
                "solver_used": solver_used,
                "polygon_mode": "max_area",
                "status": "success",
                "instance_data": instance_data if output_format == "ndjson" else None
            }
        
    except Exception as e:
        return {
            "id": instance_id,
            "status": "failed",
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Generate polygon training data (optimized version)')
    parser.add_argument('--num-instances', type=int, default=1000)
    parser.add_argument('--min-points', type=int, default=8)
    parser.add_argument('--max-points', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='data/polygon_data')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--use-ilp', action='store_true')
    parser.add_argument('--use-cpp', action='store_true', default=True)
    parser.add_argument('--no-cpp', dest='use_cpp', action='store_false')
    parser.add_argument('--skip-images', action='store_true', help='Skip image generation for maximum speed')
    parser.add_argument('--time-limit', type=float, default=30)
    
    # New arguments for random polygon generation
    parser.add_argument('--polygon-mode', type=str, choices=['max_area', 'random'], default='max_area',
                        help='Polygon generation mode: max_area (default) or random variations')
    parser.add_argument('--polygons-per-set', type=int, default=3,
                        help='Number of polygon variations to generate per point set (random mode only)')
    parser.add_argument('--min-area-ratio', type=float, default=0.1,
                        help='Minimum area ratio to filter degenerate polygons (random mode only)')
    
    # Output format arguments
    parser.add_argument('--output-format', type=str, choices=['separate', 'ndjson'], default='ndjson',
                        help='Output format: separate JSON files or single NDJSON file (default)')
    parser.add_argument('--ndjson-file', type=str, default='instances.ndjson',
                        help='Output filename for NDJSON format (default: instances.ndjson)')
    
    args = parser.parse_args()
    
    if args.skip_images:
        os.environ['SKIP_IMAGES'] = '1'
    
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 8)
    
    print(f"Polygon data generation:")
    print(f"  Instances: {args.num_instances}")
    print(f"  Points: {args.min_points}-{args.max_points}")
    print(f"  Workers: {args.num_workers}")
    print(f"  C++ enabled: {args.use_cpp and HAS_CPP}")
    print(f"  Skip images: {args.skip_images}")
    print(f"  Polygon mode: {args.polygon_mode}")
    print(f"  Output format: {args.output_format}")
    if args.output_format == "ndjson":
        print(f"  NDJSON file: {args.ndjson_file}")
    if args.polygon_mode == "random":
        print(f"  Polygons per set: {args.polygons_per_set}")
        print(f"  Min area ratio: {args.min_area_ratio}")
        print(f"  Random generator: {'available' if HAS_RANDOM else 'NOT AVAILABLE'}")
        if not HAS_RANDOM:
            print("  ERROR: Random polygon generator not available. Install required dependencies.")
            return
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup output directories based on format
    if args.output_format == "separate":
        # Create instances directory for separate JSON files
        (Path(args.output_dir) / "instances").mkdir(parents=True, exist_ok=True)
    
    # Generate tasks - Note: ndjson_writer will be passed as None for ProcessPoolExecutor
    # as we can't share the writer across processes. We'll handle this differently.
    tasks = []
    for i in range(args.num_instances):
        num_points = np.random.randint(args.min_points, args.max_points + 1)
        tasks.append((i + 1, num_points, args.output_dir, args.use_ilp, args.use_cpp, args.time_limit,
                     args.polygon_mode, args.polygons_per_set, args.min_area_ratio, args.output_format, None))
    
    # Process in parallel
    results = []
    failed_instances = []
    start_time = time.time()
    
    # For NDJSON format, we need to collect results and write them afterwards
    # since we can't share a single writer across processes
    if args.output_format == "ndjson":
        all_instance_data = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_task = {executor.submit(generate_single_instance, task): task for task in tasks}
        
        with tqdm(total=len(tasks), desc="Generating") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                
                if result["status"] == "success":
                    if result.get("polygon_mode") == "random":
                        # For random mode, add each individual variation to results
                        for file_info in result.get("saved_files", []):
                            result_data = {
                                "id": file_info["global_id"],
                                "num_points": result["num_points"],
                                "polygon_mode": "random",
                                "polygon_area": file_info["area"],
                                "solver_used": file_info["strategy"]
                            }
                            results.append(result_data)
                            
                            # Collect instance data for NDJSON format
                            if args.output_format == "ndjson" and file_info.get("instance_data"):
                                all_instance_data.append(file_info["instance_data"])
                    else:
                        # Max area mode (original)
                        result_data = {
                            "id": result["id"],
                            "num_points": result["num_points"],
                            "polygon_mode": result.get("polygon_mode", "max_area"),
                            "polygon_area": result.get("polygon_area", 0),
                            "solver_used": result.get("solver_used", "unknown")
                        }
                        results.append(result_data)
                        
                        # Collect instance data for NDJSON format
                        if args.output_format == "ndjson" and result.get("instance_data"):
                            all_instance_data.append(result["instance_data"])
                else:
                    failed_instances.append(result)
                
                pbar.update(1)
    
    end_time = time.time()
    
    # Write NDJSON file if requested
    if args.output_format == "ndjson" and all_instance_data:
        ndjson_path = Path(args.output_dir) / args.ndjson_file
        print(f"Writing {len(all_instance_data)} instances to {ndjson_path}")
        
        with open(ndjson_path, 'w') as f:
            for instance_data in all_instance_data:
                json_line = json.dumps(instance_data, separators=(',', ':'))
                f.write(json_line + '\n')
        
        print(f"NDJSON file written successfully")
    
    # Save summary
    summary = {
        "instances": results,
        "generation_time": end_time - start_time,
        "num_workers": args.num_workers,
        "failed_instances": failed_instances,
        "solver_stats": {}
    }
    
    if results:
        # Compute statistics based on polygon mode
        mode_counts = {}
        solver_counts = {}
        
        for r in results:
            mode = r.get("polygon_mode", "unknown")
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            if "solver_used" in r:
                solver = r.get("solver_used", "unknown")
                solver_counts[solver] = solver_counts.get(solver, 0) + 1
        
        summary["mode_stats"] = mode_counts
        if solver_counts:
            summary["solver_stats"] = solver_counts
    
    with open(Path(args.output_dir) / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Success: {len(results)}/{args.num_instances}")
    if results:
        if "mode_stats" in summary:
            print(f"Mode usage: {summary['mode_stats']}")
        if "solver_stats" in summary:
            print(f"Solver usage: {summary['solver_stats']}")
    print(f"Total time: {end_time - start_time:.2f}s")
    if len(results) > 0:
        print(f"Avg per instance: {(end_time - start_time) / len(results):.3f}s")


if __name__ == "__main__":
    main()