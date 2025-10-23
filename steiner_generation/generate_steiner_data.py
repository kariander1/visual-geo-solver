#!/usr/bin/env python3
"""
Generate random Steiner tree instances with optimal solutions using GeoSteiner.
Creates textual data and visualizations for each instance.
Optimized version with multiprocessing support.
"""

import os
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import logging
from tqdm import tqdm

# Configure matplotlib for multiprocessing
plt.switch_backend('Agg')

# Set matplotlib parameters for performance
plt.rcParams.update({
    'figure.max_open_warning': 0,  # Disable warning about too many figures
    'axes.linewidth': 0.5,
    'axes.edgecolor': 'black',
    'font.size': 8,
})

class SteinerDataGenerator:
    def __init__(self, geosteiner_path, output_dir):
        self.geosteiner_path = Path(geosteiner_path)
        self.output_dir = Path(output_dir)
        
        # Ensure output directories exist
        (self.output_dir / "instances").mkdir(parents=True, exist_ok=True)
        if os.getenv('SAVE_SOLUTIONS', 'false').lower() == 'true':
            (self.output_dir / "solutions").mkdir(parents=True, exist_ok=True)
        if not os.getenv('SKIP_IMAGES'):
            (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        
        # Verify GeoSteiner binaries exist
        self.rand_points = (self.geosteiner_path / "rand_points").absolute()
        self.efst = (self.geosteiner_path / "efst").absolute()
        self.bb = (self.geosteiner_path / "bb").absolute()
        self.fst2graph = (self.geosteiner_path / "fst2graph").absolute()
        
        for binary in [self.rand_points, self.efst, self.bb, self.fst2graph]:
            if not binary.exists():
                raise FileNotFoundError(f"GeoSteiner binary not found: {binary}")
    
    def generate_random_points(self, num_points, seed=None, min_dist=0.1, 
                              node_radius=2, edge_width=2, image_size=128):
        """Generate random points with padding and minimum distance constraints."""
        # Calculate padding to keep nodes away from image edges
        pad = (node_radius + edge_width) / image_size
        
        # Use seeded random generator for reproducibility
        rng = np.random.RandomState(seed) if seed is not None else np.random
        
        points = []
        attempts = 0
        max_attempts = 1000
        
        while len(points) < num_points and attempts < max_attempts:
            # Generate candidate in safe region (avoiding edges)
            candidate = rng.rand(2) * (1 - 2 * pad) + pad
            
            # Check minimum distance constraint
            if all(np.linalg.norm(candidate - np.array(p)) > min_dist for p in points):
                points.append(candidate)
            attempts += 1
        
        if len(points) < num_points:
            raise ValueError(f"Could not place {num_points} non-overlapping nodes after {max_attempts} attempts")
        
        return np.array(points)
    
    def solve_steiner_tree(self, points):
        """Solve Steiner tree using GeoSteiner pipeline."""
        # Create input for efst
        points_str = '\n'.join([f"{x} {y}" for x, y in points])
        
        try:
            # Run efst | bb pipeline with timeout
            efst_proc = subprocess.Popen([str(self.efst)], 
                                       stdin=subprocess.PIPE, 
                                       stdout=subprocess.PIPE, 
                                       text=True, cwd=self.geosteiner_path)
            
            bb_proc = subprocess.Popen([str(self.bb)], 
                                     stdin=efst_proc.stdout, 
                                     stdout=subprocess.PIPE, 
                                     text=True, cwd=self.geosteiner_path)
            
            efst_proc.stdin.write(points_str)
            efst_proc.stdin.close()
            efst_proc.stdout.close()
            
            # Set timeout for solving
            try:
                bb_output, _ = bb_proc.communicate(timeout=30)  # 30 second timeout
            except subprocess.TimeoutExpired:
                bb_proc.kill()
                efst_proc.kill()
                raise RuntimeError("GeoSteiner solving timed out")
            
            if bb_proc.returncode != 0:
                raise RuntimeError("GeoSteiner solving failed")
            
        except Exception as e:
            raise RuntimeError(f"GeoSteiner pipeline error: {e}")
        
        # Parse solution
        steiner_points = []
        length = None
        
        for line in bb_output.split('\n'):
            # Extract Steiner points
            if line.strip().startswith('% @C'):
                parts = line.strip().split()
                if len(parts) >= 3:
                    x, y = float(parts[2]), float(parts[3])
                    steiner_points.append([x, y])
            
            # Extract length
            if 'length =' in line:
                match = re.search(r'length = ([\d.]+)', line)
                if match:
                    length = float(match.group(1))
        
        return np.array(steiner_points), length, bb_output
    
    def extract_graph_structure(self, terminal_points, steiner_points, bb_output):
        """Extract optimal tree structure from GeoSteiner bb output."""
        edges = []
        edge_weights = []
        
        # Combine all points (terminals first, then Steiner points)
        if len(steiner_points) > 0:
            all_points = np.vstack([terminal_points, steiner_points])
        else:
            all_points = terminal_points
        
        # Parse connection lines from bb output
        lines = bb_output.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for connection lines with 'T' and 'S'
            if 'T' in line and 'S' in line and not line.startswith('%'):
                parts = line.split()
                
                try:
                    # Handle terminal-to-terminal edges: "0 T 3 T S"
                    if len(parts) == 5 and parts[1] == 'T' and parts[3] == 'T' and parts[4] == 'S':
                        terminal1_idx = int(parts[0])
                        terminal2_idx = int(parts[2])
                        
                        if (0 <= terminal1_idx < len(terminal_points) and 
                            0 <= terminal2_idx < len(terminal_points)):
                            edge = [terminal1_idx, terminal2_idx]
                            # Avoid duplicates
                            if edge not in edges and [terminal2_idx, terminal1_idx] not in edges:
                                dist = np.linalg.norm(all_points[terminal1_idx] - all_points[terminal2_idx])
                                edges.append(edge)
                                edge_weights.append(dist)
                    
                    # Handle terminal-to-Steiner edges
                    elif len(parts) >= 4:
                        # Two formats possible:
                        # Format 1: "5 T .2618... .3039... S" (terminal first)
                        # Format 2: ".2618... .3039... 3 T S" (steiner coords first)
                        
                        if parts[1] == 'T':
                            # Format 1: terminal index, then Steiner coordinates
                            terminal_idx = int(parts[0])
                            steiner_x = float(parts[2])
                            steiner_y = float(parts[3])
                        else:
                            # Format 2: Steiner coordinates, then terminal index
                            steiner_x = float(parts[0])
                            steiner_y = float(parts[1])
                            terminal_idx = int(parts[2])
                        
                        # Find the matching Steiner point index
                        steiner_idx = None
                        for s_idx, sp in enumerate(steiner_points):
                            if abs(sp[0] - steiner_x) < 1e-6 and abs(sp[1] - steiner_y) < 1e-6:
                                steiner_idx = len(terminal_points) + s_idx
                                break
                        
                        # Add edge if both indices are valid
                        if steiner_idx is not None and 0 <= terminal_idx < len(terminal_points):
                            edge = [terminal_idx, steiner_idx]
                            # Avoid duplicates
                            if edge not in edges and [steiner_idx, terminal_idx] not in edges:
                                dist = np.linalg.norm(all_points[terminal_idx] - all_points[steiner_idx])
                                edges.append(edge)
                                edge_weights.append(dist)
                
                except (ValueError, IndexError):
                    continue
        
        # Add Steiner-to-Steiner edges by parsing FST segments
        # Look for lines like ".5297... .6313... .7182... .6559... S"
        for line in lines:
            line = line.strip()
            if not line.startswith('%') and 'S' in line and 'T' not in line:
                parts = line.split()
                try:
                    if len(parts) >= 5 and parts[-1] == 'S':
                        # This connects two Steiner points
                        steiner1_x, steiner1_y = float(parts[0]), float(parts[1])
                        steiner2_x, steiner2_y = float(parts[2]), float(parts[3])
                        
                        # Find both Steiner point indices
                        steiner1_idx = None
                        steiner2_idx = None
                        
                        for s_idx, sp in enumerate(steiner_points):
                            if abs(sp[0] - steiner1_x) < 1e-6 and abs(sp[1] - steiner1_y) < 1e-6:
                                steiner1_idx = len(terminal_points) + s_idx
                            if abs(sp[0] - steiner2_x) < 1e-6 and abs(sp[1] - steiner2_y) < 1e-6:
                                steiner2_idx = len(terminal_points) + s_idx
                        
                        # Add edge if both indices found
                        if steiner1_idx is not None and steiner2_idx is not None:
                            edge = [steiner1_idx, steiner2_idx]
                            if edge not in edges and [steiner2_idx, steiner1_idx] not in edges:
                                dist = np.linalg.norm(all_points[steiner1_idx] - all_points[steiner2_idx])
                                edges.append(edge)
                                edge_weights.append(dist)
                
                except (ValueError, IndexError):
                    continue
        
        return edges, edge_weights
    
    def create_visualization(self, points, steiner_points, edges, instance_id):
        """Create optimized visualization with minimal overhead."""
        if os.getenv('SKIP_IMAGES'):
            return None, None
            
        try:
            # Use a single optimized figure
            plt.figure(figsize=(8, 4), dpi=100)  # Lower DPI for speed
            
            # Left subplot: terminal points only
            plt.subplot(1, 2, 1)
            plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5)
            plt.title(f"Terminals ({len(points)})", fontsize=10)
            plt.axis('equal')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            
            # Right subplot: complete Steiner tree
            plt.subplot(1, 2, 2)
            plt.scatter(points[:, 0], points[:, 1], c='red', s=50, zorder=5, label='Terminals')
            
            if len(steiner_points) > 0:
                plt.scatter(steiner_points[:, 0], steiner_points[:, 1], 
                           c='blue', s=40, zorder=5, label='Steiner', marker='s')
            
            # Draw edges efficiently
            if edges and len(points) > 0:
                all_points = np.vstack([points] + ([steiner_points] if len(steiner_points) > 0 else []))
                
                for i, j in edges:
                    if i < len(all_points) and j < len(all_points):
                        plt.plot([all_points[i, 0], all_points[j, 0]], 
                               [all_points[i, 1], all_points[j, 1]], 
                               'g-', alpha=0.7, linewidth=1.5)
            
            plt.title(f"Steiner Tree", fontsize=10)
            plt.legend(fontsize=8)
            plt.axis('equal')
            plt.xlim(-0.05, 1.05)
            plt.ylim(-0.05, 1.05)
            
            # Save with minimal options
            solution_img_path = self.output_dir / "images" / f"instance_{instance_id:04d}_solution.png"
            plt.savefig(solution_img_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
            plt.close()  # Important: close figure to free memory
            
            return None, str(solution_img_path)
            
        except Exception as e:
            plt.close()  # Make sure to close on error too
            return None, None

def generate_single_instance(args):
    """Generate a single instance - designed for multiprocessing."""
    instance_id, num_points, geosteiner_path, output_dir = args
    
    try:
        # Create generator instance for this process
        generator = SteinerDataGenerator(geosteiner_path, output_dir)
        
        # Generate random points with unique seed per instance
        points = generator.generate_random_points(num_points, seed=instance_id)
        
        # Solve Steiner tree
        steiner_points, length, raw_output = generator.solve_steiner_tree(points)
        
        # Extract graph structure
        edges, edge_weights = generator.extract_graph_structure(points, steiner_points, raw_output)
        
        # Create visualizations
        points_img, solution_img = generator.create_visualization(points, steiner_points, edges, instance_id)
        
        # Prepare data structure
        instance_data = {
            "instance_id": instance_id,
            "num_terminals": len(points),
            "num_steiner_points": len(steiner_points),
            "terminal_points": points.tolist(),
            "steiner_points": steiner_points.tolist(),
            "edges": edges,
            "edge_weights": edge_weights,
            "total_length": length,
            "solution_image": solution_img
        }
        
        # Save instance data (if using separate JSON format)
        if os.getenv('OUTPUT_FORMAT', 'json') == 'json':
            instance_file = Path(output_dir) / "instances" / f"instance_{instance_id:04d}.json"
            with open(instance_file, 'w') as f:
                if os.getenv('SKIP_IMAGES'):
                    # Compact JSON for maximum speed
                    json.dump(instance_data, f, indent=None, separators=(',', ':'))
                else:
                    # Pretty JSON with images
                    json.dump(instance_data, f, indent=2)
        
        # Save raw solution only if requested
        if os.getenv('SAVE_SOLUTIONS', 'false').lower() == 'true':
            solution_file = Path(output_dir) / "solutions" / f"solution_{instance_id:04d}.txt"
            with open(solution_file, 'w') as f:
                f.write(raw_output)
        
        return {
            "id": instance_data["instance_id"],
            "num_terminals": instance_data["num_terminals"],
            "num_steiner_points": instance_data["num_steiner_points"],
            "total_length": instance_data["total_length"],
            "status": "success",
            "data": instance_data  # Include full data for NDJSON format
        }
        
    except Exception as e:
        return {
            "id": instance_id,
            "status": "failed",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description='Generate Steiner tree training data (parallel version)')
    parser.add_argument('--num-instances', type=int, default=100, help='Number of instances to generate')
    parser.add_argument('--min-points', type=int, default=10, help='Minimum number of terminal points')
    parser.add_argument('--max-points', type=int, default=20, help='Maximum number of terminal points')
    parser.add_argument('--geosteiner-path', type=str, 
                       default='geosteiner-5.3',
                       help='Path to GeoSteiner installation')
    parser.add_argument('--output-dir', type=str,
                       default='data/steiner_data',
                       help='Output directory for generated data')
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Number of instances to process in each batch')
    parser.add_argument('--skip-images', action='store_true', 
                       help='Skip image generation for maximum speed')
    parser.add_argument('--output-format', type=str, choices=['json', 'ndjson'], default='ndjson',
                       help='Output format: separate JSON files or single NDJSON file')
    parser.add_argument('--save-solutions', action='store_true',
                       help='Save raw GeoSteiner solution files (disabled by default to save disk space)')
    
    args = parser.parse_args()
    
    if args.skip_images:
        os.environ['SKIP_IMAGES'] = '1'
    
    # Set output format environment variable
    os.environ['OUTPUT_FORMAT'] = args.output_format
    
    # Set save solutions environment variable
    if args.save_solutions:
        os.environ['SAVE_SOLUTIONS'] = 'true'
    
    # Determine number of workers
    if args.num_workers is None:
        args.num_workers = min(mp.cpu_count(), 12)
    
    print(f"Generating {args.num_instances} Steiner tree instances...")
    print(f"Terminal points range: {args.min_points}-{args.max_points}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {args.num_workers} worker processes")
    print(f"Skip images: {args.skip_images}")
    print(f"Output format: {args.output_format}")
    print(f"Save solution files: {args.save_solutions}")
    
    # Verify output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate task parameters
    tasks = []
    for i in range(args.num_instances):
        # Randomly choose number of points
        num_points = np.random.randint(args.min_points, args.max_points + 1)
        tasks.append((i + 1, num_points, args.geosteiner_path, args.output_dir))
    
    # Process in parallel with progress tracking
    results = []
    failed_instances = []
    all_instance_data = []  # For NDJSON format
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(generate_single_instance, task): task for task in tasks}
        
        # Process results as they complete
        with tqdm(total=len(tasks), desc="Generating instances") as pbar:
            for future in as_completed(future_to_task):
                result = future.result()
                
                if result["status"] == "success":
                    results.append({
                        "id": result["id"],
                        "num_terminals": result["num_terminals"],
                        "num_steiner_points": result["num_steiner_points"],
                        "total_length": result["total_length"]
                    })
                    # Store full instance data for NDJSON format
                    if args.output_format == 'ndjson' and 'data' in result:
                        all_instance_data.append(result['data'])
                else:
                    failed_instances.append(result)
                    print(f"\nFailed to generate instance {result['id']}: {result['error']}")
                
                pbar.update(1)
    
    end_time = time.time()
    
    # Save NDJSON file if requested
    if args.output_format == 'ndjson' and all_instance_data:
        ndjson_path = Path(args.output_dir) / "instances.ndjson"
        print(f"\nSaving {len(all_instance_data)} instances to NDJSON format: {ndjson_path}")
        
        # Sort by instance_id for consistent ordering
        all_instance_data.sort(key=lambda x: x['instance_id'])
        
        with open(ndjson_path, 'w') as f:
            for instance_data in all_instance_data:
                # Write each instance as a single line
                if args.skip_images:
                    json.dump(instance_data, f, separators=(',', ':'))
                else:
                    json.dump(instance_data, f)
                f.write('\n')
        
        print(f"NDJSON file saved: {ndjson_path}")
    
    # Save summary
    summary = {
        "instances": results,
        "generation_time": end_time - start_time,
        "num_workers": args.num_workers,
        "failed_instances": failed_instances
    }
    
    summary_file = Path(args.output_dir) / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGeneration complete!")
    print(f"Successfully generated: {len(results)}/{args.num_instances} instances")
    if failed_instances:
        print(f"Failed instances: {len(failed_instances)}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    if results:
        print(f"Average time per instance: {(end_time - start_time) / len(results):.3f} seconds")
    print(f"Files saved in: {args.output_dir}")

if __name__ == "__main__":
    # Set multiprocessing start method
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass  # Already set
    
    main()