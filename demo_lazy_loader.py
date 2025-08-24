#!/usr/bin/env python3
"""
Demo script showing the lazy loader with progress bars for CityD dataset
"""

import time
from pathlib import Path
from data_utils import load_data_lazy
import psutil
import os

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def demo_lazy_loader():
    """Demonstrate the lazy loader with progress bars"""
    cityD = Path().joinpath("data").joinpath("cityD-dataset.csv")
    
    if not cityD.exists():
        print(f"Dataset file not found: {cityD}")
        print("Please ensure the CityD dataset is in the data/ directory")
        return
    
    # Get file info
    file_size_mb = cityD.stat().st_size / (1024 * 1024)
    print("="*60)
    print("LAZY LOADER DEMO - CityD Dataset")
    print("="*60)
    print(f"File: {cityD}")
    print(f"Size: {file_size_mb:.1f} MB")
    
    # Monitor memory usage
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # Start timing
    print(f"\nStarting lazy load at {time.strftime('%H:%M:%S')}")
    start_time = time.time()
    
    try:
        # Load data with progress bars
        trajectories = load_data_lazy(cityD)
        
        # End timing
        end_time = time.time()
        final_memory = get_memory_usage()
        
        # Results
        print("\n" + "="*60)
        print("LOADING COMPLETE")
        print("="*60)
        print(f"[SUCCESS] Loaded {len(trajectories):,} trajectories")
        print(f"[SUCCESS] Total time: {end_time - start_time:.2f} seconds")
        print(f"[SUCCESS] Memory usage: {initial_memory:.1f} MB -> {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)")
        print(f"[SUCCESS] Processing rate: {file_size_mb / (end_time - start_time):.1f} MB/s")
        
        # Sample trajectory info
        if trajectories:
            total_points = sum(len(traj) for traj in trajectories)
            avg_points = total_points / len(trajectories)
            print(f"[SUCCESS] Total trajectory points: {total_points:,}")
            print(f"[SUCCESS] Average points per trajectory: {avg_points:.1f}")
            
            # Show sample trajectory (first 3 points):
            print(f"\nSample trajectory (first 3 points):")
            sample_traj = trajectories[0][:3]
            for i, point in enumerate(sample_traj):
                uid, x, y, is_weekend, t, duration = point
                print(f"  Point {i+1}: uid={uid}, pos=({x:.3f},{y:.3f}), weekend={is_weekend}, time={t}, duration={duration}")
    
    except Exception as e:
        print(f"\n[ERROR] Error loading data: {e}")
        return
    
    print(f"\nDemo completed at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    demo_lazy_loader()