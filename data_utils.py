import csv
import os
import pickle
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

def _is_weekend(day):
    """Check if a day is weekend (0=Sunday, 6=Saturday)"""
    temp = day % 7
    return int(temp == 0 or temp == 6)


def load_data(file_path):
    """Load trajectory data using pandas (memory-intensive for large files)"""
    df = pd.read_csv(file_path)
    data = []
    
    for _, g in df.groupby(['uid', 'd']):
        if len(g) < 1:
            continue
        
        # Sort by time to ensure chronological order
        g = g.sort_values('t')
        stay_points = []
        
        # Convert each location record to a stay point event
        for i, row in g.iterrows():
            uid, day, t, x, y = row['uid'], row['d'], row['t'], row['x'], row['y']
            
            # Calculate stay duration
            if i + 1 < len(g):
                # Duration until next location (in 30-minute slots)
                next_t = g.iloc[i + 1]['t'] if i + 1 < len(g.index) else t + 1
                duration = next_t - t
            else:
                # Last location: assume minimum 1 time slot duration
                duration = 1
            
            # Create stay point: (uid, normalized_x, normalized_y, is_weekend, start_time, duration)
            stay_point = (
                uid,
                x / 200.0,  # Normalized spatial X
                y / 200.0,  # Normalized spatial Y  
                _is_weekend(day),  # Weekend flag
                t,  # Start time slot
                duration  # Duration in time slots
            )
            stay_points.append(stay_point)
        
        if stay_points:
            data.append(stay_points)
            
    return data

def load_data_lazy(file_path):
    """
    Memory-efficient lazy loading version of load_data that reads CSV line-by-line
    without loading entire file into pandas DataFrame first.
    """
    # Group trajectories by (uid, day) while reading line by line
    user_day_data = defaultdict(list)
    
    print("Reading CSV file...")
    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Count total rows first for progress bar
        lines = sum(1 for _ in csvfile) - 1  # subtract header
        csvfile.seek(0)  # reset to beginning
        reader = csv.DictReader(csvfile)  # recreate reader
        
        for row in tqdm(reader, total=lines, desc="Loading data", unit=" rows"):
            uid = int(row['uid'])
            day = int(row['d'])
            t = int(row['t'])
            x = int(row['x'])
            y = int(row['y'])
            
            user_day_data[(uid, day)].append((t, x, y))
    
    # Process each user-day trajectory
    print("Processing trajectories...")
    data = []
    for (uid, day), trajectory in tqdm(user_day_data.items(), desc="Converting to stay points"):
        if len(trajectory) < 1:
            continue
        
        # Data is already in correct chronological order
        stay_points = []
        
        # Convert each location record to a stay point event
        for i, (t, x, y) in enumerate(trajectory):
            # Calculate stay duration
            if i + 1 < len(trajectory):
                # Duration until next location (in 30-minute slots)
                next_t = trajectory[i + 1][0]
                duration = next_t - t
            else:
                # Last location: assume minimum 1 time slot duration
                duration = 1
            
            # Create stay point: (uid, normalized_x, normalized_y, is_weekend, start_time, duration)
            stay_point = (
                uid,
                x / 200.0,  # Normalized spatial X
                y / 200.0,  # Normalized spatial Y  
                _is_weekend(day),  # Weekend flag
                t,  # Start time slot
                duration  # Duration in time slots
            )
            stay_points.append(stay_point)
        
        if stay_points:
            data.append(stay_points)
            
    return data

