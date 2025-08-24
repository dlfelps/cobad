import pandas as pd
from pathlib import Path
import pickle
from itertools import pairwise

def load_data(file_path):
  def is_weekend(day):
    # 0 sunday, 1 monday, ..., 6 saturday
    temp = day % 7
    return int(temp == 0 or temp == 6)

  df = pd.read_csv(file_path)
  df = df[df['uid'] < 4000]
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
        is_weekend(day),  # Weekend flag
        t,  # Start time slot
        duration  # Duration in time slots
      )
      stay_points.append(stay_point)
    
    if stay_points:
      data.append(stay_points)
      
  return data

