# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CoBAD (Collective Behavior Anomaly Detection) project for analyzing human mobility trajectory data. CoBAD is designed to detect anomalies in human mobility by modeling collective behaviors - how people interact and move together in space and time.

The project uses the LYMob-4Cities dataset, which contains multi-city human mobility data from 4 metropolitan areas in Japan, with movement data across a 75-day period in 30-minute intervals on 500m x 500m grid cells.

## Development Environment

This is a Python project managed with `uv` (modern Python package manager). 

### Common Commands

- **Install dependencies**: `uv sync`
- **Run Python scripts**: `uv run python <script_name>.py`
- **Launch Jupyter Lab**: `uv run jupyter lab`
- **Add new dependency**: `uv add <package_name>`

### Python Environment
- Requires Python >= 3.13
- Uses uv for dependency management (pyproject.toml + uv.lock)

## Key Dependencies

- **PyTorch**: Deep learning framework for CoBAD model implementation
- **Pandas/NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning utilities
- **Matplotlib/Plotly**: Data visualization
- **Dash**: Interactive web applications for mobility data visualization
- **TQDM**: Progress bars for long-running trajectory analysis

## Data Structure

### Core Dataset Files (in data/ directory):
- `cityD-dataset.csv`: Main mobility trajectory data for City D
- `POIdata_cityD.csv`: Points of Interest data for City D (~85 dimensional vectors)
- `POI_datacategories.csv`: List of 85 POI categories (restaurants, parks, etc.)
- `nature.pdf`: Additional documentation

### Data Characteristics:
- Grid-based: 200x200 grid cells (500m x 500m each)
- Temporal: 75-day period, 30-minute intervals  
- Multi-city: Cities A,B,C,D with 100k, 25k, 20k, 6k individuals respectively
- POI Features: 85-dimensional feature vectors per grid cell

## Architecture Notes

This is a trajectory analysis project focusing on collective behavior patterns. The CoBAD model likely involves:
- Spatial-temporal data processing for mobility trajectories
- Collective behavior modeling (group movement patterns)
- Anomaly detection in human mobility patterns
- POI-based feature engineering for location context

When working with this codebase:
- Mobility data is discretized into grid cells and time intervals
- POI features provide semantic context for locations
- Focus on collective (group) rather than individual behavior patterns
- Anomaly detection targets unusual mobility patterns in the collective behavior space