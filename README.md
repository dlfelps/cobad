# CoBAD: Collective Behavior Anomaly Detection

A Python implementation of Collective Behavior Anomaly Detection for human mobility trajectory data. CoBAD identifies anomalous collective behaviors by analyzing groups of people moving together in space and time, rather than focusing on individual anomalies.

## Project Overview

CoBAD processes human mobility data from the LYMob-4Cities dataset, which contains multi-city trajectory data from 4 metropolitan areas in Japan. The system models collective behaviors using stay point events and detects anomalous group patterns using deep learning techniques.

### Key Features

- **Stay Point Analysis**: Converts trajectory data into meaningful stay events with locations and durations
- **Collective Behavior Modeling**: Groups individuals by spatial-temporal proximity using DBSCAN clustering
- **Deep Learning Anomaly Detection**: Uses autoencoder neural networks for behavior embedding and reconstruction-based anomaly scoring
- **Dynamic Threshold Adjustment**: Multiple threshold methods to maintain target anomaly rates
- **Comprehensive Visualization**: Multi-panel plots showing spatial distributions, temporal patterns, and anomaly scores

## Installation

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. **Clone or download the project:**
   ```bash
   cd cobad
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

   This will automatically install all required dependencies including:
   - PyTorch (deep learning framework)
   - Pandas & NumPy (data processing)
   - Scikit-learn (machine learning utilities)
   - Matplotlib & Plotly (visualization)
   - TQDM (progress bars)
   - Scipy (statistical functions)

3. **Verify installation:**
   ```bash
   uv run python -c "import torch, pandas, sklearn; print('All dependencies installed successfully!')"
   ```

### Data Setup

Place your trajectory data files in the `data/` directory:
- `cityD-dataset.csv`: Main mobility trajectory data
- `POIdata_cityD.csv`: Points of Interest data (optional)
- `POI_datacategories.csv`: POI category definitions (optional)

Expected CSV format for trajectory data:
```
uid,d,t,x,y
0,0,13,133,103
0,0,14,143,98
...
```

Where:
- `uid`: User ID
- `d`: Day number (0-74 for 75-day period)
- `t`: Time slot (30-minute intervals)
- `x,y`: Grid coordinates (0-199 for 200x200 grid)

## Usage

### Basic Usage

Run the complete CoBAD pipeline:

```bash
uv run python main.py
```

### Customizing Parameters

Edit `main.py` to adjust CoBAD parameters:

```python
# Initialize with custom parameters
cobad = CoBAD(
    spatial_dim=200,                    # Grid size (200x200)
    temporal_window=48,                 # Time slots per day
    collective_threshold=3,             # Minimum people for collective behavior
    target_anomaly_rate=0.05,          # Target 5% anomaly rate
    threshold_method='adaptive'         # Threshold adjustment method
)
```

### Threshold Methods

Choose from different dynamic threshold adjustment methods:

- **`percentile`**: Uses target anomaly rate percentile
- **`mad`**: Median Absolute Deviation (robust to outliers)
- **`isolation_forest`**: Isolation Forest contamination boundary
- **`adaptive`**: Automatically selects based on data distribution

### Output Files

The system generates:
- **`cobad_model.pkl`**: Trained model for reuse
- **`cobad_results.png`**: Visualization of anomaly detection results
- Console output with detailed analysis results

## Comprehensive Anomaly Analysis

For detailed anomaly interpretation, run the comprehensive analysis tool:

```bash
uv run python anomaly_analysis.py
```

### Analysis Outputs

The `anomaly_analysis.py` script generates comprehensive outputs for in-depth anomaly interpretation:

#### 1. Interactive Visualization
- **`anomaly_temporal_analysis.html`**: Multi-panel interactive dashboard with:
  - Temporal distribution of anomalies by hour of day
  - Spatial scatter plots showing anomaly locations  
  - Time-series plots of anomaly scores over time
  - Subscore analysis across different anomaly types
  - Weekend vs weekday pattern comparisons
  - Group size correlations with anomaly scores

#### 2. Text Report
- **`anomaly_report.txt`**: Comprehensive text report containing:
  - Overview statistics (total behaviors, anomaly count, detection rate)
  - Score distribution analysis (normal vs anomalous behavior statistics)
  - Detailed subscore breakdowns for each anomaly type
  - Feature attribution analysis showing which features contribute most to anomalies
  - Link analysis results showing relationships between anomalies
  - Top 10 anomalies with complete breakdowns including subscores and feature contributions

#### 3. Numerical Data Files (in `analysis_results/` directory)
- **`anomaly_features.npy`**: Raw feature vectors for all collective behaviors
- **`anomaly_scores.npy`**: Anomaly scores for each behavior
- **`anomaly_labels.npy`**: Binary labels (1=anomaly, 0=normal)
- **`feature_attributions.npy`**: Per-feature contribution scores for each anomaly
- **`link_analysis.pkl`**: Network graph data showing relationships between anomalies
- **`subscore_*.npy`**: Individual subscore arrays for detailed analysis

### Subscore Interpretations

The analysis computes six different subscore types to provide detailed anomaly explanations:

#### **Spatial Anomaly Score**
- **What it measures**: Unusual spatial patterns and location spreads
- **High scores (>1.0)**: Collective behaviors in rare locations, unusual spatial clustering patterns, or extreme spatial spreads
- **Low scores (<0.5)**: Behaviors occurring in typical, frequently-used locations with normal spatial characteristics
- **Interpretation**: High spatial anomaly indicates groups gathering in unexpected places or with unusual spatial arrangements

#### **Temporal Anomaly Score**
- **What it measures**: Rare time-of-day patterns for collective behaviors
- **High scores (>0.8)**: Activities occurring at unusual hours when few collective behaviors typically happen
- **Low scores (<0.3)**: Activities during peak hours with high collective activity
- **Interpretation**: High temporal anomaly suggests groups active during off-peak hours or unusual daily patterns

#### **Size Anomaly Score**
- **What it measures**: Unusual group sizes relative to typical collective behavior patterns
- **High scores (>1.5)**: Extremely large or extremely small groups compared to the norm
- **Low scores (<0.5)**: Group sizes typical for the observed collective behaviors
- **Interpretation**: High size anomaly indicates unusually large gatherings or suspiciously small groups claiming to be collective

#### **Duration Anomaly Score**
- **What it measures**: Atypical stay durations for collective behaviors
- **High scores (>2.0)**: Extremely short or extremely long stay durations compared to normal patterns
- **Low scores (<1.0)**: Stay durations typical for collective behaviors
- **Interpretation**: High duration anomaly suggests rushed activities (very short stays) or extended gatherings (very long stays)

#### **Co-occurrence Anomaly Score**
- **What it measures**: Rare spatial-temporal combinations
- **High scores (>0.4)**: Unique or very infrequent combinations of location and time
- **Low scores (<0.2)**: Common location-time patterns that occur frequently
- **Interpretation**: High co-occurrence anomaly indicates groups meeting in location-time combinations that are historically rare

#### **Absence Anomaly Score**
- **What it measures**: Reconstruction error indicating how well the behavior fits learned normal patterns
- **High scores (>1.5)**: Behaviors that the model struggles to reconstruct, indicating they don't match learned normal patterns
- **Low scores (<0.8)**: Behaviors that the model can easily reconstruct from learned patterns
- **Interpretation**: High absence anomaly suggests behaviors that fundamentally differ from the model's understanding of normal collective patterns

### Feature Attribution Analysis

For each detected anomaly, the system provides feature-level attribution scores indicating which aspects contributed most to the anomalous classification:

- **`time_window`**: Unusual timing patterns
- **`center_x`, `center_y`**: Unusual spatial locations
- **`spread_x`, `spread_y`**: Unusual spatial clustering patterns
- **`rel_size`**: Unusual group size relative to other behaviors
- **`weekend_ratio`**: Unusual weekend/weekday activity patterns
- **`avg_stay_duration`**: Unusual stay duration patterns

Higher attribution scores (>2.0) indicate features that strongly contributed to the anomaly classification.

## Architecture

### Data Processing Pipeline

1. **Raw Data Loading**: Processes CSV trajectory data
2. **Stay Point Conversion**: Transforms movements into stay events with durations  
3. **Spatial-Temporal Grouping**: Groups stay points by time windows
4. **Collective Behavior Detection**: DBSCAN clustering to find spatial groups
5. **Feature Extraction**: 8-dimensional collective behavior features

### Neural Network Architecture

```
Input (8 features) → 64 → 32 → 16 (embedding) → 32 → 64 → 8 (reconstruction)
```

Features:
- Normalized time window
- Spatial cluster center (x, y)
- Spatial spread (std_x, std_y)
- Relative group size
- Weekend activity ratio
- Average stay duration

### Anomaly Detection

Uses reconstruction-based approach:
1. Train autoencoder on normal collective behaviors
2. Compute reconstruction errors for test data
3. Apply dynamic threshold adjustment
4. Flag behaviors exceeding threshold as anomalous

## API Reference

### CoBAD Class

```python
class CoBAD:
    def __init__(self, spatial_dim=200, temporal_window=48, 
                 collective_threshold=5, target_anomaly_rate=0.05, 
                 threshold_method='percentile'):
        """Initialize CoBAD model"""
    
    def fit(self, trajectories):
        """Train the model on trajectory data"""
    
    def detect_anomalies(self, trajectories):
        """Detect anomalies in new trajectory data"""
        
    def visualize_results(self, results, save_path=None):
        """Generate visualization plots"""
        
    def save_model(self, path):
        """Save trained model"""
        
    def load_model(self, path):
        """Load trained model"""
```

### Data Utilities

```python
def load_data(file_path):
    """Load and preprocess trajectory data from CSV"""
    # Returns list of stay point trajectories
```

## Development

### Project Structure

```
cobad/
├── main.py              # Main CoBAD implementation
├── data_utils.py        # Data loading and preprocessing
├── data/                # Dataset directory
├── README.md           # This file
├── CLAUDE.md           # Project instructions for Claude Code
├── pyproject.toml      # Python dependencies
└── uv.lock            # Dependency lock file
```

### Adding Dependencies

```bash
uv add package_name
```

### Running with Different Python Versions

Ensure Python >= 3.13 is installed, then:
```bash
uv python install 3.13
uv sync
```

## Dataset Information

This project uses the **LYMob-4Cities** dataset:
- **4 metropolitan areas** in Japan (Cities A, B, C, D)
- **500m × 500m grid cells** in 200×200 grid
- **75-day period** with 30-minute intervals
- **100k, 25k, 20k, 6k individuals** per city respectively
- **85-dimensional POI features** per grid cell

Dataset available at: https://zenodo.org/records/14219563

## Contributing

This is a research implementation focused on collective behavior analysis in human mobility data. Contributions should maintain the defensive security focus and avoid any potentially malicious applications.

## License

This project is for research and educational purposes in mobility data analysis and anomaly detection.