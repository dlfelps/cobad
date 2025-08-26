import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
import networkx as nx
from collections import defaultdict
from tqdm import tqdm

from data_utils import load_data_lazy
from main import CoBAD

class AnomalyAnalyzer:
    """
    Comprehensive anomaly analysis with detailed interpretations,
    subscores, feature attribution, and temporal visualizations.
    """
    
    def __init__(self, model_path="cobad_model.pkl"):
        self.cobad = CoBAD()
        self.cobad.load_model(model_path)
        self.anomaly_data = None
        self.feature_names = [
            'time_window', 'center_x', 'center_y', 'spread_x', 'spread_y', 
            'rel_size', 'weekend_ratio', 'avg_stay_duration'
        ]
        
    def load_and_analyze_test_data(self, test_data_path):
        """Load test data and perform comprehensive anomaly analysis"""
        print("Loading test data...")
        raw_data = load_data_lazy(test_data_path)
        
        sample_size = min(1000000,len(raw_data))  # Use first 1000 trajectories
        raw_data_sample = raw_data[:sample_size]
        print(f"Using sample of {len(raw_data_sample)} trajectories for demonstration")
        
        # Split raw data first to avoid preprocessing training data
        split_idx = int(0.7 * len(raw_data_sample))
        test_raw_data = raw_data_sample[split_idx:]
        
        # Preprocess only test trajectories
        test_trajectories = self.cobad.preprocess_trajectories(test_raw_data)
        
        # Detect anomalies
        print("Detecting anomalies with detailed analysis...")
        results = self.cobad.detect_anomalies(test_trajectories)
        
        if len(results) == 0:
            print("No collective behaviors found for analysis.")
            return None
            
        # Store for detailed analysis
        self.anomaly_data = results
        self.anomaly_data['trajectories'] = test_trajectories
        
        return results
    
    def load_saved_results(self, results_dir="analysis_results"):
        """Load previously saved analysis results from disk"""
        results_path = Path(results_dir)
        if not results_path.exists():
            print(f"Results directory {results_path} does not exist.")
            return None
        
        print("Loading saved analysis results...")
        
        try:
            # Load main results
            features = np.load(results_path / "anomaly_features.npy")
            scores = np.load(results_path / "anomaly_scores.npy")
            anomalies = np.load(results_path / "anomaly_labels.npy", allow_pickle=True)
            
            # Reconstruct results dictionary
            results = {
                'features': features,
                'scores': scores,
                'anomalies': anomalies.astype(bool),  # Ensure boolean type
                'embeddings': np.random.randn(len(features), 16)  # Placeholder for embeddings
            }
            
            # Load subscores if available
            subscores = {}
            for subscore_file in results_path.glob("subscore_*.npy"):
                subscore_name = subscore_file.stem.replace("subscore_", "")
                subscores[subscore_name] = np.load(subscore_file)
            
            # Load feature attributions if available
            attribution_file = results_path / "feature_attributions.npy"
            if attribution_file.exists():
                attributions = np.load(attribution_file)
            else:
                attributions = None
            
            # Load link analysis if available
            link_file = results_path / "link_analysis.pkl"
            if link_file.exists():
                with open(link_file, 'rb') as f:
                    link_analysis = pickle.load(f)
            else:
                link_analysis = None
            
            # Store for analysis
            self.anomaly_data = results
            if subscores:
                self.anomaly_data['subscores'] = subscores
            if attributions is not None:
                self.anomaly_data['feature_attributions'] = attributions
            if link_analysis is not None:
                self.anomaly_data['link_analysis'] = link_analysis
            
            print(f"Loaded results: {len(anomalies)} behaviors, {np.sum(anomalies)} anomalies")
            return results
            
        except Exception as e:
            print(f"Error loading saved results: {e}")
            return None
    
    def compute_anomaly_subscores(self):
        """Compute detailed subscores for different anomaly types"""
        if self.anomaly_data is None:
            print("No anomaly data loaded. Run load_and_analyze_test_data first.")
            return None
            
        features = self.anomaly_data['features']
        embeddings = self.anomaly_data['embeddings']
        scores = self.anomaly_data['scores']
        anomalies = self.anomaly_data['anomalies']
        
        # Initialize subscores dictionary
        subscores = {
            'spatial_anomaly': np.zeros(len(features)),
            'temporal_anomaly': np.zeros(len(features)),
            'size_anomaly': np.zeros(len(features)),
            'duration_anomaly': np.zeros(len(features)),
            'co_occurrence_anomaly': np.zeros(len(features)),
            'absence_anomaly': np.zeros(len(features))
        }
        
        # 1. Spatial Anomaly Score
        spatial_features = features[:, 1:5]  # center_x, center_y, spread_x, spread_y
        spatial_mean = np.mean(spatial_features, axis=0)
        spatial_std = np.std(spatial_features, axis=0)
        spatial_z_scores = np.abs((spatial_features - spatial_mean) / (spatial_std + 1e-8))
        subscores['spatial_anomaly'] = np.mean(spatial_z_scores, axis=1)
        
        # 2. Temporal Anomaly Score
        temporal_features = features[:, 0]  # time_window
        temporal_hist, temporal_bins = np.histogram(temporal_features, bins=24)
        temporal_probs = temporal_hist / np.sum(temporal_hist)
        temporal_bin_indices = np.digitize(temporal_features, temporal_bins) - 1
        temporal_bin_indices = np.clip(temporal_bin_indices, 0, len(temporal_probs) - 1)
        subscores['temporal_anomaly'] = 1 - temporal_probs[temporal_bin_indices]
        
        # 3. Group Size Anomaly Score
        size_features = features[:, 5]  # rel_size
        size_z_scores = np.abs((size_features - np.mean(size_features)) / (np.std(size_features) + 1e-8))
        subscores['size_anomaly'] = size_z_scores
        
        # 4. Duration Anomaly Score
        duration_features = features[:, 7]  # avg_stay_duration
        duration_z_scores = np.abs((duration_features - np.mean(duration_features)) / (np.std(duration_features) + 1e-8))
        subscores['duration_anomaly'] = duration_z_scores
        
        # 5. Co-occurrence Anomaly (spatial-temporal combinations)
        spatial_temporal = features[:, [0, 1, 2]]  # time, center_x, center_y
        
        # Create spatial-temporal grid
        time_bins = np.linspace(0, 1, 24)
        space_bins = np.linspace(0, 1, 20)
        
        co_occurrence_scores = np.zeros(len(features))
        for i, (t, x, y) in enumerate(spatial_temporal):
            t_bin = np.digitize(t, time_bins) - 1
            x_bin = np.digitize(x, space_bins) - 1
            y_bin = np.digitize(y, space_bins) - 1
            
            # Count similar spatial-temporal patterns
            similar_mask = (
                (np.abs(spatial_temporal[:, 0] - t) < 0.05) &
                (np.abs(spatial_temporal[:, 1] - x) < 0.05) &
                (np.abs(spatial_temporal[:, 2] - y) < 0.05)
            )
            co_occurrence_count = np.sum(similar_mask)
            co_occurrence_scores[i] = 1 / (co_occurrence_count + 1)
        
        subscores['co_occurrence_anomaly'] = co_occurrence_scores
        
        # 6. Absence Anomaly (reconstruction error decomposition)
        X_tensor = torch.FloatTensor(self.cobad.scaler.transform(features))
        with torch.no_grad():
            embeddings_tensor = self.cobad.behavior_encoder(X_tensor)
            reconstructed = self.cobad.reconstruction_head(embeddings_tensor)
            feature_errors = torch.abs(reconstructed - X_tensor).numpy()
            
        subscores['absence_anomaly'] = np.mean(feature_errors, axis=1)
        
        self.anomaly_data['subscores'] = subscores
        return subscores
    
    def compute_feature_attribution(self):
        """Compute feature-level attribution for anomalies"""
        if self.anomaly_data is None:
            print("No anomaly data loaded.")
            return None
            
        features = self.anomaly_data['features']
        anomalies = self.anomaly_data['anomalies']
        
        # Compute feature importance using reconstruction errors
        X_tensor = torch.FloatTensor(self.cobad.scaler.transform(features))
        
        feature_attributions = []
        
        with torch.no_grad():
            embeddings = self.cobad.behavior_encoder(X_tensor)
            reconstructed = self.cobad.reconstruction_head(embeddings)
            
            # Per-feature reconstruction errors
            feature_errors = torch.abs(reconstructed - X_tensor).numpy()
            
            for i in range(len(features)):
                if anomalies[i]:
                    # Normalize errors for this anomaly
                    normalized_errors = feature_errors[i] / (np.std(feature_errors, axis=0) + 1e-8)
                    feature_attributions.append(normalized_errors)
                else:
                    feature_attributions.append(np.zeros(len(self.feature_names)))
        
        self.anomaly_data['feature_attributions'] = np.array(feature_attributions)
        return np.array(feature_attributions)
    
    def perform_link_analysis(self):
        """Perform link-based analysis to understand anomaly relationships"""
        if self.anomaly_data is None:
            print("No anomaly data loaded.")
            return None
            
        features = self.anomaly_data['features']
        anomalies = self.anomaly_data['anomalies']
        anomaly_indices = np.where(anomalies)[0]
        
        if len(anomaly_indices) == 0:
            print("No anomalies found for link analysis.")
            return None
        
        # Create network graph
        G = nx.Graph()
        
        # Add anomaly nodes
        for idx in anomaly_indices:
            G.add_node(idx, 
                      time=features[idx, 0],
                      center_x=features[idx, 1],
                      center_y=features[idx, 2],
                      rel_size=features[idx, 5])
        
        # Add edges between similar anomalies
        anomaly_features = features[anomaly_indices]
        
        # Compute pairwise distances
        distances = cdist(anomaly_features, anomaly_features)
        
        # Connect anomalies that are similar (within top 20% of distances)
        threshold = np.percentile(distances, 20)
        
        for i, idx_i in enumerate(anomaly_indices):
            for j, idx_j in enumerate(anomaly_indices):
                if i < j and distances[i, j] < threshold:
                    G.add_edge(idx_i, idx_j, weight=1/distances[i, j])
        
        # Compute network metrics
        centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        
        # Find communities
        communities = list(nx.connected_components(G))
        
        link_analysis = {
            'graph': G,
            'centrality': centrality,
            'betweenness': betweenness,
            'clustering': clustering,
            'communities': communities,
            'anomaly_indices': anomaly_indices
        }
        
        self.anomaly_data['link_analysis'] = link_analysis
        return link_analysis
    
    def create_temporal_visualization(self):
        """Create comprehensive temporal visualizations of anomalies"""
        if self.anomaly_data is None:
            print("No anomaly data loaded.")
            return None
            
        features = self.anomaly_data['features']
        anomalies = self.anomaly_data['anomalies']
        scores = self.anomaly_data['scores']
        subscores = self.anomaly_data.get('subscores', {})
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Temporal Distribution of Anomalies',
                'Spatial Distribution of Anomalies', 
                'Anomaly Scores Over Time',
                'Subscore Analysis',
                'Anomaly Size vs Time',
                'Weekend vs Weekday Patterns'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Temporal histogram
        normal_times = features[~anomalies, 0] * 48 / 2  # Convert to hours (0-24)
        anomaly_times = features[anomalies, 0] * 48 / 2
        
        fig.add_trace(
            go.Histogram(x=normal_times, name='Normal', opacity=0.7, nbinsx=24),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=anomaly_times, name='Anomaly', opacity=0.7, nbinsx=24),
            row=1, col=1
        )
        
        # 2. Spatial scatter
        fig.add_trace(
            go.Scatter(
                x=features[~anomalies, 1], y=features[~anomalies, 2],
                mode='markers', name='Normal', opacity=0.6,
                marker=dict(size=4, color='blue')
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=features[anomalies, 1], y=features[anomalies, 2],
                mode='markers', name='Anomaly', opacity=0.8,
                marker=dict(size=8, color='red', symbol='x')
            ),
            row=1, col=2
        )
        
        # 3. Scores over time
        time_sorted_indices = np.argsort(features[:, 0])
        sorted_times = features[time_sorted_indices, 0] * 48 / 2  # Convert to hours (0-24)
        sorted_scores = scores[time_sorted_indices]
        sorted_anomalies = anomalies[time_sorted_indices]
        
        fig.add_trace(
            go.Scatter(
                x=sorted_times, y=sorted_scores,
                mode='lines+markers', name='Anomaly Score',
                marker=dict(
                    color=sorted_scores, 
                    colorscale='Viridis', 
                    size=6,
                    showscale=True,
                    colorbar=dict(
                        title="Anomaly Score",
                        x=0.48,  # Position between subplot columns
                        xanchor="left",
                        y=0.65,  # Position at row 2
                        yanchor="middle",
                        len=0.25,  # Height of colorbar
                        thickness=15
                    )
                ),
                line=dict(color='gray', width=1)
            ),
            row=2, col=1
        )
        
        # Add threshold line
        fig.add_hline(
            y=self.cobad.anomaly_threshold, 
            line_dash="dash", line_color="red",
            row=2, col=1
        )
        
        # 4. Subscore analysis (if available)
        if subscores:
            subscore_names = list(subscores.keys())
            anomaly_subscores = {name: scores[anomalies] for name, scores in subscores.items()}
            
            subscore_df = pd.DataFrame(anomaly_subscores)
            if len(subscore_df) > 0:
                fig.add_trace(
                    go.Box(y=subscore_df['spatial_anomaly'], name='Spatial'),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Box(y=subscore_df['temporal_anomaly'], name='Temporal'),
                    row=2, col=2
                )
        
        # 5. Size vs time for anomalies
        if np.any(anomalies):
            anomaly_scores = scores[anomalies]
            # Normalize marker sizes to reasonable range (5-20)
            normalized_sizes = 5 + (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8) * 15
            
            # Note: features[anomalies, 5] contains SCALED relative sizes (standardized)
            # For better interpretation, we show these as "Scaled Group Size" 
            fig.add_trace(
                go.Scatter(
                    x=features[anomalies, 0] * 48 / 2,  # Convert to hours (0-24)
                    y=features[anomalies, 5],  # Scaled relative size (standardized)
                    mode='markers', name='Anomaly Size vs Time',
                    marker=dict(
                        size=normalized_sizes,
                        color=anomaly_scores,
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(
                            title="Score",
                            x=0.48,  # Position between subplot columns
                            xanchor="left", 
                            y=0.15,  # Position at row 3 (bottom)
                            yanchor="middle",
                            len=0.25,  # Height of colorbar
                            thickness=15
                        ),
                        line=dict(width=1, color='darkred')
                    ),
                    text=[f'Score: {score:.3f}' for score in anomaly_scores],
                    hovertemplate='<b>Time:</b> %{x:.1f}h<br><b>Group Size:</b> %{y:.3f}<br>%{text}<extra></extra>'
                ),
                row=3, col=1
            )
        
        # 6. Weekend patterns
        weekend_normal = features[(~anomalies) & (features[:, 6] > 0.5), 0] * 48 / 2  # Convert to hours (0-24)
        weekend_anomaly = features[anomalies & (features[:, 6] > 0.5), 0] * 48 / 2
        weekday_normal = features[(~anomalies) & (features[:, 6] <= 0.5), 0] * 48 / 2
        weekday_anomaly = features[anomalies & (features[:, 6] <= 0.5), 0] * 48 / 2
        
        for times, name, color in [
            (weekend_normal, 'Weekend Normal', 'lightblue'),
            (weekend_anomaly, 'Weekend Anomaly', 'red'),
            (weekday_normal, 'Weekday Normal', 'lightgreen'),
            (weekday_anomaly, 'Weekday Anomaly', 'darkred')
        ]:
            if len(times) > 0:
                fig.add_trace(
                    go.Histogram(x=times, name=name, opacity=0.6, nbinsx=24),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            title="Comprehensive Anomaly Analysis Dashboard",
            height=1200,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Spatial X", row=1, col=2)
        fig.update_yaxes(title_text="Spatial Y", row=1, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
        fig.update_yaxes(title_text="Subscore Value", row=2, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
        fig.update_yaxes(title_text="Scaled Group Size (standardized)", row=3, col=1)
        fig.update_xaxes(title_text="Hour of Day", row=3, col=2)
        fig.update_yaxes(title_text="Count", row=3, col=2)
        
        fig.write_html("anomaly_temporal_analysis.html")
        print("Interactive visualization saved to anomaly_temporal_analysis.html")
        
        return fig
    
    def _explain_anomaly_in_plain_english(self, idx, features, scores, subscores, attributions):
        """Generate plain English explanation for a specific anomaly"""
        score = scores[idx]
        feature_values = features[idx]
        
        # Convert time to human readable (30-minute increments)
        time_slot = int(feature_values[0] * 48)
        hour = time_slot // 2
        minute = (time_slot % 2) * 30
        time_str = f"{hour:02d}:{minute:02d}"
        
        # Get location
        location = (feature_values[1], feature_values[2])
        
        # Get group characteristics
        weekend_ratio = feature_values[6]
        duration = feature_values[7]
        
        # Estimated duration in minutes (rough conversion)
        duration_minutes = int(duration * 60 * 30)  # Rough estimate
        
        # Analyze weekend pattern
        if weekend_ratio > 0.7:
            weekend_pattern = "primarily weekend activity (75%+ weekend)"
        elif weekend_ratio > 0.4:
            weekend_pattern = "mixed weekend/weekday activity"
        elif weekend_ratio > 0.1:
            weekend_pattern = "primarily weekday activity"
        else:
            weekend_pattern = "exclusively weekday activity"
        
        # Get top contributing factors
        explanation_parts = []
        
        if attributions is not None:
            feature_attr = attributions[idx]
            top_feature_idx = np.argmax(feature_attr)
            top_feature_name = self.feature_names[top_feature_idx]
            top_contrib_score = feature_attr[top_feature_idx]
            
            # Build explanation based on top contributing feature
            if top_feature_name == 'weekend_ratio' and top_contrib_score > 2.0:
                if weekend_ratio > 0.6:
                    explanation_parts.append(f"Has an unusually strong weekend preference ({weekend_ratio:.0%} weekend activity)")
                else:
                    explanation_parts.append(f"Has an unusual weekday concentration pattern")
            
            if top_feature_name in ['time_window'] and top_contrib_score > 2.0:
                if 6 <= hour <= 9:
                    explanation_parts.append(f"Occurs during morning hours ({time_str}) when collective activities are less common")
                elif 22 <= hour or hour <= 5:
                    explanation_parts.append(f"Occurs during late night/early morning ({time_str}) when few collective behaviors happen")
                else:
                    explanation_parts.append(f"Occurs at {time_str}, an unusual time for this type of collective behavior pattern")
            
            if top_feature_name in ['center_x', 'center_y'] and top_contrib_score > 2.0:
                explanation_parts.append(f"Located at an uncommon gathering spot ({location[0]:.3f}, {location[1]:.3f})")
            
            if top_feature_name == 'avg_stay_duration' and top_contrib_score > 2.0:
                if duration < 0.1:
                    explanation_parts.append(f"Involves unusually brief gatherings (~{duration_minutes} minutes)")
                else:
                    explanation_parts.append(f"Involves unusually long gatherings (~{duration_minutes} minutes)")
        
        # Add subscore insights
        if subscores:
            high_subscores = []
            if subscores.get('temporal_anomaly', [0])[idx] > 0.8:
                high_subscores.append("temporal")
            if subscores.get('spatial_anomaly', [0])[idx] > 0.8:
                high_subscores.append("spatial")
            if subscores.get('size_anomaly', [0])[idx] > 1.0:
                high_subscores.append("group size")
            if subscores.get('duration_anomaly', [0])[idx] > 1.0:
                high_subscores.append("duration")
            if subscores.get('co_occurrence_anomaly', [0])[idx] > 0.4:
                high_subscores.append("rare time-location combination")
        
        # Construct final explanation
        if not explanation_parts:
            explanation_parts.append("Shows multiple unusual patterns that don't match typical collective behaviors")
        
        # Create comprehensive explanation
        explanation = f"""
    PLAIN ENGLISH EXPLANATION:
    This represents a small group that {', '.join(explanation_parts).lower()}. 
    The group meets at {time_str} with {weekend_pattern}, creating a pattern that 
    significantly differs from normal urban collective behavior patterns."""
        
        # Add likely scenarios
        scenarios = []
        if weekend_ratio > 0.7 and 14 <= hour <= 18:  # Weekend afternoons
            scenarios.append("weekend recreational activity or hobby group")
        elif weekend_ratio < 0.2 and 7 <= hour <= 17:  # Weekday business hours
            scenarios.append("work-related gathering or service activity")
        elif 22 <= hour or hour <= 6:  # Night/early morning
            scenarios.append("late-night service, security, or emergency response activity")
        elif duration < 0.1:  # Very brief
            scenarios.append("brief coordination meeting or handoff activity")
        
        if scenarios:
            explanation += f"\n    Likely represents: {' or '.join(scenarios)}."
        
        return explanation
    
    def generate_anomaly_report(self, log_file="anomaly_report.txt"):
        """Generate comprehensive anomaly interpretation report"""
        if self.anomaly_data is None:
            print("No anomaly data loaded.")
            return None
            
        features = self.anomaly_data['features']
        anomalies = self.anomaly_data['anomalies']
        scores = self.anomaly_data['scores']
        subscores = self.anomaly_data.get('subscores', {})
        attributions = self.anomaly_data.get('feature_attributions')
        link_analysis = self.anomaly_data.get('link_analysis')
        
        anomaly_indices = np.where(anomalies)[0]
        
        # Helper function to write to both console and file
        def log_print(message, file_handle=None):
            print(message)
            if file_handle:
                file_handle.write(message + "\n")
        
        # Open log file for writing
        with open(log_file, 'w', encoding='utf-8') as f:
            log_print("="*80, f)
            log_print("COMPREHENSIVE ANOMALY ANALYSIS REPORT", f)
            log_print("="*80, f)
            
            log_print(f"\nOVERVIEW:", f)
            log_print(f"Total collective behaviors: {len(anomalies)}", f)
            log_print(f"Anomalies detected: {len(anomaly_indices)}", f)
            log_print(f"Anomaly rate: {len(anomaly_indices)/len(anomalies)*100:.2f}%", f)
            log_print(f"Anomaly threshold: {self.cobad.anomaly_threshold:.4f}", f)
            
            if len(anomaly_indices) == 0:
                log_print("No anomalies detected for detailed analysis.", f)
                return
            
            log_print(f"\nSCORE STATISTICS:", f)
            log_print(f"Normal behavior scores: {np.mean(scores[~anomalies]):.4f} ± {np.std(scores[~anomalies]):.4f}", f)
            log_print(f"Anomalous behavior scores: {np.mean(scores[anomalies]):.4f} ± {np.std(scores[anomalies]):.4f}", f)
            log_print(f"Score range: [{np.min(scores):.4f}, {np.max(scores):.4f}]", f)
            
            # Subscore analysis
            if subscores:
                log_print(f"\nSUBSCORE ANALYSIS:", f)
                for subscore_name, subscore_values in subscores.items():
                    anomaly_subscore = subscore_values[anomalies]
                    if len(anomaly_subscore) > 0:
                        log_print(f"{subscore_name:20s}: {np.mean(anomaly_subscore):.4f} ± {np.std(anomaly_subscore):.4f}", f)
            
            # Feature attribution analysis
            if attributions is not None:
                log_print(f"\nFEATURE ATTRIBUTION ANALYSIS:", f)
                anomaly_attributions = attributions[anomalies]
                if len(anomaly_attributions) > 0:
                    mean_attributions = np.mean(anomaly_attributions, axis=0)
                    for i, (feature_name, attribution) in enumerate(zip(self.feature_names, mean_attributions)):
                        if attribution > 0.1:  # Only show significant attributions
                            log_print(f"{feature_name:20s}: {attribution:.4f}", f)
            
            # Link analysis results
            if link_analysis:
                log_print(f"\nLINK ANALYSIS:", f)
                G = link_analysis['graph']
                communities = link_analysis['communities']
                
                log_print(f"Connected anomalies: {G.number_of_nodes()}", f)
                log_print(f"Anomaly connections: {G.number_of_edges()}", f)
                log_print(f"Anomaly communities: {len(communities)}", f)
                
                if len(communities) > 0:
                    largest_community = max(communities, key=len)
                    log_print(f"Largest community size: {len(largest_community)}", f)
            
            # Top anomalies detailed analysis  
            num_top_anomalies = min(10, len(anomaly_indices))
            log_print(f"\nTOP {num_top_anomalies} ANOMALIES DETAILED ANALYSIS:", f)
            # Sort all anomaly indices by their scores (descending order)
            sorted_anomaly_order = np.argsort(scores[anomaly_indices])[::-1]  # Descending order
            top_anomaly_indices = anomaly_indices[sorted_anomaly_order[:num_top_anomalies]]
            
            for rank, idx in enumerate(top_anomaly_indices, 1):
                log_print(f"\nRank {rank} - Anomaly #{idx}:", f)
                log_print(f"  Score: {scores[idx]:.4f}", f)
                log_print(f"  Time: {features[idx, 0]*48:.1f}h", f)
                log_print(f"  Location: ({features[idx, 1]:.3f}, {features[idx, 2]:.3f})", f)
                log_print(f"  Group size: {features[idx, 5]:.3f}", f)
                log_print(f"  Weekend ratio: {features[idx, 6]:.3f}", f)
                log_print(f"  Avg duration: {features[idx, 7]:.3f}", f)
                
                # Subscore breakdown
                if subscores:
                    log_print("  Subscores:", f)
                    for name, values in subscores.items():
                        log_print(f"    {name}: {values[idx]:.4f}", f)
                
                # Feature attribution
                if attributions is not None:
                    log_print("  Top contributing features:", f)
                    feature_attr = attributions[idx]
                    top_features = np.argsort(feature_attr)[-3:]
                    for feat_idx in reversed(top_features):
                        if feature_attr[feat_idx] > 0.1:
                            log_print(f"    {self.feature_names[feat_idx]}: {feature_attr[feat_idx]:.4f}", f)
                
                # Plain English explanation
                explanation = self._explain_anomaly_in_plain_english(idx, features, scores, subscores, attributions)
                log_print(explanation, f)
            
            log_print("\n" + "="*80, f)
            
            print(f"\nReport written to: {log_file}")
        return {
            'summary_stats': {
                'total_behaviors': len(anomalies),
                'anomaly_count': len(anomaly_indices),
                'anomaly_rate': len(anomaly_indices)/len(anomalies),
                'threshold': self.cobad.anomaly_threshold
            },
            'top_anomalies': top_anomaly_indices,
            'subscores': subscores,
            'attributions': attributions,
            'link_analysis': link_analysis
        }
    
    def save_analysis_results(self, output_dir="analysis_results"):
        """Save all analysis results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if self.anomaly_data is None:
            print("No analysis results to save.")
            return
        
        # Save main results
        np.save(output_path / "anomaly_features.npy", self.anomaly_data['features'])
        np.save(output_path / "anomaly_scores.npy", self.anomaly_data['scores'])
        np.save(output_path / "anomaly_labels.npy", self.anomaly_data['anomalies'])
        
        # Save subscores
        if 'subscores' in self.anomaly_data:
            for name, scores in self.anomaly_data['subscores'].items():
                np.save(output_path / f"subscore_{name}.npy", scores)
        
        # Save feature attributions
        if 'feature_attributions' in self.anomaly_data:
            np.save(output_path / "feature_attributions.npy", self.anomaly_data['feature_attributions'])
        
        # Save link analysis
        if 'link_analysis' in self.anomaly_data:
            with open(output_path / "link_analysis.pkl", 'wb') as f:
                pickle.dump(self.anomaly_data['link_analysis'], f)
        
        print(f"Analysis results saved to {output_path}")

def main(use_saved_results=False, results_dir="analysis_results"):
    """Main function to run comprehensive anomaly analysis
    
    Args:
        use_saved_results (bool): If True, load saved results instead of re-analyzing
        results_dir (str): Directory containing saved results
    """
    
    # Initialize analyzer
    print("Initializing Anomaly Analyzer...")
    analyzer = AnomalyAnalyzer("cobad_model.pkl")
    
    if use_saved_results:
        # Load previously saved results
        print("Using saved results mode...")
        results = analyzer.load_saved_results(results_dir)
        
        if results is None:
            print("Could not load saved results. Falling back to full analysis...")
            use_saved_results = False
    
    if not use_saved_results:
        # Load and analyze test data
        test_data_path = Path("data/cityA-dataset.csv")
        results = analyzer.load_and_analyze_test_data(test_data_path)
        
        if results is None:
            print("No results to analyze.")
            return
        
        # Compute detailed subscores
        print("Computing anomaly subscores...")
        subscores = analyzer.compute_anomaly_subscores()
        
        # Compute feature attributions
        print("Computing feature attributions...")
        attributions = analyzer.compute_feature_attribution()
        
        # Perform link analysis
        print("Performing link-based analysis...")
        link_analysis = analyzer.perform_link_analysis()
        
        # Save all results
        print("Saving analysis results...")
        analyzer.save_analysis_results()
    
    print(f"\nBasic Results:")
    print(f"Total collective behaviors: {len(results['anomalies'])}")
    print(f"Anomalies detected: {np.sum(results['anomalies'])}")
    print(f"Anomaly rate: {np.mean(results['anomalies'])*100:.2f}%")
    
    # Create temporal visualizations (always regenerate for potential changes)
    print("Creating temporal visualizations...")
    fig = analyzer.create_temporal_visualization()
    
    # Generate comprehensive report (always regenerate for potential changes)
    print("Generating comprehensive anomaly report...")
    report = analyzer.generate_anomaly_report()
    
    print("\nAnalysis complete! Check the following outputs:")
    print("- anomaly_temporal_analysis.html: Interactive visualization")
    print("- analysis_results/: Saved analysis data")
    print("- anomaly_report.txt: Detailed text report")
    print("- Console report above with detailed interpretations")

if __name__ == "__main__":
    import sys
    
    # Check for command line argument
    use_saved = "--use-saved" in sys.argv or "-s" in sys.argv
    
    if use_saved:
        print("Running with saved results...")
        main(use_saved_results=True)
    else:
        print("Running full analysis...")
        main(use_saved_results=False)