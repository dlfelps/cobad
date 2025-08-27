import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import dash
from dash import dcc, html, Input, Output, callback
import json

from cobad_paper_accurate import CoBAD, preprocess_trajectories_for_cobad
from data_utils import load_data_lazy

class TrajectoryAnomalyAnalyzer:
    """
    Comprehensive trajectory anomaly analysis dashboard for paper-accurate CoBAD model.
    Analyzes individual trajectory anomalies with detailed sub-score breakdowns.
    """
    
    def __init__(self, model_path="cobad_paper_accurate.pth"):
        """Initialize analyzer with trained CoBAD model"""
        self.model_path = model_path
        self.cobad = None
        self.trajectories = None
        self.results = None
        self.feature_names = ['norm_x', 'norm_y', 'is_weekend', 'start_time', 'duration', 'placeholder']
        
    def load_model_and_data(self, data_path, sample_size=10000, max_traj_length=20):
        """Load trained model and analyze trajectory data"""
        print("Loading CoBAD model...")
        
        # Initialize model with same config as training
        self.cobad = CoBAD(
            d_model=128,
            nhead=8,
            num_layers=2,
            dropout=0.1,
            mask_ratio=0.15,
            spatial_dim=200,
            temporal_window=48,
            collective_threshold=5,
            target_anomaly_rate=0.05,
            batch_size=32
        )
        
        if Path(self.model_path).exists():
            self.cobad.load_model(self.model_path)
            print(f"Loaded model from {self.model_path}")
        else:
            print(f"Model file not found: {self.model_path}")
            return False
        
        print("Loading trajectory data...")
        raw_data = load_data_lazy(data_path)
        raw_data_sample = raw_data[:sample_size]
        
        # Preprocess trajectories
        trajectories = preprocess_trajectories_for_cobad(raw_data_sample)
        trajectories = [traj[:max_traj_length] for traj in trajectories if len(traj) > 0]
        
        print(f"Analyzing {len(trajectories)} trajectories...")
        
        # Run anomaly detection
        self.results = self.cobad.detect_anomalies(trajectories)
        self.trajectories = trajectories
        
        print(f"Analysis complete: {np.sum(self.results['anomalies'])} anomalies detected "
              f"({np.mean(self.results['anomalies'])*100:.1f}% anomaly rate)")
        
        return True
    
    def compute_trajectory_subscores(self):
        """Compute detailed sub-scores for trajectory anomalies"""
        if self.results is None:
            print("No results available. Run load_model_and_data first.")
            return None
        
        print("Computing trajectory sub-scores...")
        
        # Extract trajectory features for analysis
        trajectory_features = []
        trajectory_lengths = []
        
        for traj in self.trajectories:
            if len(traj) > 0:
                # Aggregate trajectory-level features
                traj_array = np.array(traj)
                
                # Spatial features
                x_mean, x_std = np.mean(traj_array[:, 0]), np.std(traj_array[:, 0])
                y_mean, y_std = np.mean(traj_array[:, 1]), np.std(traj_array[:, 1])
                
                # Temporal features
                weekend_ratio = np.mean(traj_array[:, 2])
                start_time_mean = np.mean(traj_array[:, 3])
                duration_mean = np.mean(traj_array[:, 4])
                duration_total = np.sum(traj_array[:, 4])
                
                # Movement features
                if len(traj) > 1:
                    distances = np.sqrt(np.diff(traj_array[:, 0])**2 + np.diff(traj_array[:, 1])**2)
                    mobility = np.mean(distances)
                    max_distance = np.max(distances) if len(distances) > 0 else 0
                else:
                    mobility = 0
                    max_distance = 0
                
                features = [
                    x_mean, x_std, y_mean, y_std,  # Spatial
                    weekend_ratio, start_time_mean, duration_mean, duration_total,  # Temporal
                    mobility, max_distance, len(traj)  # Movement & length
                ]
                trajectory_features.append(features)
                trajectory_lengths.append(len(traj))
        
        trajectory_features = np.array(trajectory_features)
        
        # Compute sub-scores
        subscores = {}
        
        # 1. Spatial Anomaly Score (location variance)
        spatial_variance = trajectory_features[:, 1] + trajectory_features[:, 3]  # x_std + y_std
        subscores['spatial_dispersion'] = spatial_variance
        
        # 2. Temporal Anomaly Score (unusual timing patterns)
        weekend_scores = np.abs(trajectory_features[:, 4] - 0.5)  # Distance from 50% weekend
        time_scores = np.abs(trajectory_features[:, 5] - 0.5)  # Distance from midday
        subscores['temporal_pattern'] = weekend_scores + time_scores
        
        # 3. Duration Anomaly Score
        duration_z = np.abs(stats.zscore(trajectory_features[:, 6]))
        subscores['duration_anomaly'] = duration_z
        
        # 4. Mobility Anomaly Score
        mobility_z = np.abs(stats.zscore(trajectory_features[:, 8]))
        subscores['mobility_anomaly'] = mobility_z
        
        # 5. Length Anomaly Score
        length_z = np.abs(stats.zscore(trajectory_features[:, 10]))
        subscores['length_anomaly'] = length_z
        
        # 6. Reconstruction Error Components (from CoBAD results)
        subscores['event_reconstruction'] = self.results['event_recon_errors']
        subscores['link_reconstruction'] = self.results['link_recon_errors']
        subscores['pattern_learned'] = self.results['pattern_scores']
        
        self.trajectory_features = trajectory_features
        self.subscores = subscores
        
        return subscores
    
    def create_overview_dashboard(self):
        """Create comprehensive overview dashboard"""
        if self.results is None:
            print("No results available.")
            return None
        
        # Compute subscores if not done already
        if not hasattr(self, 'subscores'):
            self.compute_trajectory_subscores()
        
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Anomaly Score Distribution',
                'Sub-score Correlations', 
                'Temporal Patterns',
                'Spatial Distribution',
                'Trajectory Length vs Anomaly Score',
                'Sub-score Radar Chart (Top Anomalies)'
            ],
            specs=[
                [{"type": "histogram"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatterpolar"}]
            ]
        )
        
        # 1. Score Distribution
        fig.add_trace(
            go.Histogram(
                x=self.results['scores'],
                nbinsx=50,
                name='Normal',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        anomaly_scores = self.results['scores'][self.results['anomalies']]
        if len(anomaly_scores) > 0:
            fig.add_trace(
                go.Histogram(
                    x=anomaly_scores,
                    nbinsx=50,
                    name='Anomalies',
                    marker_color='red',
                    opacity=0.7
                ),
                row=1, col=1
            )
        
        # 2. Sub-score Correlation Heatmap
        subscore_matrix = np.array([scores for scores in self.subscores.values()]).T
        subscore_corr = np.corrcoef(subscore_matrix.T)
        
        fig.add_trace(
            go.Heatmap(
                z=subscore_corr,
                x=list(self.subscores.keys()),
                y=list(self.subscores.keys()),
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=2
        )
        
        # 3. Temporal Patterns
        if hasattr(self, 'trajectory_features'):
            fig.add_trace(
                go.Scatter(
                    x=self.trajectory_features[:, 5],  # start_time_mean
                    y=self.results['scores'],
                    mode='markers',
                    marker=dict(
                        color=self.results['anomalies'].astype(int),
                        colorscale=[[0, 'lightblue'], [1, 'red']],
                        size=8
                    ),
                    name='Trajectories'
                ),
                row=2, col=1
            )
        
        # 4. Spatial Distribution
        if hasattr(self, 'trajectory_features'):
            fig.add_trace(
                go.Scatter(
                    x=self.trajectory_features[:, 0],  # x_mean
                    y=self.trajectory_features[:, 2],  # y_mean
                    mode='markers',
                    marker=dict(
                        color=self.results['scores'],
                        colorscale='Viridis',
                        size=8,
                        colorbar=dict(
                            title="Anomaly Score",
                            x=0.48,  # Position between subplots
                            len=0.4,  # Shorter colorbar
                            y=0.7    # Upper position
                        )
                    ),
                    name='Trajectory Centers'
                ),
                row=2, col=2
            )
        
        # 5. Length vs Score
        if hasattr(self, 'trajectory_features'):
            fig.add_trace(
                go.Scatter(
                    x=self.trajectory_features[:, 10],  # trajectory length
                    y=self.results['scores'],
                    mode='markers',
                    marker=dict(
                        color=self.results['anomalies'].astype(int),
                        colorscale=[[0, 'lightblue'], [1, 'red']],
                        size=8
                    ),
                    name='Length vs Score'
                ),
                row=3, col=1
            )
        
        # 6. Radar Chart for Top Anomalies
        if len(anomaly_scores) > 0:
            # Get top 5 anomalies
            top_anomaly_indices = np.argsort(self.results['scores'])[-5:]
            
            # Normalize subscores for radar chart
            subscore_names = list(self.subscores.keys())
            for i, idx in enumerate(top_anomaly_indices):
                subscore_values = []
                for name in subscore_names:
                    value = self.subscores[name][idx]
                    # Normalize to 0-1 range
                    value_norm = (value - np.min(self.subscores[name])) / (np.max(self.subscores[name]) - np.min(self.subscores[name]) + 1e-8)
                    subscore_values.append(value_norm)
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=subscore_values + [subscore_values[0]],  # Close the polygon
                        theta=subscore_names + [subscore_names[0]],
                        fill='toself',
                        name=f'Anomaly #{idx}',
                        opacity=0.6
                    ),
                    row=3, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="CoBAD Trajectory Anomaly Analysis Dashboard",
            showlegend=True,
            legend=dict(
                x=1.05,  # Position legend to the right
                y=1.0,   # Top position
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Start Time (normalized)", row=2, col=1)
        fig.update_yaxes(title_text="Anomaly Score", row=2, col=1)
        
        fig.update_xaxes(title_text="X Coordinate", row=2, col=2)
        fig.update_yaxes(title_text="Y Coordinate", row=2, col=2)
        
        fig.update_xaxes(title_text="Trajectory Length", row=3, col=1)
        fig.update_yaxes(title_text="Anomaly Score", row=3, col=1)
        
        return fig
    
    def create_detailed_anomaly_report(self, top_n=10):
        """Create detailed report of top anomalies"""
        if self.results is None:
            print("No results available.")
            return None
        
        if not hasattr(self, 'subscores'):
            self.compute_trajectory_subscores()
        
        # Get top anomalies
        anomaly_indices = np.where(self.results['anomalies'])[0]
        if len(anomaly_indices) == 0:
            print("No anomalies detected.")
            return None
        
        top_indices = anomaly_indices[np.argsort(self.results['scores'][anomaly_indices])[-top_n:]][::-1]
        
        print(f"\n{'='*80}")
        print(f"DETAILED TRAJECTORY ANOMALY REPORT - TOP {len(top_indices)} ANOMALIES")
        print(f"{'='*80}")
        
        for rank, idx in enumerate(top_indices, 1):
            print(f"\n🚨 RANK {rank} - TRAJECTORY #{idx}")
            print(f"{'='*50}")
            
            # Basic info
            traj = self.trajectories[idx]
            score = self.results['scores'][idx]
            
            print(f"📊 Anomaly Score: {score:.4f}")
            print(f"📏 Trajectory Length: {len(traj)} points")
            
            # Trajectory summary
            if len(traj) > 0:
                traj_array = np.array(traj)
                x_range = f"[{np.min(traj_array[:, 0]):.3f}, {np.max(traj_array[:, 0]):.3f}]"
                y_range = f"[{np.min(traj_array[:, 1]):.3f}, {np.max(traj_array[:, 1]):.3f}]"
                weekend_pct = np.mean(traj_array[:, 2]) * 100
                avg_duration = np.mean(traj_array[:, 4])
                
                print(f"🗺️  Spatial Range: X{x_range}, Y{y_range}")
                print(f"📅 Weekend Activity: {weekend_pct:.1f}%")
                print(f"⏱️  Average Duration: {avg_duration:.2f}")
            
            # Sub-scores
            print(f"\n🔍 Sub-score Breakdown:")
            for name, scores in self.subscores.items():
                percentile = stats.percentileofscore(scores, scores[idx])
                print(f"   {name:20s}: {scores[idx]:7.4f} ({percentile:5.1f}th percentile)")
            
            # Trajectory pattern description
            print(f"\n📝 Pattern Analysis:")
            self._describe_trajectory_pattern(idx, traj_array if len(traj) > 0 else None)
        
        return top_indices
    
    def _describe_trajectory_pattern(self, idx, traj_array):
        """Generate human-readable description of trajectory pattern"""
        if traj_array is None:
            print("   Empty trajectory")
            return
        
        descriptions = []
        
        # Spatial pattern
        x_std, y_std = np.std(traj_array[:, 0]), np.std(traj_array[:, 1])
        if x_std > 0.1 or y_std > 0.1:
            descriptions.append(f"High spatial dispersion (σx={x_std:.3f}, σy={y_std:.3f})")
        elif x_std < 0.01 and y_std < 0.01:
            descriptions.append("Very localized activity (low spatial dispersion)")
        
        # Temporal pattern
        weekend_ratio = np.mean(traj_array[:, 2])
        if weekend_ratio > 0.8:
            descriptions.append("Predominantly weekend activity")
        elif weekend_ratio < 0.2:
            descriptions.append("Predominantly weekday activity")
        
        # Duration pattern
        durations = traj_array[:, 4]
        if np.std(durations) > np.mean(durations):
            descriptions.append("Highly variable stay durations")
        elif np.mean(durations) > 0.8:
            descriptions.append("Long stay durations")
        elif np.mean(durations) < 0.2:
            descriptions.append("Short stay durations")
        
        # Movement pattern
        if len(traj_array) > 1:
            distances = np.sqrt(np.diff(traj_array[:, 0])**2 + np.diff(traj_array[:, 1])**2)
            if np.mean(distances) > 0.1:
                descriptions.append("High mobility (large movements)")
            elif np.max(distances) < 0.01:
                descriptions.append("Very static behavior")
        
        # Sub-score insights
        subscore_insights = []
        if hasattr(self, 'subscores'):
            for name, scores in self.subscores.items():
                percentile = stats.percentileofscore(scores, scores[idx])
                if percentile > 95:
                    subscore_insights.append(f"Extreme {name.replace('_', ' ')}")
                elif percentile > 90:
                    subscore_insights.append(f"High {name.replace('_', ' ')}")
        
        # Output descriptions
        if descriptions:
            for desc in descriptions:
                print(f"   • {desc}")
        
        if subscore_insights:
            print(f"   🎯 Key anomaly drivers: {', '.join(subscore_insights)}")
        
        if not descriptions and not subscore_insights:
            print("   • Pattern within normal range across measured dimensions")
    
    def save_results(self, output_dir="trajectory_analysis_results"):
        """Save analysis results for later use"""
        if self.results is None:
            print("No results to save.")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results
        np.save(output_path / "anomaly_scores.npy", self.results['scores'])
        np.save(output_path / "anomaly_labels.npy", self.results['anomalies'])
        np.save(output_path / "pattern_scores.npy", self.results['pattern_scores'])
        np.save(output_path / "event_recon_errors.npy", self.results['event_recon_errors'])
        np.save(output_path / "link_recon_errors.npy", self.results['link_recon_errors'])
        np.save(output_path / "embeddings.npy", self.results['embeddings'])
        
        # Save subscores if computed
        if hasattr(self, 'subscores'):
            for name, scores in self.subscores.items():
                np.save(output_path / f"subscore_{name}.npy", scores)
        
        # Save trajectory features if computed
        if hasattr(self, 'trajectory_features'):
            np.save(output_path / "trajectory_features.npy", self.trajectory_features)
        
        print(f"Results saved to {output_path}")
    
    def create_embedding_analysis(self):
        """Create embedding space analysis with clustering and dimensionality reduction"""
        if self.results is None or 'embeddings' not in self.results:
            print("No embeddings available.")
            return None
        
        embeddings = self.results['embeddings']
        if len(embeddings) == 0:
            print("Empty embeddings.")
            return None
        
        print("Performing embedding analysis...")
        
        # Dimensionality reduction
        print("  Computing PCA...")
        pca = PCA(n_components=min(10, embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(embeddings)
        
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)//4))
        tsne_embeddings = tsne.fit_transform(embeddings)
        
        # Clustering
        print("  Performing clustering...")
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(embeddings)
        
        # K-means clustering
        n_clusters = min(20, len(np.unique(dbscan_labels[dbscan_labels != -1])) + 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(embeddings)
        
        # Store results
        self.embedding_analysis = {
            'pca': pca_embeddings,
            'tsne': tsne_embeddings,
            'dbscan_labels': dbscan_labels,
            'kmeans_labels': kmeans_labels,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
        
        return self.embedding_analysis
    
    def create_embedding_visualization(self):
        """Create comprehensive embedding space visualizations"""
        if not hasattr(self, 'embedding_analysis'):
            self.create_embedding_analysis()
        
        if self.embedding_analysis is None:
            return None
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                't-SNE Embedding (Colored by Anomaly)',
                't-SNE Embedding (Colored by Cluster)',
                'PCA Variance Explained',
                'PCA Components (First 2)'
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )
        
        tsne_emb = self.embedding_analysis['tsne']
        anomalies = self.results['anomalies']
        scores = self.results['scores']
        
        # 1. t-SNE colored by anomaly status
        normal_mask = ~anomalies
        anomaly_mask = anomalies
        
        if np.any(normal_mask):
            fig.add_trace(
                go.Scatter(
                    x=tsne_emb[normal_mask, 0],
                    y=tsne_emb[normal_mask, 1],
                    mode='markers',
                    marker=dict(color='lightblue', size=6, opacity=0.6),
                    name='Normal',
                    hovertemplate='Normal<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        if np.any(anomaly_mask):
            fig.add_trace(
                go.Scatter(
                    x=tsne_emb[anomaly_mask, 0],
                    y=tsne_emb[anomaly_mask, 1],
                    mode='markers',
                    marker=dict(
                        color=scores[anomaly_mask],
                        colorscale='Reds',
                        size=8,
                        opacity=0.8,
                        colorbar=dict(
                            title="Anomaly Score",
                            x=0.48,   # Position between left subplots
                            len=0.4,   # Shorter colorbar
                            y=0.85     # Upper position
                        )
                    ),
                    name='Anomalies',
                    hovertemplate='Anomaly<br>Score: %{marker.color:.3f}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. t-SNE colored by clusters
        kmeans_labels = self.embedding_analysis['kmeans_labels']
        unique_labels = np.unique(kmeans_labels)
        colors = px.colors.qualitative.Set3
        
        for i, label in enumerate(unique_labels):
            mask = kmeans_labels == label
            fig.add_trace(
                go.Scatter(
                    x=tsne_emb[mask, 0],
                    y=tsne_emb[mask, 1],
                    mode='markers',
                    marker=dict(
                        color=colors[i % len(colors)],
                        size=6,
                        opacity=0.7
                    ),
                    name=f'Cluster {label}',
                    hovertemplate=f'Cluster {label}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # 3. PCA Variance Explained
        pca_var = self.embedding_analysis['pca_explained_variance']
        fig.add_trace(
            go.Bar(
                x=[f'PC{i+1}' for i in range(len(pca_var))],
                y=pca_var,
                marker_color='steelblue',
                name='Variance Explained'
            ),
            row=2, col=1
        )
        
        # 4. PCA embedding space (first 2 components)
        pca_emb = self.embedding_analysis['pca']
        if pca_emb.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=pca_emb[:, 0],
                    y=pca_emb[:, 1],
                    mode='markers',
                    marker=dict(
                        color=scores,
                        colorscale='Viridis',
                        size=6,
                        opacity=0.7,
                        colorbar=dict(
                            title="Anomaly Score",
                            x=1.02,    # Position to the right of all plots
                            len=0.4,    # Shorter colorbar
                            y=0.3       # Lower position
                        )
                    ),
                    name='PCA Trajectories',
                    hovertemplate='Score: %{marker.color:.3f}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="CoBAD Embedding Space Analysis",
            showlegend=True,
            legend=dict(
                x=1.05,  # Position legend to the right
                y=0.8,   # Upper position
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)", 
                borderwidth=1
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=1)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=1)
        
        fig.update_xaxes(title_text="t-SNE 1", row=1, col=2)
        fig.update_yaxes(title_text="t-SNE 2", row=1, col=2)
        
        fig.update_xaxes(title_text="Principal Component", row=2, col=1)
        fig.update_yaxes(title_text="Variance Explained", row=2, col=1)
        
        fig.update_xaxes(title_text="PC1", row=2, col=2)
        fig.update_yaxes(title_text="PC2", row=2, col=2)
        
        return fig
    
    def analyze_anomaly_clusters(self):
        """Analyze characteristics of different anomaly clusters"""
        if not hasattr(self, 'embedding_analysis'):
            self.create_embedding_analysis()
        
        if self.embedding_analysis is None:
            return None
        
        if not hasattr(self, 'subscores'):
            self.compute_trajectory_subscores()
        
        kmeans_labels = self.embedding_analysis['kmeans_labels']
        anomalies = self.results['anomalies']
        scores = self.results['scores']
        
        print(f"\n{'='*60}")
        print("ANOMALY CLUSTER ANALYSIS")
        print(f"{'='*60}")
        
        unique_labels = np.unique(kmeans_labels)
        cluster_stats = {}
        
        for label in unique_labels:
            cluster_mask = kmeans_labels == label
            cluster_anomalies = anomalies[cluster_mask]
            cluster_scores = scores[cluster_mask]
            
            anomaly_rate = np.mean(cluster_anomalies) * 100
            avg_score = np.mean(cluster_scores)
            cluster_size = np.sum(cluster_mask)
            
            print(f"\n🔍 CLUSTER {label}")
            print(f"   Size: {cluster_size} trajectories")
            print(f"   Anomaly Rate: {anomaly_rate:.1f}%")
            print(f"   Average Score: {avg_score:.4f}")
            
            # Subscore analysis for this cluster
            if hasattr(self, 'subscores'):
                print(f"   Top Sub-scores:")
                cluster_subscores = {}
                for name, subscore_values in self.subscores.items():
                    cluster_subscore_mean = np.mean(subscore_values[cluster_mask])
                    cluster_subscores[name] = cluster_subscore_mean
                
                # Sort by value and show top 3
                sorted_subscores = sorted(cluster_subscores.items(), key=lambda x: x[1], reverse=True)
                for name, value in sorted_subscores[:3]:
                    print(f"     • {name}: {value:.4f}")
            
            cluster_stats[label] = {
                'size': cluster_size,
                'anomaly_rate': anomaly_rate,
                'avg_score': avg_score,
                'subscores': cluster_subscores if hasattr(self, 'subscores') else {}
            }
        
        self.cluster_stats = cluster_stats
        return cluster_stats


def main():
    """Main function to run trajectory anomaly analysis"""
    parser = argparse.ArgumentParser(description='CoBAD Trajectory Anomaly Analysis Dashboard')
    parser.add_argument('--model', type=str, default='cobad_paper_accurate.pth',
                       help='Path to trained CoBAD model')
    parser.add_argument('--data', type=str, default='data/cityD-dataset.csv',
                       help='Path to trajectory dataset')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Number of trajectories to analyze')
    parser.add_argument('--save-results', action='store_true',
                       help='Save analysis results to files')
    parser.add_argument('--show-dashboard', action='store_true',
                       help='Display interactive dashboard in browser')
    parser.add_argument('--save-html', action='store_true',
                       help='Save dashboard as HTML files')
    parser.add_argument('--output-dir', type=str, default='dashboard_output',
                       help='Directory to save HTML dashboard files')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = TrajectoryAnomalyAnalyzer(args.model)
    
    # Load model and data
    if not analyzer.load_model_and_data(args.data, args.sample_size):
        print("Failed to load model or data.")
        return
    
    # Generate detailed report
    print("Generating detailed anomaly report...")
    analyzer.create_detailed_anomaly_report(top_n=10)
    
    # Perform embedding analysis
    print("Performing embedding space analysis...")
    analyzer.create_embedding_analysis()
    analyzer.analyze_anomaly_clusters()
    
    # Create dashboard
    if args.show_dashboard or args.save_html:
        print("Creating interactive dashboard...")
        overview_fig = analyzer.create_overview_dashboard()
        embedding_fig = analyzer.create_embedding_visualization()
        
        if args.save_html:
            # Create output directory
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            
            if overview_fig:
                overview_path = output_dir / "trajectory_overview_dashboard.html"
                overview_fig.write_html(str(overview_path))
                print(f"📊 Overview dashboard saved: {overview_path}")
            
            if embedding_fig:
                embedding_path = output_dir / "trajectory_embedding_analysis.html"
                embedding_fig.write_html(str(embedding_path))
                print(f"🔍 Embedding analysis saved: {embedding_path}")
        
        if args.show_dashboard:
            if overview_fig:
                print("Displaying overview dashboard in browser...")
                overview_fig.show()
            
            if embedding_fig:
                print("Displaying embedding analysis in browser...")
                embedding_fig.show()
    
    # Save results
    if args.save_results:
        analyzer.save_results()
    
    print(f"\n{'='*60}")
    print("TRAJECTORY ANOMALY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total trajectories analyzed: {len(analyzer.results['anomalies'])}")
    print(f"🚨 Anomalies detected: {np.sum(analyzer.results['anomalies'])}")
    print(f"📈 Anomaly rate: {np.mean(analyzer.results['anomalies'])*100:.1f}%")
    print(f"🎯 Anomaly score range: [{np.min(analyzer.results['scores']):.4f}, {np.max(analyzer.results['scores']):.4f}]")
    
    if hasattr(analyzer, 'embedding_analysis'):
        n_clusters = len(np.unique(analyzer.embedding_analysis['kmeans_labels']))
        print(f"🔍 Trajectory clusters identified: {n_clusters}")
    
    print("\nUsage options:")
    print("  --show-dashboard     : View interactive visualizations in browser")
    print("  --save-html          : Save dashboards as HTML files") 
    print("  --save-results       : Save raw analysis data as .npy files")
    print(f"  --output-dir DIR     : Specify output directory (default: dashboard_output)")
    
    if args.save_html:
        print(f"\n📁 HTML dashboards saved to: {Path(args.output_dir).absolute()}")
        print("   Open the .html files in your browser to view interactive visualizations")


if __name__ == "__main__":
    main()