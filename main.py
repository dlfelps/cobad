import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from data_utils import load_data

class CoBAD:
    """
    Collective Behavior Anomaly Detection for Human Mobility Trajectories
    """
    
    def __init__(self, spatial_dim=200, temporal_window=48, collective_threshold=5):
        self.spatial_dim = spatial_dim
        self.temporal_window = temporal_window
        self.collective_threshold = collective_threshold
        self.scaler = StandardScaler()
        
        # Neural network for behavior embedding (8 features from collective stay behavior)
        self.behavior_encoder = nn.Sequential(
            nn.Linear(8, 64),  # time, center_x, center_y, spread_x, spread_y, rel_size, weekend_ratio, avg_stay_duration
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        self.collective_patterns = None
        self.anomaly_threshold = None
        self.reconstruction_head = None
        self.train_error_mean = None
        self.train_error_std = None
        
    def preprocess_trajectories(self, raw_data):
        """Convert raw trajectory data with stay point events to spatial-temporal features"""
        print("Preprocessing stay point trajectories...")
        processed_trajectories = []
        
        for user_traj in tqdm(raw_data):
            if len(user_traj) < 1:
                continue
                
            trajectory_features = []
            for stay_point in user_traj:
                # Extract features: uid, normalized_x, normalized_y, is_weekend, start_time, duration
                uid, norm_x, norm_y, is_weekend, start_time, duration = stay_point
                
                # Spatial features (already normalized to [0,1])
                spatial_x, spatial_y = norm_x, norm_y
                
                # Temporal features
                temporal_start = (start_time % self.temporal_window) / self.temporal_window
                
                # Stay duration normalized by temporal window
                normalized_duration = min(duration / self.temporal_window, 1.0)  # Cap at 1.0
                
                # Features: uid, spatial_x, spatial_y, is_weekend, temporal_start, duration
                features = [uid, spatial_x, spatial_y, is_weekend, temporal_start, normalized_duration]
                trajectory_features.append(features)
            
            if trajectory_features:
                processed_trajectories.append(np.array(trajectory_features))
        
        return processed_trajectories
    
    def extract_collective_features(self, trajectories):
        """Extract collective behavior features from trajectories"""
        print("Extracting collective behavior features...")
        collective_features = []
        
        # Group stay points by time windows
        time_windows = {}
        for traj in trajectories:
            for stay_point in traj:
                # Use start time for temporal grouping
                t_window = int(stay_point[4] * self.temporal_window)
                if t_window not in time_windows:
                    time_windows[t_window] = []
                time_windows[t_window].append(stay_point)
        
        for t_window, stay_points in tqdm(time_windows.items()):
            if len(stay_points) < self.collective_threshold:
                continue
                
            stay_points_array = np.array(stay_points)
            
            # Spatial density features
            spatial_coords = stay_points_array[:, 1:3]
            
            # Use DBSCAN to find spatial clusters (collective groups at same locations)
            clustering = DBSCAN(eps=0.05, min_samples=self.collective_threshold)
            cluster_labels = clustering.fit_predict(spatial_coords)
            
            # Extract features for each cluster
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                if label == -1:  # Noise points
                    continue
                    
                cluster_mask = cluster_labels == label
                cluster_stays = stay_points_array[cluster_mask]
                
                if len(cluster_stays) >= self.collective_threshold:
                    # Collective stay behavior features
                    cluster_center = np.mean(cluster_stays[:, 1:3], axis=0)  # Spatial center
                    cluster_spread = np.std(cluster_stays[:, 1:3], axis=0)   # Spatial spread
                    cluster_size = len(cluster_stays)  # Number of people in cluster
                    weekend_ratio = np.mean(cluster_stays[:, 3])  # Weekend activity ratio
                    avg_stay_duration = np.mean(cluster_stays[:, 5])  # Average stay duration
                    
                    collective_feature = np.concatenate([
                        [t_window / self.temporal_window],  # Normalized time window
                        cluster_center,  # Spatial center (x, y)
                        cluster_spread,  # Spatial spread (std_x, std_y)
                        [cluster_size / len(stay_points)],  # Relative cluster size
                        [weekend_ratio],  # Weekend activity ratio
                        [avg_stay_duration]  # Average stay duration
                    ])
                    
                    collective_features.append(collective_feature)
        
        return np.array(collective_features) if collective_features else np.array([])
    
    def fit(self, trajectories):
        """Fit the CoBAD model to learn normal collective behavior patterns"""
        print("Fitting CoBAD model...")
        
        # Extract collective features
        collective_features = self.extract_collective_features(trajectories)
        
        if len(collective_features) == 0:
            raise ValueError("No collective behaviors found. Try reducing collective_threshold.")
        
        # Normalize features
        collective_features_norm = self.scaler.fit_transform(collective_features)
        
        # Learn behavior embeddings using autoencoder approach
        self.behavior_encoder.train()
        optimizer = torch.optim.Adam(self.behavior_encoder.parameters(), lr=0.001)
        
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(collective_features_norm)
        
        # Train autoencoder (encoder + decoder)
        self.reconstruction_head = nn.Linear(16, collective_features_norm.shape[1])
        autoencoder = nn.Sequential(self.behavior_encoder, self.reconstruction_head)
        
        # Include reconstruction head in optimizer
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
        
        print("Training behavior encoder...")
        for epoch in tqdm(range(100)):
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = self.behavior_encoder(X_tensor)
            reconstructed = self.reconstruction_head(embeddings)
            
            # Reconstruction loss
            loss = nn.MSELoss()(reconstructed, X_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Store normal patterns (embeddings)
        self.behavior_encoder.eval()
        self.reconstruction_head.eval()
        with torch.no_grad():
            self.collective_patterns = self.behavior_encoder(X_tensor).numpy()
        
        # Set anomaly threshold more conservatively (85th percentile instead of 95th)
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            reconstruction_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            self.anomaly_threshold = np.percentile(reconstruction_errors.numpy(), 85)  # More conservative
            
            # Also store statistics for adaptive thresholding
            self.train_error_mean = np.mean(reconstruction_errors.numpy())
            self.train_error_std = np.std(reconstruction_errors.numpy())
        
        print(f"Model fitted. Anomaly threshold: {self.anomaly_threshold:.4f}")
        
    def detect_anomalies(self, trajectories):
        """Detect anomalous collective behaviors in new trajectory data"""
        print("Detecting anomalies...")
        
        # Extract collective features from new data
        collective_features = self.extract_collective_features(trajectories)
        
        if len(collective_features) == 0:
            print("No collective behaviors found in test data.")
            return []
        
        # Normalize features
        collective_features_norm = self.scaler.transform(collective_features)
        
        # Get embeddings
        X_tensor = torch.FloatTensor(collective_features_norm)
        self.behavior_encoder.eval()
        
        with torch.no_grad():
            embeddings = self.behavior_encoder(X_tensor)
            
            # Use the trained reconstruction head (not a new one!)
            reconstructed = self.reconstruction_head(embeddings)
            reconstruction_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1)
            
            # Adaptive thresholding based on training distribution
            test_error_mean = torch.mean(reconstruction_errors).item()
            test_error_std = torch.std(reconstruction_errors).item()
            
            # If test errors are much higher than training, adjust threshold
            if test_error_mean > self.train_error_mean + 2 * self.train_error_std:
                adaptive_threshold = self.train_error_mean + 3 * self.train_error_std
                print(f"Using adaptive threshold: {adaptive_threshold:.4f} (original: {self.anomaly_threshold:.4f})")
                threshold = adaptive_threshold
            else:
                threshold = self.anomaly_threshold
            
            # Identify anomalies
            anomalies = reconstruction_errors > threshold
            anomaly_scores = reconstruction_errors.numpy()
        
        return {
            'anomalies': anomalies.numpy(),
            'scores': anomaly_scores,
            'features': collective_features,
            'embeddings': embeddings.numpy()
        }
    
    def visualize_results(self, results, save_path=None):
        """Visualize anomaly detection results"""
        if len(results) == 0:
            print("No results to visualize.")
            return
            
        features = results['features']
        anomalies = results['anomalies']
        scores = results['scores']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Spatial distribution of collective behaviors
        ax1 = axes[0, 0]
        normal_mask = ~anomalies
        anomaly_mask = anomalies
        
        if np.any(normal_mask):
            ax1.scatter(features[normal_mask, 1], features[normal_mask, 2], 
                       c='blue', alpha=0.6, label='Normal', s=30)
        if np.any(anomaly_mask):
            ax1.scatter(features[anomaly_mask, 1], features[anomaly_mask, 2], 
                       c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
        
        ax1.set_xlabel('Spatial X (normalized)')
        ax1.set_ylabel('Spatial Y (normalized)')
        ax1.set_title('Spatial Distribution of Collective Behaviors')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Temporal distribution
        ax2 = axes[0, 1]
        if np.any(normal_mask):
            ax2.scatter(features[normal_mask, 0], scores[normal_mask], 
                       c='blue', alpha=0.6, label='Normal', s=30)
        if np.any(anomaly_mask):
            ax2.scatter(features[anomaly_mask, 0], scores[anomaly_mask], 
                       c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
        
        ax2.axhline(y=self.anomaly_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.anomaly_threshold:.3f})')
        ax2.set_xlabel('Time Window (normalized)')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title('Temporal Distribution of Anomaly Scores')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Collective group size vs anomaly score
        ax3 = axes[1, 0]
        if np.any(normal_mask):
            ax3.scatter(features[normal_mask, 5], scores[normal_mask], 
                       c='blue', alpha=0.6, label='Normal', s=30)
        if np.any(anomaly_mask):
            ax3.scatter(features[anomaly_mask, 5], scores[anomaly_mask], 
                       c='red', alpha=0.8, label='Anomaly', s=50, marker='x')
        
        ax3.set_xlabel('Relative Group Size')
        ax3.set_ylabel('Anomaly Score')
        ax3.set_title('Group Size vs Anomaly Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Anomaly score distribution
        ax4 = axes[1, 1]
        ax4.hist(scores[normal_mask], bins=30, alpha=0.7, color='blue', 
                label=f'Normal (n={np.sum(normal_mask)})')
        ax4.hist(scores[anomaly_mask], bins=30, alpha=0.7, color='red', 
                label=f'Anomaly (n={np.sum(anomaly_mask)})')
        ax4.axvline(x=self.anomaly_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.anomaly_threshold:.3f})')
        ax4.set_xlabel('Anomaly Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Anomaly Scores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_model(self, path):
        """Save the trained CoBAD model"""
        model_state = {
            'behavior_encoder': self.behavior_encoder.state_dict(),
            'reconstruction_head': self.reconstruction_head.state_dict(),
            'scaler': self.scaler,
            'collective_patterns': self.collective_patterns,
            'anomaly_threshold': self.anomaly_threshold,
            'train_error_mean': self.train_error_mean,
            'train_error_std': self.train_error_std,
            'spatial_dim': self.spatial_dim,
            'temporal_window': self.temporal_window,
            'collective_threshold': self.collective_threshold
        }
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained CoBAD model"""
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.behavior_encoder.load_state_dict(model_state['behavior_encoder'])
        
        # Initialize and load reconstruction head
        feature_dim = model_state['scaler'].n_features_in_
        self.reconstruction_head = nn.Linear(16, feature_dim)
        self.reconstruction_head.load_state_dict(model_state['reconstruction_head'])
        
        self.scaler = model_state['scaler']
        self.collective_patterns = model_state['collective_patterns']
        self.anomaly_threshold = model_state['anomaly_threshold']
        self.train_error_mean = model_state['train_error_mean']
        self.train_error_std = model_state['train_error_std']
        self.spatial_dim = model_state['spatial_dim']
        self.temporal_window = model_state['temporal_window']
        self.collective_threshold = model_state['collective_threshold']
        print(f"Model loaded from {path}")

def main():
    # Load trajectory data (sample for faster testing)
    cityD = Path().joinpath("data").joinpath("cityD-dataset.csv")
    print("Loading trajectory data...")
    raw_data = load_data(cityD)
    print(f"Loaded {len(raw_data)} user trajectories")
    
    # Sample data for faster processing during development
    sample_size = min(1000, len(raw_data))  # Use first 1000 trajectories
    raw_data_sample = raw_data[:sample_size]
    print(f"Using sample of {len(raw_data_sample)} trajectories for demonstration")
    
    # Initialize CoBAD model with relaxed parameters for demo
    cobad = CoBAD(spatial_dim=200, temporal_window=48, collective_threshold=3)
    
    # Preprocess trajectories
    trajectories = cobad.preprocess_trajectories(raw_data_sample)
    print(f"Processed {len(trajectories)} trajectories")
    
    # Split data for training and testing
    split_idx = int(0.7 * len(trajectories))
    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:]
    
    print(f"Training on {len(train_trajectories)} trajectories")
    print(f"Testing on {len(test_trajectories)} trajectories")
    
    # Fit the model
    try:
        cobad.fit(train_trajectories)
        
        # Detect anomalies in test data
        results = cobad.detect_anomalies(test_trajectories)
        
        # Print results summary
        if len(results) > 0:
            anomaly_count = np.sum(results['anomalies'])
            total_behaviors = len(results['anomalies'])
            anomaly_rate = anomaly_count / total_behaviors * 100
            
            print(f"\n=== CoBAD Results ===")
            print(f"Total collective behaviors analyzed: {total_behaviors}")
            print(f"Anomalous behaviors detected: {anomaly_count}")
            print(f"Anomaly rate: {anomaly_rate:.2f}%")
            print(f"Anomaly score range: {results['scores'].min():.4f} - {results['scores'].max():.4f}")
            
            # Visualize results
            cobad.visualize_results(results, "cobad_results.png")
            
            # Save model
            cobad.save_model("cobad_model.pkl")
        else:
            print("No collective behaviors found for anomaly detection.")
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Trying with lower collective_threshold...")
        
        # Try with lower threshold
        cobad_relaxed = CoBAD(spatial_dim=200, temporal_window=48, collective_threshold=2)
        trajectories_relaxed = cobad_relaxed.preprocess_trajectories(raw_data_sample)
        train_relaxed = trajectories_relaxed[:int(0.7 * len(trajectories_relaxed))]
        test_relaxed = trajectories_relaxed[int(0.7 * len(trajectories_relaxed)):]
        
        try:
            cobad_relaxed.fit(train_relaxed)
            results_relaxed = cobad_relaxed.detect_anomalies(test_relaxed)
            
            if len(results_relaxed) > 0:
                anomaly_count = np.sum(results_relaxed['anomalies'])
                total_behaviors = len(results_relaxed['anomalies'])
                anomaly_rate = anomaly_count / total_behaviors * 100
                
                print(f"\n=== CoBAD Results (Relaxed Threshold) ===")
                print(f"Total collective behaviors analyzed: {total_behaviors}")
                print(f"Anomalous behaviors detected: {anomaly_count}")
                print(f"Anomaly rate: {anomaly_rate:.2f}%")
                print(f"Anomaly score range: {results_relaxed['scores'].min():.4f} - {results_relaxed['scores'].max():.4f}")
                
                cobad_relaxed.visualize_results(results_relaxed, "cobad_results_relaxed.png")
                cobad_relaxed.save_model("cobad_model_relaxed.pkl")
            else:
                print("Still no collective behaviors found. Dataset may need different parameters.")
        
        except ValueError as e2:
            print(f"Error even with relaxed parameters: {e2}")
            print("This suggests the dataset structure or parameters need adjustment.")

if __name__ == "__main__":
    main()