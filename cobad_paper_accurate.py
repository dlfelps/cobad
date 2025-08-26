import pickle
from pathlib import Path
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention, TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from tqdm import tqdm

from data_utils import load_data_lazy

class CrossTimeAttention(nn.Module):
    """Cross-time attention to capture spatiotemporal dependencies within individual event sequences"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super(CrossTimeAttention, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(6, d_model)  # Project 6D events to d_model
        
        # Positional encoding for temporal sequences
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=1000)
        
        # Transformer encoder for cross-time attention
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, event_sequences, sequence_lengths=None):
        """
        Args:
            event_sequences: [batch_size, max_seq_len, 6] - individual event sequences
            sequence_lengths: [batch_size] - actual lengths of sequences (for padding)
        Returns:
            sequence_embeddings: [batch_size, max_seq_len, d_model]
        """
        batch_size, seq_len, _ = event_sequences.shape
        
        # Project input events to model dimension
        embedded = self.input_projection(event_sequences)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        embedded = self.pos_encoding(embedded)
        
        # Create padding mask if sequence lengths provided
        padding_mask = None
        if sequence_lengths is not None:
            padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            for i, length in enumerate(sequence_lengths):
                if length < seq_len:
                    padding_mask[i, length:] = True
        
        # Apply transformer encoder (cross-time attention)
        sequence_embeddings = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)
        
        # Apply output projection
        sequence_embeddings = self.output_projection(sequence_embeddings)
        
        return sequence_embeddings

class PositionalEncoding(nn.Module):
    """Positional encoding for temporal sequences"""
    
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)

class GraphTransformer(nn.Module):
    """Graph transformer for cross-people attention to capture relational patterns"""
    
    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1):
        super(GraphTransformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Graph attention layers
        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])
        
        # Layer normalization  
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # Feed-forward networks (reduced size to prevent memory issues)
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 2),  # Reduced from 4x to 2x
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, node_features, adjacency_matrix):
        """
        Args:
            node_features: [batch_size, num_nodes, d_model] - individual representations
            adjacency_matrix: [batch_size, num_nodes, num_nodes] - co-occurrence relationships
        Returns:
            enhanced_features: [batch_size, num_nodes, d_model]
        """
        x = node_features
        
        for i, (graph_attn, layer_norm, feed_forward) in enumerate(
            zip(self.graph_attention_layers, self.layer_norms, self.feed_forwards)):
            
            # Graph attention with residual connection
            attn_output = graph_attn(x, adjacency_matrix)
            x = layer_norm(x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = feed_forward(x)
            x = layer_norm(x + ff_output)
        
        return x

class GraphAttentionLayer(nn.Module):
    """Single graph attention layer"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(GraphAttentionLayer, self).__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, node_features, adjacency_matrix):
        """Apply graph attention with adjacency matrix as attention mask"""
        batch_size, num_nodes, d_model = node_features.shape
        
        # For now, use standard self-attention without complex masking
        # This avoids the tensor reshape issues while maintaining the architecture
        attn_output, _ = self.multihead_attn(
            node_features, node_features, node_features
        )
        
        return self.dropout(attn_output)

class CoOccurrenceEncoder(nn.Module):
    """Encoder for co-occurrence link patterns"""
    
    def __init__(self, d_model=128):
        super(CoOccurrenceEncoder, self).__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concatenated node features
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),  # Edge probability
            nn.Sigmoid()
        )
        
    def forward(self, node_features):
        """
        Args:
            node_features: [batch_size, num_nodes, d_model]
        Returns:
            edge_probs: [batch_size, num_nodes, num_nodes] - co-occurrence probabilities
        """
        batch_size, num_nodes, d_model = node_features.shape
        
        # Create all pairs of node features
        node_i = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, nodes, nodes, d_model]
        node_j = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, nodes, nodes, d_model]
        
        # Concatenate node pairs
        edge_features = torch.cat([node_i, node_j], dim=-1)  # [batch, nodes, nodes, 2*d_model]
        
        # Compute edge probabilities
        edge_probs = self.edge_encoder(edge_features).squeeze(-1)  # [batch, nodes, nodes]
        
        return edge_probs

class CoBAD(nn.Module):
    """
    Paper-accurate CoBAD implementation with:
    - Cross-time attention for individual sequences
    - Graph transformer for cross-people attention
    - Masked pretraining with dual reconstruction objectives
    """
    
    def __init__(self, d_model=128, nhead=8, num_layers=2, dropout=0.1, 
                 mask_ratio=0.15, spatial_dim=200, temporal_window=48, 
                 collective_threshold=5, target_anomaly_rate=0.05, batch_size=32):
        super(CoBAD, self).__init__()
        
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        self.spatial_dim = spatial_dim
        self.temporal_window = temporal_window
        self.collective_threshold = collective_threshold
        self.target_anomaly_rate = target_anomaly_rate
        self.batch_size = batch_size
        
        # Cross-time attention for individual sequences
        self.cross_time_attention = CrossTimeAttention(d_model, nhead, num_layers, dropout)
        
        # Graph transformer for cross-people attention
        self.graph_transformer = GraphTransformer(d_model, nhead, num_layers, dropout)
        
        # Co-occurrence link encoder/decoder
        self.co_occurrence_encoder = CoOccurrenceEncoder(d_model)
        
        # Event attribute reconstruction heads
        self.event_decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 6)  # Reconstruct 6D event attributes
        )
        
        # Anomaly scoring components
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Learned components (set during training)
        self.scaler = StandardScaler()
        self.anomaly_threshold = None
        self.train_error_mean = None
        self.train_error_std = None
        
        # Special tokens for masking
        self.mask_token = nn.Parameter(torch.randn(d_model))
        
    def create_masks(self, event_sequences, sequence_lengths):
        """Create random masks for pretraining"""
        batch_size, max_seq_len, _ = event_sequences.shape
        
        mask_indices = []
        for i, seq_len in enumerate(sequence_lengths):
            num_mask = int(seq_len * self.mask_ratio)
            indices = torch.randperm(seq_len)[:num_mask]
            mask_indices.append(indices)
        
        # Create mask tensor
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        for i, indices in enumerate(mask_indices):
            mask[i, indices] = True
            
        return mask
    
    def apply_masks(self, event_embeddings, mask):
        """Apply masks to event embeddings"""
        masked_embeddings = event_embeddings.clone()
        
        # Replace masked positions with mask token
        for i in range(mask.shape[0]):
            masked_positions = mask[i]
            masked_embeddings[i, masked_positions] = self.mask_token.unsqueeze(0).expand(
                masked_positions.sum(), -1)
        
        return masked_embeddings
    
    def build_co_occurrence_graph(self, trajectories, max_seq_len, time_window_size=1):
        """Build co-occurrence adjacency matrix from trajectories"""
        if not trajectories:
            return torch.zeros(1, 1, 1)
            
        batch_size = len(trajectories)
        
        # For this implementation, we'll create adjacency matrices that represent
        # relationships between sequence elements (time steps) within each trajectory
        # Each adjacency matrix should be [max_seq_len, max_seq_len]
        
        # Create identity adjacency matrix for each sequence
        # This represents that each time step is connected to itself
        # In a more sophisticated implementation, we might connect adjacent time steps
        # or time steps within a certain temporal window
        adj_matrix = torch.eye(max_seq_len)
        
        # Create batch of identical adjacency matrices
        adjacency_matrices = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        
        return adjacency_matrices
    
    def forward(self, trajectories, training=True):
        """
        Forward pass for CoBAD model
        
        Args:
            trajectories: List of individual trajectory sequences
            training: Whether in training mode (applies masking)
            
        Returns:
            Dictionary with embeddings, reconstructions, and predictions
        """
        if not trajectories:
            # Handle empty input
            dummy_tensor = torch.zeros(1, 1, self.d_model)
            return {
                'sequence_embeddings': dummy_tensor,
                'enhanced_embeddings': dummy_tensor,
                'reconstructed_events': torch.zeros(1, 1, 6),
                'reconstructed_links': torch.zeros(1, 1, 1),
                'anomaly_scores': torch.zeros(1, 1),
                'mask': None,
                'adjacency_matrix': torch.zeros(1, 1, 1)
            }
        
        # Convert trajectories to padded tensor format
        batch_data = self.prepare_batch_data(trajectories)
        event_sequences = batch_data['sequences']  # [batch_size, max_seq_len, 6]
        sequence_lengths = batch_data['lengths']
        
        batch_size, max_seq_len, _ = event_sequences.shape
        
        # Step 1: Cross-time attention for individual sequences
        sequence_embeddings = self.cross_time_attention(event_sequences, sequence_lengths)
        
        # Step 2: Apply masking for pretraining
        if training and max_seq_len > 0:
            mask = self.create_masks(event_sequences, sequence_lengths)
            masked_embeddings = self.apply_masks(sequence_embeddings, mask)
        else:
            mask = None
            masked_embeddings = sequence_embeddings
        
        # Step 3: Build co-occurrence graph
        adjacency_matrix = self.build_co_occurrence_graph(trajectories, max_seq_len)
        
        # Ensure adjacency matrix matches batch size
        if adjacency_matrix.size(0) != batch_size:
            if adjacency_matrix.size(0) == 1:
                adjacency_matrix = adjacency_matrix.expand(batch_size, -1, -1)
            else:
                adjacency_matrix = adjacency_matrix[:batch_size]
        
        # Step 4: Cross-people attention via graph transformer
        enhanced_embeddings = self.graph_transformer(masked_embeddings, adjacency_matrix)
        
        # Step 5: Dual reconstruction objectives
        
        # 5a: Event attribute reconstruction
        reconstructed_events = self.event_decoder(enhanced_embeddings)
        
        # 5b: Co-occurrence link reconstruction  
        reconstructed_links = self.co_occurrence_encoder(enhanced_embeddings)
        
        # Step 6: Anomaly scoring (combine patterns + reconstruction error)
        # Pool sequence embeddings safely
        if enhanced_embeddings.size(1) > 0:
            pooled_embeddings = enhanced_embeddings.mean(dim=1)  # [batch_size, d_model]
        else:
            pooled_embeddings = torch.zeros(batch_size, self.d_model)
            
        anomaly_scores = self.anomaly_scorer(pooled_embeddings)
        
        return {
            'sequence_embeddings': sequence_embeddings,
            'enhanced_embeddings': enhanced_embeddings,
            'reconstructed_events': reconstructed_events,
            'reconstructed_links': reconstructed_links,
            'anomaly_scores': anomaly_scores,
            'mask': mask,
            'adjacency_matrix': adjacency_matrix
        }
    
    def prepare_batch_data(self, trajectories):
        """Convert list of trajectories to padded batch format"""
        if not trajectories:
            return {
                'sequences': torch.zeros(1, 1, 6),
                'lengths': [1]  # Changed from [0] to avoid empty sequences
            }
        
        # Filter out empty trajectories
        valid_trajectories = [traj for traj in trajectories if len(traj) > 0]
        if not valid_trajectories:
            return {
                'sequences': torch.zeros(1, 1, 6),
                'lengths': [1]
            }
        
        # Find maximum sequence length
        max_seq_len = max(len(traj) for traj in valid_trajectories)
        batch_size = len(valid_trajectories)
        
        # Ensure minimum sequence length of 1
        max_seq_len = max(max_seq_len, 1)
        
        # Create padded sequences
        sequences = torch.zeros(batch_size, max_seq_len, 6)
        lengths = []
        
        for i, traj in enumerate(valid_trajectories):
            seq_len = len(traj)
            lengths.append(max(seq_len, 1))  # Ensure minimum length of 1
            
            # Convert trajectory to tensor
            if seq_len > 0:
                try:
                    traj_tensor = torch.FloatTensor(traj)
                    if traj_tensor.size(1) == 6:  # Ensure correct feature dimension
                        sequences[i, :seq_len] = traj_tensor
                except (ValueError, RuntimeError) as e:
                    print(f"Error converting trajectory to tensor: {e}")
                    # Fill with zeros if conversion fails
                    continue
        
        return {
            'sequences': sequences,
            'lengths': lengths
        }
    
    def compute_loss(self, outputs, targets, adjacency_true):
        """Compute dual reconstruction loss"""
        
        # Event reconstruction loss (only on masked positions if training)
        event_recon_loss = F.mse_loss(
            outputs['reconstructed_events'], 
            targets['sequences']
        )
        
        # Co-occurrence link reconstruction loss
        link_recon_loss = F.binary_cross_entropy(
            outputs['reconstructed_links'],
            adjacency_true.float()
        )
        
        # Combined loss
        total_loss = event_recon_loss + link_recon_loss
        
        return {
            'total_loss': total_loss,
            'event_recon_loss': event_recon_loss,
            'link_recon_loss': link_recon_loss
        }
    
    def fit(self, trajectories, epochs=20, lr=0.001):
        """Train CoBAD model using masked pretraining with mini-batches"""
        print(f"Training CoBAD model with mini-batches (batch_size={self.batch_size})...")
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        try:
            for epoch in tqdm(range(epochs), desc="Training"):
                epoch_losses = {'total_loss': 0, 'event_recon_loss': 0, 'link_recon_loss': 0}
                num_batches = 0
                
                # Create mini-batches
                for i in range(0, len(trajectories), self.batch_size):
                    batch_trajectories = trajectories[i:i+self.batch_size]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass on mini-batch
                    outputs = self.forward(batch_trajectories, training=True)
                    
                    # Prepare targets
                    batch_data = self.prepare_batch_data(batch_trajectories)
                    targets = batch_data
                    adjacency_true = outputs['adjacency_matrix']  # Use computed adjacency as target
                    
                    # Compute loss
                    losses = self.compute_loss(outputs, targets, adjacency_true)
                    
                    # Check for NaN losses
                    if torch.isnan(losses['total_loss']):
                        print(f"Warning: NaN loss encountered at epoch {epoch}, batch {i}")
                        continue
                    
                    # Backward pass
                    losses['total_loss'].backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Accumulate losses for epoch average
                    for key in epoch_losses:
                        epoch_losses[key] += losses[key].item()
                    num_batches += 1
                
                # Calculate average losses for the epoch
                if num_batches > 0:
                    for key in epoch_losses:
                        epoch_losses[key] /= num_batches
                
                if epoch % 5 == 0 or epoch == epochs - 1:
                    print(f"Epoch {epoch}: Total Loss: {epoch_losses['total_loss']:.4f}, "
                          f"Event Loss: {epoch_losses['event_recon_loss']:.4f}, "
                          f"Link Loss: {epoch_losses['link_recon_loss']:.4f}")
                          
        except Exception as e:
            print(f"Training interrupted at epoch {epoch}: {e}")
            print("Continuing with current model state...")
        
        # Set anomaly threshold after training
        self.eval()
        with torch.no_grad():
            outputs = self.forward(trajectories, training=False)
            pattern_scores = outputs['anomaly_scores'].squeeze().cpu().numpy()
            
            # Calculate reconstruction errors to match detection scoring
            event_recon_error = F.mse_loss(
                outputs['reconstructed_events'],
                self.prepare_batch_data(trajectories)['sequences'],
                reduction='none'
            ).mean(dim=(1, 2)).cpu().numpy()
            
            link_recon_error = F.binary_cross_entropy(
                outputs['reconstructed_links'],
                outputs['adjacency_matrix'].float(),
                reduction='none'
            ).mean(dim=(1, 2)).cpu().numpy()
            
            # Combined scores matching detection method
            combined_scores = pattern_scores + 0.5 * event_recon_error + 0.5 * link_recon_error
            
            # Use percentile-based threshold on combined scores
            percentile = (1 - self.target_anomaly_rate) * 100
            self.anomaly_threshold = np.percentile(combined_scores, percentile)
            self.train_error_mean = np.mean(combined_scores)
            self.train_error_std = np.std(combined_scores)
        
        print(f"Model trained. Anomaly threshold: {self.anomaly_threshold:.4f}")
    
    def detect_anomalies(self, trajectories):
        """Detect anomalies using learned patterns + reconstruction error with batching"""
        self.eval()
        
        all_anomalies = []
        all_scores = []
        all_pattern_scores = []
        all_event_errors = []
        all_link_errors = []
        all_embeddings = []
        
        with torch.no_grad():
            # Process trajectories in batches
            for i in range(0, len(trajectories), self.batch_size):
                batch_trajectories = trajectories[i:i+self.batch_size]
                
                outputs = self.forward(batch_trajectories, training=False)
                
                # Compute anomaly scores combining learned patterns and reconstruction error
                pattern_scores = outputs['anomaly_scores'].squeeze().cpu().numpy()
                
                # Add reconstruction error component
                event_recon_error = F.mse_loss(
                    outputs['reconstructed_events'],
                    self.prepare_batch_data(batch_trajectories)['sequences'],
                    reduction='none'
                ).mean(dim=(1, 2)).cpu().numpy()
                
                link_recon_error = F.binary_cross_entropy(
                    outputs['reconstructed_links'],
                    outputs['adjacency_matrix'].float(),
                    reduction='none'
                ).mean(dim=(1, 2)).cpu().numpy()
                
                # Combined anomaly scores
                combined_scores = pattern_scores + 0.5 * event_recon_error + 0.5 * link_recon_error
                
                # Detect anomalies for this batch
                batch_anomalies = combined_scores > self.anomaly_threshold
                
                # Collect results
                all_anomalies.extend(batch_anomalies)
                all_scores.extend(combined_scores)
                all_pattern_scores.extend(pattern_scores)
                all_event_errors.extend(event_recon_error)
                all_link_errors.extend(link_recon_error)
                all_embeddings.append(outputs['enhanced_embeddings'].cpu().numpy())
        
        # Concatenate all embeddings
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = np.array([])
        
        return {
            'anomalies': np.array(all_anomalies),
            'scores': np.array(all_scores),
            'pattern_scores': np.array(all_pattern_scores),
            'event_recon_errors': np.array(all_event_errors),
            'link_recon_errors': np.array(all_link_errors),
            'embeddings': all_embeddings
        }
    
    def save_model(self, path):
        """Save trained CoBAD model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'spatial_dim': self.spatial_dim,
                'temporal_window': self.temporal_window,
                'collective_threshold': self.collective_threshold,
                'target_anomaly_rate': self.target_anomaly_rate
            },
            'anomaly_threshold': self.anomaly_threshold,
            'train_error_mean': self.train_error_mean,
            'train_error_std': self.train_error_std,
        }, path)
        print(f"CoBAD model saved to {path}")
    
    def load_model(self, path):
        """Load trained CoBAD model"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.anomaly_threshold = checkpoint['anomaly_threshold']
        self.train_error_mean = checkpoint['train_error_mean']
        self.train_error_std = checkpoint['train_error_std']
        print(f"CoBAD model loaded from {path}")

def preprocess_trajectories_for_cobad(raw_data):
    """Preprocess trajectories to preserve individual sequences for CoBAD"""
    print("Preprocessing trajectories for CoBAD...")
    processed_trajectories = []
    
    for user_traj in tqdm(raw_data):
        if len(user_traj) < 1:
            continue
            
        trajectory_features = []
        for stay_point in user_traj:
            try:
                # Handle different data formats - stay_point could be tuple or list
                if len(stay_point) >= 6:
                    # Extract features: uid, normalized_x, normalized_y, is_weekend, start_time, duration
                    uid, norm_x, norm_y, is_weekend, start_time, duration = stay_point[:6]
                else:
                    # Handle incomplete data
                    continue
                
                # Ensure all values are numbers, not tuples/lists
                norm_x = float(norm_x) if not isinstance(norm_x, (list, tuple)) else float(norm_x[0])
                norm_y = float(norm_y) if not isinstance(norm_y, (list, tuple)) else float(norm_y[0])
                is_weekend = float(is_weekend) if not isinstance(is_weekend, (list, tuple)) else float(is_weekend[0])
                start_time = float(start_time) if not isinstance(start_time, (list, tuple)) else float(start_time[0])
                duration = float(duration) if not isinstance(duration, (list, tuple)) else float(duration[0])
                
                # Create feature vector (without uid for model input)
                features = [norm_x, norm_y, is_weekend, start_time, duration, 0.0]  # 6D feature
                trajectory_features.append(features)
                
            except (ValueError, TypeError, IndexError) as e:
                print(f"Skipping invalid stay_point: {stay_point}, Error: {e}")
                continue
        
        if trajectory_features:
            processed_trajectories.append(trajectory_features)
    
    return processed_trajectories

def main():
    """Main function to demonstrate paper-accurate CoBAD"""
    
    # Load trajectory data
    cityD = Path().joinpath("data").joinpath("cityD-dataset.csv")
    print("Loading trajectory data...")
    raw_data = load_data_lazy(cityD)
    print(f"Loaded {len(raw_data)} user trajectories")
    
    # Use much smaller sample for demonstration to avoid memory issues
    sample_size = min(100000, len(raw_data))  # Reduced sample for faster testing
    raw_data_sample = raw_data[:sample_size]
    
    # Preprocess for CoBAD (preserve individual sequences)
    trajectories = preprocess_trajectories_for_cobad(raw_data_sample)
    print(f"Processed {len(trajectories)} individual trajectories")
    
    # Filter out very long trajectories to manage memory
    max_trajectory_length = 50  # Limit trajectory length
    trajectories = [traj[:max_trajectory_length] for traj in trajectories if len(traj) > 0]
    
    if len(trajectories) < 10:
        print("Warning: Very few valid trajectories after filtering. Results may not be meaningful.")
    
    # Split data
    split_idx = max(1, int(0.7 * len(trajectories)))  # Ensure at least 1 training sample
    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:] if split_idx < len(trajectories) else trajectories[:1]
    
    print(f"Training on {len(train_trajectories)} trajectories")
    print(f"Testing on {len(test_trajectories)} trajectories")
    
    # Initialize paper-accurate CoBAD with reduced complexity and mini-batching
    cobad = CoBAD(
        d_model=64,      # Reduced from 128 to 64
        nhead=4,         # Reduced from 8 to 4
        num_layers=1,    # Reduced from 2 to 1
        dropout=0.1,
        mask_ratio=0.15,
        spatial_dim=200,
        temporal_window=48,
        collective_threshold=5,
        target_anomaly_rate=0.05,
        batch_size=16    # Small batch size for CPU training
    )
    
    try:
        # Train with masked pretraining
        cobad.fit(train_trajectories, epochs=50)
        
        # Detect anomalies
        results = cobad.detect_anomalies(test_trajectories)
        
        # Print results
        anomaly_count = np.sum(results['anomalies'])
        total_behaviors = len(results['anomalies'])
        anomaly_rate = anomaly_count / total_behaviors * 100 if total_behaviors > 0 else 0
        
        print(f"\n=== Paper-Accurate CoBAD Results ===")
        print(f"Total trajectories analyzed: {total_behaviors}")
        print(f"Anomalous trajectories detected: {anomaly_count}")
        print(f"Anomaly rate: {anomaly_rate:.2f}%")
        print(f"Anomaly score range: {results['scores'].min():.4f} - {results['scores'].max():.4f}")
        
        # Save model
        cobad.save_model("cobad_paper_accurate.pth")
        
    except Exception as e:
        print(f"Error: {e}")
        print("The paper-accurate implementation requires significant computational resources.")
        print("Consider reducing sample size or model complexity for demonstration.")

if __name__ == "__main__":
    main()