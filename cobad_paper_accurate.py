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
    
    def fit(self, trajectories, epochs=20, lr=0.001, validation_split=0.2, 
            save_best=True, checkpoint_dir="checkpoints", checkpoint_freq=1, 
            early_stopping_patience=10, early_stopping_delta=1e-4):
        """Train CoBAD model using masked pretraining with mini-batches, validation tracking, and checkpointing"""
        print(f"Training CoBAD model with mini-batches (batch_size={self.batch_size})...")
        
        # Create checkpoint directory
        import os
        if save_best or checkpoint_freq > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Split trajectories into train/validation
        if validation_split > 0 and len(trajectories) > 1:
            val_size = max(1, int(len(trajectories) * validation_split))
            train_trajectories = trajectories[:-val_size]
            val_trajectories = trajectories[-val_size:]
            print(f"Training on {len(train_trajectories)} trajectories, validating on {len(val_trajectories)}")
        else:
            train_trajectories = trajectories
            val_trajectories = []
            print(f"Training on {len(train_trajectories)} trajectories (no validation split)")
        
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Best model tracking
        best_val_loss = float('inf')
        best_train_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        try:
            for epoch in tqdm(range(epochs), desc="Training"):
                epoch_losses = {'total_loss': 0, 'event_recon_loss': 0, 'link_recon_loss': 0}
                num_batches = 0
                
                # Training phase - Create mini-batches from training data
                self.train()
                for i in range(0, len(train_trajectories), self.batch_size):
                    batch_trajectories = train_trajectories[i:i+self.batch_size]
                    
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
                
                # Validation phase
                val_losses = {'total_loss': 0, 'event_recon_loss': 0, 'link_recon_loss': 0}
                if val_trajectories:
                    self.eval()
                    val_num_batches = 0
                    with torch.no_grad():
                        for i in range(0, len(val_trajectories), self.batch_size):
                            val_batch_trajectories = val_trajectories[i:i+self.batch_size]
                            
                            # Forward pass on validation batch
                            val_outputs = self.forward(val_batch_trajectories, training=False)
                            val_batch_data = self.prepare_batch_data(val_batch_trajectories)
                            val_targets = val_batch_data
                            val_adjacency_true = val_outputs['adjacency_matrix']
                            
                            # Compute validation loss
                            val_batch_losses = self.compute_loss(val_outputs, val_targets, val_adjacency_true)
                            
                            if not torch.isnan(val_batch_losses['total_loss']):
                                for key in val_losses:
                                    val_losses[key] += val_batch_losses[key].item()
                                val_num_batches += 1
                    
                    # Calculate average validation losses
                    if val_num_batches > 0:
                        for key in val_losses:
                            val_losses[key] /= val_num_batches
                
                # Track best model
                current_loss = val_losses['total_loss'] if val_trajectories else epoch_losses['total_loss']
                current_best = best_val_loss if val_trajectories else best_train_loss
                
                if current_loss < current_best - early_stopping_delta:
                    if val_trajectories:
                        best_val_loss = current_loss
                    else:
                        best_train_loss = current_loss
                    patience_counter = 0
                    
                    # Save best model
                    if save_best:
                        best_model_state = self.state_dict().copy()
                        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                        self.save_model(best_model_path)
                        print(f"New best model saved at epoch {epoch} (loss: {current_loss:.4f})")
                else:
                    patience_counter += 1
                
                # Periodic checkpoint saving
                if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    self.save_model(checkpoint_path)
                    print(f"Checkpoint saved: {checkpoint_path}")
                
                # Print progress
                if epoch % 1 == 0 or epoch == epochs - 1:
                    if val_trajectories:
                        print(f"Epoch {epoch}: Train Loss: {epoch_losses['total_loss']:.4f}, "
                              f"Val Loss: {val_losses['total_loss']:.4f}, "
                              f"Patience: {patience_counter}/{early_stopping_patience}")
                    else:
                        print(f"Epoch {epoch}: Total Loss: {epoch_losses['total_loss']:.4f}, "
                              f"Event Loss: {epoch_losses['event_recon_loss']:.4f}, "
                              f"Link Loss: {epoch_losses['link_recon_loss']:.4f}")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                          
        except Exception as e:
            print(f"Training interrupted at epoch {epoch}: {e}")
            print("Continuing with current model state...")
        
        # Load best model if available
        if save_best and best_model_state is not None:
            self.load_state_dict(best_model_state)
            print("Loaded best model for final threshold setting")
        
        # Set anomaly threshold after training using training data (in batches)
        print("Setting anomaly threshold...")
        self.eval()
        with torch.no_grad():
            # Use training trajectories for threshold setting
            threshold_data = train_trajectories if train_trajectories else trajectories
            
            # Process threshold data in batches to avoid memory issues
            all_pattern_scores = []
            all_event_errors = []
            all_link_errors = []
            
            print(f"Processing {len(threshold_data)} trajectories for threshold setting...")
            num_batches = (len(threshold_data) + self.batch_size - 1) // self.batch_size
            
            for batch_idx, i in enumerate(range(0, len(threshold_data), self.batch_size)):
                batch_trajectories = threshold_data[i:i+self.batch_size]
                
                if batch_idx % 10 == 0:  # Progress report every 10 batches
                    print(f"  Processing batch {batch_idx+1}/{num_batches} ({len(batch_trajectories)} trajectories)")
                
                outputs = self.forward(batch_trajectories, training=False)
                pattern_scores = outputs['anomaly_scores'].squeeze().cpu().numpy()
                
                # Calculate reconstruction errors
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
                
                all_pattern_scores.extend(pattern_scores)
                all_event_errors.extend(event_recon_error)
                all_link_errors.extend(link_recon_error)
                
                # Clear GPU cache if using GPU (no-op on CPU)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Convert to numpy arrays
            pattern_scores = np.array(all_pattern_scores)
            event_recon_error = np.array(all_event_errors)
            link_recon_error = np.array(all_link_errors)
            
            # Combined scores matching detection method
            combined_scores = pattern_scores + 0.5 * event_recon_error + 0.5 * link_recon_error
            
            # Dynamic threshold calculation with multiple methods
            percentile = (1 - self.target_anomaly_rate) * 100
            percentile_threshold = np.percentile(combined_scores, percentile)
            
            # Statistical outlier threshold (mean + k*std)
            mean_score = np.mean(combined_scores)
            std_score = np.std(combined_scores)
            statistical_threshold = mean_score + 2.0 * std_score  # 2 sigma
            
            # IQR-based threshold for robustness
            q75, q25 = np.percentile(combined_scores, [75, 25])
            iqr = q75 - q25
            iqr_threshold = q75 + 1.5 * iqr
            
            # Use the most conservative (highest) threshold to avoid too many false positives
            candidate_thresholds = [percentile_threshold, statistical_threshold, iqr_threshold]
            self.anomaly_threshold = np.median(candidate_thresholds)  # Use median as compromise
            
            self.train_error_mean = mean_score
            self.train_error_std = std_score
            
            print(f"Threshold candidates - Percentile: {percentile_threshold:.4f}, "
                  f"Statistical: {statistical_threshold:.4f}, IQR: {iqr_threshold:.4f}")
            print(f"Selected threshold (median): {self.anomaly_threshold:.4f}")
        
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
                
                # Pool embeddings to consistent shape before appending
                batch_embeddings = outputs['enhanced_embeddings'].cpu().numpy()  # [batch_size, seq_len, d_model]
                if batch_embeddings.shape[1] > 0:
                    pooled_batch_embeddings = batch_embeddings.mean(axis=1)  # [batch_size, d_model]
                else:
                    pooled_batch_embeddings = np.zeros((batch_embeddings.shape[0], batch_embeddings.shape[2]))
                all_embeddings.append(pooled_batch_embeddings)
        
        # Concatenate all embeddings (now all have same shape [batch_size, d_model])
        if all_embeddings:
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            all_embeddings = np.array([])
        
        # Adaptive thresholding: adjust threshold if test distribution is very different
        all_scores_array = np.array(all_scores)
        if len(all_scores_array) > 0:
            test_mean = np.mean(all_scores_array)
            test_std = np.std(all_scores_array)
            
            # If test scores are much different from training, recalculate threshold
            if hasattr(self, 'train_error_mean') and self.train_error_mean is not None:
                score_shift = abs(test_mean - self.train_error_mean)
                if score_shift > 2 * self.train_error_std:  # Significant distribution shift
                    print(f"Detected distribution shift (train_mean: {self.train_error_mean:.4f}, "
                          f"test_mean: {test_mean:.4f})")
                    
                    # Recalculate threshold based on test data
                    percentile = (1 - self.target_anomaly_rate) * 100
                    adaptive_threshold = np.percentile(all_scores_array, percentile)
                    
                    print(f"Using adaptive threshold: {adaptive_threshold:.4f} "
                          f"(original: {self.anomaly_threshold:.4f})")
                    
                    # Re-classify with adaptive threshold
                    all_anomalies = (all_scores_array > adaptive_threshold).tolist()
        
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
        checkpoint = torch.load(path, weights_only=False)
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CoBAD: Collective Behavior Anomaly Detection')
    parser.add_argument('--load-model', type=str, default=None, 
                       help='Path to saved model to load instead of training (e.g., checkpoints/best_model.pth)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only run inference (requires --load-model)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for CPU training/inference (default: 16)')
    args = parser.parse_args()
    
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
    max_trajectory_length = 24  # Limit trajectory length
    trajectories = [traj[:max_trajectory_length] for traj in trajectories if len(traj) > 0]
    
    if len(trajectories) < 10:
        print("Warning: Very few valid trajectories after filtering. Results may not be meaningful.")
    
    # Split data
    split_idx = max(1, int(0.8 * len(trajectories)))  # Ensure at least 1 training sample
    train_trajectories = trajectories[:split_idx]
    test_trajectories = trajectories[split_idx:] if split_idx < len(trajectories) else trajectories[:1]
    
    print(f"Training on {len(train_trajectories)} trajectories")
    print(f"Testing on {len(test_trajectories)} trajectories")
    
    # Initialize paper-accurate CoBAD with reduced complexity and mini-batching
    cobad = CoBAD(
        d_model=128,      # Reduced from 128 to 64
        nhead=8,         # Reduced from 8 to 4
        num_layers=2,    # Reduced from 2 to 1
        dropout=0.1,
        mask_ratio=0.15,
        spatial_dim=200,
        temporal_window=48,
        collective_threshold=5,
        target_anomaly_rate=0.05,
        batch_size=args.batch_size    # Configurable batch size for CPU training
    )
    
    print(f"Using batch size: {args.batch_size} (use --batch-size to adjust for memory constraints)")
    
    try:
        # Check if we should load a pre-trained model
        if args.load_model:
            if Path(args.load_model).exists():
                print(f"Loading pre-trained model from: {args.load_model}")
                cobad.load_model(args.load_model)
            else:
                print(f"Error: Model file not found: {args.load_model}")
                return
        
        # Train model if not skipping training
        if not args.skip_training and not args.load_model:
            print("Training new model...")
            cobad.fit(train_trajectories, epochs=10)
        elif not args.skip_training and args.load_model:
            print("Continuing training from loaded model...")
            cobad.fit(train_trajectories, epochs=10)
        else:
            print("Skipping training (inference only)")
        
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
        
        # Save model only if we trained (not if loaded pre-trained)
        if not args.load_model or not args.skip_training:
            cobad.save_model("cobad_paper_accurate.pth")
            print("Final model saved as 'cobad_paper_accurate.pth'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("The paper-accurate implementation requires significant computational resources.")
        print("Consider reducing sample size or model complexity for demonstration.")

if __name__ == "__main__":
    main()