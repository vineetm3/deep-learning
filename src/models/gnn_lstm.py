"""
Complete GNN-LSTM model for NFL trajectory prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.models.graph_builder import GraphBuilder
from src.models.gat_encoder import GATEncoder
from src.models.lstm_decoder import RoleConditionedLSTMDecoder
from src.data.dataset import INPUT_FEATURE_COLUMNS, FEATURE_INDEX


class TemporalAttentionPooling(nn.Module):
    """Attention pooling over temporal sequences with masking support."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequences: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] (1 for valid, 0 for padded)

        Returns:
            pooled: [batch, hidden_dim]
        """
        attn_input = torch.tanh(self.proj(sequences))
        scores = self.score(attn_input).squeeze(-1)
        valid_mask = mask > 0
        has_tokens = valid_mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        scores = torch.where(
            has_tokens,
            scores,
            torch.zeros_like(scores),
        )
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.where(
            has_tokens,
            attn_weights,
            torch.zeros_like(attn_weights),
        )
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        pooled = torch.bmm(attn_weights.unsqueeze(1), sequences).squeeze(1)
        return pooled


class GNNLSTMTrajectoryPredictor(nn.Module):
    """
    Full trajectory prediction model combining:
    1. Graph construction from player states
    2. GAT encoder for spatial relationships
    3. Role-conditioned LSTM decoder for trajectory generation
    """
    
    def __init__(
        self,
        config,
    ):
        """
        Args:
            config: Model configuration object
        """
        super().__init__()
        
        self.config = config
        
        # Input feature processing
        self.num_continuous_features = len(INPUT_FEATURE_COLUMNS)
        self.temporal_hidden_dim = getattr(config, 'temporal_hidden_dim', config.gnn_hidden_dim)
        
        # Categorical features will be embedded
        self.role_embedding = nn.Embedding(config.num_roles, config.role_embedding_dim)
        self.position_embedding = nn.Embedding(config.num_positions, config.position_embedding_dim)
        self.side_embedding = nn.Embedding(2, 8)  # Offense/Defense
        self.direction_embedding = nn.Embedding(2, 8)  # Left/Right
        
        # Temporal encoder for pre-pass sequence
        self.temporal_encoder = nn.LSTM(
            input_size=self.num_continuous_features,
            hidden_size=self.temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(config.gnn_dropout)
        self.temporal_pool = TemporalAttentionPooling(self.temporal_hidden_dim)
        
        # Total node feature dimension after embedding
        total_feature_dim = (
            self.num_continuous_features +  # last frame
            self.temporal_hidden_dim +
            config.role_embedding_dim +
            config.position_embedding_dim +
            8 + 8  # side + direction
        )
        
        # Project to node_feature_dim
        self.feature_projection = nn.Linear(total_feature_dim, config.node_feature_dim)
        
        # Graph builder
        self.graph_builder = GraphBuilder(k_neighbors=config.k_nearest_neighbors)
        
        # GAT encoder
        self.gat_encoder = GATEncoder(
            in_features=config.node_feature_dim,
            hidden_features=config.gnn_hidden_dim,
            out_features=config.gnn_output_dim,
            num_layers=config.gnn_num_layers,
            num_heads=config.gnn_num_heads,
            dropout=config.gnn_dropout,
            edge_features=self.graph_builder.edge_feature_dim,
        )
        
        # LSTM decoder
        self.lstm_decoder = RoleConditionedLSTMDecoder(
            context_dim=config.gnn_output_dim,
            role_embedding_dim=config.role_embedding_dim,
            num_roles=config.num_roles,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            output_dim=config.output_dim,
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            batch: Dictionary containing:
                - input_features: [batch, num_players, input_frames, num_features]
                - categorical_features: [batch, num_players, num_categorical]
                - ball_landing: [batch, 2]
                - player_roles: [batch, num_players]
                - player_mask: [batch, num_players]
                - output_mask: [batch, num_players, output_frames]
                - output_positions: [batch, num_players, output_frames, 2] (for training)
                - num_output_frames: list of actual output frames
            teacher_forcing_ratio: Probability of using ground truth in decoder
        
        Returns:
            predictions: [batch, num_players, output_frames, 2]
        """
        # Extract batch components
        input_features = batch['input_features']  # [batch, players, frames, features]
        categorical_features = batch['categorical_features']  # [batch, players, 4]
        ball_landing = batch['ball_landing']  # [batch, 2]
        player_roles = batch['player_roles']  # [batch, players]
        player_mask = batch['player_mask']  # [batch, players]
        output_mask = batch['output_mask']  # [batch, players, output_frames]
        input_mask = batch['input_mask']  # [batch, players, frames]
        
        batch_size, num_players, input_frames, _ = input_features.shape
        _, _, max_output_frames = output_mask.shape
        
        # Get last observed positions (for initial decoder input)
        last_frame = input_features[:, :, -1, :]
        last_positions = last_frame[:, :, [FEATURE_INDEX['x'], FEATURE_INDEX['y']]]  # [batch, players, 2]
        
        # Temporal encoding using LSTM over pre-pass frames with attention pooling
        flat_sequences = input_features.view(batch_size * num_players, input_frames, -1)
        flat_mask = input_mask.view(batch_size * num_players, input_frames)
        lengths = flat_mask.sum(dim=1)
        lengths_clamped = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
        packed = pack_padded_sequence(
            flat_sequences,
            lengths_clamped.cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        temporal_outputs, _ = self.temporal_encoder(packed)
        temporal_outputs, _ = pad_packed_sequence(
            temporal_outputs,
            batch_first=True,
            total_length=input_frames,
        )
        temporal_outputs = self.temporal_dropout(temporal_outputs)
        temporal_summary_flat = self.temporal_pool(
            temporal_outputs,
            flat_mask,
        )
        zero_length_mask = lengths == 0
        temporal_summary_flat[zero_length_mask] = 0
        temporal_summary = temporal_summary_flat.view(batch_size, num_players, self.temporal_hidden_dim)
        temporal_summary = self.temporal_dropout(temporal_summary)
        
        # Rich node feature representation (current state + temporal summary)
        node_features_continuous = torch.cat([last_frame, temporal_summary], dim=-1)
        
        # Embed categorical features
        role_embeds = self.role_embedding(categorical_features[:, :, 0])  # Role
        position_embeds = self.position_embedding(categorical_features[:, :, 1])  # Position
        side_embeds = self.side_embedding(categorical_features[:, :, 2])  # Side
        direction_embeds = self.direction_embedding(categorical_features[:, :, 3])  # Direction
        
        # Concatenate all features
        node_features = torch.cat([
            node_features_continuous,
            role_embeds,
            position_embeds,
            side_embeds,
            direction_embeds,
        ], dim=-1)  # [batch, players, total_feature_dim]
        
        # Project to node_feature_dim
        node_features = self.feature_projection(node_features)  # [batch, players, node_feature_dim]
        
        # Build graph
        edge_index, edge_features, node_features_with_ball = self.graph_builder.build_graph(
            node_features,
            ball_landing,
            player_roles,
            player_mask,
            last_frame,
        )
        
        # Flatten batch for GNN processing
        # GNN expects [total_nodes, feature_dim]
        num_nodes_per_graph = num_players + 1  # Players + ball
        total_nodes = batch_size * num_nodes_per_graph
        
        node_features_flat = node_features_with_ball.view(total_nodes, -1)
        
        # Encode with GAT
        node_embeddings = self.gat_encoder(
            node_features_flat,
            edge_index,
            edge_features,
        )  # [total_nodes, gnn_output_dim]
        
        # Reshape back to batch format
        node_embeddings = node_embeddings.view(batch_size, num_nodes_per_graph, -1)
        
        # Extract player embeddings (exclude ball node)
        player_embeddings = node_embeddings[:, :num_players, :]  # [batch, players, gnn_output_dim]
        
        # Decode trajectories
        ground_truth = batch.get('output_positions', None)
        
        predictions = self.lstm_decoder(
            context=player_embeddings,
            initial_position=last_positions,
            ball_landing=ball_landing,
            roles=player_roles,
            max_steps=max_output_frames,
            ground_truth=ground_truth,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        
        return predictions


class SimplifiedGNNLSTM(nn.Module):
    """
    Simplified version without graph construction (for debugging/baseline)
    
    Uses a simple MLP encoder instead of GNN
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.num_continuous_features = len(INPUT_FEATURE_COLUMNS)
        self.temporal_hidden_dim = getattr(config, 'temporal_hidden_dim', config.gnn_hidden_dim)
        
        # Simple feature encoder
        input_dim = self.num_continuous_features + self.temporal_hidden_dim + 4 * 16  # current state + temporal summary + categorical embeddings
        
        self.role_embedding = nn.Embedding(config.num_roles, 16)
        self.position_embedding = nn.Embedding(config.num_positions, 16)
        self.side_embedding = nn.Embedding(2, 16)
        self.direction_embedding = nn.Embedding(2, 16)
        self.temporal_encoder = nn.LSTM(
            input_size=self.num_continuous_features,
            hidden_size=self.temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(config.gnn_dropout)
        self.temporal_pool = TemporalAttentionPooling(self.temporal_hidden_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.gnn_dropout),
            nn.Linear(config.gnn_hidden_dim, config.gnn_output_dim),
            nn.ReLU(),
        )
        
        # LSTM decoder
        self.lstm_decoder = RoleConditionedLSTMDecoder(
            context_dim=config.gnn_output_dim,
            role_embedding_dim=config.role_embedding_dim,
            num_roles=config.num_roles,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            output_dim=config.output_dim,
        )
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Simple forward without graph structure"""
        
        input_features = batch['input_features']
        categorical_features = batch['categorical_features']
        ball_landing = batch['ball_landing']
        player_roles = batch['player_roles']
        output_mask = batch['output_mask']
        input_mask = batch['input_mask']
        
        batch_size, num_players, input_frames, _ = input_features.shape
        max_output_frames = output_mask.shape[2]
        
        # Use last frame
        last_frame = input_features[:, :, -1, :]
        masked_inputs = input_features * input_mask.unsqueeze(-1)
        valid_lengths = input_mask.sum(dim=2, keepdim=True)
        valid_lengths_clamped = torch.clamp(valid_lengths, min=1.0)
        mean_frame = masked_inputs.sum(dim=2) / valid_lengths_clamped
        mean_frame = torch.where(
            valid_lengths > 0,
            mean_frame,
            torch.zeros_like(mean_frame),
        )
        last_positions = last_frame[:, :, [FEATURE_INDEX['x'], FEATURE_INDEX['y']]]
        
        # Temporal encoding
        flat_sequences = input_features.view(batch_size * num_players, input_frames, -1)
        flat_mask = input_mask.view(batch_size * num_players, input_frames)
        lengths = flat_mask.sum(dim=1)
        lengths_clamped = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
        packed = pack_padded_sequence(
            flat_sequences,
            lengths_clamped.cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        temporal_outputs, _ = self.temporal_encoder(packed)
        temporal_outputs, _ = pad_packed_sequence(
            temporal_outputs,
            batch_first=True,
            total_length=input_frames,
        )
        temporal_outputs = self.temporal_dropout(temporal_outputs)
        temporal_summary_flat = self.temporal_pool(
            temporal_outputs,
            flat_mask,
        )
        zero_length_mask = lengths == 0
        temporal_summary_flat[zero_length_mask] = 0
        temporal_summary = temporal_summary_flat.view(batch_size, num_players, self.temporal_hidden_dim)
        temporal_summary = self.temporal_dropout(temporal_summary)
        
        # Embed categoricals
        role_embeds = self.role_embedding(categorical_features[:, :, 0])
        position_embeds = self.position_embedding(categorical_features[:, :, 1])
        side_embeds = self.side_embedding(categorical_features[:, :, 2])
        direction_embeds = self.direction_embedding(categorical_features[:, :, 3])
        
        # Concatenate
        node_features = torch.cat([
            last_frame,
            temporal_summary,
            role_embeds,
            position_embeds,
            side_embeds,
            direction_embeds,
        ], dim=-1)
        
        # Encode
        embeddings = self.encoder(node_features)
        
        # Decode
        ground_truth = batch.get('output_positions', None)
        
        predictions = self.lstm_decoder(
            context=embeddings,
            initial_position=last_positions,
            ball_landing=ball_landing,
            roles=player_roles,
            max_steps=max_output_frames,
            ground_truth=ground_truth,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        
        return predictions


