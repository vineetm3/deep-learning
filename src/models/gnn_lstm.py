"""
Complete GNN-Transformer model for NFL trajectory prediction
"""

import torch
import torch.nn as nn
from typing import Dict

from torch.nn.utils.rnn import pack_padded_sequence

from src.models.graph_builder import GraphBuilder
from src.models.gat_encoder import GATEncoder
from src.models.transformer_decoder import TransformerTrajectoryDecoder
from src.data.dataset import INPUT_FEATURE_COLUMNS, FEATURE_INDEX


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
        
        # Total node feature dimension after embedding
        total_feature_dim = (
            self.num_continuous_features +
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
            edge_features=9,  # extended relational features
        )
        
        # Transformer decoder
        self.decoder = TransformerTrajectoryDecoder(
            context_dim=config.gnn_output_dim,
            role_dim=config.role_embedding_dim,
            embed_dim=config.decoder_embed_dim,
            num_steps=config.max_output_frames,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
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
        player_mask = batch['player_mask'].to(input_features.dtype)  # [batch, players]
        output_mask = batch['output_mask'].to(input_features.dtype)  # [batch, players, output_frames]
        input_mask = batch['input_mask']  # [batch, players, frames]
        
        batch_size, num_players, input_frames, _ = input_features.shape
        _, _, max_output_frames = output_mask.shape
        
        # Get last observed positions (for initial decoder input)
        last_frame = input_features[:, :, -1, :]
        last_positions = last_frame[:, :, [FEATURE_INDEX['x'], FEATURE_INDEX['y']]]  # [batch, players, 2]
        
        # Average input features across time (simple temporal aggregation)
        # Alternatively, could use last frame or LSTM over frames
        # For now: use last frame for node features
        node_features_continuous = last_frame  # [batch, players, num_features]
        
        # Temporal encoding using LSTM over pre-pass frames
        flat_sequences = input_features.view(batch_size * num_players, input_frames, -1)
        flat_mask = input_mask.view(batch_size * num_players, input_frames)
        lengths = flat_mask.sum(dim=1)
        lengths_clamped = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
        packed = pack_padded_sequence(flat_sequences, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False)
        _, (temporal_hidden, _) = self.temporal_encoder(packed)
        temporal_context = temporal_hidden[-1]  # [batch*players, hidden]
        temporal_context[lengths == 0] = 0
        temporal_context = temporal_context.view(batch_size, num_players, self.temporal_hidden_dim)
        temporal_context = self.temporal_dropout(temporal_context)
        
        # Embed categorical features
        role_embeds = self.role_embedding(categorical_features[:, :, 0])  # Role
        position_embeds = self.position_embedding(categorical_features[:, :, 1])  # Position
        side_embeds = self.side_embedding(categorical_features[:, :, 2])  # Side
        direction_embeds = self.direction_embedding(categorical_features[:, :, 3])  # Direction
        
        # Concatenate all features
        node_features = torch.cat([
            node_features_continuous,
            temporal_context,
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
            node_features_continuous,
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
        
        # Decode trajectories with transformer
        mask = output_mask * player_mask.unsqueeze(-1)
        residuals = self.decoder(
            context=player_embeddings,
            role_embed=role_embeds,
            ball_landing=ball_landing,
            mask=mask,
        )
        residuals = residuals * mask.unsqueeze(-1)
        residuals_cumsum = residuals.cumsum(dim=2)
        predictions = last_positions.unsqueeze(2) + residuals_cumsum
        predictions = predictions * mask.unsqueeze(-1) + last_positions.unsqueeze(2) * (1 - mask.unsqueeze(-1))
        
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
        cat_dim = (
            config.role_embedding_dim +
            config.position_embedding_dim +
            8 +
            8
        )
        input_dim = self.num_continuous_features + self.temporal_hidden_dim + cat_dim
        
        self.role_embedding = nn.Embedding(config.num_roles, config.role_embedding_dim)
        self.position_embedding = nn.Embedding(config.num_positions, config.position_embedding_dim)
        self.side_embedding = nn.Embedding(2, 8)
        self.direction_embedding = nn.Embedding(2, 8)
        self.temporal_encoder = nn.LSTM(
            input_size=self.num_continuous_features,
            hidden_size=self.temporal_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.temporal_dropout = nn.Dropout(config.gnn_dropout)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.gnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.gnn_dropout),
            nn.Linear(config.gnn_hidden_dim, config.gnn_output_dim),
            nn.ReLU(),
        )
        
        # Transformer decoder
        self.decoder = TransformerTrajectoryDecoder(
            context_dim=config.gnn_output_dim,
            role_dim=config.role_embedding_dim,
            embed_dim=config.decoder_embed_dim,
            num_steps=config.max_output_frames,
            num_layers=config.decoder_num_layers,
            num_heads=config.decoder_num_heads,
            dropout=config.decoder_dropout,
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
        player_mask = batch.get('player_mask', torch.ones_like(player_roles, dtype=input_features.dtype)).to(input_features.dtype)
        output_mask = output_mask.to(input_features.dtype)
        output_mask = batch['output_mask']
        input_mask = batch['input_mask']
        
        batch_size, num_players, input_frames, _ = input_features.shape
        max_output_frames = output_mask.shape[2]
        
        # Use last frame
        node_features_continuous = input_features[:, :, -1, :]
        last_positions = node_features_continuous[:, :, [FEATURE_INDEX['x'], FEATURE_INDEX['y']]]
        
        # Temporal encoding
        flat_sequences = input_features.view(batch_size * num_players, input_frames, -1)
        flat_mask = input_mask.view(batch_size * num_players, input_frames)
        lengths = flat_mask.sum(dim=1)
        lengths_clamped = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
        packed = pack_padded_sequence(flat_sequences, lengths_clamped.cpu(), batch_first=True, enforce_sorted=False)
        _, (temporal_hidden, _) = self.temporal_encoder(packed)
        temporal_context = temporal_hidden[-1]
        temporal_context[lengths == 0] = 0
        temporal_context = temporal_context.view(batch_size, num_players, self.temporal_hidden_dim)
        temporal_context = self.temporal_dropout(temporal_context)
        
        # Embed categoricals
        role_embeds = self.role_embedding(categorical_features[:, :, 0])
        position_embeds = self.position_embedding(categorical_features[:, :, 1])
        side_embeds = self.side_embedding(categorical_features[:, :, 2])
        direction_embeds = self.direction_embedding(categorical_features[:, :, 3])
        
        # Concatenate
        node_features = torch.cat([
            node_features_continuous,
            temporal_context,
            role_embeds,
            position_embeds,
            side_embeds,
            direction_embeds,
        ], dim=-1)
        
        # Encode
        embeddings = self.encoder(node_features)
        
        # Decode with transformer
        mask = output_mask * player_mask.unsqueeze(-1)
        residuals = self.decoder(
            context=embeddings,
            role_embed=role_embeds,
            ball_landing=ball_landing,
            mask=mask,
        )
        residuals = residuals * mask.unsqueeze(-1)
        residuals_cumsum = residuals.cumsum(dim=2)
        predictions = last_positions.unsqueeze(2) + residuals_cumsum
        predictions = predictions * mask.unsqueeze(-1) + last_positions.unsqueeze(2) * (1 - mask.unsqueeze(-1))
        
        return predictions


