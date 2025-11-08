"""
Complete GNN-LSTM model for NFL trajectory prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from src.models.graph_builder import GraphBuilder
from src.models.gat_encoder import GATEncoder
from src.models.lstm_decoder import RoleConditionedLSTMDecoder


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
        # Continuous features: x, y, s, a, dir, o, height, weight, dist_to_ball (9 features)
        self.num_continuous_features = 9
        
        # Categorical features will be embedded
        self.role_embedding = nn.Embedding(config.num_roles, config.role_embedding_dim)
        self.position_embedding = nn.Embedding(config.num_positions, config.position_embedding_dim)
        self.side_embedding = nn.Embedding(2, 8)  # Offense/Defense
        self.direction_embedding = nn.Embedding(2, 8)  # Left/Right
        
        # Total node feature dimension after embedding
        total_feature_dim = (
            self.num_continuous_features +
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
            edge_features=3,  # distance, rel_x, rel_y
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
        
        batch_size, num_players, input_frames, _ = input_features.shape
        _, _, max_output_frames = output_mask.shape
        
        # Get last observed positions (for initial decoder input)
        last_positions = input_features[:, :, -1, :2]  # [batch, players, 2]
        
        # Average input features across time (simple temporal aggregation)
        # Alternatively, could use last frame or LSTM over frames
        # For now: use last frame for node features
        node_features_continuous = input_features[:, :, -1, :]  # [batch, players, 9]
        
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
        
        # Simple feature encoder
        input_dim = 9 + 4 * 16  # continuous + categorical embeddings
        
        self.role_embedding = nn.Embedding(config.num_roles, 16)
        self.position_embedding = nn.Embedding(config.num_positions, 16)
        self.side_embedding = nn.Embedding(2, 16)
        self.direction_embedding = nn.Embedding(2, 16)
        
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
        
        batch_size, num_players, input_frames, _ = input_features.shape
        max_output_frames = output_mask.shape[2]
        
        # Use last frame
        node_features_continuous = input_features[:, :, -1, :]
        last_positions = input_features[:, :, -1, :2]
        
        # Embed categoricals
        role_embeds = self.role_embedding(categorical_features[:, :, 0])
        position_embeds = self.position_embedding(categorical_features[:, :, 1])
        side_embeds = self.side_embedding(categorical_features[:, :, 2])
        direction_embeds = self.direction_embedding(categorical_features[:, :, 3])
        
        # Concatenate
        node_features = torch.cat([
            node_features_continuous,
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


