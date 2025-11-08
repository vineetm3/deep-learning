"""
Graph construction for GNN-based trajectory prediction
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from src.data.dataset import FEATURE_INDEX

IDX_X = FEATURE_INDEX['x']
IDX_Y = FEATURE_INDEX['y']
IDX_VX = FEATURE_INDEX.get('vx')
IDX_VY = FEATURE_INDEX.get('vy')
IDX_AX = FEATURE_INDEX.get('ax')
IDX_AY = FEATURE_INDEX.get('ay')
IDX_BALL_SIN = FEATURE_INDEX.get('ball_angle_sin')
IDX_BALL_COS = FEATURE_INDEX.get('ball_angle_cos')


class GraphBuilder:
    """
    Build graphs for multi-agent trajectory prediction
    
    Creates a graph with:
    - Player nodes (22 per play, actual number varies)
    - Ball landing node (1 per play)
    - Edges: player-to-ball, k-nearest neighbors, role-specific
    """
    
    def __init__(self, k_neighbors: int = 5):
        """
        Args:
            k_neighbors: Number of nearest neighbors for each player
        """
        self.k_neighbors = k_neighbors
    
    def build_graph(
        self,
        node_features: torch.Tensor,
        ball_landing: torch.Tensor,
        player_roles: torch.Tensor,
        player_mask: torch.Tensor,
        continuous_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build graph structure for a batch of plays
        
        Args:
            node_features: [batch, num_players, feature_dim] - player features
            ball_landing: [batch, 2] - ball landing position (x, y)
            player_roles: [batch, num_players] - role indices
            player_mask: [batch, num_players] - 1 for real players, 0 for padding
        
        Returns:
            Tuple of:
                - edge_index: [2, num_edges] - edge connectivity
                - edge_features: [num_edges, edge_feature_dim] - edge features
                - node_features_with_ball: [batch, num_players+1, feature_dim] - includes ball node
        """
        batch_size, num_players, feature_dim = node_features.shape
        device = node_features.device
        
        # We'll build a single large graph with disconnected components (one per batch item)
        # Each component has num_players + 1 nodes (players + ball)
        
        all_edge_indices = []
        all_edge_features = []
        
        for b in range(batch_size):
            # Get actual number of players (non-padded)
            num_real_players = int(player_mask[b].sum().item())
            
            if num_real_players == 0:
                continue
            
            # Node offset for this batch item
            node_offset = b * (num_players + 1)
            
            player_continuous = continuous_features[b, :num_real_players]  # [num_real_players, feat]
            player_positions = player_continuous[:, [IDX_X, IDX_Y]]
            ball_pos = ball_landing[b]  # [2]

            # Extract velocity and acceleration components if available
            def get_feature(idx: Optional[int]) -> torch.Tensor:
                if idx is None or idx >= player_continuous.size(-1):
                    return torch.zeros_like(player_positions[:, 0])
                return player_continuous[:, idx]

            vx = get_feature(IDX_VX)
            vy = get_feature(IDX_VY)
            ax = get_feature(IDX_AX)
            ay = get_feature(IDX_AY)
            ball_angle_sin = get_feature(IDX_BALL_SIN)
            ball_angle_cos = get_feature(IDX_BALL_COS)
            
            # Edge list for this play
            edge_index = []
            edge_features = []
            
            # 1. Player-to-ball edges (bidirectional)
            ball_node_idx = num_players  # Ball is last node
            for i in range(num_real_players):
                # Player to ball
                edge_index.append([i, ball_node_idx])
                # Ball to player
                edge_index.append([ball_node_idx, i])
                
                # Edge features: distance, relative position, velocity and acceleration towards ball
                rel_pos = player_positions[i] - ball_pos
                dist = torch.norm(rel_pos).item()

                rel_features = torch.tensor([
                    dist,
                    rel_pos[0].item(),
                    rel_pos[1].item(),
                    vx[i].item(),
                    vy[i].item(),
                    ax[i].item(),
                    ay[i].item(),
                    ball_angle_sin[i].item(),
                    ball_angle_cos[i].item(),
                ], device=device)
                edge_features.append(rel_features)
                edge_features.append(rel_features)
            
            # 2. K-nearest neighbor edges
            if num_real_players > 1:
                # Compute pairwise distances
                dist_matrix = torch.cdist(player_positions, player_positions)  # [num_real_players, num_real_players]
                
                # For each player, find k nearest neighbors
                for i in range(num_real_players):
                    # Get distances to all other players
                    distances = dist_matrix[i].clone()
                    distances[i] = float('inf')  # Exclude self
                    
                    # Get k nearest
                    k = min(self.k_neighbors, num_real_players - 1)
                    _, nearest_indices = torch.topk(distances, k, largest=False)
                    
                    # Add edges
                    for j in nearest_indices:
                        j = j.item()
                        edge_index.append([i, j])
                        
                        # Edge features
                        dist = distances[j].item()
                        rel_pos = player_positions[j] - player_positions[i]
                        rel_vel = torch.tensor([vx[j] - vx[i], vy[j] - vy[i]], device=device)
                        rel_acc = torch.tensor([ax[j] - ax[i], ay[j] - ay[i]], device=device)
                        edge_feat = torch.tensor([
                            dist,
                            rel_pos[0].item(),
                            rel_pos[1].item(),
                            rel_vel[0].item(),
                            rel_vel[1].item(),
                            rel_acc[0].item(),
                            rel_acc[1].item(),
                            0.0,
                            0.0,
                        ], device=device)
                        edge_features.append(edge_feat)
            
            # 3. Role-specific edges (defensive coverage to targeted receiver)
            targeted_receiver_idx = None
            coverage_indices = []
            
            for i in range(num_real_players):
                role = player_roles[b, i].item()
                if role == 0:  # Targeted Receiver
                    targeted_receiver_idx = i
                elif role == 1:  # Defensive Coverage
                    coverage_indices.append(i)
            
            # Add bidirectional edges between coverage and targeted receiver
            if targeted_receiver_idx is not None:
                for coverage_idx in coverage_indices:
                    # Coverage to receiver
                    edge_index.append([coverage_idx, targeted_receiver_idx])
                    # Receiver to coverage
                    edge_index.append([targeted_receiver_idx, coverage_idx])
                    
                    # Edge features
                    rel_pos = player_positions[targeted_receiver_idx] - player_positions[coverage_idx]
                    dist = torch.norm(rel_pos).item()
                    rel_vel = torch.tensor([
                        vx[targeted_receiver_idx] - vx[coverage_idx],
                        vy[targeted_receiver_idx] - vy[coverage_idx],
                    ], device=device)
                    rel_acc = torch.tensor([
                        ax[targeted_receiver_idx] - ax[coverage_idx],
                        ay[targeted_receiver_idx] - ay[coverage_idx],
                    ], device=device)
                    edge_feat = torch.tensor([
                        dist,
                        rel_pos[0].item(),
                        rel_pos[1].item(),
                        rel_vel[0].item(),
                        rel_vel[1].item(),
                        rel_acc[0].item(),
                        rel_acc[1].item(),
                        0.0,
                        0.0,
                    ], device=device)
                    edge_features.append(edge_feat)
                    edge_features.append(edge_feat)
            
            # Offset edge indices by batch
            if len(edge_index) > 0:
                edge_index_tensor = torch.tensor(edge_index, device=device).t() + node_offset
                all_edge_indices.append(edge_index_tensor)
                all_edge_features.extend(edge_features)
        
        # Combine all edges
        if len(all_edge_indices) > 0:
            edge_index = torch.cat(all_edge_indices, dim=1)  # [2, total_edges]
            edge_features = torch.stack(all_edge_features)  # [total_edges, edge_dim]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_features = torch.zeros((0, 9), device=device)
        
        # Create ball node features (use ball landing position + zeros for other features)
        ball_features = torch.zeros(batch_size, 1, feature_dim, device=device)
        ball_features[:, 0, :2] = ball_landing  # Set position to ball landing
        
        # Concatenate player and ball nodes
        node_features_with_ball = torch.cat([node_features, ball_features], dim=1)
        
        return edge_index, edge_features, node_features_with_ball
    
    def build_graph_pyg(
        self,
        node_features: torch.Tensor,
        ball_landing: torch.Tensor,
        player_roles: torch.Tensor,
        player_mask: torch.Tensor,
        continuous_features: torch.Tensor,
    ):
        """
        Build graph in PyTorch Geometric format
        
        Args:
            node_features: [batch, num_players, feature_dim]
            ball_landing: [batch, 2]
            player_roles: [batch, num_players]
            player_mask: [batch, num_players]
        
        Returns:
            PyG Data object or Batch
        """
        from torch_geometric.data import Data, Batch
        
        batch_size, num_players, feature_dim = node_features.shape
        
        data_list = []
        
        for b in range(batch_size):
            # Get actual number of players
            num_real_players = int(player_mask[b].sum().item())
            
            if num_real_players == 0:
                continue
            
            # Extract features for this play
            player_feats = node_features[b, :num_real_players]  # [num_real_players, feature_dim]
            ball_pos = ball_landing[b]  # [2]
            
            # Create ball node features
            ball_feat = torch.zeros(1, feature_dim, device=node_features.device)
            ball_feat[0, :2] = ball_pos
            
            # Combine nodes
            x = torch.cat([player_feats, ball_feat], dim=0)  # [num_real_players+1, feature_dim]
            
            # Build edges for this play
            edge_index, edge_attr, _ = self.build_graph(
                node_features[b:b+1, :num_real_players],
                ball_landing[b:b+1],
                player_roles[b:b+1, :num_real_players],
                player_mask[b:b+1, :num_real_players],
                continuous_features[b:b+1, :num_real_players],
            )
            # Remove batch offset (since build_graph assumes full batch)
            edge_index = edge_index - edge_index.min()
            
            # Create PyG Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_nodes=num_real_players + 1,
            )
            data_list.append(data)
        
        # Batch the graphs
        if len(data_list) > 0:
            return Batch.from_data_list(data_list)
        else:
            return None


def compute_edge_features(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    velocities: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute edge features from node positions
    
    Args:
        positions: [num_nodes, 2] - node positions
        edge_index: [2, num_edges] - edge connectivity
        velocities: [num_nodes, 2] - node velocities (optional)
    
    Returns:
        edge_features: [num_edges, feature_dim]
    """
    # Source and target nodes
    src_nodes = edge_index[0]
    tgt_nodes = edge_index[1]
    
    # Relative positions
    rel_pos = positions[tgt_nodes] - positions[src_nodes]  # [num_edges, 2]
    
    # Distances
    distances = torch.norm(rel_pos, dim=1, keepdim=True)  # [num_edges, 1]
    
    # Combine features
    edge_features = torch.cat([distances, rel_pos], dim=1)  # [num_edges, 3]
    
    # Add relative velocities if provided
    if velocities is not None:
        rel_vel = velocities[tgt_nodes] - velocities[src_nodes]
        edge_features = torch.cat([edge_features, rel_vel], dim=1)  # [num_edges, 5]
    
    return edge_features


