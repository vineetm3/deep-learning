"""
Graph construction for GNN-based trajectory prediction
"""

import torch
from typing import Tuple, Optional

from src.data.dataset import FEATURE_INDEX

IDX_X = FEATURE_INDEX['x']
IDX_Y = FEATURE_INDEX['y']
IDX_VX = FEATURE_INDEX.get('vx')
IDX_VY = FEATURE_INDEX.get('vy')
IDX_AX = FEATURE_INDEX.get('ax')
IDX_AY = FEATURE_INDEX.get('ay')


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
        self.edge_feature_dim = 9
        self.eps = 1e-6

    def _get_vector(
        self,
        data: torch.Tensor,
        idx_x: Optional[int],
        idx_y: Optional[int],
    ) -> torch.Tensor:
        if (
            idx_x is None or idx_y is None or
            idx_x >= data.size(-1) or idx_y >= data.size(-1)
        ):
            return torch.zeros(data.size(0), 2, device=data.device)
        return torch.stack([data[:, idx_x], data[:, idx_y]], dim=-1)

    def _pairwise_features(
        self,
        src_pos: torch.Tensor,
        tgt_pos: torch.Tensor,
        src_vel: torch.Tensor,
        tgt_vel: torch.Tensor,
        src_acc: torch.Tensor,
        tgt_acc: torch.Tensor,
    ) -> torch.Tensor:
        rel_vec = tgt_pos - src_pos  # [N, 2]
        dist = torch.norm(rel_vec, dim=-1, keepdim=True).clamp(min=self.eps)
        rel_dir = rel_vec / dist

        rel_vel = tgt_vel - src_vel
        speed_along = (rel_vel * rel_dir).sum(dim=-1, keepdim=True)
        speed_mag = torch.norm(rel_vel, dim=-1, keepdim=True)

        rel_acc = tgt_acc - src_acc
        acc_along = (rel_acc * rel_dir).sum(dim=-1, keepdim=True)
        acc_mag = torch.norm(rel_acc, dim=-1, keepdim=True)

        src_speed = torch.norm(src_vel, dim=-1, keepdim=True)
        heading_alignment = ( (src_vel / (src_speed + self.eps)) * rel_dir ).sum(dim=-1, keepdim=True)
        heading_alignment = torch.clamp(heading_alignment, -1.0, 1.0)

        time_to_target = dist / (speed_mag + self.eps)

        features = torch.cat([
            dist,
            rel_dir,
            speed_along,
            speed_mag,
            acc_along,
            acc_mag,
            heading_alignment,
            time_to_target,
        ], dim=-1)

        return features
    
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
        
        all_edge_indices: list[torch.Tensor] = []
        all_edge_features: list[torch.Tensor] = []
        
        for b in range(batch_size):
            num_real_players = int(player_mask[b].sum().item())
            if num_real_players == 0:
                continue

            node_offset = b * (num_players + 1)

            player_continuous = continuous_features[b, :num_real_players]
            player_positions = player_continuous[:, [IDX_X, IDX_Y]]
            player_velocities = self._get_vector(player_continuous, IDX_VX, IDX_VY)
            player_accelerations = self._get_vector(player_continuous, IDX_AX, IDX_AY)

            ball_position = ball_landing[b].unsqueeze(0)
            ball_repeated = ball_position.expand(num_real_players, -1)
            ball_zero = torch.zeros_like(ball_repeated)

            edge_chunks = []
            feature_chunks = []

            player_indices = torch.arange(num_real_players, device=device, dtype=torch.long)
            ball_node_idx = torch.full((num_real_players,), num_players, device=device, dtype=torch.long)

            # Player -> Ball
            pb_features = self._pairwise_features(
                player_positions,
                ball_repeated,
                player_velocities,
                ball_zero,
                player_accelerations,
                ball_zero,
            )
            edge_pb = torch.stack([player_indices, ball_node_idx], dim=0)
            edge_chunks.append(edge_pb)
            feature_chunks.append(pb_features)

            # Ball -> Player
            bp_features = self._pairwise_features(
                ball_repeated,
                player_positions,
                ball_zero,
                player_velocities,
                ball_zero,
                player_accelerations,
            )
            edge_bp = torch.stack([ball_node_idx, player_indices], dim=0)
            edge_chunks.append(edge_bp)
            feature_chunks.append(bp_features)

            # K-Nearest Neighbor edges with symmetry
            if num_real_players > 1 and self.k_neighbors > 0:
                pairwise_dist = torch.cdist(player_positions, player_positions, p=2)
                pairwise_dist.fill_diagonal_(float('inf'))
                k = min(self.k_neighbors, num_real_players - 1)
                knn_indices = pairwise_dist.topk(k, largest=False).indices  # [N, k]

                src_idx = player_indices.unsqueeze(1).expand(-1, k).reshape(-1)
                tgt_idx = knn_indices.reshape(-1)

                src_pos = player_positions[src_idx]
                tgt_pos = player_positions[tgt_idx]
                src_vel = player_velocities[src_idx]
                tgt_vel = player_velocities[tgt_idx]
                src_acc = player_accelerations[src_idx]
                tgt_acc = player_accelerations[tgt_idx]

                features_knn = self._pairwise_features(src_pos, tgt_pos, src_vel, tgt_vel, src_acc, tgt_acc)
                edges_knn = torch.stack([src_idx, tgt_idx], dim=0)
                edge_chunks.append(edges_knn)
                feature_chunks.append(features_knn)

                features_knn_rev = self._pairwise_features(tgt_pos, src_pos, tgt_vel, src_vel, tgt_acc, src_acc)
                edges_knn_rev = torch.stack([tgt_idx, src_idx], dim=0)
                edge_chunks.append(edges_knn_rev)
                feature_chunks.append(features_knn_rev)

            # Role-specific coverage edges
            roles = player_roles[b, :num_real_players]
            targeted_candidates = (roles == 0).nonzero(as_tuple=False).squeeze(-1)
            coverage_indices = (roles == 1).nonzero(as_tuple=False).squeeze(-1)

            if targeted_candidates.numel() > 0 and coverage_indices.numel() > 0:
                targeted_idx = targeted_candidates[0]
                coverage_idx = coverage_indices
                target_rep = torch.full_like(coverage_idx, targeted_idx)

                cov_features = self._pairwise_features(
                    player_positions[coverage_idx],
                    player_positions[target_rep],
                    player_velocities[coverage_idx],
                    player_velocities[target_rep],
                    player_accelerations[coverage_idx],
                    player_accelerations[target_rep],
                )
                cov_edges = torch.stack([coverage_idx, target_rep], dim=0)
                edge_chunks.append(cov_edges)
                feature_chunks.append(cov_features)

                rec_features = self._pairwise_features(
                    player_positions[target_rep],
                    player_positions[coverage_idx],
                    player_velocities[target_rep],
                    player_velocities[coverage_idx],
                    player_accelerations[target_rep],
                    player_accelerations[coverage_idx],
                )
                rec_edges = torch.stack([target_rep, coverage_idx], dim=0)
                edge_chunks.append(rec_edges)
                feature_chunks.append(rec_features)

            if edge_chunks:
                local_edges = torch.cat(edge_chunks, dim=1)
                local_features = torch.cat(feature_chunks, dim=0)
                all_edge_indices.append(local_edges + node_offset)
                all_edge_features.append(local_features)
        
        # Combine all edges
        if len(all_edge_indices) > 0:
            edge_index = torch.cat(all_edge_indices, dim=1)  # [2, total_edges]
            edge_features = torch.cat(all_edge_features, dim=0)  # [total_edges, edge_dim]
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            edge_features = torch.zeros((0, self.edge_feature_dim), device=device)
        
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


