"""
Graph Attention Network (GAT) encoder for spatial relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GATLayer(nn.Module):
    """
    Graph Attention Layer
    
    Implements attention mechanism to weight neighbor node importance
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True,
        edge_features: int = 0,
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension (per head)
            num_heads: Number of attention heads
            dropout: Dropout rate
            concat: Whether to concatenate multi-head outputs (True) or average (False)
            edge_features: Dimension of edge features (if > 0, will be used in attention)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.edge_features = edge_features
        
        # Linear transformations for source and target nodes
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        
        # Attention mechanism parameters
        self.a_src = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        self.a_tgt = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        
        # Edge feature attention (if edge features provided)
        if edge_features > 0:
            self.W_edge = nn.Linear(edge_features, num_heads * out_features)
            self.a_edge = nn.Parameter(torch.zeros(size=(1, num_heads, out_features)))
        
        # Bias
        if concat:
            self.bias = nn.Parameter(torch.zeros(num_heads * out_features))
        else:
            self.bias = nn.Parameter(torch.zeros(out_features))
        
        # Initialization
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_tgt)
        if edge_features > 0:
            nn.init.xavier_uniform_(self.W_edge.weight)
            nn.init.xavier_uniform_(self.a_edge)
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            return_attention: Whether to return attention weights
        
        Returns:
            Updated node features and optionally attention weights
        """
        num_nodes = x.size(0)
        
        # Linear transformation
        h = self.W(x)  # [num_nodes, num_heads * out_features]
        h = h.view(num_nodes, self.num_heads, self.out_features)  # [num_nodes, num_heads, out_features]
        
        # Get source and target nodes
        edge_src = edge_index[0]
        edge_tgt = edge_index[1]
        
        # Compute attention logits
        # e_ij = LeakyReLU(a_src^T h_i + a_tgt^T h_j + a_edge^T e_ij)
        h_src = h[edge_src]  # [num_edges, num_heads, out_features]
        h_tgt = h[edge_tgt]  # [num_edges, num_heads, out_features]
        
        # Attention scores
        e_src = (h_src * self.a_src).sum(dim=-1)  # [num_edges, num_heads]
        e_tgt = (h_tgt * self.a_tgt).sum(dim=-1)  # [num_edges, num_heads]
        e = e_src + e_tgt
        
        # Add edge features to attention if provided
        if edge_attr is not None and self.edge_features > 0:
            h_edge = self.W_edge(edge_attr)  # [num_edges, num_heads * out_features]
            h_edge = h_edge.view(-1, self.num_heads, self.out_features)
            e_edge = (h_edge * self.a_edge).sum(dim=-1)  # [num_edges, num_heads]
            e = e + e_edge
        
        e = self.leakyrelu(e)  # [num_edges, num_heads]
        
        # Normalize attention coefficients using softmax
        # For each target node, normalize over all incoming edges
        attention = torch.zeros(edge_index[1].max().item() + 1, edge_index.size(1), self.num_heads, device=x.device)
        attention = attention.index_add_(0, edge_tgt, e)
        
        # Compute softmax per target node
        e_exp = torch.exp(e - e.max())  # Numerical stability
        
        # Sum of exponentials for each target node
        e_exp_sum = torch.zeros(num_nodes, self.num_heads, device=x.device)
        e_exp_sum = e_exp_sum.index_add_(0, edge_tgt, e_exp)
        
        # Normalize
        alpha = e_exp / (e_exp_sum[edge_tgt] + 1e-16)  # [num_edges, num_heads]
        alpha = self.dropout_layer(alpha)
        
        # Aggregate neighbor features
        # h'_i = sum_j alpha_ij * h_j
        h_prime = torch.zeros(num_nodes, self.num_heads, self.out_features, device=x.device)
        
        # Weighted sum
        weighted_h = alpha.unsqueeze(-1) * h_tgt  # [num_edges, num_heads, out_features]
        h_prime = h_prime.index_add_(0, edge_tgt, weighted_h)
        
        # Concatenate or average multi-head outputs
        if self.concat:
            h_prime = h_prime.view(num_nodes, self.num_heads * self.out_features)
        else:
            h_prime = h_prime.mean(dim=1)
        
        # Add bias
        h_prime = h_prime + self.bias
        
        if return_attention:
            return h_prime, alpha
        else:
            return h_prime, None


class GATEncoder(nn.Module):
    """
    Multi-layer GAT encoder
    
    Processes graph-structured data through multiple attention layers
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_features: int = 3,
    ):
        """
        Args:
            in_features: Input node feature dimension
            hidden_features: Hidden layer dimension
            out_features: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_features: Edge feature dimension
        """
        super().__init__()
        
        self.num_layers = num_layers
        
        # Build layers
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(
            GATLayer(
                in_features,
                hidden_features,
                num_heads=num_heads,
                dropout=dropout,
                concat=True,
                edge_features=edge_features,
            )
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(
                GATLayer(
                    hidden_features * num_heads,
                    hidden_features,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_features=edge_features,
                )
            )
        
        # Output layer
        if num_layers > 1:
            self.layers.append(
                GATLayer(
                    hidden_features * num_heads,
                    out_features,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=False,  # Average heads for final layer
                    edge_features=edge_features,
                )
            )
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_features * num_heads if i < num_layers - 1 else out_features)
            for i in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through all GAT layers
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
        
        Returns:
            Node embeddings [num_nodes, out_features]
        """
        h = x
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            h_new, _ = layer(h, edge_index, edge_attr)
            h_new = norm(h_new)
            
            # ReLU activation (except last layer)
            if i < self.num_layers - 1:
                h_new = F.elu(h_new)
            
            h = h_new
        
        return h


class TemporalGATEncoder(nn.Module):
    """
    Temporal GAT encoder that processes sequences of graphs
    
    For each timestep, builds a graph and processes it with GAT,
    then aggregates temporal information
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        edge_features: int = 3,
    ):
        """
        Args:
            node_feature_dim: Dimension of node features per timestep
            hidden_dim: Hidden dimension for GAT
            output_dim: Output embedding dimension
            num_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            edge_features: Edge feature dimension
        """
        super().__init__()
        
        self.gat = GATEncoder(
            node_feature_dim,
            hidden_dim,
            output_dim,
            num_layers,
            num_heads,
            dropout,
            edge_features,
        )
        
        # Temporal aggregation (LSTM over time)
        self.temporal_lstm = nn.LSTM(
            output_dim,
            output_dim,
            num_layers=1,
            batch_first=True,
        )
    
    def forward(
        self,
        node_features_seq: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process temporal sequence of graphs
        
        Args:
            node_features_seq: [batch, num_nodes, seq_len, feature_dim]
            edge_index: [2, num_edges] (same graph structure for all timesteps)
            edge_attr: [num_edges, edge_features]
        
        Returns:
            Final node embeddings [batch, num_nodes, output_dim]
        """
        batch_size, num_nodes, seq_len, feature_dim = node_features_seq.shape
        
        # Process each timestep
        all_embeddings = []
        
        for t in range(seq_len):
            # Get features for this timestep
            x_t = node_features_seq[:, :, t, :].reshape(-1, feature_dim)  # [batch*num_nodes, feature_dim]
            
            # Process with GAT
            h_t = self.gat(x_t, edge_index, edge_attr)  # [batch*num_nodes, output_dim]
            h_t = h_t.view(batch_size, num_nodes, -1)  # [batch, num_nodes, output_dim]
            
            all_embeddings.append(h_t)
        
        # Stack temporal dimension
        embeddings_seq = torch.stack(all_embeddings, dim=2)  # [batch, num_nodes, seq_len, output_dim]
        
        # Apply LSTM to capture temporal patterns (per node)
        batch_size, num_nodes, seq_len, embed_dim = embeddings_seq.shape
        
        # Reshape for LSTM
        embeddings_seq = embeddings_seq.view(batch_size * num_nodes, seq_len, embed_dim)
        
        # LSTM
        _, (h_n, _) = self.temporal_lstm(embeddings_seq)  # h_n: [1, batch*num_nodes, output_dim]
        
        # Reshape back
        final_embeddings = h_n.squeeze(0).view(batch_size, num_nodes, embed_dim)
        
        return final_embeddings


