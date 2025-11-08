"""
LSTM decoder with role conditioning for trajectory generation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class RoleConditionedLSTMDecoder(nn.Module):
    """
    LSTM decoder that generates trajectories conditioned on:
    - GNN encoding of the scene
    - Player role
    - Ball landing location
    
    Uses scheduled sampling during training
    """
    
    def __init__(
        self,
        context_dim: int,
        role_embedding_dim: int,
        num_roles: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        """
        Args:
            context_dim: Dimension of context vector from GNN encoder
            role_embedding_dim: Dimension of role embeddings
            num_roles: Number of unique roles
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_dim: Output dimension (2 for x, y)
        """
        super().__init__()
        
        self.context_dim = context_dim
        self.role_embedding_dim = role_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Role embeddings
        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        
        # Input dimension: previous position (2) + ball landing (2) + context + role embedding
        input_dim = output_dim + 2 + context_dim + role_embedding_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Initial position encoder
        self.initial_pos_encoder = nn.Linear(output_dim, hidden_dim)
    
    def forward(
        self,
        context: torch.Tensor,
        initial_position: torch.Tensor,
        ball_landing: torch.Tensor,
        roles: torch.Tensor,
        max_steps: int,
        ground_truth: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate trajectory predictions
        
        Args:
            context: Context vector from encoder [batch, num_players, context_dim]
            initial_position: Initial position [batch, num_players, 2]
            ball_landing: Ball landing position [batch, 2]
            roles: Player roles [batch, num_players]
            max_steps: Maximum number of steps to generate
            ground_truth: Ground truth trajectory [batch, num_players, max_steps, 2] (for training)
            teacher_forcing_ratio: Probability of using ground truth instead of prediction
        
        Returns:
            predictions: [batch, num_players, max_steps, 2]
        """
        batch_size, num_players, _ = context.shape
        device = context.device
        
        # Expand ball landing for all players
        ball_landing_expanded = ball_landing.unsqueeze(1).expand(-1, num_players, -1)  # [batch, num_players, 2]
        
        # Get role embeddings
        role_embeds = self.role_embedding(roles)  # [batch, num_players, role_embedding_dim]
        
        # Initialize LSTM hidden state
        h0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
        
        # Initialize with initial position
        init_encoding = self.initial_pos_encoder(initial_position.view(-1, self.output_dim))
        h0[0] = init_encoding
        
        hidden = (h0, c0)
        
        # Reshape for processing all players in parallel
        # [batch * num_players, feature_dim]
        context_flat = context.reshape(-1, self.context_dim)
        role_embeds_flat = role_embeds.reshape(-1, self.role_embedding_dim)
        ball_landing_flat = ball_landing_expanded.reshape(-1, 2)
        
        # Auto-regressive generation
        predictions = []
        current_pos = initial_position.reshape(-1, self.output_dim)  # [batch * num_players, 2]
        
        use_teacher_forcing = ground_truth is not None and teacher_forcing_ratio > 0
        
        for t in range(max_steps):
            # Prepare input: [prev_pos, ball_landing, context, role_embedding]
            lstm_input = torch.cat([
                current_pos,
                ball_landing_flat,
                context_flat,
                role_embeds_flat,
            ], dim=1)  # [batch * num_players, input_dim]
            
            # Add time dimension
            lstm_input = lstm_input.unsqueeze(1)  # [batch * num_players, 1, input_dim]
            
            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)  # lstm_out: [batch * num_players, 1, hidden_dim]
            
            # Predict next position
            next_pos = self.output_fc(lstm_out.squeeze(1))  # [batch * num_players, 2]
            
            predictions.append(next_pos)
            
            # Scheduled sampling: decide whether to use prediction or ground truth
            if use_teacher_forcing and t < ground_truth.shape[2] - 1:
                # Randomly use ground truth or prediction
                use_gt = torch.rand(1).item() < teacher_forcing_ratio
                
                if use_gt:
                    # Use ground truth for next step
                    gt_flat = ground_truth[:, :, t + 1, :].reshape(-1, self.output_dim)
                    current_pos = gt_flat
                else:
                    current_pos = next_pos
            else:
                # Use prediction
                current_pos = next_pos
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # [batch * num_players, max_steps, 2]
        
        # Reshape back to batch format
        predictions = predictions.view(batch_size, num_players, max_steps, self.output_dim)
        
        return predictions
    
    def predict_single_step(
        self,
        context: torch.Tensor,
        current_position: torch.Tensor,
        ball_landing: torch.Tensor,
        roles: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict a single step (useful for inference)
        
        Args:
            context: [batch, num_players, context_dim]
            current_position: [batch, num_players, 2]
            ball_landing: [batch, 2]
            roles: [batch, num_players]
            hidden: LSTM hidden state tuple (h, c)
        
        Returns:
            next_position: [batch, num_players, 2]
            hidden: Updated LSTM hidden state
        """
        batch_size, num_players, _ = context.shape
        device = context.device
        
        # Expand ball landing
        ball_landing_expanded = ball_landing.unsqueeze(1).expand(-1, num_players, -1)
        
        # Get role embeddings
        role_embeds = self.role_embedding(roles)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
            c0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
            
            init_encoding = self.initial_pos_encoder(current_position.view(-1, self.output_dim))
            h0[0] = init_encoding
            
            hidden = (h0, c0)
        
        # Flatten
        context_flat = context.reshape(-1, self.context_dim)
        role_embeds_flat = role_embeds.reshape(-1, self.role_embedding_dim)
        ball_landing_flat = ball_landing_expanded.reshape(-1, 2)
        current_pos_flat = current_position.reshape(-1, self.output_dim)
        
        # Prepare input
        lstm_input = torch.cat([
            current_pos_flat,
            ball_landing_flat,
            context_flat,
            role_embeds_flat,
        ], dim=1).unsqueeze(1)  # [batch * num_players, 1, input_dim]
        
        # LSTM step
        lstm_out, hidden = self.lstm(lstm_input, hidden)
        
        # Predict
        next_pos = self.output_fc(lstm_out.squeeze(1))  # [batch * num_players, 2]
        
        # Reshape
        next_pos = next_pos.view(batch_size, num_players, self.output_dim)
        
        return next_pos, hidden


class AttentionLSTMDecoder(nn.Module):
    """
    LSTM decoder with attention over the input sequence
    
    Can attend to different timesteps of the input trajectory
    """
    
    def __init__(
        self,
        context_dim: int,
        role_embedding_dim: int,
        num_roles: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 2,
    ):
        """Similar to RoleConditionedLSTMDecoder but with attention"""
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Role embeddings
        self.role_embedding = nn.Embedding(num_roles, role_embedding_dim)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Input dimension
        input_dim = output_dim + 2 + role_embedding_dim
        
        # LSTM
        self.lstm = nn.LSTM(
            input_dim + context_dim,  # Add attended context
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # Output projection
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Context projection for attention
        self.context_proj = nn.Linear(context_dim, hidden_dim)
    
    def forward(
        self,
        context_sequence: torch.Tensor,
        initial_position: torch.Tensor,
        ball_landing: torch.Tensor,
        roles: torch.Tensor,
        max_steps: int,
        ground_truth: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Generate trajectory with attention over context sequence
        
        Args:
            context_sequence: [batch, num_players, seq_len, context_dim]
            initial_position: [batch, num_players, 2]
            ball_landing: [batch, 2]
            roles: [batch, num_players]
            max_steps: Number of steps to generate
            ground_truth: [batch, num_players, max_steps, 2]
            teacher_forcing_ratio: Teacher forcing probability
        
        Returns:
            predictions: [batch, num_players, max_steps, 2]
        """
        batch_size, num_players, seq_len, context_dim = context_sequence.shape
        device = context_sequence.device
        
        # Project context for attention
        context_seq_proj = self.context_proj(context_sequence.reshape(-1, seq_len, context_dim))
        context_seq_proj = context_seq_proj.view(batch_size * num_players, seq_len, self.hidden_dim)
        
        # Get role embeddings
        role_embeds = self.role_embedding(roles).reshape(-1, self.role_embedding_dim)
        
        # Expand ball landing
        ball_landing_expanded = ball_landing.unsqueeze(1).expand(-1, num_players, -1).reshape(-1, 2)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size * num_players, self.hidden_dim, device=device)
        hidden = (h0, c0)
        
        # Auto-regressive generation
        predictions = []
        current_pos = initial_position.reshape(-1, self.output_dim)
        
        use_teacher_forcing = ground_truth is not None and teacher_forcing_ratio > 0
        
        for t in range(max_steps):
            # Attend to context sequence
            # Query: current decoder state (use current pos as simple query)
            query = current_pos.unsqueeze(1)  # [batch * num_players, 1, 2]
            query_proj = self.context_proj(
                torch.cat([query, torch.zeros_like(query).expand(-1, -1, context_dim - 2)], dim=-1)
            )
            
            attended_context, _ = self.attention(
                query_proj,
                context_seq_proj,
                context_seq_proj,
            )  # [batch * num_players, 1, hidden_dim]
            
            # Prepare LSTM input
            lstm_input = torch.cat([
                current_pos,
                ball_landing_expanded,
                role_embeds,
            ], dim=1).unsqueeze(1)  # [batch * num_players, 1, input_dim]
            
            # Concatenate with attended context
            lstm_input = torch.cat([lstm_input, attended_context], dim=-1)
            
            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            # Predict
            next_pos = self.output_fc(lstm_out.squeeze(1))
            predictions.append(next_pos)
            
            # Scheduled sampling
            if use_teacher_forcing and t < ground_truth.shape[2] - 1:
                use_gt = torch.rand(1).item() < teacher_forcing_ratio
                if use_gt:
                    current_pos = ground_truth[:, :, t + 1, :].reshape(-1, self.output_dim)
                else:
                    current_pos = next_pos
            else:
                current_pos = next_pos
        
        # Stack and reshape
        predictions = torch.stack(predictions, dim=1)
        predictions = predictions.view(batch_size, num_players, max_steps, self.output_dim)
        
        return predictions


