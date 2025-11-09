"""
Transformer-based trajectory decoder predicting positional residuals.
"""

import torch
import torch.nn as nn
from typing import Optional


class TransformerTrajectoryDecoder(nn.Module):
    """
    Decoder that predicts Δx/Δy residuals for each future timestep using a Transformer.
    """

    def __init__(
        self,
        context_dim: int,
        role_dim: int,
        embed_dim: int,
        num_steps: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim

        self.context_proj = nn.Linear(context_dim + role_dim + 2, embed_dim)
        self.query_embed = nn.Parameter(torch.randn(num_steps, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.context_proj.weight)
        if self.context_proj.bias is not None:
            nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        context: torch.Tensor,
        role_embed: torch.Tensor,
        ball_landing: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, players, context_dim] - GNN embeddings.
            role_embed: [batch, players, role_dim] - role embeddings.
            ball_landing: [batch, 2] - ball landing coordinates.
            mask: Optional [batch, players, steps] mask (1 valid, 0 padded).

        Returns:
            residuals: [batch, players, steps, 2]
        """
        batch, players, _ = context.shape
        device = context.device

        ball_expand = ball_landing.unsqueeze(1).expand(-1, players, -1)
        decoder_context = torch.cat([context, role_embed, ball_expand], dim=-1)
        memory = self.context_proj(decoder_context)  # [batch, players, embed_dim]
        memory = memory.reshape(batch * players, 1, self.embed_dim).transpose(0, 1)  # [1, batch*players, embed_dim]

        queries = self.query_embed.unsqueeze(1).expand(-1, batch * players, -1)  # [steps, batch*players, embed_dim]
        decoded = self.transformer(queries, memory)  # [steps, batch*players, embed_dim]
        decoded = decoded.transpose(0, 1)  # [batch*players, steps, embed_dim]
        decoded = self.dropout(decoded)
        residuals = self.output_proj(decoded)  # [batch*players, steps, 2]
        residuals = residuals.view(batch, players, self.num_steps, 2)

        if mask is not None:
            residuals = residuals * mask.unsqueeze(-1)

        return residuals

