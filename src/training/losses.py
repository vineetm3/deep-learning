"""
Loss functions for trajectory prediction
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class WeightedMSELoss(nn.Module):
    """
    MSE loss with role-based weighting
    
    Gives higher weight to predictions for targeted receivers and defensive coverage
    """
    
    def __init__(self, role_weights: Dict[str, float]):
        """
        Args:
            role_weights: Dictionary mapping role names to weights
        """
        super().__init__()
        
        # Convert role names to indices and weights
        # Assuming: 0=Targeted Receiver, 1=Defensive Coverage, 2=Other Route Runner, 3=Passer
        self.role_weight_tensor = torch.ones(4)
        
        role_to_idx = {
            'Targeted Receiver': 0,
            'Defensive Coverage': 1,
            'Other Route Runner': 2,
            'Passer': 3,
        }
        
        for role_name, weight in role_weights.items():
            if role_name in role_to_idx:
                idx = role_to_idx[role_name]
                self.role_weight_tensor[idx] = weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        roles: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            roles: [batch, players] - role indices
            mask: [batch, players, frames] - validity mask
        
        Returns:
            Scalar loss
        """
        # Move role weights to same device
        self.role_weight_tensor = self.role_weight_tensor.to(predictions.device)
        
        # Compute squared errors
        squared_errors = (predictions - targets) ** 2  # [batch, players, frames, 2]
        squared_errors = squared_errors.sum(dim=-1)  # [batch, players, frames]
        
        # Get role weights for each player
        role_weights = self.role_weight_tensor[roles]  # [batch, players]
        role_weights = role_weights.unsqueeze(-1)  # [batch, players, 1]
        
        # Apply role weights
        weighted_errors = squared_errors * role_weights
        
        # Apply validity mask
        if mask is not None:
            weighted_errors = weighted_errors * mask
            num_valid = mask.sum()
        else:
            num_valid = squared_errors.numel()
        
        # Compute mean
        if num_valid > 0:
            loss = weighted_errors.sum() / num_valid
        else:
            loss = torch.tensor(0.0, device=predictions.device)
        
        return loss


class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction
    
    Can include position loss, velocity loss, and collision avoidance
    """
    
    def __init__(
        self,
        role_weights: Dict[str, float],
        velocity_weight: float = 0.0,
        collision_weight: float = 0.0,
    ):
        """
        Args:
            role_weights: Weights for different player roles
            velocity_weight: Weight for velocity consistency loss
            collision_weight: Weight for collision avoidance loss
        """
        super().__init__()
        
        self.position_loss = WeightedMSELoss(role_weights)
        self.velocity_weight = velocity_weight
        self.collision_weight = collision_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        roles: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined loss
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            roles: [batch, players]
            mask: [batch, players, frames]
        
        Returns:
            Scalar loss
        """
        # Position loss
        loss = self.position_loss(predictions, targets, roles, mask)
        
        # Velocity consistency loss (optional)
        if self.velocity_weight > 0:
            # Penalize large changes in velocity
            pred_velocities = predictions[:, :, 1:] - predictions[:, :, :-1]
            velocity_changes = pred_velocities[:, :, 1:] - pred_velocities[:, :, :-1]
            velocity_loss = (velocity_changes ** 2).mean()
            loss = loss + self.velocity_weight * velocity_loss
        
        # Collision avoidance loss (optional)
        if self.collision_weight > 0:
            # Penalize when players get too close
            # This would require computing pairwise distances
            # Skipping for now as it's computationally expensive
            pass
        
        return loss


