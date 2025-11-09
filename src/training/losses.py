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
        timestep_weights: Optional[torch.Tensor] = None,
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
        
        # Apply timestep weights if provided
        frame_weights = None
        if timestep_weights is not None:
            frame_weights = timestep_weights.to(predictions.device).view(1, 1, -1)
            weighted_errors = weighted_errors * frame_weights
        
        # Apply validity mask
        if mask is not None:
            mask = mask.to(predictions.device)
            weighted_errors = weighted_errors * mask
            if frame_weights is not None:
                num_valid = (mask * frame_weights).sum()
            else:
                num_valid = mask.sum()
        else:
            if frame_weights is not None:
                num_valid = frame_weights.sum() * predictions.size(0) * predictions.size(1)
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
        late_timestep_gamma: float = 1.5,
        late_timestep_max: float = 1.0,
        receiver_aux_weight: float = 0.0,
        coverage_aux_weight: float = 0.0,
        coverage_window: int = 5,
    ):
        """
        Args:
            role_weights: Weights for different player roles
            velocity_weight: Weight for velocity consistency loss
            collision_weight: Weight for collision avoidance loss
            late_timestep_gamma: Exponent controlling steepness of late timestep weighting
            late_timestep_max: Maximum multiplier applied to the final timestep
            receiver_aux_weight: Extra weight for targeted receiver window loss
            coverage_aux_weight: Extra weight for defensive coverage window loss
            coverage_window: Number of timesteps (from the end) to include in role-focused losses
        """
        super().__init__()
        
        self.position_loss = WeightedMSELoss(role_weights)
        self.velocity_weight = velocity_weight
        self.collision_weight = collision_weight
        self.late_timestep_gamma = late_timestep_gamma
        self.late_timestep_max = late_timestep_max
        self.receiver_aux_weight = receiver_aux_weight
        self.coverage_aux_weight = coverage_aux_weight
        self.coverage_window = coverage_window
    
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
        timestep_weights = None
        if self.late_timestep_max > 1.0:
            num_frames = targets.size(2)
            if num_frames > 0:
                steps = torch.arange(num_frames, device=predictions.device, dtype=predictions.dtype)
                denom = max(num_frames - 1, 1)
                normalized = steps / denom
                timestep_weights = 1.0 + (self.late_timestep_max - 1.0) * (normalized ** self.late_timestep_gamma)
        
        # Position loss
        loss = self.position_loss(
            predictions,
            targets,
            roles,
            mask,
            timestep_weights=timestep_weights,
        )
        
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
        
        # Receiver-focused final window loss
        if self.receiver_aux_weight > 0 and mask is not None:
            receiver_loss = self._window_role_loss(
                predictions,
                targets,
                mask,
                roles,
                role_idx=0,
                window=min(self.coverage_window, targets.size(2)),
            )
            if receiver_loss is not None:
                loss = loss + self.receiver_aux_weight * receiver_loss
        
        # Coverage-focused final window loss
        if self.coverage_aux_weight > 0 and mask is not None:
            coverage_loss = self._window_role_loss(
                predictions,
                targets,
                mask,
                roles,
                role_idx=1,
                window=min(self.coverage_window, targets.size(2)),
            )
            if coverage_loss is not None:
                loss = loss + self.coverage_aux_weight * coverage_loss
        
        return loss

    def _window_role_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        roles: torch.Tensor,
        role_idx: int,
        window: int,
    ) -> Optional[torch.Tensor]:
        """Compute mean squared error over the last `window` timesteps for a specific role."""
        if window <= 0:
            return None
        pred_window = predictions[:, :, -window:, :]
        target_window = targets[:, :, -window:, :]
        mask_window = mask[:, :, -window:]
        role_mask = (roles == role_idx).unsqueeze(-1).unsqueeze(-1).float()
        combined_mask = role_mask * mask_window.unsqueeze(-1)
        total_weight = combined_mask.sum()
        if total_weight.item() == 0:
            return None
        diff = (pred_window - target_window) ** 2 * combined_mask
        return diff.sum() / total_weight


