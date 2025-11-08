"""
Evaluation metrics for NFL trajectory prediction
"""

import numpy as np
import torch
from typing import Dict, Optional


def compute_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Root Mean Squared Error
    
    Args:
        predictions: Predicted positions [batch, players, frames, 2]
        targets: Target positions [batch, players, frames, 2]
        mask: Binary mask [batch, players, frames] indicating valid positions
    
    Returns:
        RMSE value
    """
    if mask is not None:
        # Expand mask to cover both x and y coordinates
        mask = mask.unsqueeze(-1)  # [batch, players, frames, 1]
        
        # Compute squared errors only for valid positions
        squared_errors = ((predictions - targets) ** 2).sum(dim=-1, keepdim=True)  # [batch, players, frames, 1]
        masked_errors = squared_errors * mask
        
        # Mean over all valid positions
        total_error = masked_errors.sum()
        num_valid = mask.sum()
        
        if num_valid == 0:
            return 0.0
        
        mse = total_error / num_valid
        rmse = torch.sqrt(mse)
    else:
        # All positions are valid
        mse = ((predictions - targets) ** 2).mean()
        rmse = torch.sqrt(mse)
    
    return rmse.item()


def compute_rmse_by_role(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    roles: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    role_names: Optional[Dict[int, str]] = None,
) -> Dict[str, float]:
    """
    Compute RMSE separately for each player role
    
    Args:
        predictions: Predicted positions [batch, players, frames, 2]
        targets: Target positions [batch, players, frames, 2]
        roles: Player roles [batch, players] (integer indices)
        mask: Binary mask [batch, players, frames]
        role_names: Mapping from role index to name
    
    Returns:
        Dictionary mapping role name to RMSE
    """
    if role_names is None:
        role_names = {
            0: 'Targeted Receiver',
            1: 'Defensive Coverage',
            2: 'Other Route Runner',
            3: 'Passer',
        }
    
    results = {}
    unique_roles = torch.unique(roles)
    
    for role_idx in unique_roles:
        role_idx = role_idx.item()
        role_name = role_names.get(role_idx, f'Role {role_idx}')
        
        # Create mask for this role
        role_mask_2d = (roles == role_idx)  # [batch, players]
        role_mask_3d = role_mask_2d.unsqueeze(-1).expand_as(mask) if mask is not None else role_mask_2d.unsqueeze(-1).unsqueeze(-1)
        
        # Combine with validity mask
        if mask is not None:
            combined_mask = role_mask_3d & (mask > 0)
        else:
            combined_mask = role_mask_3d
        
        # Compute RMSE for this role
        rmse = compute_rmse(predictions, targets, combined_mask.float())
        results[role_name] = rmse
    
    return results


def compute_rmse_by_timestep(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    max_timesteps: int = 20,
) -> np.ndarray:
    """
    Compute RMSE for each timestep (to see how error accumulates)
    
    Args:
        predictions: Predicted positions [batch, players, frames, 2]
        targets: Target positions [batch, players, frames, 2]
        mask: Binary mask [batch, players, frames]
        max_timesteps: Maximum number of timesteps to compute
    
    Returns:
        Array of RMSE values per timestep
    """
    batch_size, num_players, num_frames, _ = predictions.shape
    max_timesteps = min(max_timesteps, num_frames)
    
    rmse_per_step = np.zeros(max_timesteps)
    
    for t in range(max_timesteps):
        # Get predictions and targets for this timestep
        pred_t = predictions[:, :, t, :]  # [batch, players, 2]
        target_t = targets[:, :, t, :]  # [batch, players, 2]
        
        if mask is not None:
            mask_t = mask[:, :, t]  # [batch, players]
            
            # Compute RMSE
            squared_errors = ((pred_t - target_t) ** 2).sum(dim=-1)  # [batch, players]
            masked_errors = squared_errors * mask_t
            
            total_error = masked_errors.sum()
            num_valid = mask_t.sum()
            
            if num_valid > 0:
                mse = total_error / num_valid
                rmse_per_step[t] = torch.sqrt(mse).item()
        else:
            mse = ((pred_t - target_t) ** 2).mean()
            rmse_per_step[t] = torch.sqrt(mse).item()
    
    return rmse_per_step


def compute_ade(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Average Displacement Error
    
    Args:
        predictions: Predicted positions [batch, players, frames, 2]
        targets: Target positions [batch, players, frames, 2]
        mask: Binary mask [batch, players, frames]
    
    Returns:
        ADE value
    """
    # Compute Euclidean distance for each position
    distances = torch.sqrt(((predictions - targets) ** 2).sum(dim=-1))  # [batch, players, frames]
    
    if mask is not None:
        masked_distances = distances * mask
        total_distance = masked_distances.sum()
        num_valid = mask.sum()
        
        if num_valid == 0:
            return 0.0
        
        ade = total_distance / num_valid
    else:
        ade = distances.mean()
    
    return ade.item()


def compute_fde(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Compute Final Displacement Error (error at last timestep)
    
    Args:
        predictions: Predicted positions [batch, players, frames, 2]
        targets: Target positions [batch, players, frames, 2]
        mask: Binary mask [batch, players, frames]
    
    Returns:
        FDE value
    """
    batch_size, num_players, num_frames, _ = predictions.shape
    
    # Get the last valid timestep for each player
    if mask is not None:
        # Find last valid frame for each player
        last_valid_indices = mask.sum(dim=-1) - 1  # [batch, players]
        last_valid_indices = last_valid_indices.long().clamp(min=0)
        
        # Gather predictions and targets at last valid indices
        batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(last_valid_indices)
        player_indices = torch.arange(num_players).view(1, -1).expand_as(last_valid_indices)
        
        pred_final = predictions[batch_indices, player_indices, last_valid_indices]  # [batch, players, 2]
        target_final = targets[batch_indices, player_indices, last_valid_indices]  # [batch, players, 2]
        
        # Compute distances
        distances = torch.sqrt(((pred_final - target_final) ** 2).sum(dim=-1))  # [batch, players]
        
        # Only count players that have valid data
        valid_players = (mask.sum(dim=-1) > 0).float()  # [batch, players]
        masked_distances = distances * valid_players
        
        total_distance = masked_distances.sum()
        num_valid = valid_players.sum()
        
        if num_valid == 0:
            return 0.0
        
        fde = total_distance / num_valid
    else:
        # Use last timestep
        pred_final = predictions[:, :, -1, :]
        target_final = targets[:, :, -1, :]
        
        distances = torch.sqrt(((pred_final - target_final) ** 2).sum(dim=-1))
        fde = distances.mean()
    
    return fde.item()


class MetricsTracker:
    """Track metrics during training/evaluation"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_rmse = 0.0
        self.total_ade = 0.0
        self.total_fde = 0.0
        self.num_batches = 0
        self.role_rmses = {}
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        roles: Optional[torch.Tensor] = None,
    ):
        """Update metrics with a new batch"""
        self.total_rmse += compute_rmse(predictions, targets, mask)
        self.total_ade += compute_ade(predictions, targets, mask)
        self.total_fde += compute_fde(predictions, targets, mask)
        self.num_batches += 1
        
        # Per-role metrics
        if roles is not None:
            role_rmses = compute_rmse_by_role(predictions, targets, roles, mask)
            for role, rmse in role_rmses.items():
                if role not in self.role_rmses:
                    self.role_rmses[role] = []
                self.role_rmses[role].append(rmse)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get average metrics"""
        if self.num_batches == 0:
            return {}
        
        metrics = {
            'rmse': self.total_rmse / self.num_batches,
            'ade': self.total_ade / self.num_batches,
            'fde': self.total_fde / self.num_batches,
        }
        
        # Add per-role RMSE
        for role, rmses in self.role_rmses.items():
            metrics[f'rmse_{role.lower().replace(" ", "_")}'] = np.mean(rmses)
        
        return metrics


