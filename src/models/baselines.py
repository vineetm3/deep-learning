"""
Baseline models for NFL trajectory prediction
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict


class ConstantVelocityBaseline:
    """
    Constant Velocity baseline model
    
    Predicts future positions by extrapolating using the velocity at the last observed frame:
        x(t + Δt) = x(t) + v_x * Δt
        y(t + Δt) = y(t) + v_y * Δt
    """
    
    def __init__(self, fps: float = 10.0):
        """
        Args:
            fps: Frames per second (default 10 for NFL tracking data)
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time step in seconds
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict future trajectories using constant velocity
        
        Args:
            batch: Dictionary containing:
                - input_features: [batch, players, frames, features]
                - output_mask: [batch, players, output_frames]
                - num_output_frames: list of actual output frames per play
        
        Returns:
            predictions: [batch, players, output_frames, 2] (x, y positions)
        """
        input_features = batch['input_features']
        output_mask = batch['output_mask']
        
        batch_size, num_players, input_frames, num_features = input_features.shape
        output_frames = output_mask.shape[2]
        
        predictions = torch.zeros(batch_size, num_players, output_frames, 2)
        
        for b in range(batch_size):
            for p in range(num_players):
                # Get last observed position and velocity
                # Assume x, y are first two features and s, dir are features 2, 4
                # From our preprocessing: ['x', 'y', 's', 'a', 'dir', 'o', ...]
                
                # Find last valid input frame
                player_input = input_features[b, p]  # [input_frames, features]
                
                # Get last position (already normalized)
                last_x = player_input[-1, 0].item()
                last_y = player_input[-1, 1].item()
                
                # Get velocity components
                # s is speed (feature 2), dir is direction in degrees (feature 4)
                # Note: features are normalized, need to handle this
                # For simplicity in baseline, we'll compute velocity from position changes
                
                # Compute velocity from last two frames
                if input_frames >= 2:
                    dx = player_input[-1, 0].item() - player_input[-2, 0].item()
                    dy = player_input[-1, 1].item() - player_input[-2, 1].item()
                else:
                    dx = 0.0
                    dy = 0.0
                
                # Predict future positions
                for t in range(output_frames):
                    predictions[b, p, t, 0] = last_x + dx * (t + 1)
                    predictions[b, p, t, 1] = last_y + dy * (t + 1)
        
        return predictions
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make the baseline callable like a nn.Module"""
        return self.predict(batch)


class MeanPositionBaseline:
    """
    Simple baseline that predicts the mean position of the last observed frame
    
    All future positions are the same as the last observed position (no movement)
    """
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict future trajectories as constant position
        
        Args:
            batch: Dictionary containing:
                - input_features: [batch, players, frames, features]
                - output_mask: [batch, players, output_frames]
        
        Returns:
            predictions: [batch, players, output_frames, 2] (x, y positions)
        """
        input_features = batch['input_features']
        output_mask = batch['output_mask']
        
        batch_size, num_players, input_frames, num_features = input_features.shape
        output_frames = output_mask.shape[2]
        
        # Get last position (x, y are first two features)
        last_positions = input_features[:, :, -1, :2]  # [batch, players, 2]
        
        # Repeat for all future timesteps
        predictions = last_positions.unsqueeze(2).expand(-1, -1, output_frames, -1)
        
        return predictions
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make the baseline callable like a nn.Module"""
        return self.predict(batch)


class LinearExtrapolationBaseline:
    """
    Linear extrapolation baseline using least squares fit on recent trajectory
    
    Fits a line to the last N observed positions and extrapolates
    """
    
    def __init__(self, lookback_frames: int = 5):
        """
        Args:
            lookback_frames: Number of recent frames to use for linear fit
        """
        self.lookback_frames = lookback_frames
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Predict future trajectories using linear extrapolation
        
        Args:
            batch: Dictionary containing:
                - input_features: [batch, players, frames, features]
                - output_mask: [batch, players, output_frames]
        
        Returns:
            predictions: [batch, players, output_frames, 2] (x, y positions)
        """
        input_features = batch['input_features']
        output_mask = batch['output_mask']
        input_mask = batch['input_mask']
        
        batch_size, num_players, input_frames, num_features = input_features.shape
        output_frames = output_mask.shape[2]
        
        predictions = torch.zeros(batch_size, num_players, output_frames, 2)
        
        for b in range(batch_size):
            for p in range(num_players):
                # Get valid input frames for this player
                valid_mask = input_mask[b, p] > 0
                valid_indices = torch.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    continue
                
                # Use last N frames for fitting
                n_frames = min(self.lookback_frames, len(valid_indices))
                recent_indices = valid_indices[-n_frames:]
                
                # Get positions
                x_positions = input_features[b, p, recent_indices, 0].cpu().numpy()
                y_positions = input_features[b, p, recent_indices, 1].cpu().numpy()
                
                # Time indices
                t = np.arange(len(x_positions))
                
                # Fit linear model
                if len(t) >= 2:
                    # Fit x = a*t + b
                    x_coeffs = np.polyfit(t, x_positions, deg=1)
                    y_coeffs = np.polyfit(t, y_positions, deg=1)
                    
                    # Extrapolate
                    future_t = np.arange(len(t), len(t) + output_frames)
                    x_pred = np.polyval(x_coeffs, future_t)
                    y_pred = np.polyval(y_coeffs, future_t)
                    
                    predictions[b, p, :, 0] = torch.FloatTensor(x_pred)
                    predictions[b, p, :, 1] = torch.FloatTensor(y_pred)
                else:
                    # Not enough frames, use last position
                    last_x = x_positions[-1]
                    last_y = y_positions[-1]
                    predictions[b, p, :, 0] = last_x
                    predictions[b, p, :, 1] = last_y
        
        return predictions
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make the baseline callable like a nn.Module"""
        return self.predict(batch)


