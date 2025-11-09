"""
Training utilities for trajectory prediction models
"""

import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from src.evaluation.metrics import MetricsTracker
from src.data.dataset import FEATURE_INDEX


class Trainer:
    """Train trajectory prediction models"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        config,
        device: str = 'cuda',
        norm_ranges: Optional[Dict[str, tuple]] = None,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.config = config
        self.device = device
        self.norm_ranges = norm_ranges
        self.use_mirroring = getattr(config, 'use_left_right_mirroring', False)
        self.mirror_probability = getattr(config, 'mirror_probability', 0.0)
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        scheduler_choice = getattr(config, 'lr_scheduler', 'reduce_on_plateau').lower()
        self.scheduler_type = scheduler_choice
        if scheduler_choice == 'cosine':
            t_max = getattr(config, 'cosine_t_max', 10)
            eta_min = getattr(config, 'cosine_eta_min', 1e-6)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=t_max,
                eta_min=eta_min,
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_factor,
                patience=config.lr_patience,
            )
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    def _compute_teacher_forcing_ratio(self, epoch: int) -> float:
        strategy = getattr(self.config, 'teacher_forcing_strategy', 'linear').lower()
        initial = getattr(self.config, 'initial_teacher_forcing_ratio', 1.0)
        final = getattr(self.config, 'final_teacher_forcing_ratio', 0.5)
        decay_epochs = max(1, getattr(self.config, 'teacher_forcing_decay_epochs', 50))
        
        if strategy == 'exponential':
            gamma = getattr(self.config, 'teacher_forcing_gamma', 0.9)
            ratio = initial * (gamma ** epoch)
        elif strategy == 'cosine':
            progress = min(epoch / decay_epochs, 1.0)
            ratio = final + 0.5 * (initial - final) * (1 + math.cos(progress * math.pi))
        else:  # linear
            progress = min(epoch / decay_epochs, 1.0)
            ratio = initial * (1 - progress) + final * progress
        
        return max(final, min(initial, ratio))
    
    def train_epoch(self, teacher_forcing_ratio: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            teacher_forcing_ratio: Ratio of teacher forcing (1.0 = always use ground truth)
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        tracker = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")
        for batch in pbar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            self._maybe_apply_mirroring(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(
                batch,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            
        # Compute residual targets
        last_positions = self._get_last_positions(batch)
        target_residuals = self._compute_residual_sequences(batch['output_positions'], last_positions)
        pred_residuals = self._compute_residual_sequences(predictions, last_positions)
        
        # Compute loss on residuals
        loss = self.criterion(
            pred_residuals,
            target_residuals,
            batch['player_roles'],
            batch['output_mask'],
        )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            tracker.update(
                predictions.detach(),
                batch['output_positions'],
                batch['output_mask'],
                batch['player_roles'],
            )
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Get metrics
        metrics = tracker.get_metrics()
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        tracker = MetricsTracker()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")
            for batch in pbar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass (no teacher forcing during validation)
                predictions = self.model(batch, teacher_forcing_ratio=0.0)
                
                last_positions = self._get_last_positions(batch)
                target_residuals = self._compute_residual_sequences(batch['output_positions'], last_positions)
                pred_residuals = self._compute_residual_sequences(predictions, last_positions)
                
                # Compute loss
                loss = self.criterion(
                    pred_residuals,
                    target_residuals,
                    batch['player_roles'],
                    batch['output_mask'],
                )
                
                # Track metrics
                total_loss += loss.item()
                tracker.update(
                    predictions,
                    batch['output_positions'],
                    batch['output_mask'],
                    batch['player_roles'],
                )
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Get metrics
        metrics = tracker.get_metrics()
        metrics['loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self, num_epochs: Optional[int] = None):
        """
        Full training loop
        
        Args:
            num_epochs: Number of epochs (defaults to config.num_epochs)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters())}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Compute teacher forcing ratio (decay from initial to final)
            teacher_forcing_ratio = self._compute_teacher_forcing_ratio(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch(teacher_forcing_ratio)
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate scheduling
            if self.scheduler_type == 'cosine':
                self.scheduler.step()
            else:
                self.scheduler.step(val_metrics['loss'])
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
            print(f"  Teacher forcing ratio: {teacher_forcing_ratio:.3f}")
            print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pt', val_metrics)
                print(f"  ✓ New best model saved!")
            else:
                self.epochs_without_improvement += 1
            
            # Regular checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', val_metrics)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _get_last_positions(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract last observed positions (x, y) prior to the throw."""
        positions = batch['input_features'][..., [FEATURE_INDEX['x'], FEATURE_INDEX['y']]]
        mask = batch['input_mask'].bool()
        last_positions = torch.zeros(
            positions.size(0),
            positions.size(1),
            2,
            device=positions.device,
            dtype=positions.dtype,
        )
        for t in range(positions.size(2)):
            update_mask = mask[:, :, t].unsqueeze(-1)
            last_positions = torch.where(update_mask, positions[:, :, t], last_positions)
        return last_positions

    def _compute_residual_sequences(self, positions: torch.Tensor, last_positions: torch.Tensor) -> torch.Tensor:
        """Compute Δ positions relative to previous timestep (first step relative to last observed)."""
        prev_positions = torch.cat([last_positions.unsqueeze(2), positions[:, :, :-1]], dim=2)
        residuals = positions - prev_positions
        return residuals

    def _maybe_apply_mirroring(self, batch: Dict[str, torch.Tensor]):
        """Randomly mirror plays left/right to augment training data."""
        if not self.use_mirroring or self.norm_ranges is None or self.mirror_probability <= 0.0:
            return
        if torch.rand(1, device=self.device).item() > self.mirror_probability:
            return
        self._mirror_batch_inplace(batch)

    def _mirror_batch_inplace(self, batch: Dict[str, torch.Tensor]):
        """Mirror all tensors in batch across the field width."""
        input_features = batch['input_features']
        output_positions = batch['output_positions']
        ball_landing = batch['ball_landing']
        categorical = batch['categorical_features']

        # Mirror positional coordinates
        input_features[..., FEATURE_INDEX['y']] = self._mirror_feature(input_features[..., FEATURE_INDEX['y']], 'y')
        input_features[..., FEATURE_INDEX['ball_land_y']] = self._mirror_feature(input_features[..., FEATURE_INDEX['ball_land_y']], 'ball_land_y')
        output_positions[..., 1] = self._mirror_feature(output_positions[..., 1], 'y')
        ball_landing[..., 1] = self._mirror_feature(ball_landing[..., 1], 'ball_land_y')

        # Flip velocity/acceleration components in Y
        input_features[..., FEATURE_INDEX['vy']] = self._flip_feature_sign(input_features[..., FEATURE_INDEX['vy']], 'vy')
        input_features[..., FEATURE_INDEX['ay']] = self._flip_feature_sign(input_features[..., FEATURE_INDEX['ay']], 'ay')

        # Flip ball-relative Y features
        input_features[..., FEATURE_INDEX['ball_dy']] = self._flip_feature_sign(input_features[..., FEATURE_INDEX['ball_dy']], 'ball_dy')
        input_features[..., FEATURE_INDEX['ball_angle_sin']] = self._flip_feature_sign(input_features[..., FEATURE_INDEX['ball_angle_sin']], 'ball_angle_sin')
        input_features[..., FEATURE_INDEX['route_width']] = self._flip_feature_sign(input_features[..., FEATURE_INDEX['route_width']], 'route_width')

        # Mirror directional angles
        input_features[..., FEATURE_INDEX['dir']] = self._mirror_angle_feature(input_features[..., FEATURE_INDEX['dir']], 'dir')
        input_features[..., FEATURE_INDEX['o']] = self._mirror_angle_feature(input_features[..., FEATURE_INDEX['o']], 'o')

        # Flip play direction categorical label (0 <-> 1)
        categorical[:, :, 3] = 1 - categorical[:, :, 3]

    def _mirror_feature(self, values: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Mirror value across midpoint between min and max."""
        min_val, max_val = self.norm_ranges[feature_name]
        min_t = torch.as_tensor(min_val, device=values.device, dtype=values.dtype)
        max_t = torch.as_tensor(max_val, device=values.device, dtype=values.dtype)
        real = values * (max_t - min_t) + min_t
        mirrored = (min_t + max_t) - real
        normalized = (mirrored - min_t) / (max_t - min_t)
        return normalized.clamp(0.0, 1.0)

    def _flip_feature_sign(self, values: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Flip the sign of a feature with symmetric range."""
        min_val, max_val = self.norm_ranges[feature_name]
        min_t = torch.as_tensor(min_val, device=values.device, dtype=values.dtype)
        max_t = torch.as_tensor(max_val, device=values.device, dtype=values.dtype)
        real = values * (max_t - min_t) + min_t
        flipped = -real
        normalized = (flipped - min_t) / (max_t - min_t)
        return normalized.clamp(0.0, 1.0)

    def _mirror_angle_feature(self, values: torch.Tensor, feature_name: str) -> torch.Tensor:
        """Mirror directional angle (in degrees) across the vertical axis."""
        min_val, max_val = self.norm_ranges[feature_name]
        min_t = torch.as_tensor(min_val, device=values.device, dtype=values.dtype)
        max_t = torch.as_tensor(max_val, device=values.device, dtype=values.dtype)
        period = max_t - min_t
        real = values * period + min_t
        mirrored = (torch.as_tensor(180.0, device=values.device, dtype=values.dtype) - real + period) % period
        mirrored = torch.where(mirrored < 0, mirrored + period, mirrored)
        normalized = (mirrored - min_t) / period
        return normalized.clamp(0.0, 1.0)


