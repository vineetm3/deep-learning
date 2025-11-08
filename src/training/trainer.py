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
            
            # Forward pass
            self.optimizer.zero_grad()
            
            predictions = self.model(
                batch,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            
            # Compute loss
            loss = self.criterion(
                predictions,
                batch['output_positions'],
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
                
                # Compute loss
                loss = self.criterion(
                    predictions,
                    batch['output_positions'],
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


