"""
Evaluation utilities for models
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional
from src.evaluation.metrics import MetricsTracker


class ModelEvaluator:
    """Evaluate models on trajectory prediction"""
    
    def __init__(self, device: str = 'cpu'):
        """
        Args:
            device: Device to run evaluation on
        """
        self.device = device
    
    def evaluate(
        self,
        model,
        dataloader: torch.utils.data.DataLoader,
        desc: str = "Evaluating",
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate a model on a dataset
        
        Args:
            model: Model to evaluate (can be nn.Module or baseline)
            dataloader: DataLoader for evaluation data
            desc: Description for progress bar
            return_predictions: If True, also return predictions
        
        Returns:
            Dictionary of metrics
        """
        # Check if model is a nn.Module
        is_neural_net = isinstance(model, torch.nn.Module)
        
        if is_neural_net:
            model.eval()
        
        tracker = MetricsTracker()
        all_predictions = [] if return_predictions else None
        all_targets = [] if return_predictions else None
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=desc):
                # Move batch to device
                if is_neural_net:
                    batch = self._move_batch_to_device(batch)
                
                # Get predictions
                predictions = model(batch)
                
                # Get targets and masks
                targets = batch['output_positions']
                output_mask = batch['output_mask']
                player_roles = batch['player_roles']
                
                # Move to CPU for metrics computation if needed
                if is_neural_net:
                    predictions = predictions.cpu()
                    targets = targets.cpu()
                    output_mask = output_mask.cpu()
                    player_roles = player_roles.cpu()
                
                # Update metrics
                tracker.update(predictions, targets, output_mask, player_roles)
                
                # Store predictions if requested
                if return_predictions:
                    all_predictions.append(predictions)
                    all_targets.append(targets)
        
        # Get metrics
        metrics = tracker.get_metrics()
        
        if return_predictions:
            return metrics, torch.cat(all_predictions, dim=0), torch.cat(all_targets, dim=0)
        else:
            return metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def compare_models(
        self,
        models: Dict[str, any],
        dataloader: torch.utils.data.DataLoader,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models
        
        Args:
            models: Dictionary mapping model name to model
            dataloader: DataLoader for evaluation data
        
        Returns:
            Dictionary mapping model name to metrics
        """
        results = {}
        
        for name, model in models.items():
            print(f"\nEvaluating {name}...")
            metrics = self.evaluate(model, dataloader, desc=name)
            results[name] = metrics
            
            # Print results
            print(f"\n{name} Results:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  ADE: {metrics['ade']:.4f}")
            print(f"  FDE: {metrics['fde']:.4f}")
            
            # Per-role metrics
            for key, value in metrics.items():
                if key.startswith('rmse_'):
                    role = key.replace('rmse_', '').replace('_', ' ').title()
                    print(f"  RMSE ({role}): {value:.4f}")
        
        return results
    
    def create_submission_csv(
        self,
        model,
        test_loader: torch.utils.data.DataLoader,
        preprocessor,
        output_path: str,
    ):
        """
        Create submission CSV for Kaggle
        
        Args:
            model: Trained model
            test_loader: Test data loader
            preprocessor: Data preprocessor (for denormalization)
            output_path: Path to save submission CSV
        """
        import pandas as pd
        
        is_neural_net = isinstance(model, torch.nn.Module)
        if is_neural_net:
            model.eval()
        
        all_rows = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                # Move batch to device if neural net
                if is_neural_net:
                    batch = self._move_batch_to_device(batch)
                
                # Get predictions
                predictions = model(batch)
                
                # Move to CPU
                if is_neural_net:
                    predictions = predictions.cpu()
                
                # Denormalize predictions
                predictions_np = predictions.numpy()
                
                # Process each play in batch
                for b in range(len(batch['game_ids'])):
                    game_id = batch['game_ids'][b]
                    play_id = batch['play_ids'][b]
                    num_players = batch['num_players'][b].item()
                    num_output_frames = batch['num_output_frames'][b]
                    
                    # Get predictions for this play
                    play_preds = predictions_np[b, :num_players, :num_output_frames]  # [players, frames, 2]
                    
                    # Denormalize
                    x_denorm, y_denorm = preprocessor.denormalize_positions(
                        play_preds[:, :, 0], play_preds[:, :, 1]
                    )
                    
                    # Create rows for CSV
                    # Note: Need to get nfl_ids from batch
                    # This would require storing them in the dataset
                    # For now, placeholder structure:
                    for p in range(num_players):
                        for f in range(num_output_frames):
                            row = {
                                'game_id': game_id,
                                'play_id': play_id,
                                'nfl_id': 0,  # Would need to get from batch
                                'frame_id': f + 1,
                                'x': x_denorm[p, f],
                                'y': y_denorm[p, f],
                            }
                            all_rows.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        print(f"\nSubmission saved to {output_path}")


