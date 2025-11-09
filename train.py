"""
Main training script for NFL trajectory prediction
"""

import torch
import argparse
import random
import numpy as np
from pathlib import Path

from src.utils.config import get_config
from src.data.preprocessing import NFLDataPreprocessor, load_multiple_weeks
from src.data.dataset import create_dataloaders
from src.models.gnn_lstm import GNNLSTMTrajectoryPredictor, SimplifiedGNNLSTM
from src.training.losses import TrajectoryLoss
from src.training.trainer import Trainer


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args):
    """Main training function"""
    
    # Load configuration
    config = get_config()
    
    # Override with command line arguments
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    
    # Set random seed
    set_seed(config.seed)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    config.training.device = device
    print(f"Using device: {device}")
    
    # Create preprocessor
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    preprocessor = NFLDataPreprocessor(config.data)
    
    # Load training data to fit categorical mappings
    print("\nLoading training data to fit preprocessor...")
    train_input, _ = load_multiple_weeks(config.data.train_dir, config.data.train_weeks)
    preprocessor.fit_categorical_mappings(train_input)
    
    # Ensure model config has enough capacity for learned vocab sizes
    num_positions = len(preprocessor.position_to_idx)
    if num_positions > config.model.num_positions:
        config.model.num_positions = num_positions
    
    # Save preprocessor
    preprocessor_path = Path("checkpoints") / "preprocessor.pkl"
    preprocessor_path.parent.mkdir(exist_ok=True, parents=True)
    preprocessor.save(preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config.data,
        preprocessor,
        batch_size=config.training.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    if args.simple:
        print("\nUsing SimplifiedGNNLSTM (MLP encoder)")
        model = SimplifiedGNNLSTM(config.model)
    else:
        print("\nUsing GNNLSTMTrajectoryPredictor (full GAT encoder)")
        model = GNNLSTMTrajectoryPredictor(config.model)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")
    
    # Create loss function
    criterion = TrajectoryLoss(
        role_weights=config.training.role_weights,
        velocity_weight=config.training.velocity_weight,
        collision_weight=config.training.collision_weight,
        late_timestep_gamma=config.training.late_timestep_gamma,
        late_timestep_max=config.training.late_timestep_max,
        receiver_aux_weight=config.training.receiver_aux_weight,
        coverage_aux_weight=config.training.coverage_aux_weight,
        coverage_window=config.training.coverage_focus_window,
    )
    
    # Create trainer
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        config=config.training,
        device=device,
        norm_ranges=config.data.norm_ranges,
    )
    
    # Train
    trainer.train(num_epochs=config.training.num_epochs)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NFL trajectory prediction model")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    # Model arguments
    parser.add_argument("--simple", action="store_true", help="Use simplified model (MLP encoder)")
    
    # Experiment arguments
    parser.add_argument("--experiment-name", type=str, help="Experiment name for checkpoints")
    
    args = parser.parse_args()
    
    main(args)


