"""
Evaluation script to compare baseline and GNN-LSTM models
"""

import torch
import argparse
from pathlib import Path

from src.utils.config import get_config
from src.data.preprocessing import NFLDataPreprocessor, load_multiple_weeks
from src.data.dataset import create_dataloaders
from src.models.baselines import ConstantVelocityBaseline, LinearExtrapolationBaseline
from src.models.gnn_lstm import GNNLSTMTrajectoryPredictor, SimplifiedGNNLSTM
from src.evaluation.evaluator import ModelEvaluator


def load_trained_model(checkpoint_path: str, config, device: str = 'cpu', simple: bool = False):
    """Load a trained model from checkpoint"""
    
    if simple:
        model = SimplifiedGNNLSTM(config.model)
    else:
        model = GNNLSTMTrajectoryPredictor(config.model)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model


def main(args):
    """Main evaluation function"""
    
    # Load configuration
    config = get_config()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load preprocessor
    print("\n" + "="*80)
    print("LOADING PREPROCESSOR AND DATA")
    print("="*80)
    
    preprocessor_path = Path("checkpoints") / "preprocessor.pkl"
    preprocessor = NFLDataPreprocessor(config.data)
    preprocessor.load(preprocessor_path)
    print(f"Loaded preprocessor from {preprocessor_path}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    _, _, test_loader = create_dataloaders(
        config.data,
        preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"  Test batches: {len(test_loader)}")
    
    # Create evaluator
    evaluator = ModelEvaluator(device=device)
    
    # Dictionary to store models
    models = {}
    
    # Add baseline models
    if args.eval_baselines:
        print("\n" + "="*80)
        print("CREATING BASELINE MODELS")
        print("="*80)
        
        models['Constant Velocity'] = ConstantVelocityBaseline()
        models['Linear Extrapolation'] = LinearExtrapolationBaseline(lookback_frames=5)
        print("  ✓ Constant Velocity baseline")
        print("  ✓ Linear Extrapolation baseline")
    
    # Add trained model
    if args.checkpoint:
        print("\n" + "="*80)
        print("LOADING TRAINED MODEL")
        print("="*80)
        
        model = load_trained_model(
            args.checkpoint,
            config,
            device=device,
            simple=args.simple,
        )
        
        model_name = "GNN-LSTM" if not args.simple else "SimplifiedGNN-LSTM"
        models[model_name] = model
    
    # Evaluate all models
    if len(models) > 0:
        print("\n" + "="*80)
        print("EVALUATING MODELS")
        print("="*80)
        
        results = evaluator.compare_models(models, test_loader)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        print("\n{:<30} {:<10} {:<10} {:<10}".format("Model", "RMSE", "ADE", "FDE"))
        print("-" * 70)
        
        for model_name, metrics in results.items():
            print("{:<30} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                model_name,
                metrics['rmse'],
                metrics['ade'],
                metrics['fde'],
            ))
        
        # Per-role breakdown
        print("\n" + "="*80)
        print("PER-ROLE RMSE BREAKDOWN")
        print("="*80)
        
        for model_name, metrics in results.items():
            print(f"\n{model_name}:")
            for key, value in metrics.items():
                if key.startswith('rmse_') and key != 'rmse':
                    role = key.replace('rmse_', '').replace('_', ' ').title()
                    print(f"  {role}: {value:.4f}")
    
    else:
        print("\nNo models to evaluate!")
        print("  Use --checkpoint to evaluate a trained model")
        print("  Use --eval-baselines to evaluate baseline models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NFL trajectory prediction models")
    
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--eval-baselines", action="store_true", help="Evaluate baseline models")
    parser.add_argument("--simple", action="store_true", help="Use simplified model architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    main(args)


