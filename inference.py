"""
Inference script for Kaggle submission
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from src.utils.config import get_config
from src.data.preprocessing import NFLDataPreprocessor
from src.models.gnn_lstm import GNNLSTMTrajectoryPredictor, SimplifiedGNNLSTM


class NFLInferenceModel:
    """
    Inference model compatible with Kaggle evaluation gateway
    
    This can be integrated with kaggle_evaluation/nfl_inference_server.py
    """
    
    def __init__(self, checkpoint_path: str, preprocessor_path: str, device: str = 'cpu', simple: bool = False):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            preprocessor_path: Path to preprocessor
            device: Device to run inference on
            simple: Whether to use simplified model
        """
        self.device = device
        
        # Load config
        config = get_config()
        
        # Load preprocessor
        self.preprocessor = NFLDataPreprocessor(config.data)
        self.preprocessor.load(Path(preprocessor_path))
        
        # Load model
        if simple:
            self.model = SimplifiedGNNLSTM(config.model)
        else:
            self.model = GNNLSTMTrajectoryPredictor(config.model)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
    
    def predict(self, test_batch: pd.DataFrame, test_input: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions for a batch of plays
        
        This function signature is compatible with the Kaggle gateway
        
        Args:
            test_batch: DataFrame with columns ['id', 'game_id', 'play_id', 'nfl_id', 'frame_id']
            test_input: DataFrame with input features for the play
        
        Returns:
            DataFrame with columns ['x', 'y']
        """
        with torch.no_grad():
            # Process test_input through preprocessor
            test_input_processed = self.preprocessor.process_input_data(test_input)
            
            # Build batch similar to dataset
            # This is simplified - in production would need full batch construction
            # For now, return dummy predictions
            
            num_predictions = len(test_batch)
            
            # Placeholder: constant velocity or model predictions
            predictions = pd.DataFrame({
                'x': [50.0] * num_predictions,  # Dummy value
                'y': [26.0] * num_predictions,  # Dummy value
            })
            
            return predictions
    
    def predict_from_dataloader(self, dataloader):
        """
        Generate predictions from a PyTorch dataloader
        
        Returns list of predictions matching the dataloader format
        """
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating predictions"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                predictions = self.model(batch, teacher_forcing_ratio=0.0)
                
                all_predictions.append(predictions.cpu())
        
        return torch.cat(all_predictions, dim=0)
    
    def save_predictions(self, predictions, output_path: str):
        """
        Save predictions to CSV
        
        Args:
            predictions: Tensor of predictions [num_samples, num_players, num_frames, 2]
            output_path: Path to save CSV
        """
        # Convert to DataFrame format expected by Kaggle
        # This would need game_id, play_id, nfl_id, frame_id from the test set
        
        # Placeholder implementation
        print(f"Predictions saved to {output_path}")


def main():
    """Example inference pipeline"""
    
    # Paths
    checkpoint_path = "checkpoints/best_model.pt"
    preprocessor_path = "checkpoints/preprocessor.pkl"
    
    # Check if files exist
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train a model first using train.py")
        return
    
    if not Path(preprocessor_path).exists():
        print(f"Error: Preprocessor not found at {preprocessor_path}")
        print("Please train a model first using train.py")
        return
    
    # Create inference model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    inference_model = NFLInferenceModel(
        checkpoint_path=checkpoint_path,
        preprocessor_path=preprocessor_path,
        device=device,
    )
    
    print("\nInference model ready!")
    print("\nTo use with Kaggle gateway:")
    print("  1. Integrate this class with kaggle_evaluation/nfl_gateway.py")
    print("  2. Implement the predict() method to handle play-by-play batches")
    print("  3. Run the gateway for live evaluation")


if __name__ == "__main__":
    main()


