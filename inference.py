"""
Inference script for Kaggle submission
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import polars as pl
import pandas as pd
import torch

from src.utils.config import get_config
from src.data.preprocessing import NFLDataPreprocessor
from src.data.dataset import collate_fn, INPUT_FEATURE_COLUMNS
from src.models.gnn_lstm import GNNLSTMTrajectoryPredictor, SimplifiedGNNLSTM

CATEGORICAL_COLS = ['player_role_idx', 'player_position_idx', 'player_side_idx', 'play_direction_idx']


class NFLInferenceModel:
    """
    Inference model compatible with Kaggle evaluation gateway
    
    This can be integrated with kaggle_evaluation/nfl_inference_server.py
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        preprocessor_path: Path = Path("checkpoints/preprocessor.pkl"),
        device: str = 'cpu',
        simple: bool = False,
        checkpoint_paths: Optional[List[Path]] = None,
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            preprocessor_path: Path to preprocessor
            device: Device to run inference on
            simple: Whether to use simplified model
        """
        self.device = device
        
        # Load config
        self.config = get_config()
        self.max_input_frames = self.config.data.max_input_frames
        self.max_output_frames = self.config.data.max_output_frames
        
        # Load preprocessor
        self.preprocessor = NFLDataPreprocessor(self.config.data)
        self.preprocessor.load(preprocessor_path)
        
        # Adjust model config to match preprocessor vocab sizes
        num_positions = len(self.preprocessor.position_to_idx)
        if num_positions > self.config.model.num_positions:
            self.config.model.num_positions = num_positions
        
        # Determine checkpoints to load
        if checkpoint_paths is None:
            if checkpoint_path is None:
                raise ValueError("Either checkpoint_path or checkpoint_paths must be provided.")
            checkpoint_paths = [Path(checkpoint_path)]
        self.checkpoint_paths = [Path(p) for p in checkpoint_paths]
        
        # Load model(s)
        self.models: List[torch.nn.Module] = []
        model_cls = SimplifiedGNNLSTM if simple else GNNLSTMTrajectoryPredictor
        for path in self.checkpoint_paths:
            model = model_cls(self.config.model)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            self.models.append(model)
            print(f"Loaded model from {path}")
        
        # Primary reference model
        self.model = self.models[0]

    def _build_play_sample(self, play_df: pd.DataFrame) -> Dict:
        player_ids = play_df['nfl_id'].drop_duplicates().astype(int).tolist()
        num_players = len(player_ids)
        
        input_features = np.zeros((num_players, self.max_input_frames, len(INPUT_FEATURE_COLUMNS)), dtype=np.float32)
        categorical_features = np.zeros((num_players, len(CATEGORICAL_COLS)), dtype=np.int64)
        input_mask = np.zeros((num_players, self.max_input_frames), dtype=np.float32)
        output_mask = np.zeros((num_players, self.max_output_frames), dtype=np.float32)
        player_roles = np.zeros(num_players, dtype=np.int64)
        player_to_predict = np.zeros(num_players, dtype=np.int64)
        
        ball_landing = play_df[['ball_land_x', 'ball_land_y']].iloc[0].to_numpy(dtype=np.float32)
        num_output_frames = int(play_df['num_frames_output'].iloc[0])
        frames_to_predict = min(num_output_frames, self.max_output_frames)
        
        for i, player_id in enumerate(player_ids):
            player_rows = play_df[play_df['nfl_id'] == player_id].sort_values('frame_id')
            n_input_frames = min(len(player_rows), self.max_input_frames)
            if len(player_rows) > self.max_input_frames:
                player_rows = player_rows.iloc[-self.max_input_frames:]
            
            input_features[i, :n_input_frames] = player_rows[INPUT_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            input_mask[i, :n_input_frames] = 1.0
            categorical_features[i] = player_rows[CATEGORICAL_COLS].iloc[0].to_numpy(dtype=np.int64)
            player_roles[i] = int(player_rows['player_role_idx'].iloc[0])
            player_to_predict[i] = int(bool(player_rows['player_to_predict'].iloc[0]))
            output_mask[i, :frames_to_predict] = 1.0
        
        return {
            'input_features': torch.FloatTensor(input_features),
            'categorical_features': torch.LongTensor(categorical_features),
            'output_positions': torch.zeros(num_players, self.max_output_frames, 2, dtype=torch.float32),
            'ball_landing': torch.FloatTensor(ball_landing),
            'num_players': num_players,
            'input_mask': torch.FloatTensor(input_mask),
            'output_mask': torch.FloatTensor(output_mask),
            'player_roles': torch.LongTensor(player_roles),
            'player_to_predict': torch.LongTensor(player_to_predict),
            'num_output_frames': frames_to_predict,
            'game_id': int(play_df['game_id'].iloc[0]),
            'play_id': int(play_df['play_id'].iloc[0]),
            'player_ids': torch.LongTensor(player_ids),
        }

    def _prepare_batch(self, processed_input: pd.DataFrame) -> Tuple[Dict, List[Tuple[int, int]], Dict[Tuple[int, int], pd.DataFrame]]:
        samples = []
        play_keys: List[Tuple[int, int]] = []
        play_data_map: Dict[Tuple[int, int], pd.DataFrame] = {}
        
        for (game_id, play_id), play_df in processed_input.groupby(['game_id', 'play_id']):
            play_df_sorted = play_df.sort_values(['nfl_id', 'frame_id'])
            sample = self._build_play_sample(play_df_sorted)
            samples.append(sample)
            key = (int(game_id), int(play_id))
            play_keys.append(key)
            play_data_map[key] = play_df_sorted
        
        if not samples:
            return {}, play_keys, play_data_map
        
        batch = collate_fn(samples)
        return batch, play_keys, play_data_map
    
    def predict(self, test_batch, test_input):
        """
        Make predictions for a batch of plays. Supports Polars or Pandas DataFrames.
        """
        if isinstance(test_batch, pl.DataFrame):
            test_df = test_batch.to_pandas()
        else:
            test_df = test_batch.copy()
        
        if isinstance(test_input, pl.DataFrame):
            test_input_df = test_input.to_pandas()
        else:
            test_input_df = test_input.copy()
        
        if len(test_df) == 0:
            empty_df = pd.DataFrame({'x': [], 'y': []})
            return pl.DataFrame(empty_df) if isinstance(test_batch, pl.DataFrame) else empty_df
        
        with torch.no_grad():
            processed_input = self.preprocessor.process_input_data(test_input_df)
            batch, play_keys, play_data_map = self._prepare_batch(processed_input)
            
            if not batch:
                fallback = np.zeros((len(test_df), 2), dtype=np.float32)
                x_denorm, y_denorm = self.preprocessor.denormalize_positions(fallback[:, 0], fallback[:, 1])
                out_df = pd.DataFrame({'x': x_denorm, 'y': y_denorm})
                return pl.DataFrame(out_df) if isinstance(test_batch, pl.DataFrame) else out_df
            
            num_players = batch['num_players'].numpy().astype(int)
            game_ids = batch['game_ids']
            play_ids = batch['play_ids']
            player_ids_list = batch['player_ids']
            
            batch_tensors = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_tensors[key] = value.to(self.device)
                else:
                    batch_tensors[key] = value
            
            ensemble_outputs = []
            for model in self.models:
                ensemble_outputs.append(model(batch_tensors, teacher_forcing_ratio=0.0))
            predictions = torch.stack(ensemble_outputs, dim=0).mean(dim=0).cpu().numpy()
            output_mask_np = batch_tensors['output_mask'].detach().cpu().numpy()
            
            play_index_map = { (game_ids[i], play_ids[i]): i for i in range(len(game_ids)) }
            pred_norm = np.zeros((len(test_df), 2), dtype=np.float32)
            
            for idx, row in enumerate(test_df.itertuples(index=False)):
                key = (int(row.game_id), int(row.play_id))
                play_idx = play_index_map.get(key)
                if play_idx is None:
                    pred_norm[idx] = 0.0
                    continue
                
                player_list = player_ids_list[play_idx]
                player_map = {pid: i for i, pid in enumerate(player_list)}
                player_idx = player_map.get(int(row.nfl_id))
                
                if player_idx is None or player_idx >= num_players[play_idx]:
                    play_input_rows = play_data_map[key]
                    player_rows = play_input_rows[play_input_rows['nfl_id'] == row.nfl_id]
                    if len(player_rows) == 0:
                        pred_norm[idx] = 0.0
                    else:
                        last_row = player_rows.sort_values('frame_id').iloc[-1]
                        pred_norm[idx, 0] = last_row['x']
                        pred_norm[idx, 1] = last_row['y']
                    continue
                
                frame_idx = int(row.frame_id) - 1
                effective_frames = int(output_mask_np[play_idx, player_idx].sum())
                if effective_frames > 0:
                    frame_idx = max(0, min(frame_idx, effective_frames - 1))
                    pred_norm[idx] = predictions[play_idx, player_idx, frame_idx]
                else:
                    play_input_rows = play_data_map[key]
                    player_rows = play_input_rows[play_input_rows['nfl_id'] == row.nfl_id]
                    if len(player_rows) == 0:
                        pred_norm[idx] = 0.0
                    else:
                        last_row = player_rows.sort_values('frame_id').iloc[-1]
                        pred_norm[idx, 0] = last_row['x']
                        pred_norm[idx, 1] = last_row['y']
            
            x_denorm, y_denorm = self.preprocessor.denormalize_positions(pred_norm[:, 0], pred_norm[:, 1])
            out_df = pd.DataFrame({'x': x_denorm, 'y': y_denorm})
            
            return pl.DataFrame(out_df) if isinstance(test_batch, pl.DataFrame) else out_df


# Global inference model for Kaggle entry point
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'checkpoints'))
CHECKPOINT_PATH = MODEL_DIR / os.getenv('MODEL_FILENAME', 'best_model.pt')
PREPROCESSOR_PATH = MODEL_DIR / os.getenv('PREPROCESSOR_FILENAME', 'preprocessor.pkl')
USE_SIMPLE_MODEL = os.getenv('USE_SIMPLE_MODEL', '0') == '1'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

_inference_model = NFLInferenceModel(
    checkpoint_path=CHECKPOINT_PATH,
    preprocessor_path=PREPROCESSOR_PATH,
    device=DEVICE,
    simple=USE_SIMPLE_MODEL,
)


def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:
    """
    Kaggle evaluation entry point.
    """
    return _inference_model.predict(test, test_input)


def main():
    """Example inference pipeline"""
    
    if not CHECKPOINT_PATH.exists():
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train a model first using train.py")
        return
    
    if not PREPROCESSOR_PATH.exists():
        print(f"Error: Preprocessor not found at {PREPROCESSOR_PATH}")
        print("Please train a model first using train.py")
        return
    
    print(f"Using device: {DEVICE}")
    print("Inference model ready. Use the `predict` function with Kaggle's evaluation gateway.")


if __name__ == "__main__":
    main()
    
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
                ensemble = []
                for model in self.models:
                    ensemble.append(model(batch, teacher_forcing_ratio=0.0))
                predictions = torch.stack(ensemble, dim=0).mean(dim=0)
                
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


