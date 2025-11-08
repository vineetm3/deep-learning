"""
PyTorch Dataset for NFL trajectory prediction
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.data.preprocessing import NFLDataPreprocessor, get_play_ids

INPUT_FEATURE_COLUMNS = [
    'x',
    'y',
    's',
    'a',
    'dir',
    'o',
    'player_height_inches',
    'player_weight',
    'dist_to_ball',
    'vx',
    'vy',
    'ax',
    'ay',
    'ball_dx',
    'ball_dy',
    'ball_dist',
    'ball_angle_sin',
    'ball_angle_cos',
]
FEATURE_INDEX = {name: idx for idx, name in enumerate(INPUT_FEATURE_COLUMNS)}


class NFLTrajectoryDataset(Dataset):
    """
    Dataset for NFL player trajectory prediction
    
    Each sample represents one play with multiple players
    """
    
    def __init__(
        self,
        input_df: pd.DataFrame,
        output_df: pd.DataFrame,
        preprocessor: NFLDataPreprocessor,
        max_input_frames: int = 80,
        max_output_frames: int = 100,
    ):
        """
        Args:
            input_df: Input tracking data (before pass)
            output_df: Output tracking data (during ball flight)
            preprocessor: Fitted preprocessor
            max_input_frames: Maximum number of input frames (will pad if less)
            max_output_frames: Maximum number of output frames (will pad if less)
        """
        self.preprocessor = preprocessor
        self.max_input_frames = max_input_frames
        self.max_output_frames = max_output_frames
        
        # Preprocess data
        print("Preprocessing input data...")
        self.input_df = preprocessor.process_input_data(input_df)
        
        print("Preprocessing output data...")
        self.output_df = preprocessor.process_output_data(output_df)
        
        # Get unique plays
        self.play_ids = get_play_ids(self.input_df)
        print(f"Loaded {len(self.play_ids)} plays")
        
        # Feature columns for input
        self.input_feature_cols = INPUT_FEATURE_COLUMNS.copy()
        
        # Categorical feature columns
        self.categorical_cols = [
            'player_role_idx', 'player_position_idx',
            'player_side_idx', 'play_direction_idx'
        ]
        
    def __len__(self) -> int:
        return len(self.play_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single play
        
        Returns dictionary with:
            - input_features: [num_players, max_input_frames, num_features]
            - categorical_features: [num_players, num_categorical]
            - output_positions: [num_players, max_output_frames, 2]  (x, y)
            - ball_landing: [2]  (ball_land_x, ball_land_y)
            - num_players: int
            - input_mask: [num_players, max_input_frames]  (1 for valid, 0 for padding)
            - output_mask: [num_players, max_output_frames]  (1 for valid, 0 for padding)
            - player_roles: [num_players]  (role indices)
            - player_to_predict: [num_players]  (boolean, 1 if should predict)
            - num_output_frames: int (actual number of output frames)
        """
        game_id, play_id = self.play_ids[idx]
        
        # Get input data for this play
        input_play = self.input_df[
            (self.input_df['game_id'] == game_id) & 
            (self.input_df['play_id'] == play_id)
        ].copy()
        
        # Get output data for this play
        output_play = self.output_df[
            (self.output_df['game_id'] == game_id) & 
            (self.output_df['play_id'] == play_id)
        ].copy()
        
        # Get unique players
        player_ids = input_play['nfl_id'].unique()
        num_players = len(player_ids)
        
        # Initialize tensors
        input_features = np.zeros((num_players, self.max_input_frames, len(self.input_feature_cols)))
        categorical_features = np.zeros((num_players, len(self.categorical_cols)))
        output_positions = np.zeros((num_players, self.max_output_frames, 2))
        input_mask = np.zeros((num_players, self.max_input_frames))
        output_mask = np.zeros((num_players, self.max_output_frames))
        player_roles = np.zeros(num_players, dtype=np.int64)
        player_to_predict = np.zeros(num_players, dtype=np.int64)
        
        # Get ball landing position (same for all players in a play)
        ball_landing = input_play[['ball_land_x', 'ball_land_y']].iloc[0].values
        num_output_frames = input_play['num_frames_output'].iloc[0]
        
        # Fill in data for each player
        for i, player_id in enumerate(player_ids):
            # Input data
            player_input = input_play[input_play['nfl_id'] == player_id].sort_values('frame_id')
            n_input_frames = min(len(player_input), self.max_input_frames)
            
            # Take the last max_input_frames (most recent before pass)
            if len(player_input) > self.max_input_frames:
                player_input = player_input.iloc[-self.max_input_frames:]
            
            input_features[i, :n_input_frames] = player_input[self.input_feature_cols].values
            input_mask[i, :n_input_frames] = 1
            
            # Categorical features (same across all frames)
            categorical_features[i] = player_input[self.categorical_cols].iloc[0].values
            
            # Role and prediction flag
            player_roles[i] = player_input['player_role_idx'].iloc[0]
            player_to_predict[i] = player_input['player_to_predict'].iloc[0]
            
            # Output data
            player_output = output_play[output_play['nfl_id'] == player_id].sort_values('frame_id')
            n_output_frames = min(len(player_output), self.max_output_frames)
            
            if len(player_output) > 0:
                output_positions[i, :n_output_frames] = player_output[['x', 'y']].values[:n_output_frames]
                output_mask[i, :n_output_frames] = 1
        
        return {
            'input_features': torch.FloatTensor(input_features),
            'categorical_features': torch.LongTensor(categorical_features),
            'output_positions': torch.FloatTensor(output_positions),
            'ball_landing': torch.FloatTensor(ball_landing),
            'num_players': num_players,
            'input_mask': torch.FloatTensor(input_mask),
            'output_mask': torch.FloatTensor(output_mask),
            'player_roles': torch.LongTensor(player_roles),
            'player_to_predict': torch.LongTensor(player_to_predict),
            'num_output_frames': num_output_frames,
            'game_id': game_id,
            'play_id': play_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable number of players per play
    
    Pads to max number of players in batch
    """
    max_players = max(sample['num_players'] for sample in batch)
    batch_size = len(batch)
    
    # Get dimensions
    max_input_frames = batch[0]['input_features'].shape[1]
    max_output_frames = batch[0]['output_positions'].shape[1]
    num_input_features = batch[0]['input_features'].shape[2]
    num_categorical = batch[0]['categorical_features'].shape[1]
    
    # Initialize batched tensors
    batched = {
        'input_features': torch.zeros(batch_size, max_players, max_input_frames, num_input_features),
        'categorical_features': torch.zeros(batch_size, max_players, num_categorical, dtype=torch.long),
        'output_positions': torch.zeros(batch_size, max_players, max_output_frames, 2),
        'ball_landing': torch.zeros(batch_size, 2),
        'num_players': torch.zeros(batch_size, dtype=torch.long),
        'input_mask': torch.zeros(batch_size, max_players, max_input_frames),
        'output_mask': torch.zeros(batch_size, max_players, max_output_frames),
        'player_mask': torch.zeros(batch_size, max_players),  # Mask for actual players vs padding
        'player_roles': torch.zeros(batch_size, max_players, dtype=torch.long),
        'player_to_predict': torch.zeros(batch_size, max_players, dtype=torch.long),
        'num_output_frames': [],
        'game_ids': [],
        'play_ids': [],
    }
    
    # Fill in data
    for i, sample in enumerate(batch):
        n_players = sample['num_players']
        
        batched['input_features'][i, :n_players] = sample['input_features']
        batched['categorical_features'][i, :n_players] = sample['categorical_features']
        batched['output_positions'][i, :n_players] = sample['output_positions']
        batched['ball_landing'][i] = sample['ball_landing']
        batched['num_players'][i] = n_players
        batched['input_mask'][i, :n_players] = sample['input_mask']
        batched['output_mask'][i, :n_players] = sample['output_mask']
        batched['player_mask'][i, :n_players] = 1
        batched['player_roles'][i, :n_players] = sample['player_roles']
        batched['player_to_predict'][i, :n_players] = sample['player_to_predict']
        batched['num_output_frames'].append(sample['num_output_frames'])
        batched['game_ids'].append(sample['game_id'])
        batched['play_ids'].append(sample['play_id'])
    
    return batched


def create_dataloaders(
    config,
    preprocessor: NFLDataPreprocessor,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: DataConfig object
        preprocessor: Fitted NFLDataPreprocessor
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from src.data.preprocessing import load_multiple_weeks
    
    # Load data
    print("Loading training data...")
    train_input, train_output = load_multiple_weeks(config.train_dir, config.train_weeks)
    
    print("Loading validation data...")
    val_input, val_output = load_multiple_weeks(config.train_dir, config.val_weeks)
    
    print("Loading test data...")
    test_input, test_output = load_multiple_weeks(config.train_dir, config.test_weeks)
    
    # Create datasets
    train_dataset = NFLTrajectoryDataset(
        train_input, train_output, preprocessor,
        config.max_input_frames, config.max_output_frames
    )
    
    val_dataset = NFLTrajectoryDataset(
        val_input, val_output, preprocessor,
        config.max_input_frames, config.max_output_frames
    )
    
    test_dataset = NFLTrajectoryDataset(
        test_input, test_output, preprocessor,
        config.max_input_frames, config.max_output_frames
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader, test_loader


