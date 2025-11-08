"""
Data preprocessing utilities for NFL trajectory prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import pickle


class NFLDataPreprocessor:
    """Preprocessor for NFL tracking data"""
    
    def __init__(self, config):
        """
        Args:
            config: DataConfig object with normalization ranges and feature definitions
        """
        self.config = config
        self.role_to_idx = config.role_to_idx
        self.position_to_idx = {}
        self.side_to_idx = {'Offense': 0, 'Defense': 1}
        self.direction_to_idx = {'left': 0, 'right': 1}
        
    def fit_categorical_mappings(self, df: pd.DataFrame):
        """Learn mappings for categorical features from training data"""
        # Position mapping
        unique_positions = df['player_position'].unique()
        self.position_to_idx = {pos: idx for idx, pos in enumerate(sorted(unique_positions))}
        
        print(f"Learned {len(self.position_to_idx)} unique positions")
        print(f"Position mapping: {self.position_to_idx}")
        
    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features to [0, 1] range"""
        df = df.copy()
        
        for feature, (min_val, max_val) in self.config.norm_ranges.items():
            if feature in df.columns:
                # Min-max normalization
                df[feature] = (df[feature] - min_val) / (max_val - min_val)
                # Clip to [0, 1] to handle outliers
                df[feature] = df[feature].clip(0, 1)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features as integers"""
        df = df.copy()
        
        # Role - default to "Other Route Runner" if unseen
        default_role = self.role_to_idx.get('Other Route Runner', 0)
        df['player_role_idx'] = (
            df['player_role']
            .map(self.role_to_idx)
            .fillna(default_role)
            .astype(int)
        )
        
        # Position - extend mapping dynamically for unseen positions
        unknown_positions = df.loc[~df['player_position'].isin(self.position_to_idx.keys()), 'player_position'].unique()
        for pos in unknown_positions:
            self.position_to_idx[pos] = len(self.position_to_idx)
        df['player_position_idx'] = (
            df['player_position']
            .map(self.position_to_idx)
            .fillna(0)
            .astype(int)
        )
        
        # Side - default to Offense (0) if missing
        df['player_side_idx'] = (
            df['player_side']
            .map(self.side_to_idx)
            .fillna(self.side_to_idx.get('Offense', 0))
            .astype(int)
        )
        
        # Direction - default to 'right' (1) if missing
        default_direction = self.direction_to_idx.get('right', 1)
        df['play_direction_idx'] = (
            df['play_direction']
            .str.lower()
            .map(self.direction_to_idx)
            .fillna(default_direction)
            .astype(int)
        )
        
        return df
    
    def parse_height(self, height_str: str) -> float:
        """Convert height string (e.g., '6-1') to inches"""
        try:
            feet, inches = height_str.split('-')
            return int(feet) * 12 + int(inches)
        except:
            return 72  # Default ~6 feet
    
    def process_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process input tracking data (before pass)"""
        df = df.copy()
        
        # Parse height
        df['player_height_inches'] = df['player_height'].apply(self.parse_height)
        
        # Normalize height
        df['player_height_inches'] = (df['player_height_inches'] - 60) / 24  # ~5'0" to ~7'0"
        df['player_height_inches'] = df['player_height_inches'].clip(0, 1)
        
        # Normalize numerical features
        df = self.normalize_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Compute derived features
        df['dist_to_ball'] = np.sqrt(
            (df['x'] - df['ball_land_x'])**2 + 
            (df['y'] - df['ball_land_y'])**2
        )
        
        return df
    
    def process_output_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process output tracking data (during ball flight)"""
        df = df.copy()
        
        # Normalize positions
        df['x'] = (df['x'] - self.config.norm_ranges['x'][0]) / \
                  (self.config.norm_ranges['x'][1] - self.config.norm_ranges['x'][0])
        df['y'] = (df['y'] - self.config.norm_ranges['y'][0]) / \
                  (self.config.norm_ranges['y'][1] - self.config.norm_ranges['y'][0])
        
        df['x'] = df['x'].clip(0, 1)
        df['y'] = df['y'].clip(0, 1)
        
        return df
    
    def denormalize_positions(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert normalized positions back to yards"""
        x_range = self.config.norm_ranges['x']
        y_range = self.config.norm_ranges['y']
        
        x_denorm = x * (x_range[1] - x_range[0]) + x_range[0]
        y_denorm = y * (y_range[1] - y_range[0]) + y_range[0]
        
        return x_denorm, y_denorm
    
    def save(self, path: Path):
        """Save preprocessor state"""
        state = {
            'position_to_idx': self.position_to_idx,
            'role_to_idx': self.role_to_idx,
            'side_to_idx': self.side_to_idx,
            'direction_to_idx': self.direction_to_idx,
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load preprocessor state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.position_to_idx = state['position_to_idx']
        self.role_to_idx = state['role_to_idx']
        self.side_to_idx = state['side_to_idx']
        self.direction_to_idx = state['direction_to_idx']


def load_week_data(data_dir: Path, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load input and output data for a specific week
    
    Args:
        data_dir: Path to train directory
        week: Week number (1-18)
    
    Returns:
        Tuple of (input_df, output_df)
    """
    input_path = data_dir / f"input_2023_w{week:02d}.csv"
    output_path = data_dir / f"output_2023_w{week:02d}.csv"
    
    input_df = pd.read_csv(input_path)
    output_df = pd.read_csv(output_path)
    
    return input_df, output_df


def load_multiple_weeks(data_dir: Path, weeks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate data from multiple weeks
    
    Args:
        data_dir: Path to train directory
        weeks: List of week numbers
    
    Returns:
        Tuple of (combined_input_df, combined_output_df)
    """
    input_dfs = []
    output_dfs = []
    
    for week in weeks:
        input_df, output_df = load_week_data(data_dir, week)
        input_dfs.append(input_df)
        output_dfs.append(output_df)
    
    combined_input = pd.concat(input_dfs, ignore_index=True)
    combined_output = pd.concat(output_dfs, ignore_index=True)
    
    return combined_input, combined_output


def get_play_ids(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Extract unique (game_id, play_id) tuples from dataframe"""
    return df[['game_id', 'play_id']].drop_duplicates().values.tolist()


