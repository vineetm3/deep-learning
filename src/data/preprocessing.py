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
        
        # Position - map to known index, fallback to 0 if unseen
        default_position_idx = 0
        df['player_position_idx'] = (
            df['player_position']
            .map(self.position_to_idx)
            .fillna(default_position_idx)
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
        
        # Derived kinematic features
        dir_rad = np.deg2rad(df['dir'])
        df['vx'] = df['s'] * np.cos(dir_rad)
        df['vy'] = df['s'] * np.sin(dir_rad)
        df['ax'] = df['a'] * np.cos(dir_rad)
        df['ay'] = df['a'] * np.sin(dir_rad)
        
        # Ball-relative features
        df['ball_dx'] = df['ball_land_x'] - df['x']
        df['ball_dy'] = df['ball_land_y'] - df['y']
        df['ball_dist'] = np.sqrt(df['ball_dx'] ** 2 + df['ball_dy'] ** 2)
        ball_angle = np.arctan2(df['ball_dy'], df['ball_dx'])
        df['ball_angle_sin'] = np.sin(ball_angle)
        df['ball_angle_cos'] = np.cos(ball_angle)
        
        # Advanced feature engineering
        df = self._add_opponent_features(df)
        df = self._add_route_features(df)
        df = self._add_geometric_features(df)
        
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
    
    def _add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add nearest opponent distance and pressure-related metrics"""
        df['nearest_opp_dist'] = 30.0
        df['near_opp_count_3'] = 0.0
        df['near_opp_count_5'] = 0.0
        df['closing_speed'] = 0.0
        
        group_cols = ['game_id', 'play_id', 'frame_id']
        for _, group in df.groupby(group_cols, sort=False):
            idx = group.index
            coords = group[['x', 'y']].to_numpy(dtype=np.float32)
            vx = group['vx'].to_numpy(dtype=np.float32)
            vy = group['vy'].to_numpy(dtype=np.float32)
            sides = (group['player_side'] == 'Offense').to_numpy()
            n = len(group)
            
            if n <= 1:
                continue
            
            nearest = np.full(n, 30.0, dtype=np.float32)
            cnt3 = np.zeros(n, dtype=np.float32)
            cnt5 = np.zeros(n, dtype=np.float32)
            closing = np.zeros(n, dtype=np.float32)
            
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                opp_mask = mask & (sides != sides[i])
                if not np.any(opp_mask):
                    continue
                diffs = coords[opp_mask] - coords[i]
                dists = np.linalg.norm(diffs, axis=1)
                nearest[i] = float(dists.min())
                cnt3[i] = float((dists < 3.0).sum())
                cnt5[i] = float((dists < 5.0).sum())
                opp_indices = np.where(opp_mask)[0]
                nearest_idx = opp_indices[dists.argmin()]
                rel_vx = vx[nearest_idx] - vx[i]
                rel_vy = vy[nearest_idx] - vy[i]
                unit = (coords[nearest_idx] - coords[i]) / (nearest[i] + 1e-6)
                closing[i] = float(-(rel_vx * unit[0] + rel_vy * unit[1]))
            
            df.loc[idx, 'nearest_opp_dist'] = nearest
            df.loc[idx, 'near_opp_count_3'] = cnt3
            df.loc[idx, 'near_opp_count_5'] = cnt5
            df.loc[idx, 'closing_speed'] = closing
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-player route statistics across the pre-pass window"""
        df['route_depth'] = 0.0
        df['route_width'] = 0.0
        df['route_straightness'] = 0.0
        df['route_speed_mean'] = 0.0
        df['route_speed_change'] = 0.0
        
        for _, group in df.groupby(['game_id', 'play_id', 'nfl_id'], sort=False):
            group_sorted = group.sort_values('frame_id')
            idx = group_sorted.index
            if len(group_sorted) < 2:
                df.loc[idx, 'route_speed_mean'] = float(group_sorted['s'].iloc[-1])
                continue
            
            x_vals = group_sorted['x'].to_numpy(dtype=np.float32)
            y_vals = group_sorted['y'].to_numpy(dtype=np.float32)
            s_vals = group_sorted['s'].to_numpy(dtype=np.float32)
            
            dx = float(x_vals[-1] - x_vals[0])
            dy = float(y_vals[-1] - y_vals[0])
            steps = np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2)
            total_dist = float(steps.sum())
            displacement = float(np.sqrt(dx**2 + dy**2))
            straightness = float(displacement / (total_dist + 1e-6))
            speed_mean = float(s_vals.mean())
            speed_change = float(s_vals[-1] - s_vals[0])
            
            df.loc[idx, 'route_depth'] = dx
            df.loc[idx, 'route_width'] = dy
            df.loc[idx, 'route_straightness'] = straightness
            df.loc[idx, 'route_speed_mean'] = speed_mean
            df.loc[idx, 'route_speed_change'] = speed_change
        
        return df
    
    def _add_geometric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate geometric endpoints and alignment toward the ball"""
        t_total = (df['num_frames_output'].clip(lower=1).astype(float)) / 10.0
        geo_endpoint_x = df['x'] + df['vx'] * t_total
        geo_endpoint_y = df['y'] + df['vy'] * t_total
        
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        geo_endpoint_x = np.where(receiver_mask, df['ball_land_x'], geo_endpoint_x)
        geo_endpoint_y = np.where(receiver_mask, df['ball_land_y'], geo_endpoint_y)
        
        geo_vector_x = geo_endpoint_x - df['x']
        geo_vector_y = geo_endpoint_y - df['y']
        geo_distance = np.sqrt(geo_vector_x ** 2 + geo_vector_y ** 2)
        
        velocity_mag = np.sqrt(df['vx'] ** 2 + df['vy'] ** 2) + 1e-6
        geo_alignment = (
            (df['vx'] * geo_vector_x + df['vy'] * geo_vector_y) /
            ((geo_distance + 1e-6) * velocity_mag)
        )
        
        df['geo_distance'] = geo_distance
        df['geo_alignment'] = geo_alignment
        
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


