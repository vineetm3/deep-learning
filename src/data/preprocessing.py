"""
Data preprocessing utilities for NFL trajectory prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
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
        self.num_workers = max(1, getattr(config, 'preprocessing_workers', 1))
        
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
        groups = list(df.groupby(group_cols, sort=False))
        if not groups:
            return df
        
        if self.num_workers > 1:
            results = Parallel(n_jobs=self.num_workers)(
                delayed(self._compute_opponent_metrics_for_group)(group)
                for _, group in groups
            )
        else:
            results = [self._compute_opponent_metrics_for_group(group) for _, group in groups]
        
        for idx, nearest_dist, within_3, within_5, closing_speed in results:
            if idx.size == 0:
                continue
            df.loc[idx, 'nearest_opp_dist'] = nearest_dist
            df.loc[idx, 'near_opp_count_3'] = within_3
            df.loc[idx, 'near_opp_count_5'] = within_5
            df.loc[idx, 'closing_speed'] = closing_speed
        
        return df
    
    def _add_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add per-player route statistics across the pre-pass window"""
        df['route_depth'] = 0.0
        df['route_width'] = 0.0
        df['route_straightness'] = 0.0
        df['route_speed_mean'] = 0.0
        df['route_speed_change'] = 0.0
        
        groups = list(df.groupby(['game_id', 'play_id', 'nfl_id'], sort=False))
        if self.num_workers > 1:
            results = Parallel(n_jobs=self.num_workers)(
                delayed(self._compute_route_features_for_group)(group)
                for _, group in groups
            )
        else:
            results = [self._compute_route_features_for_group(group) for _, group in groups]
        
        for idx, depth, width, straightness, speed_mean, speed_change in results:
            if idx.size == 0:
                continue
            df.loc[idx, 'route_depth'] = depth
            df.loc[idx, 'route_width'] = width
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

    @staticmethod
    def _compute_opponent_metrics_for_group(group: pd.DataFrame):
        idx = group.index.to_numpy(dtype=np.int64)
        n_players = len(idx)
        if n_players == 0:
            return idx, np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        if n_players == 1:
            return (
                idx,
                np.full(1, 30.0, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
            )
        
        coords = group[['x', 'y']].to_numpy(dtype=np.float32)
        velocities = group[['vx', 'vy']].to_numpy(dtype=np.float32)
        sides = (group['player_side'] == 'Offense').to_numpy(dtype=bool)
        
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        opp_mask = sides[:, None] != sides[None, :]
        opp_dists = np.where(opp_mask, dists, np.inf)
        
        nearest_idx = np.argmin(opp_dists, axis=1)
        nearest_dist = opp_dists[np.arange(n_players), nearest_idx]
        has_opp = np.isfinite(nearest_dist)
        
        within_3 = np.sum((opp_dists < 3.0), axis=1, dtype=np.int32).astype(np.float32)
        within_5 = np.sum((opp_dists < 5.0), axis=1, dtype=np.int32).astype(np.float32)
        
        rel_vectors = coords[nearest_idx] - coords
        rel_unit = rel_vectors / (nearest_dist[:, None] + 1e-6)
        rel_vel = velocities[nearest_idx] - velocities
        closing_speed = -np.einsum('ij,ij->i', rel_vel, rel_unit).astype(np.float32)
        closing_speed[~has_opp] = 0.0
        
        nearest_dist = nearest_dist.astype(np.float32)
        nearest_dist[~has_opp] = 30.0
        within_3[~has_opp] = 0.0
        within_5[~has_opp] = 0.0
        
        return idx, nearest_dist, within_3, within_5, closing_speed

    @staticmethod
    def _compute_route_features_for_group(group: pd.DataFrame):
        group_sorted = group.sort_values('frame_id')
        idx = group_sorted.index.to_numpy(dtype=np.int64)
        values = group_sorted[['x', 'y', 's']].to_numpy(dtype=np.float32)
        n_frames = len(values)
        if n_frames == 0:
            zeros = np.zeros(0, dtype=np.float32)
            return idx, zeros, zeros, zeros, zeros, zeros
        if n_frames == 1:
            speed = np.full(1, values[0, 2], dtype=np.float32)
            zeros = np.zeros(1, dtype=np.float32)
            return idx, zeros, zeros, zeros, speed, zeros
        
        delta = values[-1] - values[0]
        steps = values[1:, :2] - values[:-1, :2]
        step_lengths = np.linalg.norm(steps, axis=1)
        total_distance = float(step_lengths.sum())
        displacement = float(np.linalg.norm(delta[:2]))
        straightness = float(displacement / (total_distance + 1e-6))
        speed_mean = float(values[:, 2].mean())
        speed_change = float(values[-1, 2] - values[0, 2])
        
        depth = np.full(n_frames, float(delta[0]), dtype=np.float32)
        width = np.full(n_frames, float(delta[1]), dtype=np.float32)
        straightness_arr = np.full(n_frames, straightness, dtype=np.float32)
        speed_mean_arr = np.full(n_frames, speed_mean, dtype=np.float32)
        speed_change_arr = np.full(n_frames, speed_change, dtype=np.float32)
        
        return idx, depth, width, straightness_arr, speed_mean_arr, speed_change_arr
    
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


