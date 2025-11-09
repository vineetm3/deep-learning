my work based on these excellent notebooks:

https://www.kaggle.com/code/yusufsertkayaysk/nfl-big-data-bowl-2026-lb-0-604

https://www.kaggle.com/code/ryanadamsai/nfl-big-data-bowl-geometric-gnn-lb-586

https://www.kaggle.com/code/muran169633/baseline-0-583

huge respect to the authors. I was also insipired by the ideas on the discussions.

My train notebook is https://www.kaggle.com/code/goose666/1109mytrain0579

The score after inference using the trained weight file is 0.579. But I lost my 0.577 train notebook.I only adjusted some parameters here, but I've forgotten the details.

About
This solution integrates a variety of effective feature engineering techniques, which I've encapsulated into functions and called during the creation of frame sequences, resulting in a total of 187 features. The model employs a STtransformer with residual MLP blocks, and has undergone some hyperparameter tuning. It predicts positional residuals. Training takes approximately one and a half hours, and inference followed by evaluation requires about two hours.

Future
The current model is overfitting. My friend suggested that this is because we are predicting the residual of the position rather than the position itself. Also, some of the features created during the feature engineering phase may be redundant and ineffective. I will spend more time exploring data preprocessing and feature engineering next. Perhaps this task does not require a very complex model, we need to refocus on the data itself.(my English is not good)

Import
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
import warnings
import os
import pickle
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings('ignore')
Config
class Config:
    DATA_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction/")
    
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 256
    WINDOW_SIZE = 12
    HIDDEN_DIM = 128
    MAX_FUTURE_HORIZON = 94

    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(Config.SEED)
Model Part
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.activation(self.net(x) + x)

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, horizon, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_pos_encoding = nn.Parameter(torch.randn(1, Config.WINDOW_SIZE, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.prediction_head = ResidualMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=horizon,
            num_layers=3,
            dropout=dropout
        )
        
        self.output_norm = nn.LayerNorm(horizon)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0.0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x):
        batch_size, window_size, _ = x.shape
        
        x = self.input_projection(x)
        x = x + self.temporal_pos_encoding[:, :window_size, :]
        x = self.transformer_encoder(x)
        
        attention_weights = torch.softmax(torch.mean(x, dim=-1), dim=-1)
        x_pooled = torch.sum(x * attention_weights.unsqueeze(-1), dim=1)
        
        pred = self.prediction_head(x_pooled)
        pred = self.output_norm(pred)
        pred = torch.cumsum(pred, dim=1)
        
        return pred

class ImprovedSeqModel(nn.Module):
    def __init__(self, input_dim, horizon):
        super().__init__()
        self.horizon = horizon
        self.model = SpatioTemporalTransformer(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        )
    
    def forward(self, x):
        return self.model(x)
Feature
def height_to_feet(height_str):
    try:
        ft, inches = map(int, str(height_str).split('-'))
        return ft + inches/12
    except:
        return 6.0

def get_velocity(speed, direction_deg):
    theta = np.deg2rad(direction_deg)
    return speed * np.sin(theta), speed * np.cos(theta)

def create_base_features(input_df):
    df = input_df.copy()
    
    df['player_height_feet'] = df['player_height'].apply(height_to_feet)

    height_parts = df['player_height'].str.split('-', expand=True)
    df['height_inches'] = height_parts[0].astype(float) * 12 + height_parts[1].astype(float)
    df['bmi'] = (df['player_weight'] / (df['height_inches']**2)) * 703

    dir_rad = np.deg2rad(df['dir'].fillna(0))
    df['velocity_x'] = df['s'] * np.sin(dir_rad)
    df['velocity_y'] = df['s'] * np.cos(dir_rad)
    df['acceleration_x'] = df['a'] * np.cos(dir_rad)
    df['acceleration_y'] = df['a'] * np.sin(dir_rad)

    df['is_offense'] = (df['player_side'] == 'Offense').astype(int)
    df['is_defense'] = (df['player_side'] == 'Defense').astype(int)
    df['is_receiver'] = (df['player_role'] == 'Targeted Receiver').astype(int)
    df['is_coverage'] = (df['player_role'] == 'Defensive Coverage').astype(int)
    df['is_passer'] = (df['player_role'] == 'Passer').astype(int)

    df['role_targeted_receiver'] = df['is_receiver']
    df['role_defensive_coverage'] = df['is_coverage']
    df['role_passer'] = df['is_passer']
    df['side_offense'] = df['is_offense']

    mass_kg = df['player_weight'].fillna(200.0) / 2.20462
    df['momentum_x'] = df['velocity_x'] * df['player_weight']
    df['momentum_y'] = df['velocity_y'] * df['player_weight']
    df['kinetic_energy'] = 0.5 * df['player_weight'] * (df['s'] ** 2)

    df['speed_squared'] = df['s'] ** 2
    df['accel_magnitude'] = np.sqrt(df['acceleration_x']**2 + df['acceleration_y']**2)
    df['orientation_diff'] = np.abs(df['o'] - df['dir'])
    df['orientation_diff'] = np.minimum(df['orientation_diff'], 360 - df['orientation_diff'])

    if 'ball_land_x' in df.columns:
        ball_dx = df['ball_land_x'] - df['x']
        ball_dy = df['ball_land_y'] - df['y']
        df['distance_to_ball'] = np.sqrt(ball_dx**2 + ball_dy**2)
        df['dist_to_ball'] = df['distance_to_ball']
        df['dist_squared'] = df['distance_to_ball'] ** 2
        df['angle_to_ball'] = np.arctan2(ball_dy, ball_dx)
        df['ball_direction_x'] = ball_dx / (df['distance_to_ball'] + 1e-6)
        df['ball_direction_y'] = ball_dy / (df['distance_to_ball'] + 1e-6)
        df['closing_speed_ball'] = (
            df['velocity_x'] * df['ball_direction_x'] +
            df['velocity_y'] * df['ball_direction_y']
        )
        df['velocity_toward_ball'] = (
            df['velocity_x'] * np.cos(df['angle_to_ball']) + 
            df['velocity_y'] * np.sin(df['angle_to_ball'])
        )
        df['velocity_alignment'] = np.cos(df['angle_to_ball'] - dir_rad)
        df['angle_diff'] = np.abs(df['o'] - np.degrees(df['angle_to_ball']))
        df['angle_diff'] = np.minimum(df['angle_diff'], 360 - df['angle_diff'])
    
    return df

def create_lag_features(df, window_size=8):
    df = df.copy()
    
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']

    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            if col in df.columns:
                df[f'{col}_lag{lag}'] = df.groupby(gcols)[col].shift(lag)

    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            if col in df.columns:
                df[f'{col}_rolling_mean_{window}'] = (
                    df.groupby(gcols)[col]
                    .rolling(window, min_periods=1).mean()
                    .reset_index(level=[0,1,2], drop=True)
                )
                df[f'{col}_rolling_std_{window}'] = (
                    df.groupby(gcols)[col]
                    .rolling(window, min_periods=1).std()
                    .reset_index(level=[0,1,2], drop=True)
                )

    for col in ['velocity_x', 'velocity_y']:
        if col in df.columns:
            df[f'{col}_delta'] = df.groupby(gcols)[col].diff()

    df['velocity_x_ema'] = df.groupby(gcols)['velocity_x'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    df['velocity_y_ema'] = df.groupby(gcols)['velocity_y'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    df['speed_ema'] = df.groupby(gcols)['s'].transform(
        lambda x: x.ewm(alpha=0.3, adjust=False).mean()
    )
    
    return df

def get_opponent_features(input_df):
    features = []
    
    for (gid, pid), group in tqdm(input_df.groupby(['game_id', 'play_id']),
                                desc="🏈 Opponents", leave=False):
        last = group.sort_values('frame_id').groupby('nfl_id').last()

        if len(last) < 2:
            continue

        positions = last[['x', 'y']].values
        sides = last['player_side'].values
        speeds = last['s'].values
        directions = last['dir'].values
        roles = last['player_role'].values

        receiver_mask = np.isin(roles, ['Targeted Receiver', 'Other Route Runner'])

        for i, (nid, side, role) in enumerate(zip(last.index, sides, roles)):
            opp_mask = sides != side

            feat = {
                'game_id': gid, 'play_id': pid, 'nfl_id': nid,
                'nearest_opp_dist': 50.0, 'closing_speed': 0.0,
                'num_nearby_opp_3': 0, 'num_nearby_opp_5': 0,
                'mirror_wr_vx': 0.0, 'mirror_wr_vy': 0.0,
                'mirror_offset_x': 0.0, 'mirror_offset_y': 0.0,
                'mirror_wr_dist': 50.0,
            }

            if not opp_mask.any():
                features.append(feat)
                continue

            opp_positions = positions[opp_mask]
            distances = np.sqrt(((positions[i] - opp_positions) ** 2).sum(axis=1))

            if len(distances) == 0:
                features.append(feat)
                continue

            nearest_idx = distances.argmin()
            feat['nearest_opp_dist'] = distances[nearest_idx]
            feat['num_nearby_opp_3'] = (distances < 3.0).sum()
            feat['num_nearby_opp_5'] = (distances < 5.0).sum()

            my_vx, my_vy = get_velocity(speeds[i], directions[i])
            opp_speeds = speeds[opp_mask]
            opp_dirs = directions[opp_mask]
            opp_vx, opp_vy = get_velocity(opp_speeds[nearest_idx], opp_dirs[nearest_idx])

            rel_vx = my_vx - opp_vx
            rel_vy = my_vy - opp_vy
            to_me = positions[i] - opp_positions[nearest_idx]
            to_me_norm = to_me / (np.linalg.norm(to_me) + 0.1)
            feat['closing_speed'] = -(rel_vx * to_me_norm[0] + rel_vy * to_me_norm[1])

            if role == 'Defensive Coverage' and receiver_mask.any():
                rec_positions = positions[receiver_mask]
                rec_distances = np.sqrt(((positions[i] - rec_positions) ** 2).sum(axis=1))

                if len(rec_distances) > 0:
                    closest_rec_idx = rec_distances.argmin()
                    rec_indices = np.where(receiver_mask)[0]
                    actual_rec_idx = rec_indices[closest_rec_idx]

                    rec_vx, rec_vy = get_velocity(speeds[actual_rec_idx], directions[actual_rec_idx])

                    feat['mirror_wr_vx'] = rec_vx
                    feat['mirror_wr_vy'] = rec_vy
                    feat['mirror_wr_dist'] = rec_distances[closest_rec_idx]
                    feat['mirror_offset_x'] = positions[i][0] - rec_positions[closest_rec_idx][0]
                    feat['mirror_offset_y'] = positions[i][1] - rec_positions[closest_rec_idx][1]

            features.append(feat)

    return pd.DataFrame(features)

def extract_route_patterns(input_df, kmeans=None, scaler=None, fit=False):
    route_features = []
    
    for (gid, pid, nid), group in tqdm(input_df.groupby(['game_id', 'play_id', 'nfl_id']), 
                                      desc="🛣️ Routes", leave=False):
        traj = group.sort_values('frame_id').tail(5)
        
        if len(traj) < 3:
            continue
        
        positions = traj[['x', 'y']].values
        speeds = traj['s'].values
        
        total_dist = np.sum(np.sqrt(np.diff(positions[:, 0])**2 + np.diff(positions[:, 1])**2))
        displacement = np.sqrt((positions[-1, 0] - positions[0, 0])**2 + 
                               (positions[-1, 1] - positions[0, 1])**2)
        straightness = displacement / (total_dist + 0.1)
        
        angles = np.arctan2(np.diff(positions[:, 1]), np.diff(positions[:, 0]))
        if len(angles) > 1:
            angle_changes = np.abs(np.diff(angles))
            max_turn = np.max(angle_changes)
            mean_turn = np.mean(angle_changes)
        else:
            max_turn = mean_turn = 0
        
        speed_mean = speeds.mean()
        speed_change = speeds[-1] - speeds[0] if len(speeds) > 1 else 0
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        route_features.append({
            'game_id': gid, 'play_id': pid, 'nfl_id': nid,
            'traj_straightness': straightness,
            'traj_max_turn': max_turn,
            'traj_mean_turn': mean_turn,
            'traj_depth': abs(dx),
            'traj_width': abs(dy),
            'speed_mean': speed_mean,
            'speed_change': speed_change,
        })
    
    route_df = pd.DataFrame(route_features)
    if route_df.empty or 'traj_straightness' not in route_df.columns:
        return pd.DataFrame()
            
    feat_cols = ['traj_straightness', 'traj_max_turn', 'traj_mean_turn',
                 'traj_depth', 'traj_width', 'speed_mean', 'speed_change']
    X = route_df[feat_cols].fillna(0)
    
    if kmeans is None or scaler is None:
        return route_df
    
    X_scaled = scaler.transform(X)
    route_df['route_pattern'] = kmeans.predict(X_scaled)
    return route_df

def compute_neighbor_embeddings(input_df, k_neigh=Config.K_NEIGH, 
                                radius=Config.RADIUS, tau=Config.TAU):
    
    cols_needed = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y", 
                   "velocity_x", "velocity_y", "player_side"]
    src = input_df[cols_needed].copy()
    
    last = (src.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
               .groupby(["game_id", "play_id", "nfl_id"], as_index=False)
               .tail(1)
               .rename(columns={"frame_id": "last_frame_id"})
               .reset_index(drop=True))
    
    tmp = last.merge(
        src.rename(columns={
            "frame_id": "nb_frame_id", "nfl_id": "nfl_id_nb",
            "x": "x_nb", "y": "y_nb", 
            "velocity_x": "vx_nb", "velocity_y": "vy_nb", 
            "player_side": "player_side_nb"
        }),
        left_on=["game_id", "play_id", "last_frame_id"],
        right_on=["game_id", "play_id", "nb_frame_id"],
        how="left"
    )
    
    tmp = tmp[tmp["nfl_id_nb"] != tmp["nfl_id"]]
    tmp["dx"] = tmp["x_nb"] - tmp["x"]
    tmp["dy"] = tmp["y_nb"] - tmp["y"]
    tmp["dvx"] = tmp["vx_nb"] - tmp["velocity_x"]
    tmp["dvy"] = tmp["vy_nb"] - tmp["velocity_y"]
    tmp["dist"] = np.sqrt(tmp["dx"]**2 + tmp["dy"]**2)
    
    tmp = tmp[np.isfinite(tmp["dist"]) & (tmp["dist"] > 1e-6)]
    if radius is not None:
        tmp = tmp[tmp["dist"] <= radius]
    
    tmp["is_ally"] = (tmp["player_side_nb"] == tmp["player_side"]).astype(np.float32)
    
    keys = ["game_id", "play_id", "nfl_id"]
    tmp["rnk"] = tmp.groupby(keys)["dist"].rank(method="first")
    if k_neigh is not None:
        tmp = tmp[tmp["rnk"] <= float(k_neigh)]
    
    tmp["w"] = np.exp(-tmp["dist"] / float(tau))
    sum_w = tmp.groupby(keys)["w"].transform("sum")
    tmp["wn"] = np.where(sum_w > 0, tmp["w"] / sum_w, 0.0)
    
    tmp["wn_ally"] = tmp["wn"] * tmp["is_ally"]
    tmp["wn_opp"] = tmp["wn"] * (1.0 - tmp["is_ally"])
    
    for col in ["dx", "dy", "dvx", "dvy"]:
        tmp[f"{col}_ally_w"] = tmp[col] * tmp["wn_ally"]
        tmp[f"{col}_opp_w"] = tmp[col] * tmp["wn_opp"]
    
    tmp["dist_ally"] = np.where(tmp["is_ally"] > 0.5, tmp["dist"], np.nan)
    tmp["dist_opp"] = np.where(tmp["is_ally"] < 0.5, tmp["dist"], np.nan)
    
    ag = tmp.groupby(keys).agg(
        gnn_ally_dx_mean=("dx_ally_w", "sum"),
        gnn_ally_dy_mean=("dy_ally_w", "sum"),
        gnn_ally_dvx_mean=("dvx_ally_w", "sum"),
        gnn_ally_dvy_mean=("dvy_ally_w", "sum"),
        gnn_opp_dx_mean=("dx_opp_w", "sum"),
        gnn_opp_dy_mean=("dy_opp_w", "sum"),
        gnn_opp_dvx_mean=("dvx_opp_w", "sum"),
        gnn_opp_dvy_mean=("dvy_opp_w", "sum"),
        gnn_ally_cnt=("is_ally", "sum"),
        gnn_opp_cnt=("is_ally", lambda s: float(len(s) - s.sum())),
        gnn_ally_dmin=("dist_ally", "min"),
        gnn_ally_dmean=("dist_ally", "mean"),
        gnn_opp_dmin=("dist_opp", "min"),
        gnn_opp_dmean=("dist_opp", "mean"),
    ).reset_index()
    
    near = tmp.loc[tmp["rnk"] <= 3, keys + ["rnk", "dist"]].copy()
    if len(near) > 0:
        near["rnk"] = near["rnk"].astype(int)
        dwide = near.pivot_table(index=keys, columns="rnk", values="dist", aggfunc="first")
        dwide = dwide.rename(columns={1: "gnn_d1", 2: "gnn_d2", 3: "gnn_d3"}).reset_index()
        ag = ag.merge(dwide, on=keys, how="left")
    
    for c in ["gnn_ally_dx_mean", "gnn_ally_dy_mean", "gnn_ally_dvx_mean", "gnn_ally_dvy_mean",
              "gnn_opp_dx_mean", "gnn_opp_dy_mean", "gnn_opp_dvx_mean", "gnn_opp_dvy_mean"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_cnt", "gnn_opp_cnt"]:
        ag[c] = ag[c].fillna(0.0)
    for c in ["gnn_ally_dmin", "gnn_opp_dmin", "gnn_ally_dmean", "gnn_opp_dmean", 
              "gnn_d1", "gnn_d2", "gnn_d3"]:
        ag[c] = ag[c].fillna(radius if radius is not None else 30.0)
    
    return ag

def compute_geometric_endpoint(df):
    """基于几何规则计算每个球员的终点位置"""
    df = df.copy()

    if 'num_frames_output' in df.columns:
        t_total = df['num_frames_output'] / 10.0
    else:
        t_total = 3.0
    
    df['time_to_endpoint'] = t_total

    df['geo_endpoint_x'] = df['x'] + df['velocity_x'] * t_total
    df['geo_endpoint_y'] = df['y'] + df['velocity_y'] * t_total

    if 'ball_land_x' in df.columns:
        receiver_mask = df['player_role'] == 'Targeted Receiver'
        df.loc[receiver_mask, 'geo_endpoint_x'] = df.loc[receiver_mask, 'ball_land_x']
        df.loc[receiver_mask, 'geo_endpoint_y'] = df.loc[receiver_mask, 'ball_land_y']

        defender_mask = df['player_role'] == 'Defensive Coverage'
        has_mirror = df.get('mirror_offset_x', 0).notna() & (df.get('mirror_wr_dist', 50) < 15)
        coverage_mask = defender_mask & has_mirror
        
        df.loc[coverage_mask, 'geo_endpoint_x'] = (
            df.loc[coverage_mask, 'ball_land_x'] + 
            df.loc[coverage_mask, 'mirror_offset_x'].fillna(0)
        )
        df.loc[coverage_mask, 'geo_endpoint_y'] = (
            df.loc[coverage_mask, 'ball_land_y'] + 
            df.loc[coverage_mask, 'mirror_offset_y'].fillna(0)
        )

    df['geo_endpoint_x'] = df['geo_endpoint_x'].clip(Config.FIELD_X_MIN, Config.FIELD_X_MAX)
    df['geo_endpoint_y'] = df['geo_endpoint_y'].clip(Config.FIELD_Y_MIN, Config.FIELD_Y_MAX)
    
    return df

def add_geometric_features(df):
    df = compute_geometric_endpoint(df)

    df['geo_vector_x'] = df['geo_endpoint_x'] - df['x']
    df['geo_vector_y'] = df['geo_endpoint_y'] - df['y']
    df['geo_distance'] = np.sqrt(df['geo_vector_x']**2 + df['geo_vector_y']**2)

    t = df['time_to_endpoint'] + 0.1
    df['geo_required_vx'] = df['geo_vector_x'] / t
    df['geo_required_vy'] = df['geo_vector_y'] / t

    df['geo_velocity_error_x'] = df['geo_required_vx'] - df['velocity_x']
    df['geo_velocity_error_y'] = df['geo_required_vy'] - df['velocity_y']
    df['geo_velocity_error'] = np.sqrt(
        df['geo_velocity_error_x']**2 + df['geo_velocity_error_y']**2
    )

    t_sq = t * t
    df['geo_required_ax'] = 2 * df['geo_vector_x'] / t_sq
    df['geo_required_ay'] = 2 * df['geo_vector_y'] / t_sq
    df['geo_required_ax'] = df['geo_required_ax'].clip(-10, 10)
    df['geo_required_ay'] = df['geo_required_ay'].clip(-10, 10)

    velocity_mag = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)
    geo_unit_x = df['geo_vector_x'] / (df['geo_distance'] + 0.1)
    geo_unit_y = df['geo_vector_y'] / (df['geo_distance'] + 0.1)
    df['geo_alignment'] = (
        df['velocity_x'] * geo_unit_x + df['velocity_y'] * geo_unit_y
    ) / (velocity_mag + 0.1)

    df['geo_receiver_urgency'] = df['is_receiver'] * df['geo_distance'] / (t + 0.1)
    df['geo_defender_coupling'] = df['is_coverage'] * (1.0 / (df.get('mirror_wr_dist', 50) + 1.0))
    
    return df

def add_advanced_features(df):
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    gcols = ['game_id', 'play_id', 'nfl_id']

    if 'distance_to_ball' in df.columns:
        df['distance_to_ball_change'] = df.groupby(gcols)['distance_to_ball'].diff().fillna(0)
        df['distance_to_ball_accel'] = df.groupby(gcols)['distance_to_ball_change'].diff().fillna(0)
        df['time_to_intercept'] = (df['distance_to_ball'] / 
                                  (np.abs(df['distance_to_ball_change']) + 0.1)).clip(0, 10)

    if 'ball_direction_x' in df.columns:
        df['velocity_alignment'] = (
            df['velocity_x'] * df['ball_direction_x'] +
            df['velocity_y'] * df['ball_direction_y']
        )
        df['velocity_perpendicular'] = (
            df['velocity_x'] * (-df['ball_direction_y']) +
            df['velocity_y'] * df['ball_direction_x']
        )
        if 'acceleration_x' in df.columns:
            df['accel_alignment'] = (
                df['acceleration_x'] * df['ball_direction_x'] +
                df['acceleration_y'] * df['ball_direction_y']
            )

    if 'velocity_x' in df.columns:
        df['velocity_x_change'] = df.groupby(gcols)['velocity_x'].diff().fillna(0)
        df['velocity_y_change'] = df.groupby(gcols)['velocity_y'].diff().fillna(0)
        df['speed_change'] = df.groupby(gcols)['s'].diff().fillna(0)
        df['direction_change'] = df.groupby(gcols)['dir'].diff().fillna(0)
        df['direction_change'] = df['direction_change'].apply(
            lambda x: x if abs(x) < 180 else x - 360 * np.sign(x)
        )

    df['dist_from_left'] = df['y']
    df['dist_from_right'] = 53.3 - df['y']
    df['dist_from_sideline'] = np.minimum(df['dist_from_left'], df['dist_from_right'])
    df['dist_from_endzone'] = np.minimum(df['x'], 120 - df['x'])

    if 'is_receiver' in df.columns and 'velocity_alignment' in df.columns:
        df['receiver_optimality'] = df['is_receiver'] * df['velocity_alignment']
        df['receiver_deviation'] = df['is_receiver'] * np.abs(df.get('velocity_perpendicular', 0))
    if 'is_coverage' in df.columns and 'closing_speed' in df.columns:
        df['defender_closing_speed'] = df['is_coverage'] * df['closing_speed']

    df['frames_elapsed'] = df.groupby(gcols).cumcount()
    df['normalized_time'] = df.groupby(gcols)['frames_elapsed'].transform(
        lambda x: x / (x.max() + 1)
    )

    if 'nearest_opp_dist' in df.columns:
        df['pressure'] = 1 / np.maximum(df['nearest_opp_dist'], 0.5)
        df['under_pressure'] = (df['nearest_opp_dist'] < 3).astype(int)
        df['pressure_x_speed'] = df['pressure'] * df['s']
        
    if 'mirror_wr_vx' in df.columns:
        s_safe = np.maximum(df['s'], 0.1)
        df['mirror_similarity'] = (
                df['velocity_x'] * df['mirror_wr_vx'] +
                df['velocity_y'] * df['mirror_wr_vy']
        ) / s_safe
        df['mirror_offset_dist'] = np.sqrt(
            df['mirror_offset_x'] ** 2 + df['mirror_offset_y'] ** 2
        )
        df['mirror_alignment'] = df['mirror_similarity'] * df['is_coverage']

    return df

def add_time_features(df):
    if 'num_frames_output' not in df.columns:
        return df
        
    max_frames = df['num_frames_output']
    
    df['max_play_duration'] = max_frames / 10.0
    df['frame_time'] = df['frame_id'] / 10.0
    df['progress_ratio'] = df['frame_id'] / np.maximum(max_frames, 1)
    df['time_remaining'] = (max_frames - df['frame_id']) / 10.0
    df['frames_remaining'] = max_frames - df['frame_id']
    
    df['expected_x_at_ball'] = df['x'] + df['velocity_x'] * df['frame_time']
    df['expected_y_at_ball'] = df['y'] + df['velocity_y'] * df['frame_time']
    
    if 'ball_land_x' in df.columns:
        df['error_from_ball_x'] = df['expected_x_at_ball'] - df['ball_land_x']
        df['error_from_ball_y'] = df['expected_y_at_ball'] - df['ball_land_y']
        df['error_from_ball'] = np.sqrt(
            df['error_from_ball_x']**2 + df['error_from_ball_y']**2
        )
        
        df['weighted_dist_by_time'] = df['dist_to_ball'] / (df['frame_time'] + 0.1)
        df['dist_scaled_by_progress'] = df['dist_to_ball'] * (1 - df['progress_ratio'])
    
    df['time_squared'] = df['frame_time'] ** 2
    df['velocity_x_progress'] = df['velocity_x'] * df['progress_ratio']
    df['velocity_y_progress'] = df['velocity_y'] * df['progress_ratio']
    df['speed_scaled_by_time_left'] = df['s'] * df['time_remaining']
    
    df['actual_play_length'] = max_frames
    df['length_ratio'] = max_frames / 30.0
    
    return df

def get_feature_columns(df):
    base_feature_cols = [
        'x', 'y', 's', 'a', 'o', 'dir', 'frame_id',
        'ball_land_x', 'ball_land_y',
        'player_height_feet', 'player_weight', 'height_inches', 'bmi',
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',
        'momentum_x', 'momentum_y', 'kinetic_energy',
        'speed_squared', 'accel_magnitude', 'orientation_diff',
        'is_offense', 'is_defense', 'is_receiver', 'is_coverage', 'is_passer',
        'role_targeted_receiver', 'role_defensive_coverage', 'role_passer', 'side_offense',
        'distance_to_ball', 'dist_to_ball', 'dist_squared', 'angle_to_ball', 
        'ball_direction_x', 'ball_direction_y', 'closing_speed_ball',
        'velocity_toward_ball', 'velocity_alignment', 'angle_diff',
    ]
    
    opponent_cols = [
        'nearest_opp_dist', 'closing_speed', 'num_nearby_opp_3', 'num_nearby_opp_5',
        'mirror_wr_vx', 'mirror_wr_vy', 'mirror_offset_x', 'mirror_offset_y', 'mirror_wr_dist',
    ]
    
    route_cols = [
        'route_pattern', 'traj_straightness', 'traj_max_turn', 'traj_mean_turn',
        'traj_depth', 'traj_width', 'speed_mean', 'speed_change',
    ]
    
    gnn_cols = [
        'gnn_ally_dx_mean', 'gnn_ally_dy_mean', 'gnn_ally_dvx_mean', 'gnn_ally_dvy_mean',
        'gnn_opp_dx_mean', 'gnn_opp_dy_mean', 'gnn_opp_dvx_mean', 'gnn_opp_dvy_mean',
        'gnn_ally_cnt', 'gnn_opp_cnt', 'gnn_ally_dmin', 'gnn_ally_dmean', 
        'gnn_opp_dmin', 'gnn_opp_dmean', 'gnn_d1', 'gnn_d2', 'gnn_d3',
    ]
    
    temporal_cols = []
    for lag in [1, 2, 3, 4, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's', 'a']:
            temporal_cols.append(f'{col}_lag{lag}')
    
    for window in [3, 5]:
        for col in ['x', 'y', 'velocity_x', 'velocity_y', 's']:
            temporal_cols.append(f'{col}_rolling_mean_{window}')
            temporal_cols.append(f'{col}_rolling_std_{window}')
    
    temporal_cols.extend(['velocity_x_delta', 'velocity_y_delta'])
    temporal_cols.extend(['velocity_x_ema', 'velocity_y_ema', 'speed_ema'])
    
    time_cols = [
        'max_play_duration', 'frame_time', 'progress_ratio', 'time_remaining', 'frames_remaining',
        'expected_x_at_ball', 'expected_y_at_ball', 
        'error_from_ball_x', 'error_from_ball_y', 'error_from_ball',
        'time_squared', 'weighted_dist_by_time', 
        'velocity_x_progress', 'velocity_y_progress', 'dist_scaled_by_progress',
        'speed_scaled_by_time_left', 'actual_play_length', 'length_ratio',
    ]
    
    advanced_cols = [
        'distance_to_ball_change', 'distance_to_ball_accel', 'time_to_intercept',
        'velocity_alignment', 'velocity_perpendicular', 'accel_alignment',
        'velocity_x_change', 'velocity_y_change', 'speed_change', 'direction_change',
        'dist_from_sideline', 'dist_from_endzone',
        'receiver_optimality', 'receiver_deviation', 'defender_closing_speed',
        'frames_elapsed', 'normalized_time',
        'pressure', 'under_pressure', 'pressure_x_speed',
        'mirror_similarity', 'mirror_offset_dist', 'mirror_alignment'
    ]
    
    geometric_cols = [
        'geo_endpoint_x', 'geo_endpoint_y',
        'geo_vector_x', 'geo_vector_y', 'geo_distance',
        'geo_required_vx', 'geo_required_vy',
        'geo_velocity_error_x', 'geo_velocity_error_y', 'geo_velocity_error',
        'geo_required_ax', 'geo_required_ay',
        'geo_alignment', 'geo_receiver_urgency', 'geo_defender_coupling'
    ]
    
    all_feature_cols = (base_feature_cols + opponent_cols + route_cols + gnn_cols + 
                       temporal_cols + time_cols + advanced_cols + geometric_cols)
    
    return [c for c in all_feature_cols if c in df.columns]

def wrap_angle_deg(s):
    return ((s + 180.0) % 360.0) - 180.0

def unify_left_direction(df: pd.DataFrame) -> pd.DataFrame:
    if 'play_direction' not in df.columns:
        return df
    df = df.copy()
    right = df['play_direction'].eq('right')
    if 'x' in df.columns: df.loc[right, 'x'] = Config.FIELD_X_MAX - df.loc[right, 'x']
    if 'y' in df.columns: df.loc[right, 'y'] = Config.FIELD_Y_MAX - df.loc[right, 'y']
    for col in ('dir','o'):
        if col in df.columns:
            df.loc[right, col] = (df.loc[right, col] + 180.0) % 360.0
    if 'ball_land_x' in df.columns:
        df.loc[right, 'ball_land_x'] = Config.FIELD_X_MAX - df.loc[right, 'ball_land_x']
    if 'ball_land_y' in df.columns:
        df.loc[right, 'ball_land_y'] = Config.FIELD_Y_MAX - df.loc[right, 'ball_land_y']
    return df

def build_play_direction_map(df_in: pd.DataFrame) -> pd.Series:
    s = (
        df_in[['game_id','play_id','play_direction']]
        .drop_duplicates()
        .set_index(['game_id','play_id'])['play_direction']
    )
    return s

def apply_direction_to_df(df: pd.DataFrame, dir_map: pd.Series) -> pd.DataFrame:
    if 'play_direction' not in df.columns:
        dir_df = dir_map.reset_index()
        df = df.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
    return unify_left_direction(df)

def invert_to_original_direction(x_u, y_u, play_dir_right: bool):
    if not play_dir_right:
        return float(x_u), float(y_u)
    return float(Config.FIELD_X_MAX - x_u), float(Config.FIELD_Y_MAX - y_u)
Seqences
def prepare_sequences_fixed(input_df, output_df=None, test_template=None, 
                           is_training=False, window_size=Config.WINDOW_SIZE,
                           route_kmeans=None, route_scaler=None):
    dir_map = build_play_direction_map(input_df)
    input_df_u = apply_direction_to_df(input_df, dir_map)
    
    if is_training:
        out_u = apply_direction_to_df(output_df, dir_map)
        target_rows = out_u
        target_groups = out_u[['game_id','play_id','nfl_id']].drop_duplicates()
    else:
        if 'play_direction' not in test_template.columns:
            dir_df = dir_map.reset_index()
            test_template = test_template.merge(dir_df, on=['game_id','play_id'], how='left', validate='many_to_one')
        target_rows = test_template
        target_groups = target_rows[['game_id','play_id','nfl_id']].drop_duplicates()

    input_df_u = create_base_features(input_df_u)
    input_df_u = create_lag_features(input_df_u, window_size)
    opponent_features = get_opponent_features(input_df_u)
    input_df_u = input_df_u.merge(opponent_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    route_features = extract_route_patterns(input_df_u, route_kmeans, route_scaler, fit=False)
    if not route_features.empty:
        input_df_u = input_df_u.merge(route_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    gnn_features = compute_neighbor_embeddings(input_df_u)
    input_df_u = input_df_u.merge(gnn_features, on=['game_id', 'play_id', 'nfl_id'], how='left')
    input_df_u = add_advanced_features(input_df_u)
    input_df_u = add_time_features(input_df_u)
    input_df_u = add_geometric_features(input_df_u)
    feature_cols = get_feature_columns(input_df_u)

    input_df_u.set_index(['game_id', 'play_id', 'nfl_id'], inplace=True)
    grouped = input_df_u.groupby(level=['game_id', 'play_id', 'nfl_id'])
    
    sequences, sequence_ids = [], []
    geo_endpoints_x, geo_endpoints_y = [], []
    
    for _, row in tqdm(target_groups.iterrows(), desc="sequences ing"):
        key = (row['game_id'], row['play_id'], row['nfl_id'])
        try:
            group_df = grouped.get_group(key)
        except KeyError:
            continue

        input_window = group_df.tail(window_size)
        if len(input_window) < window_size:
            pad_len = window_size - len(input_window)
            pad_df = pd.DataFrame(np.nan, index=range(pad_len), columns=input_window.columns)
            input_window = pd.concat([pad_df, input_window], ignore_index=True)

        input_window = input_window.fillna(group_df.mean(numeric_only=True))
        seq = input_window[feature_cols].values

        if np.isnan(seq).any():
            seq = np.nan_to_num(seq, nan=0.0)
        
        sequences.append(seq)

        geo_x = input_window.iloc[-1]['geo_endpoint_x']
        geo_y = input_window.iloc[-1]['geo_endpoint_y']
        geo_endpoints_x.append(geo_x)
        geo_endpoints_y.append(geo_y)
        
        sequence_ids.append({
            'game_id': key[0],
            'play_id': key[1],
            'nfl_id': key[2],
            'frame_id': input_window.iloc[-1]['frame_id'],
            'play_direction': input_window.iloc[-1]['play_direction'],
            'last_x': input_window.iloc[-1]['x'],
            'last_y': input_window.iloc[-1]['y']
        })
    
    return sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y, feature_cols
Predict
def predict(test: pl.DataFrame, test_input: pl.DataFrame) -> pl.DataFrame:

    test_pd = test.to_pandas()
    test_input_pd = test_input.to_pandas()
    
    MODEL_DIR = Path("/kaggle/input/1103new-all-all-all/pytorch/default/1/1103new_all_all_all") 
    try:
        route_kmeans = joblib.load(MODEL_DIR / "route_kmeans.pkl")
        route_scaler = joblib.load(MODEL_DIR / "route_scaler.pkl")
    except:
        route_kmeans = None
        route_scaler = None

    sequences, sequence_ids, geo_endpoints_x, geo_endpoints_y, feature_cols = prepare_sequences_fixed(
        test_input_pd, test_template=test_pd, is_training=False, 
        window_size=Config.WINDOW_SIZE, route_kmeans=route_kmeans, route_scaler=route_scaler
    )
    
    if not sequences:
        return pl.DataFrame({"x": [], "y": []})
    
    X_test = np.array(sequences, dtype=object)

    x_last_u = np.array([s[-1, 0] for s in X_test]) 
    y_last_u = np.array([s[-1, 1] for s in X_test]) 

    model_x_paths = [MODEL_DIR / f"model_x_fold{i+1}.pth" for i in range(Config.N_FOLDS)]
    model_y_paths = [MODEL_DIR / f"model_y_fold{i+1}.pth" for i in range(Config.N_FOLDS)]
    scaler_paths = [MODEL_DIR / f"scaler_fold{i+1}.pkl" for i in range(Config.N_FOLDS)]

    for p in model_x_paths + model_y_paths + scaler_paths:
        if not p.exists():
            raise FileNotFoundError(f"loss: {p.name}")

    models_x = []
    models_y = []
    scalers = []
    
    for i in range(Config.N_FOLDS):

        model_x = ImprovedSeqModel(input_dim=X_test[0].shape[1], horizon=Config.MAX_FUTURE_HORIZON).to(Config.DEVICE)
        model_x.load_state_dict(torch.load(model_x_paths[i], map_location=Config.DEVICE))
        model_x.eval()
        models_x.append(model_x)

        model_y = ImprovedSeqModel(input_dim=X_test[0].shape[1], horizon=Config.MAX_FUTURE_HORIZON).to(Config.DEVICE)
        model_y.load_state_dict(torch.load(model_y_paths[i], map_location=Config.DEVICE))
        model_y.eval()
        models_y.append(model_y)

        scaler = joblib.load(scaler_paths[i])
        scalers.append(scaler)

    all_dx, all_dy = [], []
    for mx, my, sc in zip(models_x, models_y, scalers):
        X_scaled = np.stack([sc.transform(s) for s in X_test])
        X_tensor = torch.tensor(X_scaled.astype(np.float32)).to(Config.DEVICE)
        
        mx.eval()
        my.eval()
        with torch.no_grad():
            dx = mx(X_tensor).cpu().numpy()
            dy = my(X_tensor).cpu().numpy()
        all_dx.append(dx)
        all_dy.append(dy)
    
    ens_dx = np.mean(all_dx, axis=0)
    ens_dy = np.mean(all_dy, axis=0)

    rows = []
    H = ens_dx.shape[1]
    
    for i, sid in enumerate(sequence_ids):
        fids = test_pd[
            (test_pd['game_id'] == sid['game_id']) &
            (test_pd['play_id'] == sid['play_id']) &
            (test_pd['nfl_id'] == sid['nfl_id'])
        ]['frame_id'].sort_values().tolist()
        
        play_dir_right = (sid['play_direction'] == 'right')
        
        for t, fid in enumerate(fids):
            tt = min(t, H - 1)

            x_u = np.clip(x_last_u[i] + ens_dx[i, tt], 0, 120)
            y_u = np.clip(y_last_u[i] + ens_dy[i, tt], 0, 53.3)
            
            x_orig, y_orig = invert_to_original_direction(x_u, y_u, play_dir_right)
            
            rows.append({
                'x': float(x_orig),
                'y': float(y_orig)
            })
    
    return pl.DataFrame(rows)
Infer
try:
    from kaggle_evaluation.nfl_inference_server import NFLInferenceServer
    inference_server = NFLInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
except ImportError as e:
    if __name__ == "__main__":
        pass
sequences ing: 
 3/? [00:00<00:00, 19.29it/s]
sequences ing: 
 2/? [00:00<00:00, 20.15it/s]
sequences ing: 
 3/? [00:00<00:00, 17.88it/s]
sequences ing: 
 4/? [00:00<00:00, 18.31it/s]
sequences ing: 
 5/? [00:00<00:00, 20.98it/s]
sequences ing: 
 1/? [00:00<00:00, 16.08it/s]
sequences ing: 
 3/? [00:00<00:00, 21.56it/s]
sequences ing: 
 6/? [00:00<00:00, 21.99it/s]
sequences ing: 
 3/? [00:00<00:00, 20.88it/s]
sequences ing: 
 2/? [00:00<00:00, 20.69it/s]
sequences ing: 
 3/? [00:00<00:00, 19.80it/s]
sequences ing: 
 5/? [00:00<00:00, 21.70it/s]
sequences ing: 
 3/? [00:00<00:00, 19.85it/s]
sequences ing: 
 5/? [00:00<00:00, 10.23it/s]
sequences ing: 
 3/? [00:00<00:00, 19.02it/s]
sequences ing: 
 3/? [00:00<00:00, 19.70it/s]
sequences ing: 
 3/? [00:00<00:00, 19.95it/s]
sequences ing: 
 6/? [00:00<00:00, 22.43it/s]
sequences ing: 
 2/? [00:00<00:00, 16.27it/s]
sequences ing: 
 5/? [00:00<00:00, 21.14it/s]
sequences ing: 
 3/? [00:00<00:00, 19.78it/s]
sequences ing: 
 3/? [00:00<00:00, 19.46it/s]
sequences ing: 
 3/? [00:00<00:00, 19.53it/s]
sequences ing: 
 4/? [00:00<00:00, 19.98it/s]
sequences ing: 
 3/? [00:00<00:00, 19.81it/s]
sequences ing: 
 4/? [00:00<00:00, 21.20it/s]
sequences ing: 
 3/? [00:00<00:00, 16.11it/s]
sequences ing: 
 2/? [00:00<00:00, 17.40it/s]
sequences ing: 
 3/? [00:00<00:00, 19.61it/s]
sequences ing: 
 5/? [00:00<00:00, 20.28it/s]
sequences ing: 
 3/? [00:00<00:00, 19.44it/s]
sequences ing: 
 5/? [00:00<00:00, 20.42it/s]
sequences ing: 
 4/? [00:00<00:00, 18.94it/s]
sequences ing: 
 3/? [00:00<00:00, 16.54it/s]
sequences ing: 
 6/? [00:00<00:00, 21.79it/s]
sequences ing: 
 4/? [00:00<00:00,  9.80it/s]
sequences ing: 
 3/? [00:00<00:00, 18.95it/s]
sequences ing: 
 3/? [00:00<00:00, 19.44it/s]
sequences ing: 
 3/? [00:00<00:00, 19.86it/s]
sequences ing: 
 2/? [00:00<00:00, 20.76it/s]
sequences ing: 
 4/? [00:00<00:00, 18.87it/s]
sequences ing: 
 2/? [00:00<00:00, 19.63it/s]
sequences ing: 
 2/? [00:00<00:00, 16.53it/s]
sequences ing: 
 2/? [00:00<00:00,  7.03it/s]
sequences ing: 
 2/? [00:00<00:00, 19.82it/s]
sequences ing: 
 2/? [00:00<00:00, 19.41it/s]
sequences ing: 
 2/? [00:00<00:00, 20.48it/s]
sequences ing: 
 2/? [00:00<00:00, 19.91it/s]
sequences ing: 
 1/? [00:00<00:00, 18.60it/s]
sequences ing: 
 3/? [00:00<00:00, 19.28it/s]
sequences ing: 
 2/? [00:00<00:00, 17.44it/s]
sequences ing: 
 3/? [00:00<00:00, 20.36it/s]
sequences ing: 
 6/? [00:00<00:00, 13.33it/s]
sequences ing: 
 2/? [00:00<00:00, 19.32it/s]
sequences ing: 
 5/? [00:00<00:00, 20.73it/s]
sequences ing: 
 2/? [00:00<00:00, 18.93it/s]
sequences ing: 
 2/? [00:00<00:00, 19.12it/s]
sequences ing: 
 2/? [00:00<00:00, 18.17it/s]
sequences ing: 
 4/? [00:00<00:00, 16.04it/s]
sequences ing: 
 3/? [00:00<00:00, 21.91it/s]
sequences ing: 
 4/? [00:00<00:00, 19.75it/s]
sequences ing: 
 4/? [00:00<00:00, 19.51it/s]
sequences ing: 
 6/? [00:00<00:00, 21.22it/s]
sequences ing: 
 4/? [00:00<00:00, 21.33it/s]
sequences ing: 
 4/? [00:00<00:00, 21.40it/s]
sequences ing: 
 3/? [00:00<00:00, 18.32it/s]
sequences ing: 
 3/? [00:00<00:00, 21.35it/s]
sequences ing: 
 2/? [00:00<00:00, 19.32it/s]
sequences ing: 
 5/? [00:00<00:00, 20.57it/s]
sequences ing: 
 3/? [00:00<00:00, 19.17it/s]
sequences ing: 
 2/? [00:00<00:00, 19.97it/s]
sequences ing: 
 3/? [00:00<00:00, 21.58it/s]
sequences ing: 
 3/? [00:00<00:00, 19.33it/s]
sequences ing: 
 5/? [00:00<00:00, 21.15it/s]
sequences ing: 
 3/? [00:00<00:00, 20.54it/s]
sequences ing: 
 2/? [00:00<00:00, 20.83it/s]
sequences ing: 
 1/? [00:00<00:00, 17.71it/s]
sequences ing: 
 4/? [00:00<00:00, 18.86it/s]
sequences ing: 
 4/? [00:00<00:00, 19.60it/s]
sequences ing: 
 5/? [00:00<00:00, 22.04it/s]
sequences ing: 
 1/? [00:00<00:00, 17.40it/s]
sequences ing: 
 2/? [00:00<00:00, 19.00it/s]
sequences ing: 
 7/? [00:00<00:00, 21.54it/s]
sequences ing: 
 3/? [00:00<00:00, 19.10it/s]
sequences ing: 
 4/? [00:00<00:00, 17.46it/s]
sequences ing: 
 3/? [00:00<00:00, 18.78it/s]
sequences ing: 
 3/? [00:00<00:00, 19.18it/s]
sequences ing: 
 3/? [00:00<00:00, 21.68it/s]
sequences ing: 
 3/? [00:00<00:00, 18.31it/s]
sequences ing: 
 4/? [00:00<00:00, 19.17it/s]
sequences ing: 
 4/? [00:00<00:00, 20.31it/s]
sequences ing: 
 4/? [00:00<00:00, 19.72it/s]
sequences ing: 
 4/? [00:00<00:00, 19.10it/s]
sequences ing: 
 3/? [00:00<00:00, 22.18it/s]
sequences ing: 
 3/? [00:00<00:00, 19.30it/s]
sequences ing: 
 3/? [00:00<00:00, 21.04it/s]
sequences ing: 
 1/? [00:00<00:00, 16.41it/s]
sequences ing: 
 3/? [00:00<00:00, 18.85it/s]
sequences ing: 
 5/? [00:00<00:00, 19.21it/s]
sequences ing: 
 3/? [00:00<00:00, 18.89it/s]
sequences ing: 
 3/? [00:00<00:00, 21.14it/s]
sequences ing: 
 1/? [00:00<00:00, 17.67it/s]
sequences ing: 
 3/? [00:00<00:00, 19.17it/s]
sequences ing: 
 3/? [00:00<00:00, 19.92it/s]
sequences ing: 
 3/? [00:00<00:00, 21.40it/s]
sequences ing: 
 3/? [00:00<00:00, 17.84it/s]
sequences ing: 
 3/? [00:00<00:00, 18.15it/s]
sequences ing: 
 1/? [00:00<00:00, 17.55it/s]
sequences ing: 
 3/? [00:00<00:00, 19.95it/s]
sequences ing: 
 4/? [00:00<00:00, 19.62it/s]
sequences ing: 
 4/? [00:00<00:00, 21.40it/s]
sequences ing: 
 3/? [00:00<00:00, 18.88it/s]
sequences ing: 
 4/? [00:00<00:00, 19.37it/s]
sequences ing: 
 2/? [00:00<00:00, 19.88it/s]
sequences ing: 
 2/? [00:00<00:00, 19.32it/s]
sequences ing: 
 2/? [00:00<00:00, 17.98it/s]
sequences ing: 
 3/? [00:00<00:00, 19.60it/s]
sequences ing: 
 5/? [00:00<00:00, 21.53it/s]
sequences ing: 
 4/? [00:00<00:00, 11.66it/s]
sequences ing: 
 3/? [00:00<00:00, 18.72it/s]
sequences ing: 
 5/? [00:00<00:00, 21.54it/s]
sequences ing: 
 3/? [00:00<00:00, 21.23it/s]
sequences ing: 
 4/? [00:00<00:00, 19.24it/s]
sequences ing: 
 4/? [00:00<00:00, 21.63it/s]
sequences ing: 
 3/? [00:00<00:00, 19.93it/s]
sequences ing: 
 4/? [00:00<00:00, 19.87it/s]
sequences ing: 
 1/? [00:00<00:00, 16.56it/s]
sequences ing: 
 4/? [00:00<00:00, 19.92it/s]
sequences ing: 
 2/? [00:00<00:00, 18.80it/s]
sequences ing: 
 4/? [00:00<00:00, 17.66it/s]
sequences ing: 
 4/? [00:00<00:00, 19.28it/s]
sequences ing: 
 3/? [00:00<00:00, 21.20it/s]
sequences ing: 
 5/? [00:00<00:00, 21.46it/s]
sequences ing: 
 4/? [00:00<00:00, 21.43it/s]
sequences ing: 
 4/? [00:00<00:00, 21.34it/s]
sequences ing: 
 3/? [00:00<00:00, 19.53it/s]
sequences ing: 
 4/? [00:00<00:00, 20.96it/s]
sequences ing: 
 3/? [00:00<00:00, 20.88it/s]
sequences ing: 
 3/? [00:00<00:00, 18.95it/s]
sequences ing: 
 4/? [00:00<00:00, 18.50it/s]
sequences ing: 
 3/? [00:00<00:00, 19.01it/s]
sequences ing: 
 3/? [00:00<00:00, 19.45it/s]
sequences ing: 
 8/? [00:00<00:00, 22.29it/s]