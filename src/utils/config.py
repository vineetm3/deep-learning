"""
Configuration file for NFL trajectory prediction
"""

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Data configuration"""
    # Paths
    data_dir: Path = Path("nfl-big-data-bowl-2026-prediction")
    train_dir: Path = field(init=False)
    
    # Week splits
    train_weeks: list = field(default_factory=lambda: list(range(1, 15)))  # Weeks 1-14
    val_weeks: list = field(default_factory=lambda: [15, 16])  # Weeks 15-16
    test_weeks: list = field(default_factory=lambda: [17, 18])  # Weeks 17-18
    
    # Sequence lengths
    max_input_frames: int = 80  # Max frames before pass (handle up to 8 seconds @ 10fps)
    max_output_frames: int = 100  # Max frames during ball flight
    
    # Feature columns
    position_features: list = field(default_factory=lambda: ['x', 'y'])
    motion_features: list = field(default_factory=lambda: ['s', 'a', 'dir', 'o', 'vx', 'vy', 'ax', 'ay'])
    static_features: list = field(default_factory=lambda: ['player_weight', 'absolute_yardline_number'])
    ball_features: list = field(default_factory=lambda: ['ball_land_x', 'ball_land_y', 'ball_dx', 'ball_dy', 'ball_dist', 'ball_angle_sin', 'ball_angle_cos'])
    
    # Categorical features
    categorical_features: list = field(default_factory=lambda: ['player_role', 'player_position', 'player_side', 'play_direction'])
    
    # Role mappings
    role_to_idx: dict = field(default_factory=lambda: {
        'Targeted Receiver': 0,
        'Defensive Coverage': 1,
        'Other Route Runner': 2,
        'Passer': 3,
    })
    
    # Normalization ranges (from exploration)
    norm_ranges: dict = field(default_factory=lambda: {
        'x': (0.0, 120.0),
        'y': (0.0, 53.3),
        's': (0.0, 15.0),  # Speed in yards/sec (max observed ~12.5, add buffer)
        'a': (0.0, 20.0),  # Acceleration (max observed ~17, add buffer)
        'dir': (0.0, 360.0),  # Direction in degrees
        'o': (0.0, 360.0),  # Orientation in degrees
        'vx': (-15.0, 15.0),
        'vy': (-15.0, 15.0),
        'ax': (-20.0, 20.0),
        'ay': (-20.0, 20.0),
        'ball_land_x': (0.0, 120.0),
        'ball_land_y': (-10.0, 60.0),  # Can be out of bounds
        'ball_dx': (-60.0, 60.0),
        'ball_dy': (-30.0, 30.0),
        'ball_dist': (0.0, 70.0),
        'ball_angle_sin': (-1.0, 1.0),
        'ball_angle_cos': (-1.0, 1.0),
        'player_weight': (150.0, 350.0),
        'absolute_yardline_number': (0.0, 100.0),
    })
    
    def __post_init__(self):
        self.train_dir = self.data_dir / "train"


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Node feature dimensions
    node_feature_dim: int = 64  # After embedding categorical features
    
    # GNN configuration
    gnn_hidden_dim: int = 128
    gnn_output_dim: int = 128
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4  # For GAT
    gnn_dropout: float = 0.3
    
    # Graph construction
    k_nearest_neighbors: int = 5
    
    # LSTM configuration
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.35
    
    # Role embeddings
    role_embedding_dim: int = 16
    num_roles: int = 4
    
    # Position embeddings
    position_embedding_dim: int = 16
    num_positions: int = 15  # Approximate number of unique positions
    
    # Output
    output_dim: int = 2  # (x, y) coordinates


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 5e-5
    batch_size: int = 32
    num_epochs: int = 100
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Scheduled sampling
    initial_teacher_forcing_ratio: float = 1.0
    final_teacher_forcing_ratio: float = 0.2
    teacher_forcing_decay_epochs: int = 10
    teacher_forcing_strategy: str = "exponential"  # "linear", "exponential", "cosine"
    teacher_forcing_gamma: float = 0.9
    
    # Loss weighting
    role_weights: dict = field(default_factory=lambda: {
        'Targeted Receiver': 1.5,
        'Defensive Coverage': 1.5,
        'Other Route Runner': 1.0,
        'Passer': 1.0,
    })
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "reduce_on_plateau" or "cosine"
    lr_patience: int = 5
    lr_factor: float = 0.5
    cosine_t_max: int = 10
    cosine_eta_min: float = 1e-5
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every_n_epochs: int = 5
    
    # Device
    device: str = "cuda"  # Will be set to "cpu" if CUDA unavailable


@dataclass
class Config:
    """Main configuration container"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Random seed
    seed: int = 42
    
    # Experiment name
    experiment_name: str = "gnn_lstm_baseline"


def get_config():
    """Get default configuration"""
    return Config()


