# NFL Player Trajectory Prediction with GNN-LSTM

This project implements a Graph Neural Network (GNN) with LSTM decoders to predict NFL player trajectories during pass plays, as proposed in the research proposal for the NFL Big Data Bowl 2026.

## Overview

The model predicts the (x, y) coordinates of all players on the field during the time the football is in the air after a quarterback releases a pass. It uses:

1. **Graph Attention Network (GAT)** to model spatial relationships between players
2. **Role-conditioned LSTM Decoder** to generate future trajectories
3. **Multi-agent prediction** with explicit modeling of player interactions

## Project Structure

```
deep-learning/
├── nfl-big-data-bowl-2026-prediction/  # Data directory
│   ├── train/                          # Training data (weeks 1-18, 2023 season)
│   │   ├── input_2023_w01.csv         # Input features (before pass)
│   │   └── output_2023_w01.csv        # Ground truth (ball in air)
│   ├── test_input.csv                 # Test input features
│   └── kaggle_evaluation/             # Kaggle evaluation framework
├── src/
│   ├── data/
│   │   ├── dataset.py                 # PyTorch Dataset
│   │   └── preprocessing.py           # Data preprocessing utilities
│   ├── models/
│   │   ├── baselines.py              # Baseline models (Constant Velocity, etc.)
│   │   ├── graph_builder.py          # Graph construction from player states
│   │   ├── gat_encoder.py            # Graph Attention Network encoder
│   │   ├── lstm_decoder.py           # Role-conditioned LSTM decoder
│   │   └── gnn_lstm.py               # Complete GNN-LSTM model
│   ├── training/
│   │   ├── trainer.py                # Training loop
│   │   └── losses.py                 # Loss functions (role-weighted MSE)
│   ├── evaluation/
│   │   ├── metrics.py                # RMSE, ADE, FDE metrics
│   │   └── evaluator.py              # Model evaluation utilities
│   └── utils/
│       └── config.py                 # Configuration management
├── train.py                          # Main training script
├── evaluate.py                       # Evaluation and comparison script
├── inference.py                      # Inference for Kaggle submission
├── explore_data.py                   # Data exploration script
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

### 1. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch Geometric requires additional installation steps. Follow the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your system.

For CPU-only installation:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

For CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
```

## Quick Start

### 1. Explore the Data

```bash
python3 explore_data.py
```

This will analyze the structure of the training data and print statistics.

### 2. Train a Model

#### Train the full GNN-LSTM model:
```bash
python3 train.py
```

#### Train with custom parameters:
```bash
python3 train.py --batch-size 16 --lr 0.0005 --epochs 50
```

#### Train the simplified model (MLP encoder, faster):
```bash
python3 train.py --simple --epochs 30
```

### 3. Evaluate Models

#### Evaluate baseline models only:
```bash
python3 evaluate.py --eval-baselines
```

#### Evaluate trained model:
```bash
python3 evaluate.py --checkpoint checkpoints/best_model.pt
```

#### Compare baselines and trained model:
```bash
python3 evaluate.py --checkpoint checkpoints/best_model.pt --eval-baselines
```

## Model Architecture

### Graph Construction

For each play, we create a graph with:
- **Nodes:** 12-14 players (actual number varies) + 1 ball landing location
- **Node features:** Position (x, y), velocity (s, dir), acceleration, orientation, role, etc.
- **Edges:**
  - Player-to-ball edges (bidirectional)
  - K-nearest neighbor edges (k=5)
  - Role-specific edges (defensive coverage ↔ targeted receiver)
- **Edge features:** Distance, relative position

### GAT Encoder

- 2-3 layers of Graph Attention Networks
- Multi-head attention (4-8 heads) to learn interaction importance
- Processes spatial relationships between players
- Outputs rich player embeddings (128-256 dim)

### LSTM Decoder

- Role-conditioned trajectory generation
- Auto-regressive prediction with scheduled sampling
- Inputs: GNN embedding + role embedding + ball landing location
- Outputs: Sequence of (x, y) positions for each player

### Training

- **Loss:** Role-weighted MSE (1.5× weight for targeted receiver and defensive coverage)
- **Optimizer:** Adam with learning rate decay
- **Batch size:** 32 plays
- **Gradient clipping:** Max norm 1.0
- **Scheduled sampling:** Teacher forcing ratio decays from 1.0 → 0.5

## Data Format

### Input CSV (before pass)
- **Columns:** game_id, play_id, nfl_id, frame_id, x, y, s, a, dir, o, player_role, ball_land_x, ball_land_y, etc.
- **Sequence length:** ~10-30 frames per player (1-3 seconds @ 10 fps)

### Output CSV (ball in air)
- **Columns:** game_id, play_id, nfl_id, frame_id, x, y
- **Sequence length:** ~5-100 frames (variable based on throw distance)

## Configuration

Edit `src/utils/config.py` to customize:

- **Data splits:** Train/val/test week ranges
- **Model architecture:** GNN/LSTM dimensions, number of layers
- **Training:** Learning rate, batch size, epochs
- **Graph construction:** Number of nearest neighbors

## Evaluation Metrics

- **RMSE (Root Mean Squared Error):** Primary metric, measures position error
- **ADE (Average Displacement Error):** Mean Euclidean distance over trajectory
- **FDE (Final Displacement Error):** Error at final timestep
- **Per-role RMSE:** Breakdown by player role (targeted receiver, coverage, etc.)

## Baselines

The project includes several baseline models for comparison:

1. **Constant Velocity:** Linear extrapolation using last observed velocity
2. **Linear Extrapolation:** Least-squares fit on recent trajectory
3. **Mean Position:** Assumes no movement (last position repeated)

## Results

After training, you'll see results like:

```
Model                          RMSE       ADE        FDE
----------------------------------------------------------------
Constant Velocity              X.XXXX     X.XXXX     X.XXXX
Linear Extrapolation           X.XXXX     X.XXXX     X.XXXX
GNN-LSTM                       X.XXXX     X.XXXX     X.XXXX
```

**Per-role breakdown:**
- Targeted Receiver: X.XXXX
- Defensive Coverage: X.XXXX
- Other Route Runner: X.XXXX
- Passer: X.XXXX

## Kaggle Integration

To integrate with the Kaggle evaluation gateway:

1. The `inference.py` script provides an `NFLInferenceModel` class
2. Modify `kaggle_evaluation/nfl_inference_server.py` to use this model
3. The model will receive play-by-play batches and return predictions

## Hardware Requirements

- **Minimum:** CPU with 8GB RAM
- **Recommended:** GPU with 8GB+ VRAM (NVIDIA RTX 3060 or better)
- **Training time:** 
  - CPU: ~6-12 hours for 50 epochs
  - GPU: ~1-2 hours for 50 epochs

## Troubleshooting

### Out of Memory Errors
- Reduce batch size: `--batch-size 8`
- Use simplified model: `--simple`
- Reduce sequence length in `config.py`

### Slow Training
- Use GPU if available
- Reduce number of workers: `--num-workers 0`
- Use simplified model for faster iteration

### Import Errors
- Ensure you're in the project root directory
- Check that all dependencies are installed
- Try: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

## Citation

If you use this code, please cite:

```
@misc{nfl-trajectory-gnn-lstm,
  author = {Your Name},
  title = {Spatio-Temporal Player Trajectory Prediction in the NFL using Graph Neural Networks},
  year = {2025},
  publisher = {NFL Big Data Bowl 2026}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

- NFL Big Data Bowl 2026 for the dataset and competition
- PyTorch Geometric team for the GNN framework
- Graph Attention Networks (Veličković et al., 2018)

## Contact

For questions or issues, please open a GitHub issue or contact [your contact info].

---

**Note:** This is a research implementation. For production use, additional optimization and testing are recommended.


