"""
Quick test to verify the data pipeline and model instantiation
"""

import torch
from src.utils.config import get_config
from src.data.preprocessing import NFLDataPreprocessor, load_week_data
from src.data.dataset import NFLTrajectoryDataset
from src.models.gnn_lstm import SimplifiedGNNLSTM
from src.models.baselines import ConstantVelocityBaseline

print("Testing NFL Trajectory Prediction Pipeline")
print("=" * 80)

# 1. Test configuration
print("\n1. Loading configuration...")
config = get_config()
print(f"   ✓ Config loaded")
print(f"   - Train weeks: {config.data.train_weeks}")
print(f"   - Val weeks: {config.data.val_weeks}")
print(f"   - Test weeks: {config.data.test_weeks}")

# 2. Test data loading
print("\n2. Loading data (week 1)...")
try:
    input_df, output_df = load_week_data(config.data.train_dir, 1)
    print(f"   ✓ Data loaded")
    print(f"   - Input shape: {input_df.shape}")
    print(f"   - Output shape: {output_df.shape}")
    print(f"   - Unique plays: {input_df[['game_id', 'play_id']].drop_duplicates().shape[0]}")
except Exception as e:
    print(f"   ✗ Error loading data: {e}")
    exit(1)

# 3. Test preprocessor
print("\n3. Testing preprocessor...")
try:
    preprocessor = NFLDataPreprocessor(config.data)
    preprocessor.fit_categorical_mappings(input_df)
    print(f"   ✓ Preprocessor fitted")
    print(f"   - Position mappings: {len(preprocessor.position_to_idx)}")
except Exception as e:
    print(f"   ✗ Error with preprocessor: {e}")
    exit(1)

# 4. Test dataset
print("\n4. Creating dataset...")
try:
    dataset = NFLTrajectoryDataset(
        input_df.head(1000),  # Just first 1000 rows for quick test
        output_df.head(1000),
        preprocessor,
        max_input_frames=80,
        max_output_frames=100,
    )
    print(f"   ✓ Dataset created")
    print(f"   - Number of plays: {len(dataset)}")
    
    # Get one sample
    sample = dataset[0]
    print(f"   - Sample keys: {list(sample.keys())}")
    print(f"   - Input features shape: {sample['input_features'].shape}")
    print(f"   - Output positions shape: {sample['output_positions'].shape}")
    print(f"   - Number of players: {sample['num_players']}")
except Exception as e:
    print(f"   ✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. Test model instantiation
print("\n5. Creating model...")
try:
    model = SimplifiedGNNLSTM(config.model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Model created")
    print(f"   - Number of parameters: {num_params:,}")
except Exception as e:
    print(f"   ✗ Error creating model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 6. Test forward pass
print("\n6. Testing forward pass...")
try:
    # Create a small batch
    from torch.utils.data import DataLoader
    from src.data.dataset import collate_fn
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    print(f"   - Batch size: {batch['input_features'].shape[0]}")
    print(f"   - Max players: {batch['input_features'].shape[1]}")
    
    # Forward pass
    with torch.no_grad():
        predictions = model(batch, teacher_forcing_ratio=0.0)
    
    print(f"   ✓ Forward pass successful")
    print(f"   - Predictions shape: {predictions.shape}")
except Exception as e:
    print(f"   ✗ Error in forward pass: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 7. Test baseline
print("\n7. Testing baseline model...")
try:
    baseline = ConstantVelocityBaseline()
    baseline_preds = baseline(batch)
    print(f"   ✓ Baseline predictions")
    print(f"   - Shape: {baseline_preds.shape}")
except Exception as e:
    print(f"   ✗ Error with baseline: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 8. Test metrics
print("\n8. Testing metrics...")
try:
    from src.evaluation.metrics import compute_rmse
    
    rmse = compute_rmse(
        predictions,
        batch['output_positions'],
        batch['output_mask'],
    )
    print(f"   ✓ RMSE computed: {rmse:.4f}")
except Exception as e:
    print(f"   ✗ Error computing metrics: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✓ All tests passed! Pipeline is working correctly.")
print("=" * 80)
print("\nNext steps:")
print("  1. Train the model: python3 train.py --simple --epochs 5")
print("  2. Evaluate baselines: python3 evaluate.py --eval-baselines")
print("  3. Train full model: python3 train.py --epochs 50")


