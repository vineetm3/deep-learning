## Anticipated Gains from the Transformer Roadmap

We are introducing five major upgrades to move beyond the current GNN-LSTM baseline. Below is a quick rationale and the expected impact for each.

### 1. Feature Engineering Expansion
- **What:** incorporate opponent proximity, geometric endpoints, route statistics, temporal lags/rolling features, and ball-aligned signals into the per-player node (and edge) features.
- **Why:** the baseline input set (pos, velocity, acceleration, ball distance) leaves a lot of domain knowledge untapped. Encoding richer clues directly in the features reduces the burden on the network and mirrors the best public Kaggle solutions.
- **Expected improvement:** cleaner separation of offensive vs defensive patterns, better coverage modeling, and lower variance in short horizons. This should shave error on defensive coverage where we currently lag behind the linear baseline.

### 2. Transformer Decoder
- **What:** replace the role-conditioned LSTM decoder with a transformer-based residual decoder (self-attention + cross-attention to the GNN embeddings).
- **Why:** transformers handle short horizon trajectories better than an autoregressive LSTM. Every step can attend to earlier predictions, mitigating error accumulation; residual prediction stabilizes the rollout.
- **Expected improvement:** better temporal modeling, fewer compounding errors, and stronger coverage/receiver tracking on the hidden test set.

### 3. Residual Training
- **What:** predict Δx/Δy residuals and accumulate them instead of predicting absolute positions; tune teacher forcing to support residual decoding.
- **Why:** residual targets make it easier to model small adjustments around the last observed frame and reduce drift. Teacher forcing now stabilizes early epochs but decays quickly so the model trains in the same regime used at inference.
- **Expected improvement:** closer alignment between training and inference behavior, lower RMSE in the final frames (where linear extrapolation currently wins).

### 4. Role-Specific Losses/Decoders
- **What:** expand the loss (and potentially decoder heads) for targeted receivers and coverage defenders, emphasize late timesteps, and consider auxiliary velocity/collision terms.
- **Why:** the competition score is dominated by tight coverage vs receiver interactions. Focusing model capacity and loss weight on these roles helps narrow the gap with physics baselines and top leaderboard entries.
- **Expected improvement:** reduced error on coverage defenders (our weakest role), helping overall RMSE move toward sub-yard territory.

### 5. Ensemble & Augmentation
- **What:** augment data via left/right mirroring and train multiple seeds or folds, averaging predictions.
- **Why:** top Kaggle solutions leverage data augmentation and ensembling. Mirroring doubles the effective data (symmetry in football field), while ensembling reduces variance.
- **Expected improvement:** more robust inference, more consistent Kaggle scores (mitigating random seed swings), and incremental RMSE gains (~0.05–0.1 yards).

Taken together, these changes aim to combine the relational power of our graph encoder with the accuracy of transformer-based decoders and feature-rich inputs. The expectation is to dramatically close the gap with the linear baseline on the hidden leaderboard and move toward the sub-yard RMSE achieved by top published solutions.  

