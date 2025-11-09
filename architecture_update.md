## Transformer Decoder Upgrade – Rationale and Expectations

### Why move beyond the LSTM decoder?

Our original pipeline used a role-conditioned LSTM decoder that predicted absolute positions autoregressively. While straightforward, it showed several limitations during experimentation and Kaggle evaluation:

1. **Short-horizon dynamics**: In the 0.5–1.5 second window while the ball is airborne, players often adjust their paths sharply (e.g., receiver breaks, defensive mirroring). The LSTM struggled to capture these moment-to-moment changes, especially when teacher forcing dropped below ~0.5.

2. **Gradient bottlenecks**: Even with scheduled sampling, the LSTM generates one step at a time. Errors compound, and gradients must flow through the entire sequence. This made convergence fragile and encouraged very high teacher forcing ratios, which, in turn, caused inference-time degradation.

3. **Limited context**: The LSTM state primarily carries the last hidden vector. It has difficulty directly referencing earlier timesteps—everything is “summed” into the hidden state—so patterns like “frame t+5 depends on frame t+1” are only learned indirectly.

These issues produced a large gap between our local validation performance and the physical baselines, and ultimately the hidden-test RMSE (~2.7 yards) trailed far behind top leaderboard entries (< 0.6 yards).

### Why a transformer-based decoder?

Transformer decoders process all future timesteps in parallel, using multi-head self-attention and residual connections. Replacing the LSTM with a transformer brings several advantages:

- **Richer temporal relationships**  
  Every future step can directly attend to every previous predicted step, allowing the model to learn fine-grained temporal dependencies (e.g., a defender reacting to a receiver’s break two frames earlier).

- **Stable residual prediction**  
  We predict Δx and Δy (residuals) for each future timestep and accumulate them. Transformers are well suited to modeling these residual sequences, reducing error drift and making it easier for the model to correct itself mid-trajectory.

- **Parallel training efficiency**  
  During training the decoder ingests the entire horizon at once. Even with causal masks, gradients flow across all timesteps simultaneously, encouraging consistency and speeding up convergence.

- **Compatibility with the GNN encoder**  
  The transformer decoder can include cross-attention to the GNN embeddings. The GAT still provides spatial context, while the transformer handles the finer temporal ordering, giving us the best of both worlds.

### Expected benefits

1. **Lower inference error**: By modeling residuals with attention over all previous timesteps, we expect less error accumulation and a closer match to the linear-extrapolation baseline—especially on coverage defenders, which were previously our weak spot.

2. **Better alignment with top leaderboard setups**: Most leading Kaggle solutions use transformer-style decoders or residual MLPs over engineered features. Adopting a transformer brings our architecture closer to that proven design while preserving our graph-based strengths.

3. **Stronger generalization**: The decoder no longer relies on high teacher forcing ratios, making its predictions more robust when deployed autoregressively on the hidden test set.

4. **Flexibility for future enhancements**: Transformer layers are modular. We can easily add additional heads (e.g., velocity prediction), auxiliary losses, or role-specific decoder blocks without reworking recurrent loops.

### Summary

By swapping the LSTM decoder for a transformer-based decoder that predicts positional residuals, we address the core weaknesses revealed by the Kaggle leaderboard: compounding errors, limited temporal context, and overreliance on teacher forcing. Coupled with richer feature engineering and tighter regularization, this architectural shift should narrow—and ideally close—the gap between our GNN-based pipeline and the top-performing transformer ensembles.  

