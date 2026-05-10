# 📊 Experimental Results

## Study Area
| Property | Value |
|----------|-------|
| Location | Lokoja, Kogi State, Nigeria |
| Coordinates | 6.4°E–7.1°E, 7.4°N–8.1°N |
| Coverage | ~6,000 km² |
| Resolution | 500m × 500m tiles |

## Graph Statistics
| Property | Value |
|----------|-------|
| Nodes | 24,336 |
| Edges | 196,566 (undirected) |
| Mean node degree | 8.08 |
| No Risk nodes | 23,593 (96.9%) |
| Flood Risk nodes | 743 (3.1%) |

## Phase 3 — GATv2

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Architecture | 3-layer GATv2, heads: 4/4/2 |
| Input features | 276 (CNN + spectral + coords) |
| Hidden dim | 128 |
| Embed dim | 64 |
| Dropout | 0.15 |
| Edge dropout | 0.05 |
| Loss | CrossEntropyLoss (weighted, ε=0.05) |
| Optimizer | AdamW (lr=5e-4, wd=1e-5) |
| Scheduler | CosineAnnealingLR |
| Best epoch | 157 |
| Patience | 50 |

### Test Results
| Metric | Value |
|--------|-------|
| Accuracy | 99.2% |
| Flood class F1 | **0.922** |
| Macro F1 | 0.959 |

## Phase 4 — Temporal Transformer

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Architecture | 4-layer Transformer encoder |
| Input dim | 7 (per time step) |
| d_model | 64 |
| Attention heads | 4 |
| Feedforward dim | 256 |
| Time steps | 3 (dry / wet / flood) |
| GAT fusion dim | 64 |
| Dropout | 0.1 |
| Loss | CrossEntropyLoss (weighted, ε=0.05) |
| Optimizer | AdamW (lr=5e-4, wd=1e-5) |
| Batch size | 512 |
| Best epoch | 3 |
| Patience | 25 |

### Test Results
| Metric | Value |
|--------|-------|
| Accuracy | 99.73% |
| Flood class F1 | **0.956** |
| Macro F1 | 0.977 |

## Ablation Study

| Model | Accuracy | Flood F1 | Macro F1 | Δ Flood F1 |
|-------|----------|----------|----------|------------|
| Pixel MLP baseline | 97.1% | 0.712 | 0.843 | — |
| + CNN features | 98.3% | 0.801 | 0.893 | +0.089 |
| + GATv2 spatial graph | 99.2% | 0.922 | 0.959 | +0.121 |
| + Temporal Transformer | **99.73%** | **0.956** | **0.977** | +0.034 |

**Finding:** Spatial graph reasoning (GATv2) contributed the largest improvement (+12.1% F1), confirming flood risk is a network phenomenon.

## Runtime (Free Google Colab T4 GPU)
| Phase | Time |
|-------|------|
| Phase 1 — GEE data export | ~15 min |
| Phase 2 — CNN features | ~20 min |
| Phase 3 — GAT training | ~25 min |
| Phase 4 — Transformer | ~10 min |
| **Total** | **~70 min** |
