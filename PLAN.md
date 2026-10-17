# PLAN — tabular-shift

A pilot on tabular foundation models for inverse materials design.

## Question
Do tabular foundation models (TFMs) keep their advantage over gradient boosting when test materials
differ from training ones — and can they find materials with a record critical temperature when no
such materials were in training (one-shot "inverse design" as screening)?

## Data
UCI Superconductivity (Hamidieh, 2018): 21 263 records, 81 features computed from elemental
properties, the chemical formula, and the target — critical temperature Tc (K). 5 721 records repeat
an already seen formula, so every honest split is grouped by formula.

## Splits
| Split | Train | Test |
|---|---|---|
| `row-random` (leaky) | random rows | random rows — formulas overlap |
| `group-random` (IID) | random formulas | other formulas |
| `no-cuprate` | everything except cuprates (Cu + O) | cuprates |
| `no-iron` | everything except iron-based (Fe + As / Se / P) | iron-based |
| `tc-extrapolation` | formulas with max Tc ≤ 74 K (80th percentile) | held-out formulas, full Tc range |
| `cluster-k` (k = 0, 1, 2) | all KMeans(8) clusters except *k* | cluster *k* |

Training on 3000 records (a typical TFM context), testing on up to 2000, 3 seeds per split.

## Models
TabICL (Inria) and TabDPT (Layer6) — open TFMs without fine-tuning; CatBoost (1000 iterations);
Ridge as a linear baseline. TabPFN is not used: its weights require a Prior Labs account login.

## Metrics
RMSE, Spearman ρ; screening — precision@50 for the test pool's top-10% Tc; for `tc-extrapolation` —
discovery rate: the share of materials with Tc > 74 K among the 50 best predictions.

## Outcome (2026-10-16, 19:05 MSK)
1 — not confirmed (difference ≤ 0.006 in ρ); 2 — confirmed; 3 — confirmed for every model,
including the linear one. Details in the [README](README.md).

## Predictions (fixed on 2026-10-16, before the run)
1. `row-random` inflates every model's score relative to `group-random`.
2. On `group-random` TFMs are no worse than CatBoost; under family shift their advantage shrinks or
   disappears.
3. On `tc-extrapolation` trees do not predict Tc above the training maximum; every model's discovery
   rate is clearly below what perfect ranking would allow.
