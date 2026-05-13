# FFCA Report

_Generated 2026-05-12 13:27:24 — 30 checkpoint(s), 187 features._

## Diagnostics — model health overview

### ⚠ `trust_instability` · **warn** — 135/187 features are unstable across checkpoints

**What was observed.** More than half (72%) of features changed archetype between checkpoints (similarity-weighted stability < 0.5).

**Why it matters.** High INVESTIGATE rate suggests the model has not settled into stable feature roles — either it is still training, or the data is noisy enough that different epochs use the features differently.

**What to do.** Train for more epochs, or use a learning-rate schedule that converges sooner. If accuracy is good but features are unstable, the model is an ensemble in disguise.

### ℹ `capacity` · **info** — Healthy archetype distribution

**What was observed.** Noise 14%, Catalyst 23%, Complex 29%.

**Why it matters.** No archetype dominates — the model has a diverse mix of feature roles.

**What to do.** None.

### ℹ `co_sensitivity` · **info** — Co-Sensitivity refused to recommend any prune (2 groups found, but no prune-safe group)

**What was observed.** k=2 functional groups, silhouette=0.41, permutation p=0.000, bootstrap ARI=0.10, best group NC fraction=15%.  Abort triggered because: no group has >50% Noise Candidates (best=15%) ; bootstrap stability low (ARI=0.10<0.5).

**Why it matters.** Co-Sensitivity clusters features by gradient similarity. It only recommends pruning when a whole group is noise-dominated (>50%) AND the clustering itself is statistically supported. Refusing to prune is the SAFE outcome — the alternative is removing useful features.

**What to do.** Pruning is not warranted from this run. If you need to compress the model, use magnitude-based or movement pruning instead.

### ℹ `overfitting` · **info** — No volatility spike detected across 30 checkpoints

**What was observed.** Final-checkpoint mean Volatility = 6.587e-06 ; median of earlier checkpoints = 4.243e-06 (ratio = 1.55×).

**Why it matters.** Volatility = Var(∂f/∂x_i) measures how context-dependent each feature's effect is. A late-training jump usually means the model is memorising sample-specific quirks rather than learning a stable rule.

**What to do.** Within healthy training; no action required.

### ℹ `trust_keep_recommended` · **info** — 33 features are stably important across all checkpoints

**What was observed.** These features retained the same useful archetype (Workhorse / Catalyst / Stable Contributor) at every checkpoint.

**Why it matters.** High-stability + high-importance features are the backbone of the model. Removing or changing them will be felt in accuracy.

**What to do.** Treat these as load-bearing. Protect with extra logging in production; do not prune.

## The FFCA 4-D signature

FFCA decomposes each feature's influence on the model into four independent axes. Higher values mean stronger / less linear / more context-dependent / more entangled.

| Axis | Definition | Reads as |
|---|---|---|
| **Impact** | E[\|∂f/∂x_i\|] | how much the feature moves the output |
| **Volatility** | Var[∂f/∂x_i] | how context-dependent that effect is |
| **Non-linearity** | E[\|∂²f/∂x_i²\|] | how curved the response is |
| **Interaction** | Σ E[\|∂²f/∂x_i∂x_j\|] | how much the feature acts through others |

### Summary at the final checkpoint

| Dim | mean | std | min | max |
|-----|------|-----|-----|-----|
| impact | 0.0109 | 0.0327 | 0.0001 | 0.3000 |
| volatility | 0.0000 | 0.0000 | 0.0000 | 0.0001 |
| nonlinearity | 0.0049 | 0.0141 | 0.0002 | 0.1521 |
| interaction | 0.1300 | 0.2155 | 0.0124 | 1.8736 |

_Interaction column computed via_ **cauchy_hvp**.

### Top 10 features by Impact

Sorted by Impact (mean absolute gradient). The archetype column is FFCA's high-level label for what role each feature plays in the model. See `docs/adapters.md` for the full archetype table.

| Rank | Feature | Impact | Interaction | Archetype |
|--|--|--|--|--|
| 1 | gwl | 0.3000 | 1.8736 | Catalyst |
| 2 | gwl_t-1 | 0.2230 | 1.3803 | Catalyst |
| 3 | gwl_t-2 | 0.1613 | 0.9780 | Catalyst |
| 4 | rain | 0.1360 | 0.8593 | Catalyst |
| 5 | gwl_t-3 | 0.1099 | 0.7295 | Catalyst |
| 6 | gwl_t-4 | 0.0714 | 0.4715 | Catalyst |
| 7 | rain_t-1 | 0.0664 | 0.5605 | Catalyst |
| 8 | rain_t-2 | 0.0510 | 0.3087 | Catalyst |
| 9 | rain_t-3 | 0.0453 | 0.2557 | Catalyst |
| 10 | gwl_t-5 | 0.0406 | 0.3317 | Catalyst |

## Archetype distribution

How FFCA categorises every feature in the model. Healthy models have a spread; a single dominant bucket suggests either under- or over-fitting (see Diagnostics above).

| Archetype | Count | % | What it means |
|--|--|--|--|
| Noise | 27 | 14.4% | low everywhere — candidate for pruning |
| Hidden Interactor | 4 | 2.1% | weak alone, strong via interactions |
| Catalyst | 43 | 23.0% | strong AND interacting — load-bearing |
| Nonlinear Driver | 12 | 6.4% | strong with curved relationship |
| Volatile Specialist | 18 | 9.6% | strong but context-dependent |
| Stable Contributor | 29 | 15.5% | moderate, reliable |
| Complex Driver | 54 | 28.9% | complex behaviour across all four axes |

## Trust Score (similarity-weighted stability across 30 checkpoints)

Each feature is tracked across training checkpoints. A feature that keeps the same archetype every time gets high stability; one that flips around gets low stability. The decision combines that stability with the feature's mean Impact.

| Decision | Count | What it means |
|--|--|--|
| INVESTIGATE (unstable) | 135 | archetype flipped — role uncertain |
| CONFIDENTLY KEEP | 32 | stable + important — load-bearing |
| MONITOR (borderline) | 19 | stability between 0.5 and 0.7 |
| KEEP (stable) | 1 | stable but moderate importance |

**Investigate** (135): `gwl_t-24`, `gwl_t-23`, `gwl_t-22`, `gwl_t-21`, `gwl_t-20`, `gwl_t-19`, `gwl_t-16`, `gwl_t-15`, `gwl_t-14`, `gwl_t-10`

## Co-Sensitivity functional groups — ❌ ABORT — no group is safe to prune

Features are clustered by gradient-correlation distance (1 − |ρ|) using k-medoids. The package only recommends pruning when (a) a group is dominated by Noise Candidates (>50%), (b) the clustering is statistically distinguishable from random shuffling (perm-p < 0.05), and (c) the clustering is stable across 80%-bootstrap resamples (ARI ≥ 0.5).

- **k** = 2 groups
- **silhouette** = 0.411  _(higher = tighter clusters; >0.5 is strong, >0.2 is moderate)_
- **permutation p-value** = 0.000  _(<0.05 means clusters aren't random)_
- **bootstrap ARI** = 0.103  _(≥0.5 means clustering is stable)_
- **best NC fraction** = 15.1%  _(needs >50% to prune)_

| Group | Size | NC % | Mean Impact | Recommendation |
|--|--|--|--|--|
| 0 | 41 | 12.2% | 0.018 | KEEP — mostly useful |
| 1 | 146 | 15.1% | 0.0089 | KEEP — mostly useful |

## Plots — what's in each one

### `plots/01_signature_radar.png`

Radar of the four FFCA axes for the top features. Lines that hug the outer ring on all four axes are Complex Drivers; ones that bulge only on the Interaction axis are Hidden Interactors.

### `plots/02_archetype_distribution.png`

How many features fall into each of the eight archetypes. The shape of this distribution is the model's high-level health signature.

### `plots/03_impact_ranking.png`

Top features by mean absolute gradient, colour-coded by archetype. The colour bar at the top is the same key as the archetype-distribution plot.

### `plots/04_interaction_ci.png`

Per-feature interaction score with its 95% Cauchy-HVP confidence interval. Features whose error bars do NOT cross zero are reliably interacting.

### `plots/05_channel_archetype_grid.png`

One coloured square per channel, numbered left-to-right, top-to-bottom; colour encodes the channel's archetype. Useful for spotting clusters of redundant or Noise channels.

### `plots/05_pixel_interaction_map.png`

Pixel-level interaction reshaped back into the image grid. Bright regions are where the model's decision is driven by interactions between pixels.

### `plots/06_fbr_diagnostic.png`

Foreground/Background ratio: mean interaction in the centre half of the image vs the surrounding ring. FBR < 0.5 is the Waterbirds-style background-shortcut signature.

### `plots/10_impact_evolution.png`

Top features' Impact across all checkpoints. Curves that diverge late in training are becoming more important; ones that collapse to zero have been forgotten.

### `plots/11_ranking_evolution.png`

Bump chart of feature rank (by Impact) over time. Lines that cross frequently indicate unstable feature importance — see the Trust Score above.

### `plots/12_archetype_evolution.png`

Heatmap of each feature's archetype (colour) at each checkpoint (column). Vertical stripes = stable archetype; rainbow rows = unstable.

### `plots/13_trust_scatter.png`

Stability vs Importance scatter. Top-right quadrant = confidently keep; bottom-left + stable = prune; anything on the left (stability < 0.5) = investigate.

### `plots/20_co_sensitivity_groups.png`

Cluster sizes and Noise-Candidate fractions. Bars are coloured red if a group is a prune candidate (NC > 50%), orange if it's borderline (>30%), green otherwise.


## Timing

- signatures_s: 0.36s
- trust_s: 0.02s
- cosens_s: 0.06s
- diagnostics_s: 0.00s
