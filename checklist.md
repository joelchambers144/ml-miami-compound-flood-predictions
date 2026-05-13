# FFCA-Guided Model Retraining Checklist

## What We Know

### Paper (TAMUCC_BetoEstrada_Draft_Journal_Paper_March24.pdf)
- Study predicts groundwater level (GWL) as a proxy for compound flooding in Miami-Dade County
- 7-year dataset: Oct 2017 – Dec 2024
- Features: `gwl`, `wl` (ocean), `rain`, `stgH`, `stgT`, `gate1`, `gate2`
- Five model variants (Models 1–5) each tested at 4 lead times (3, 6, 12, 24 hr) = 20 experiment configs
- Best model: "Predictions All Inputs" (Model 5) with rain + gate + ocean WL predictions as input
- Key paper finding: rain predictions are most important; ocean WL predictions add minimal value

### Existing Experiment Infrastructure
- 20 baseline experiment JSONs in `experiments/` (one per model × lead time), all using the `_sigmoid` naming convention that matches the FFCA results folders
- `input_specifications` in each JSON uses **contiguous `lag_range`** format: `{"column": "gwl", "lag_range": [-24, 0]}`
- Pipeline entry point: `python run_experiments.py -e experiments/<name>.json`
- Results cache: if `results/<experiment_name>/` already exists, training is skipped

### FFCA Analysis Results (`FFCA_results/`)
- 20 `report.json` files — one per model × lead time — each containing per-feature trust decisions across 30 ensemble members
- Five decision categories:
  - **CONFIDENTLY KEEP** — stable and important across all 30 models
  - **KEEP (stable)** — stable and useful
  - **MONITOR (borderline)** — borderline; include conservatively
  - **INVESTIGATE (unstable)** — role changes across ensemble members; uncertain
  - **CONFIDENTLY PRUNE** — not important, safe to remove

### Key FFCA Findings Across All 20 Models
| Variable | Verdict |
|----------|---------|
| `rain` past lags | Always CONFIDENTLY KEEP — full lag window important |
| `rain_t+N` future predictions | ALL CONFIDENTLY KEEP in Models 3, 4, 5 — strongest signal |
| `gwl` past lags | CONFIDENTLY KEEP for recent ~9–14 lags; older lags degrade |
| `stgH` past lags | Moderate — a few lags keep (typically t0..t-15 range) |
| `wl` past lags | Low — 0–5 lags keep, scattered |
| `wl_t+N` future predictions | 100% INVESTIGATE — completely useless |
| `stgT` past lags | Mostly INVESTIGATE; 0–3 lags keep occasionally |
| `gate1`/`gate2` past lags | Mostly INVESTIGATE — only t0/t-1 sometimes keeps |
| `gate1_t+N`/`gate2_t+N` future predictions | Essentially all INVESTIGATE — no value |

### Trained Model Specifications (extracted from saved .h5 files)

All 20 original models share the same training protocol. Hyperparameters were selected by GridSearch k-fold CV across the grid below; the winning config is frozen into the saved ensemble.

**Grid search space** (`src/models/MLP.py`):
| Hyperparameter | Values searched |
|----------------|----------------|
| Hidden layers | 1, 2 |
| Neurons per layer | 50, 100, 200 |
| Learning rate | 1e-3, 1e-4, 1e-5 |
| Activation | ReLU |

**Selected hyperparameters per experiment** (from saved model weights):

| Experiment | Hidden Layers | Neurons | Learning Rate | Dropout | n_inputs |
|------------|--------------|---------|--------------|---------|---------|
| 3hr Measurements Only | 1 | 100 | 0.001 | 0.4 | 175 |
| 6hr Measurements Only | 1 | 200 | 0.001 | 0.4 | 175 |
| 12hr Measurements Only | 1 | 100 | 0.001 | 0.4 | 175 |
| 24hr Measurements Only | 1 | 200 | 0.001 | 0.4 | 175 |
| 3hr Predicted Ocean WL | 1 | 200 | 0.001 | 0.4 | 178 |
| 6hr Predicted Ocean WL | 1 | 200 | **0.0001** | 0.4 | 181 |
| 12hr Predicted Ocean WL | 1 | 100 | 0.001 | 0.4 | 187 |
| 24hr Predicted Ocean WL | 1 | 100 | 0.001 | 0.4 | 199 |
| 3hr Predicted Rainfall | 1 | 200 | 0.001 | 0.4 | 178 |
| 6hr Predicted Rainfall | 1 | 200 | 0.001 | 0.4 | 181 |
| 12hr Predicted Rainfall | 1 | 100 | 0.001 | 0.4 | 187 |
| 24hr Predicted Rainfall | 1 | 200 | 0.001 | 0.4 | 199 |
| 3hr Predicted Gate Opening | 1 | 200 | 0.001 | 0.4 | 181 |
| 6hr Predicted Gate Opening | 1 | 200 | 0.001 | 0.4 | 187 |
| 12hr Predicted Gate Opening | 1 | 200 | 0.001 | 0.4 | 199 |
| 24hr Predicted Gate Opening | 1 | 100 | 0.001 | 0.4 | 223 |
| 3hr Predictions All Inputs | 1 | 200 | 0.001 | 0.4 | 187 |
| 6hr Predictions All Inputs | 1 | 100 | 0.001 | 0.4 | 199 |
| 12hr Predictions All Inputs | 1 | 200 | 0.001 | 0.4 | 223 |
| 24hr Predictions All Inputs | 1 | 200 | 0.001 | 0.4 | 271 |

**Key observations:**
- Grid search **always selected 1 hidden layer** — deeper architectures did not help
- LR was 0.001 in 19 of 20 cases; only 6hr Ocean WL used 0.0001
- Dropout is 0.4 on all hidden layers (fixed, not tuned)
- Batch size: 64 (fixed)
- Early stopping patience: 20 epochs, monitoring `val_mean_squared_error`
- Max epochs during tuning: 2,000; during final ensemble training: 10,000
- Ensemble: 30 models, each with a different `tf.random.set_seed(i)` for weight init diversity
- Final ensemble validation year (for early stopping): 2023

**Architecture layout** (Dense → Dropout → Dense[sigmoid] → Lambda[denorm]):
```
Input(n_features) → Dense(neurons, relu, HeNormal) → Dropout(0.4)
                 → Dense(1, sigmoid) → Lambda(x * (y_max - y_min) + y_min)
```

**Comparability requirement for FFCA-pruned models:**
To make a fair apples-to-apples comparison, the FFCA-pruned models must use the **exact same hyperparameters** as their corresponding originals (same neurons, LR, dropout, batch size, early stopping, ensemble size). The only thing that changes is `n_inputs` (the feature set). No re-tuning should be done for the pruned models.

### Current Code Gap
- **No existing code reads `FFCA_results/` or uses trust decisions for anything.**
- The `lag_range` format in experiment JSONs is contiguous; FFCA-selected lags are non-contiguous (e.g., gwl keeps `t-9, t-5, t-2, t0` with gaps between)
- `create_input_columns()` in `src/data/preprocessing.py` only supports contiguous ranges today
- The tuning step would re-run by default if a new experiment name is used — need a way to **skip tuning and use fixed hyperparameters** from the original model's tuning run

---

## Plan

### Step 1 — Add explicit lag list support to preprocessing
- [ ] Modify `src/data/preprocessing.py` → `create_input_columns()` to accept an optional `"lags"` key in each input specification entry
  - If `"lags": [-9, -5, -2, 0]` is present, use that explicit list instead of expanding the `lag_range`
  - If only `"lag_range"` is present, existing behavior is unchanged (backwards compatible)
- [ ] Update `create_lagged_columns()` if needed to support non-contiguous lag lists

### Step 2 — Add fixed-hyperparameter support to MLP training
- [ ] Add a `"fixed_hyperparams"` key to the experiment JSON schema (e.g. `{"num_layers": 1, "neurons": 200, "lr": 0.001, "activation": "relu"}`)
- [ ] Modify `src/models/MLP.py` → `model_training()`: if `experiment.fixed_hyperparams` is set, skip `kfold_cross_validation()` and `find_best_trial_id()`, use the fixed params directly in `train_final_models()`
- [ ] This lets pruned experiments reuse the original model's winning hyperparams without re-running the expensive GridSearch

### Step 3 — Write FFCA-to-experiment-JSON generator script
- [ ] Create `generate_ffca_experiments.py` at repo root
  - Reads all 20 `FFCA_results/<model>/<folder>/report.json` files
  - For each report, collects features with decision in `{CONFIDENTLY KEEP, KEEP (stable), MONITOR (borderline)}`
  - Groups selected features by variable prefix, extracts lag values as explicit lists
  - Reads the corresponding baseline experiment JSON to inherit data path, train/test years, loss function, objective metric
  - Injects the original model's selected hyperparameters (from the table above) as `"fixed_hyperparams"`
  - Writes 20 new experiment JSONs to `experiments/ffca_pruned/`
- [ ] Naming convention: `experiments/ffca_pruned/<lead>_<model_variant>_ffca.json`
- [ ] Experiment name field: `<original_name>_ffca` so results land in `results/<original_name>_ffca/`

### Step 4 — Retrain the 20 pruned MLP-only models
- [ ] Run `python run_experiments.py -e experiments/ffca_pruned/*.json`
  - Restrict to MLP only (add `"model_architectures": ["MLP"]` in generator) since comparison is MLP vs MLP
- [ ] Confirm 30 `.h5` files appear in each `results/<name>_ffca/MLP/models/`

### Step 5 — Compare performance: full vs. pruned
- [ ] Update `export_results_table.py` to include both original and `_ffca` experiment names side by side
- [ ] Generate comparison table: CF_15CM, CF_5CM, CF_1CM, RMSE, R² — original vs. FFCA-pruned for all 20
- [ ] Run `python visuals.py` to render as PNG

### Step 6 — Validate and interpret
- [ ] Check that pruned models maintain or improve performance vs. originals
- [ ] Flag any regressions — possible sign that MONITOR features were load-bearing
- [ ] Document input reduction: full feature count → pruned feature count per model (see table below)

---

## Feature Count Summary (Full vs. Pruned Target)

| Model | Lead | Full Features | C.KEEP+KEEP | C.KEEP+KEEP+MONITOR |
|-------|------|--------------|-------------|---------------------|
| Measurements Only | 3hr | 175 | 39 | 66 |
| Measurements Only | 6hr | 175 | 36 | 60 |
| Measurements Only | 12hr | 175 | 31 | 51 |
| Measurements Only | 24hr | 175 | 37 | 58 |
| Predicted Ocean WL | 3hr | 178 | 29 | 57 |
| Predicted Ocean WL | 6hr | 181 | 26 | 44 |
| Predicted Ocean WL | 12hr | 187 | 33 | 52 |
| Predicted Ocean WL | 24hr | 199 | 40 | 50 |
| Predicted Rainfall | 3hr | 178 | 35 | 64 |
| Predicted Rainfall | 6hr | 181 | 38 | 66 |
| Predicted Rainfall | 12hr | 187 | 48 | 74 |
| Predicted Rainfall | 24hr | 199 | 54 | 74 |
| Predicted Gate Opening | 3hr | 181 | 30 | 54 |
| Predicted Gate Opening | 6hr | 187 | 38 | 62 |
| Predicted Gate Opening | 12hr | 199 | 47 | 72 |
| Predicted Gate Opening | 24hr | 223 | 48 | 62 |
| Predictions All Inputs | 3hr | 187 | 30 | 55 |
| Predictions All Inputs | 6hr | 199 | 39 | 77 |
| Predictions All Inputs | 12hr | 223 | 58 | 96 |
| Predictions All Inputs | 24hr | 271 | 69 | 95 |
