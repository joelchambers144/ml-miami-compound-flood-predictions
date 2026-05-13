# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository implements machine learning models to predict groundwater levels in Miami, Florida as a proxy for compound flooding events. The project uses Multi-Layer Perceptron (MLP), Random Forest (RF), and Linear Regression (LR) models with various input combinations (groundwater level, ocean water level, rainfall, rainfall predictions, and gate operations) at different lead times (1, 3, 6, 12, 24 hours).

## Setup & Environment

**Create Conda environment:**
```bash
conda env create -f environment.yml
conda activate conda-flood
```

**Update environment after modifying environment.yml:**
```bash
conda env update --name conda-flood --file environment.yml --prune
```

**Python version:** 3.12.11

**Key dependencies:**
- Keras 3.11.2 with Keras Tuner for hyperparameter tuning
- TensorFlow/scikit-learn for model implementations
- Pandas/NumPy for data processing
- TensorBoard for visualizing hyperparameter tuning results

## Data Pipeline

The data pipeline consists of several stages:

1. **Raw Data Sources:** Located in `data/GWL/`, `data/WL/`, `data/GATE/`, `data/Rain/`, `data/STG/`, `data/FLOW/`
2. **Cleaning:** Individual scripts clean raw data into hourly UTC-indexed CSVs
   - `clean_flow_data.py` - Processes SFWMD FLOW data (skips 34 header rows, resamples to hourly mean)
3. **Merging:** Combined into `data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv` (main dataset used for experiments)
   - `merge_flow_data.py` - Adds FLOW data as a new column

**Key dataset columns:**
- `gwl` - Groundwater level (target variable)
- `wl` - Ocean water level
- `rain` - Rainfall measurements
- `stgH`, `stgT` - Stage measurements
- `gate1`, `gate2` - Gate operations
- `flow` - Flow measurements (added via merge)

## Experiment Architecture

**Configuration-driven approach:**
1. Experiments are defined in JSON files in `experiments/` directory
2. Each JSON specifies:
   - Data file path
   - Input specifications (column names + lag ranges, e.g., `gwl_t-24` to `gwl_t0`)
   - Target column and lead time (future prediction offset in hours)
   - Train/test years for temporal split
   - Model architectures to train (LR, RF, MLP)
   - Loss function and optimization metric

**Example experiment structure:**
```json
{
  "experiments": [{
    "data_file_path": "data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv",
    "experiment_name": "test_experiment",
    "lead_time": 24,
    "target_column": "gwl",
    "input_specifications": [
      {"column": "gwl", "lag_range": [-24, 0]},
      {"column": "wl", "lag_range": [-24, 0]}
    ],
    "model_architectures": ["LR", "RF", "MLP"],
    "train_years": [2017, 2018],
    "test_years": [2024],
    "loss_function": "mean_squared_error",
    "objective_metric": "loss"
  }]
}
```

## Running Experiments

**Run one or more experiments:**
```bash
python run_experiments.py -e experiments/test_experiment.json
python run_experiments.py -e experiments/3hr_models.json experiments/24hr_models.json
```

The pipeline automatically:
1. Creates lagged input columns based on input specifications
2. Splits data by years (train/test)
3. For MLP: Performs k-fold cross-validation with Keras Tuner GridSearch to find optimal hyperparameters
4. For RF/LR: Performs GridSearch with custom k-fold splits
5. Trains ensemble of models (30 models for MLP) using best hyperparameters on full training set
6. Evaluates on test set and saves results

**Results structure:**
```
results/{experiment_name}/
├── {model_type}/ (LR, RF, or MLP)
│   ├── test/
│   │   ├── results.csv (metrics: CF_15CM, CF_5CM, CF_1CM, RMSE, MAE, R2, MAPE)
│   │   └── predictions.csv
│   ├── tuning/ (MLP/RF only)
│   │   ├── best_cv_metrics.csv (or best_results.csv)
│   │   └── {validation_year}/tb_logs/ (TensorBoard logs for MLP)
│   └── models/
│       ├── hypermodel.pkl (RF/LR)
│       └── hypermodel1.h5, hypermodel2.h5, ... (MLP ensemble)
└── test/
    └── results.csv (combined metrics across all models)
```

## Model Implementations

**MLP (`src/models/MLP.py`):**
- Keras Sequential model with configurable layers and neurons
- Sigmoid output layer with Lambda denormalization using global min/max bounds
- Ensemble prediction via median of 30 models
- GridSearch hyperparameter tuning with options:
  - `num_layers`: [1, 2]
  - `neurons`: [50, 100, 200]
  - `lr`: [1e-3, 1e-4, 1e-5]
  - `activation`: ['relu']
- Custom weighted MSE loss available (prioritizes high groundwater levels > 0.5m)
- K-fold cross-validation with early stopping

**RF (`src/models/RF.py`):**
- scikit-learn RandomForestRegressor
- GridSearch tuning parameters: n_estimators, max_depth, min_samples_split, max_features
- K-fold via custom_folds (based on train_years)

**LR (`src/models/LR.py`):**
- Simple scikit-learn LinearRegression baseline
- No hyperparameter tuning

## Data Preprocessing Pipeline

**Key functions in `src/data/preprocessing.py`:**

1. **create_input_columns()** - Creates lagged columns based on input_specifications, drops NaNs
2. **create_lagged_columns()** - Adds past/future shifted columns
   - Lag range [-24, 0] creates columns from t-24 to t0 (past 24 hours)
   - Lag range [24, 24] creates target column (t+24 for 24-hour lead time)
3. **split_df_by_years()** - Temporal train/test split
4. **create_kfolds_keras()** - K-fold splits for Keras (returns train/valid data tuples)
5. **create_partitions_sklearn()** - K-fold splits for scikit-learn (returns index arrays)
6. **order_input_arrays()** - Orders columns by prefix name, then ascending lag time

**Normalization (`src/data/normalization.py`):**
- MLP output layer denormalized using global min/max bounds (max increased by 20% for extrapolation)
- Sigmoid activates in [0, 1], then scaled to [y_min, y_max * 1.2]

## Evaluation Metrics

**Calculated in `src/evaluation/metrics.py`:**
- **CF_15CM, CF_5CM, CF_1CM** - Central Frequency: % of predictions within N cm of true value
- **RMSE, MAE, MEDAE, MAPE** - Error metrics
- **R2** - Coefficient of determination
- **MSE** - Mean squared error (not typically used in results, but intermediate metric)
- **Weighted MSE** - Custom loss function that assigns higher weight (20x) to high groundwater levels

## Visualization & Results Export

**Export results table:**
```bash
python export_results_table.py
```
Outputs `results/flow_comparison_table.csv` with metrics for specified experiments and models.

**Visualize results table as PNG:**
```bash
python visuals.py
```
Generates `table.png` from `flow_comparison_table.csv`.

**TensorBoard for MLP hyperparameter tuning:**
```bash
tensorboard --logdir results/{experiment_name}/MLP/tuning/{validation_year}/tb_logs --port 6006
```

## Notebooks

Located in `notebooks/`:
- `clean_flow_data.ipynb` - Data cleaning exploration
- `24hr_flow_model_training.ipynb` - Flow study experiment training
- `flow_visualization.ipynb` - Visualization of flow experiment results
- `shap_analysis.ipynb` - SHAP feature importance analysis

## GPU Usage

Control GPU usage via `.env` file:
```
USE_GPU=false  # Set to 'false' to use CPU only, anything else enables GPU
```
This is read by `run_experiments.py` to set `CUDA_VISIBLE_DEVICES`.

## Key Design Patterns

1. **Experiment objects** encapsulate all configuration parameters, passed through the entire pipeline
2. **Model classes** (MLP, RF, LR) have consistent interfaces:
   - `model_training(df_data, experiment, results_directory)` - Train and return best model
   - `predict(model, X)` - Make predictions, return dict with 'predictions' key
3. **Temporal data splits** by year (not random) to avoid data leakage
4. **Ensemble methods**: MLP uses median of 30 models; RF/LR use single best model
5. **Lazy evaluation**: Test results cached - if they exist, pipeline skips training
