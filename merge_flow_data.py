"""
Merges the clean FLOW CSV into the main merged dataset, adding a 'flow' column.

Inputs:
    data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv
    data/FLOW/clean/FLOW_S28_S_clean.csv

Output:
    data/Merged/Miami_GWL_WL_RAIN_GATE_FLOW_2017_2024.csv

Usage:
    python merge_flow_data.py
"""

import pandas as pd

MERGED_IN  = "data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv"
FLOW_CLEAN = "data/FLOW/clean/FLOW_S28_S_clean.csv"
MERGED_OUT = "data/Merged/Miami_GWL_WL_RAIN_GATE_FLOW_2017_2024.csv"


merged = pd.read_csv(MERGED_IN, index_col=0, parse_dates=True)
flow   = pd.read_csv(FLOW_CLEAN, index_col=0, parse_dates=True)

if merged.index.tz is None:
    merged.index = merged.index.tz_localize("UTC")

merged = merged.join(flow, how="left")

print(f"Columns:        {merged.columns.tolist()}")
print(f"Shape:          {merged.shape}")
print(f"Flow NaN count: {merged['flow'].isna().sum()} / {len(merged)}")
print(f"\nSample:\n{merged[['gate1', 'gate2', 'flow']].head(5)}")

merged.to_csv(MERGED_OUT)
print(f"\nSaved → {MERGED_OUT}")
