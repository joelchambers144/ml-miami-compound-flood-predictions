"""
Script to process raw FLOW data into a clean hourly CSV and create a new merged dataset.
"""

import pandas as pd
import os
import glob

FLOW_DIR = "data/FLOW/FLOW_S28_S"
CLEAN_OUT = "data/FLOW/clean/FLOW_S28_S_clean.csv"
MERGED_IN = "data/Merged/Miami_GWL_WL_RAIN_GATE_2017_2024.csv"
MERGED_OUT = "data/Merged/Miami_GWL_WL_RAIN_GATE_FLOW_2017_2024.csv"

# -------------------------------------------------------------------
# Step 1: Process raw FLOW CSVs → clean hourly CSV
# -------------------------------------------------------------------
print("Reading raw FLOW CSV files...")
flow_files = sorted(glob.glob(os.path.join(FLOW_DIR, "*.csv")))
print(f"  Found {len(flow_files)} files")

dfs = []
for f in flow_files:
    df = pd.read_csv(f, skiprows=34, usecols=["TIMESTAMP", "VALUE"], low_memory=False)
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    dfs.append(df)

flow = pd.concat(dfs, ignore_index=True)
print(f"  Total rows before resampling: {len(flow)}")

# Set index, sort, resample to hourly mean
flow = flow.set_index("TIMESTAMP").sort_index()
flow_hourly = flow["VALUE"].resample("h").mean()
flow_hourly.index = flow_hourly.index.tz_localize("UTC")
flow_hourly.name = "flow"

print(f"  Hourly rows: {len(flow_hourly)}")
print(f"  Date range: {flow_hourly.index[0]} → {flow_hourly.index[-1]}")
print(f"  NaN count: {flow_hourly.isna().sum()}")

flow_hourly.to_csv(CLEAN_OUT, index_label="")
print(f"  Saved → {CLEAN_OUT}\n")

# -------------------------------------------------------------------
# Step 2: Merge flow into the main dataset
# -------------------------------------------------------------------
print("Creating merged dataset with flow column...")
merged = pd.read_csv(MERGED_IN, index_col=0, parse_dates=True)
print(f"  Existing merged shape: {merged.shape}")
print(f"  Columns: {merged.columns.tolist()}")

# Align timezones
if merged.index.tz is None:
    merged.index = merged.index.tz_localize("UTC")

merged = merged.join(flow_hourly, how="left")
print(f"  New merged shape: {merged.shape}")
print(f"  Flow NaN count: {merged['flow'].isna().sum()} / {len(merged)}")

merged.to_csv(MERGED_OUT)
print(f"  Saved → {MERGED_OUT}\n")
print("Done.")
