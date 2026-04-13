"""
Cleans the raw SFWMD FLOW data into an hourly CSV matching the format of
the gate and stage clean CSVs (UTC-indexed, hourly mean, single 'flow' column).

Output: data/FLOW/clean/FLOW_S28_S_clean.csv

Usage:
    python clean_flow_data.py
"""

import glob
import os
import pandas as pd

RAW_PATTERN = "data/FLOW/FLOW_S28_S/*.csv"
OUTPUT_PATH = "data/FLOW/clean/FLOW_S28_S_clean.csv"
HEADER_ROWS = 34  # SFWMD warning/metadata rows before the CSV header


def clean_sfwmd_to_hourly(file_pattern, column_name, header_rows=34):
    """
    Reads one or more raw SFWMD CSV files, coerces VALUE to numeric,
    resamples to hourly mean, and returns a UTC-indexed DataFrame with
    a single column named column_name — matching the format of the gate
    and stage clean CSVs.

    Parameters:
        file_pattern (str): Glob pattern matching the raw CSV files to read
        column_name (str):  Name to give the output value column (e.g. 'flow')
        header_rows (int):  Number of metadata/warning rows to skip before the
                            CSV header (default 34 for SFWMD exports)

    Returns:
        pd.DataFrame: Hourly DataFrame with UTC DatetimeTzDtype index and a
                      single column named column_name
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {file_pattern}")

    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, skiprows=header_rows, usecols=["TIMESTAMP", "VALUE"], low_memory=False)
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True).set_index("TIMESTAMP").sort_index()
    print(f"\nTotal rows before resampling: {len(combined)}")

    hourly = combined["VALUE"].resample("h").mean()
    hourly.index = hourly.index.tz_localize("UTC")
    hourly.name = column_name

    return hourly.to_frame()


if __name__ == "__main__":
    df_clean = clean_sfwmd_to_hourly(RAW_PATTERN, column_name="flow", header_rows=HEADER_ROWS)

    print(f"Hourly rows: {len(df_clean)}")
    print(f"Date range: {df_clean.index[0]} → {df_clean.index[-1]}")
    print(f"NaN count: {df_clean['flow'].isna().sum()}")
    print(f"\nSample output:\n{df_clean.head(5)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"\nSaved → {OUTPUT_PATH}")
