"""
Exports flow experiment results + baseline to a single CSV table.
Output: results/flow_comparison_table.csv

Usage:
    python export_results_table.py
"""

import pandas as pd

experiments = [
    ("Baseline: Perfect Prog All Inputs",         "results/24hr_perfect_prog_all_inputs_sigmoid"),
    ("Flow: All Inputs",                           "results/24hr_flow_all_inputs"),
    ("Flow: All Inputs (no STG prog)",             "results/24hr_flow_all_inputs_noSTG"),
    ("Flow: GWL + WL + Flow",                      "results/24hr_flow_gwl_wl_flow"),
    ("Flow: GWL + Flow",                           "results/24hr_flow_gwl_flow"),
    ("Flow: GWL + Rain + Gates",                   "results/24hr_flow_gwl_rain_gates"),
]

models = ["LR", "RF", "MLP"]

rows = []
for exp_label, exp_dir in experiments:
    for model in models:
        path = f"{exp_dir}/{model}/test/results.csv"
        try:
            df = pd.read_csv(path, index_col=0)
            r = df.iloc[0]
            rows.append({
                "Experiment":  exp_label,
                "Model":       model,
                "CF 15cm (%)": round(r["CF_15CM"], 2),
                "CF 5cm (%)":  round(r["CF_5CM"],  2),
                "CF 1cm (%)":  round(r["CF_1CM"],  2),
                "RMSE":        round(r["RMSE"],     4),
                "MAE":         round(r["MAE"],      4),
                "R2":          round(r["R2"],        3),
            })
        except FileNotFoundError:
            print(f"WARNING: not found — {path}")

out = pd.DataFrame(rows)
out_path = "results/flow_comparison_table.csv"
out.to_csv(out_path, index=False)
print(f"Saved → {out_path}")
print(out.to_string(index=False))
