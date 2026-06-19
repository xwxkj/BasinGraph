#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

root = Path(__file__).resolve().parents[1]
result_root = root / "cutest_results" / "protocol_v2"

print("Result root:", result_root)

progress = result_root / "progress.json"
if progress.exists():
    print("\nCurrent progress:")
    print(progress.read_text())

raw = result_root / "cutest_raw_results_all_available.csv"
if raw.exists():
    df = pd.read_csv(raw)
    print("\nAll available rows:", len(df))
    print("Seeds:", sorted(df["protocol_seed_index"].unique().tolist()))
    print("Algorithms:", sorted(df["algorithm"].unique().tolist()))
    print("Problem instances:", df["instance_id"].nunique())
    print("\nRows by seed:")
    print(df.groupby("protocol_seed_index").size())
    print("\nFailures:")
    print(df["runner_status"].value_counts(dropna=False))
else:
    print("\nNo merged result CSV yet.")

failure_dir = result_root / "job_failures"
print("\nJob failure files:", len(list(failure_dir.glob("*.txt"))) if failure_dir.exists() else 0)
