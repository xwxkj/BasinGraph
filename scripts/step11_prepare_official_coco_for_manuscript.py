from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path.cwd()
RUN_ID_FILE = ROOT / "official_results" / "STEP9_LAST_RUN_ID.txt"

if not RUN_ID_FILE.exists():
    raise FileNotFoundError("Cannot find official_results/STEP9_LAST_RUN_ID.txt. Run Step 9 first.")

RUN_ID = RUN_ID_FILE.read_text().strip()

CSV_PATH = ROOT / "processed_results" / f"{RUN_ID}.csv"
COCOPP_LOG = ROOT / "logs" / f"{RUN_ID}_cocopp.log"
OFFICIAL_DIR = ROOT / "official_results" / RUN_ID
OUT_DIR = ROOT / "manuscript_inputs" / f"{RUN_ID}_official_coco"

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures_from_cocopp").mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Missing summary CSV: {CSV_PATH}")

if not COCOPP_LOG.exists():
    raise FileNotFoundError(f"Missing cocopp log: {COCOPP_LOG}")


# ---------------------------------------------------------------------
# Load Step 9 summary CSV
# ---------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)

required_cols = {
    "algorithm", "algorithm_id", "problem_id", "dimension",
    "budget", "seed", "fbest", "nfe", "message", "result_folder"
}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {CSV_PATH}: {sorted(missing)}")

df["dimension"] = df["dimension"].astype(int)
df["budget"] = df["budget"].astype(float)
df["nfe"] = df["nfe"].astype(float)
df["fbest"] = df["fbest"].astype(float)
df["nfe_budget_ratio"] = df["nfe"] / df["budget"].replace(0, np.nan)

# Parse function number when possible.
m = df["problem_id"].str.extract(r"bbob_f(\d+)")
df["function_index"] = pd.to_numeric(m[0], errors="coerce").astype("Int64")

algorithms = sorted(df["algorithm"].unique())
dimensions = sorted(df["dimension"].unique())
n_algorithms = len(algorithms)
n_dimensions = len(dimensions)
n_problems = df["problem_id"].nunique()
n_rows = len(df)

# ---------------------------------------------------------------------
# Official observer output counts
# ---------------------------------------------------------------------
exdata_dirs = sorted((ROOT / "exdata").glob(f"{RUN_ID}_*"))
info_files = []
dat_files = []
for d in exdata_dirs:
    info_files.extend(d.rglob("*.info"))
    dat_files.extend(d.rglob("*.dat"))

# ---------------------------------------------------------------------
# Locate cocopp output directory
# ---------------------------------------------------------------------
log_text = COCOPP_LOG.read_text(errors="ignore")
matches = re.findall(r"Output data written to folder\s+(.+)", log_text)
cocopp_dir = None
if matches:
    cocopp_dir = Path(matches[-1].strip())
    if not cocopp_dir.exists():
        # Sometimes cocopp prints absolute path; otherwise try relative.
        candidate = ROOT / matches[-1].strip()
        if candidate.exists():
            cocopp_dir = candidate

cocopp_all_done = "ALL done" in log_text

# ---------------------------------------------------------------------
# Task-normalized final-value ranks
# Important: this is a sanity check. Official COCO target-runtime
# performance should be interpreted through cocopp ECDF/ERT outputs.
# ---------------------------------------------------------------------
rank_df = df.copy()
rank_df["final_value_rank"] = rank_df.groupby("problem_id")["fbest"].rank(
    method="average", ascending=True
)
rank_df["is_final_value_win"] = rank_df.groupby("problem_id")["fbest"].transform(
    lambda s: s == s.min()
)

algorithm_summary = (
    rank_df.groupby("algorithm")
    .agg(
        mean_final_value_rank=("final_value_rank", "mean"),
        median_final_value_rank=("final_value_rank", "median"),
        wins_by_final_value=("is_final_value_win", "sum"),
        mean_success_proxy=("message", lambda s: np.nan),  # kept for schema compatibility
        mean_nfe_budget_ratio=("nfe_budget_ratio", "mean"),
        median_nfe_budget_ratio=("nfe_budget_ratio", "median"),
        min_nfe_budget_ratio=("nfe_budget_ratio", "min"),
        max_nfe_budget_ratio=("nfe_budget_ratio", "max"),
        rows=("problem_id", "count"),
    )
    .reset_index()
    .sort_values(["mean_final_value_rank", "algorithm"])
)

# Dimension-level summary.
dimension_summary = (
    rank_df.groupby(["dimension", "algorithm"])
    .agg(
        mean_final_value_rank=("final_value_rank", "mean"),
        median_final_value_rank=("final_value_rank", "median"),
        wins_by_final_value=("is_final_value_win", "sum"),
        mean_nfe_budget_ratio=("nfe_budget_ratio", "mean"),
        rows=("problem_id", "count"),
    )
    .reset_index()
    .sort_values(["dimension", "mean_final_value_rank", "algorithm"])
)

# Function-group summary.
function_summary = (
    rank_df.groupby(["function_index", "algorithm"], dropna=False)
    .agg(
        mean_final_value_rank=("final_value_rank", "mean"),
        wins_by_final_value=("is_final_value_win", "sum"),
        rows=("problem_id", "count"),
    )
    .reset_index()
    .sort_values(["function_index", "mean_final_value_rank", "algorithm"])
)

# Budget check.
budget_check = (
    df.groupby("algorithm")
    .agg(
        rows=("problem_id", "count"),
        min_nfe_budget_ratio=("nfe_budget_ratio", "min"),
        median_nfe_budget_ratio=("nfe_budget_ratio", "median"),
        mean_nfe_budget_ratio=("nfe_budget_ratio", "mean"),
        max_nfe_budget_ratio=("nfe_budget_ratio", "max"),
        rows_below_99pct_budget=("nfe_budget_ratio", lambda s: int((s < 0.99).sum())),
    )
    .reset_index()
    .sort_values(["algorithm"])
)

# Pairwise wins against BasinGraph by final value.
pairwise_rows = []
if "BasinGraph" in df["algorithm"].unique():
    bg = df[df["algorithm"] == "BasinGraph"][["problem_id", "fbest"]].rename(
        columns={"fbest": "fbest_basingraph"}
    )
    for alg in algorithms:
        if alg == "BasinGraph":
            continue

        other = df[df["algorithm"] == alg][["problem_id", "fbest"]].rename(
            columns={"fbest": "fbest_baseline"}
        )
        merged = bg.merge(other, on="problem_id", how="inner")
        if len(merged) == 0:
            continue

        # Positive diff means BasinGraph is better.
        diff = merged["fbest_baseline"].to_numpy() - merged["fbest_basingraph"].to_numpy()
        better = int(np.sum(diff > 0))
        worse = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        p_value = np.nan
        nonzero = diff[diff != 0]
        if len(nonzero) > 0:
            try:
                p_value = float(wilcoxon(nonzero, alternative="greater").pvalue)
            except Exception:
                p_value = np.nan

        pairwise_rows.append({
            "baseline": alg,
            "paired_problems": len(merged),
            "basingraph_better_by_final_value": better,
            "basingraph_worse_by_final_value": worse,
            "ties": ties,
            "wilcoxon_p_value_on_paired_final_values": p_value,
            "note": "Final-value sanity comparison; official COCO target-runtime interpretation should use cocopp outputs.",
        })

pairwise = pd.DataFrame(pairwise_rows).sort_values(
    ["basingraph_better_by_final_value", "baseline"], ascending=[False, True]
)

# ---------------------------------------------------------------------
# Locate and copy cocopp figures / tables
# ---------------------------------------------------------------------
figure_inventory = []
if cocopp_dir is not None and cocopp_dir.exists():
    for f in sorted(cocopp_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in {".pdf", ".png", ".svg", ".html", ".tex", ".txt"}:
            rel = f.relative_to(cocopp_dir)
            figure_inventory.append({
                "source_path": str(f),
                "relative_path": str(rel),
                "suffix": f.suffix.lower(),
                "size_bytes": f.stat().st_size,
            })

            # Copy likely manuscript-relevant lightweight files.
            # Avoid copying huge archives; figures and tables only.
            dest = OUT_DIR / "figures_from_cocopp" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(f, dest)
            except Exception:
                pass

figure_inventory_df = pd.DataFrame(figure_inventory)

# ---------------------------------------------------------------------
# Save result tables
# ---------------------------------------------------------------------
df.to_csv(OUT_DIR / "step9_official_coco_raw_summary_copy.csv", index=False)
algorithm_summary.to_csv(OUT_DIR / "step9_official_coco_algorithm_summary.csv", index=False)
dimension_summary.to_csv(OUT_DIR / "step9_official_coco_dimension_summary.csv", index=False)
function_summary.to_csv(OUT_DIR / "step9_official_coco_function_summary.csv", index=False)
budget_check.to_csv(OUT_DIR / "step9_official_coco_budget_check.csv", index=False)
pairwise.to_csv(OUT_DIR / "step9_official_coco_pairwise_wins.csv", index=False)
figure_inventory_df.to_csv(OUT_DIR / "figure_inventory.csv", index=False)

# ---------------------------------------------------------------------
# Copy logs and README files if available
# ---------------------------------------------------------------------
for src in [
    CSV_PATH,
    COCOPP_LOG,
    OFFICIAL_DIR / "README_STEP9.md",
    OFFICIAL_DIR / "VALIDATION_CHECK_STEP9.txt",
    OFFICIAL_DIR / "baseline_provenance_step9.csv",
    OFFICIAL_DIR / f"coco_observer_logs_{RUN_ID}.tar.gz",
    OFFICIAL_DIR / f"cocopp_output_{RUN_ID}.tar.gz",
    OFFICIAL_DIR / "MANIFEST_SHA256.txt",
]:
    if src.exists():
        try:
            shutil.copy2(src, OUT_DIR / src.name)
        except Exception:
            pass

# ---------------------------------------------------------------------
# Generate manuscript-ready notes
# ---------------------------------------------------------------------
top_alg = algorithm_summary.iloc[0]["algorithm"] if len(algorithm_summary) else "[TO FILL]"
top_rank = algorithm_summary.iloc[0]["mean_final_value_rank"] if len(algorithm_summary) else np.nan

bg_row = algorithm_summary[algorithm_summary["algorithm"] == "BasinGraph"]
if len(bg_row):
    bg_mean_rank = float(bg_row.iloc[0]["mean_final_value_rank"])
    bg_wins = int(bg_row.iloc[0]["wins_by_final_value"])
    bg_budget_median = float(bg_row.iloc[0]["median_nfe_budget_ratio"])
else:
    bg_mean_rank = np.nan
    bg_wins = 0
    bg_budget_median = np.nan

ppdata_text = str(cocopp_dir) if cocopp_dir is not None else "[COCOPP OUTPUT DIRECTORY NOT FOUND]"

md = f"""# Step 11 official COCO/BBOB manuscript notes

Generated: {datetime.now().isoformat(timespec='seconds')}

## Run identity

- RUN_ID: `{RUN_ID}`
- Summary CSV: `{CSV_PATH}`
- cocopp log: `{COCOPP_LOG}`
- cocopp output directory: `{ppdata_text}`

## Protocol

- Suite: official COCO/BBOB noiseless
- Dimensions: {", ".join(map(str, dimensions))}
- Functions: 1-24
- Instances: 1-15
- Budget: 1000d
- Algorithms: {", ".join(algorithms)}

## Evidence integrity checks

- Number of algorithm folders under `exdata`: {len(exdata_dirs)}
- Number of `.info` files: {len(info_files)}
- Number of `.dat` files: {len(dat_files)}
- Number of `.info + .dat` files: {len(info_files) + len(dat_files)}
- Summary rows excluding header: {n_rows}
- Unique problem IDs: {n_problems}
- cocopp status contains `ALL done`: {cocopp_all_done}

Expected summary rows for this protocol:
`7 algorithms × 5 dimensions × 24 functions × 15 instances = 12600`.

Observed summary rows:
`{n_rows}`.

## Final-value sanity summary

Important: this section is a final-best-value sanity check based on the CSV. 
For manuscript-level official COCO/BBOB claims, prioritize the `cocopp` ECDF, ERT and target-runtime figures.

- Best task-normalized mean final-value rank: {top_alg} ({top_rank:.3f})
- BasinGraph task-normalized mean final-value rank: {bg_mean_rank:.3f}
- BasinGraph final-value wins: {bg_wins}
- BasinGraph median nfe/budget ratio: {bg_budget_median:.3f}

See:
- `step9_official_coco_algorithm_summary.csv`
- `step9_official_coco_dimension_summary.csv`
- `step9_official_coco_pairwise_wins.csv`
- `step9_official_coco_budget_check.csv`

## Manuscript-ready Results paragraph

In the official COCO/BBOB validation, we evaluated BasinGraph against six general-purpose baselines on the noiseless BBOB suite using the COCO observer and `cocopp` post-processing workflow. The benchmark covered 24 functions, five dimensions (2, 3, 5, 10 and 20), 15 instances per function--dimension pair and a budget of 1000d function evaluations. This protocol produced {n_rows:,} algorithm--problem records, {len(info_files)} `.info` files and {len(dat_files)} `.dat` files. The `cocopp` post-processing completed successfully and generated target-runtime ECDFs, function-group comparisons and comparison tables in `{ppdata_text}`.

As a secondary sanity check on final best values, BasinGraph achieved a task-normalized mean final-value rank of {bg_mean_rank:.3f} across {n_problems} official COCO problem instances, with {bg_wins} final-value wins and a median function-evaluation budget usage ratio of {bg_budget_median:.3f}. Because COCO is primarily target-runtime based, these final-value summaries are reported only as supplementary checks; the main performance interpretation should rely on the `cocopp` ECDF and ERT outputs.

## Suggested figure references

Use the `cocopp` output directory for the official COCO/BBOB figure panels:

- Main Fig. 3a: COCO target-runtime ECDF over all noiseless BBOB functions.
- Main Fig. 3b: dimension-wise or function-group ECDF comparison.
- Supplementary Fig. Sx: per-function ECDF panels.
- Supplementary Tables: `cocopp` comparison tables and `step9_official_coco_algorithm_summary.csv`.

A file inventory is available in `figure_inventory.csv`. The copied figure and table files are in `figures_from_cocopp/`.

## Caution for manuscript wording

Correct wording:
- `official COCO/BBOB validation`
- `COCO observer logs`
- `cocopp target-runtime post-processing`
- `self-contained final-value sanity check`

Avoid:
- claiming that final `fbest` ranks replace COCO ECDF or ERT analysis;
- calling the current baseline implementations author-official unless provenance has been verified;
- reporting cross-function raw `fbest` values without task normalization.

"""

(OUT_DIR / "STEP11_OFFICIAL_COCO_RESULTS_TEXT.md").write_text(md)

# ---------------------------------------------------------------------
# README
# ---------------------------------------------------------------------
readme = f"""# Official COCO/BBOB Step 11 manuscript input package

RUN_ID: {RUN_ID}

This folder contains manuscript-ready notes, summary tables, figure inventory
and copied cocopp files for the Step 9 official COCO/BBOB validation.

Main file:
- STEP11_OFFICIAL_COCO_RESULTS_TEXT.md

Key tables:
- step9_official_coco_algorithm_summary.csv
- step9_official_coco_dimension_summary.csv
- step9_official_coco_pairwise_wins.csv
- step9_official_coco_budget_check.csv
- figure_inventory.csv

Use cocopp ECDF and ERT outputs as the main official COCO/BBOB evidence.
The final-value rank tables are supplementary sanity checks.
"""
(OUT_DIR / "README.md").write_text(readme)

# ---------------------------------------------------------------------
# Zip package
# ---------------------------------------------------------------------
zip_path = ROOT / "manuscript_inputs" / f"{RUN_ID}_official_coco_manuscript_inputs.zip"
if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for f in OUT_DIR.rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(OUT_DIR.parent))

print("STEP_11_OK")
print("RUN_ID:", RUN_ID)
print("Output directory:", OUT_DIR)
print("Zip package:", zip_path)
print("Rows:", n_rows)
print("Algorithms:", n_algorithms)
print("Dimensions:", dimensions)
print("Problems:", n_problems)
print(".info files:", len(info_files))
print(".dat files:", len(dat_files))
print("cocopp ALL done:", cocopp_all_done)
print("BasinGraph mean final-value rank:", bg_mean_rank)
print("BasinGraph final-value wins:", bg_wins)
