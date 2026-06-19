from __future__ import annotations

import csv
import hashlib
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =====================================================================
# 1. Locate the frozen final COCO/BBOB run
# =====================================================================

ROOT = Path.cwd()

RUN_ID_FILE = (
    ROOT
    / "official_results"
    / "OFFICIAL_FINAL_CORE_RUN_ID.txt"
)

if not RUN_ID_FILE.exists():
    raise FileNotFoundError(
        "Missing official_results/OFFICIAL_FINAL_CORE_RUN_ID.txt"
    )

RUN_ID = RUN_ID_FILE.read_text().strip()

FINAL_ARCHIVE_DIR = (
    ROOT
    / "official_results"
    / f"{RUN_ID}_FINAL"
)

CSV_PATH = (
    ROOT
    / "processed_results"
    / f"{RUN_ID}.csv"
)

COCOPP_LOG = (
    ROOT
    / "logs"
    / f"{RUN_ID}_cocopp.log"
)

if not FINAL_ARCHIVE_DIR.exists():
    raise FileNotFoundError(
        f"Frozen archive directory not found: {FINAL_ARCHIVE_DIR}"
    )

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Final result CSV not found: {CSV_PATH}"
    )

if not COCOPP_LOG.exists():
    raise FileNotFoundError(
        f"cocopp log not found: {COCOPP_LOG}"
    )


# =====================================================================
# 2. Create output directory
# =====================================================================

OUT_DIR = (
    ROOT
    / "manuscript_inputs"
    / f"{RUN_ID}_FINAL_manuscript_inputs"
)

if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

for subdirectory in [
    "tables",
    "source_data",
    "cocopp_selected_figures",
    "cocopp_tables",
    "logs_and_protocols",
    "manuscript_text",
]:
    (OUT_DIR / subdirectory).mkdir(
        parents=True,
        exist_ok=True,
    )


# =====================================================================
# 3. Load and validate final result table
# =====================================================================

df = pd.read_csv(CSV_PATH)

required_columns = {
    "algorithm",
    "algorithm_id",
    "problem_id",
    "dimension",
    "budget",
    "seed",
    "fbest",
    "nfe",
    "message",
    "result_folder",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

expected_algorithms = {
    "BasinGraph",
    "CMA-ES",
    "BIPOP-CMA-ES",
    "Differential Evolution",
    "Multi-start L-BFGS-B",
    "Latin Hypercube Sampling",
    "Random Search",
}

assert len(df) == 12600
assert set(df["algorithm"].unique()) == expected_algorithms
assert sorted(df["dimension"].unique().tolist()) == [
    2, 3, 5, 10, 20
]
assert df["problem_id"].nunique() == 1800

df["budget_ratio"] = (
    df["nfe"].astype(float)
    / df["budget"].astype(float)
)

# Parse BBOB function number.
function_match = df["problem_id"].str.extract(
    r"bbob_f(\d+)"
)

df["function_index"] = pd.to_numeric(
    function_match[0],
    errors="coerce",
).astype("Int64")


# =====================================================================
# 4. Add standard BBOB function-group labels
# =====================================================================

def bbob_group(function_index: int) -> str:
    """Return the standard broad BBOB function group."""
    if 1 <= function_index <= 5:
        return "Separable functions (f1–f5)"
    if 6 <= function_index <= 9:
        return "Low/moderate conditioning (f6–f9)"
    if 10 <= function_index <= 14:
        return "High conditioning and unimodal (f10–f14)"
    if 15 <= function_index <= 19:
        return "Multimodal with global structure (f15–f19)"
    if 20 <= function_index <= 24:
        return "Multimodal with weak global structure (f20–f24)"
    return "Unknown"


df["function_group"] = df["function_index"].map(
    lambda x: bbob_group(int(x))
    if pd.notna(x)
    else "Unknown"
)


# =====================================================================
# 5. Compute task-normalized final-value ranks
# =====================================================================

# COCO's primary evidence is target-runtime/ERT from cocopp.
# These final-value ranks are supplementary sanity checks only.
rank_df = df.copy()

rank_df["final_value_rank"] = (
    rank_df.groupby("problem_id")["fbest"]
    .rank(
        method="average",
        ascending=True,
    )
)

rank_df["final_value_win"] = (
    rank_df.groupby("problem_id")["fbest"]
    .transform(lambda values: values == values.min())
)


# =====================================================================
# 6. Algorithm-level summary
# =====================================================================

algorithm_summary = (
    rank_df.groupby("algorithm")
    .agg(
        mean_final_value_rank=(
            "final_value_rank",
            "mean",
        ),
        median_final_value_rank=(
            "final_value_rank",
            "median",
        ),
        final_value_wins=(
            "final_value_win",
            "sum",
        ),
        median_budget_ratio=(
            "budget_ratio",
            "median",
        ),
        min_budget_ratio=(
            "budget_ratio",
            "min",
        ),
        mean_budget_ratio=(
            "budget_ratio",
            "mean",
        ),
        result_rows=(
            "problem_id",
            "count",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "mean_final_value_rank",
            "algorithm",
        ]
    )
)

algorithm_summary.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_algorithm_summary.csv",
    index=False,
)


# =====================================================================
# 7. Dimension-level summary
# =====================================================================

dimension_summary = (
    rank_df.groupby(
        [
            "dimension",
            "algorithm",
        ]
    )
    .agg(
        mean_final_value_rank=(
            "final_value_rank",
            "mean",
        ),
        median_final_value_rank=(
            "final_value_rank",
            "median",
        ),
        final_value_wins=(
            "final_value_win",
            "sum",
        ),
        median_budget_ratio=(
            "budget_ratio",
            "median",
        ),
        result_rows=(
            "problem_id",
            "count",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "dimension",
            "mean_final_value_rank",
            "algorithm",
        ]
    )
)

dimension_summary.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_dimension_summary.csv",
    index=False,
)


# =====================================================================
# 8. Function-group summary
# =====================================================================

function_group_summary = (
    rank_df.groupby(
        [
            "function_group",
            "algorithm",
        ]
    )
    .agg(
        mean_final_value_rank=(
            "final_value_rank",
            "mean",
        ),
        median_final_value_rank=(
            "final_value_rank",
            "median",
        ),
        final_value_wins=(
            "final_value_win",
            "sum",
        ),
        result_rows=(
            "problem_id",
            "count",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "function_group",
            "mean_final_value_rank",
            "algorithm",
        ]
    )
)

function_group_summary.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_function_group_summary.csv",
    index=False,
)


# =====================================================================
# 9. Per-function summary
# =====================================================================

function_summary = (
    rank_df.groupby(
        [
            "function_index",
            "algorithm",
        ],
        dropna=False,
    )
    .agg(
        mean_final_value_rank=(
            "final_value_rank",
            "mean",
        ),
        median_final_value_rank=(
            "final_value_rank",
            "median",
        ),
        final_value_wins=(
            "final_value_win",
            "sum",
        ),
        result_rows=(
            "problem_id",
            "count",
        ),
    )
    .reset_index()
    .sort_values(
        [
            "function_index",
            "mean_final_value_rank",
            "algorithm",
        ]
    )
)

function_summary.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_per_function_summary.csv",
    index=False,
)


# =====================================================================
# 10. Pairwise final-value comparisons against BasinGraph
# =====================================================================

pairwise_rows = []

bg = (
    rank_df[
        rank_df["algorithm"] == "BasinGraph"
    ][
        [
            "problem_id",
            "fbest",
        ]
    ]
    .rename(
        columns={
            "fbest": "fbest_basingraph",
        }
    )
)

for baseline in sorted(
    expected_algorithms.difference({"BasinGraph"})
):
    other = (
        rank_df[
            rank_df["algorithm"] == baseline
        ][
            [
                "problem_id",
                "fbest",
            ]
        ]
        .rename(
            columns={
                "fbest": "fbest_baseline",
            }
        )
    )

    paired = bg.merge(
        other,
        on="problem_id",
        how="inner",
    )

    difference = (
        paired["fbest_baseline"].to_numpy()
        - paired["fbest_basingraph"].to_numpy()
    )

    better = int(np.sum(difference > 0))
    worse = int(np.sum(difference < 0))
    ties = int(np.sum(difference == 0))

    nonzero = difference[difference != 0]

    if len(nonzero) > 0:
        try:
            p_value = float(
                wilcoxon(
                    nonzero,
                    alternative="greater",
                ).pvalue
            )
        except Exception:
            p_value = np.nan
    else:
        p_value = np.nan

    pairwise_rows.append({
        "baseline": baseline,
        "paired_problem_instances": len(paired),
        "basingraph_better_final_value": better,
        "basingraph_worse_final_value": worse,
        "ties": ties,
        "one_sided_wilcoxon_p_value": p_value,
        "interpretation_note":
            "Supplementary final-value comparison; "
            "official COCO conclusions should rely on cocopp ECDF/ERT.",
    })

pairwise_df = pd.DataFrame(pairwise_rows)

pairwise_df.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_pairwise_final_value_comparisons.csv",
    index=False,
)


# =====================================================================
# 11. Budget usage summary
# =====================================================================

budget_summary = (
    df.groupby("algorithm")
    .agg(
        result_rows=(
            "problem_id",
            "count",
        ),
        min_budget_ratio=(
            "budget_ratio",
            "min",
        ),
        median_budget_ratio=(
            "budget_ratio",
            "median",
        ),
        mean_budget_ratio=(
            "budget_ratio",
            "mean",
        ),
        max_budget_ratio=(
            "budget_ratio",
            "max",
        ),
        rows_below_99_percent_budget=(
            "budget_ratio",
            lambda values: int(
                (values < 0.99).sum()
            ),
        ),
    )
    .reset_index()
)

budget_summary.to_csv(
    OUT_DIR
    / "tables"
    / "official_coco_budget_usage.csv",
    index=False,
)


# =====================================================================
# 12. Locate final cocopp output
# =====================================================================

cocopp_log_text = COCOPP_LOG.read_text(
    errors="ignore"
)

if "ALL done" not in cocopp_log_text:
    raise RuntimeError(
        "Final cocopp log does not contain 'ALL done'."
    )

output_matches = re.findall(
    r"Output data written to folder\s+(.+)",
    cocopp_log_text,
)

if not output_matches:
    raise RuntimeError(
        "Cannot locate cocopp output directory."
    )

COCOPP_DIR = Path(
    output_matches[-1].strip()
)

if not COCOPP_DIR.exists():
    candidate = ROOT / output_matches[-1].strip()
    if candidate.exists():
        COCOPP_DIR = candidate
    else:
        raise FileNotFoundError(
            f"cocopp directory not found: {COCOPP_DIR}"
        )


# =====================================================================
# 13. Inventory and copy manuscript-relevant cocopp outputs
# =====================================================================

figure_inventory = []

figure_suffixes = {
    ".pdf",
    ".png",
    ".svg",
}

table_suffixes = {
    ".html",
    ".tex",
    ".txt",
}

figure_keywords = (
    "pprldmany",
    "pprldistr",
    "ppfigdim",
    "ppfigs",
    "ppscatter",
    "ecdf",
    "rld",
)

table_keywords = (
    "pptable",
    "pptex",
    "table",
)


def classify_file(file_path: Path) -> str:
    name = file_path.name.lower()

    if any(keyword in name for keyword in figure_keywords):
        return "figure_candidate"

    if any(keyword in name for keyword in table_keywords):
        return "table_candidate"

    return "supplementary_other"


for file_path in sorted(
    COCOPP_DIR.rglob("*")
):
    if not file_path.is_file():
        continue

    suffix = file_path.suffix.lower()

    if suffix not in (
        figure_suffixes
        | table_suffixes
    ):
        continue

    relative_path = file_path.relative_to(
        COCOPP_DIR
    )

    classification = classify_file(
        file_path
    )

    figure_inventory.append({
        "classification": classification,
        "source_relative_path": str(relative_path),
        "suffix": suffix,
        "size_bytes": file_path.stat().st_size,
    })

    if classification == "figure_candidate":
        destination = (
            OUT_DIR
            / "cocopp_selected_figures"
            / relative_path
        )
    elif classification == "table_candidate":
        destination = (
            OUT_DIR
            / "cocopp_tables"
            / relative_path
        )
    else:
        # Do not copy all peripheral files.
        continue

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        file_path,
        destination,
    )


figure_inventory_df = pd.DataFrame(
    figure_inventory
)

figure_inventory_df.to_csv(
    OUT_DIR
    / "tables"
    / "cocopp_figure_and_table_inventory.csv",
    index=False,
)


# =====================================================================
# 14. Copy final logs, protocols and provenance
# =====================================================================

files_to_copy = [
    FINAL_ARCHIVE_DIR / "README.md",
    FINAL_ARCHIVE_DIR / "MANIFEST_SHA256.csv",
    FINAL_ARCHIVE_DIR
    / "protocols"
    / "final_baseline_provenance.csv",
    FINAL_ARCHIVE_DIR
    / "protocols"
    / "final_run_metadata.csv",
    FINAL_ARCHIVE_DIR
    / "tables"
    / CSV_PATH.name,
    FINAL_ARCHIVE_DIR
    / "tables"
    / "final_budget_usage_summary.csv",
    FINAL_ARCHIVE_DIR
    / "logs"
    / COCOPP_LOG.name,
]

for source in files_to_copy:
    if source.exists():
        shutil.copy2(
            source,
            OUT_DIR
            / "logs_and_protocols"
            / source.name,
        )


# =====================================================================
# 15. Generate Source Data files
# =====================================================================

algorithm_summary.to_csv(
    OUT_DIR
    / "source_data"
    / "SourceData_Fig3_algorithm_summary.csv",
    index=False,
)

dimension_summary.to_csv(
    OUT_DIR
    / "source_data"
    / "SourceData_Fig3_dimension_summary.csv",
    index=False,
)

function_group_summary.to_csv(
    OUT_DIR
    / "source_data"
    / "SourceData_Fig3_function_groups.csv",
    index=False,
)

pairwise_df.to_csv(
    OUT_DIR
    / "source_data"
    / "SourceData_Supplementary_pairwise.csv",
    index=False,
)


# =====================================================================
# 16. Extract headline numbers
# =====================================================================

bg_row = algorithm_summary[
    algorithm_summary["algorithm"]
    == "BasinGraph"
].iloc[0]

best_row = algorithm_summary.iloc[0]

bg_mean_rank = float(
    bg_row["mean_final_value_rank"]
)

bg_median_rank = float(
    bg_row["median_final_value_rank"]
)

bg_wins = int(
    bg_row["final_value_wins"]
)

bg_budget_ratio = float(
    bg_row["median_budget_ratio"]
)

best_algorithm = str(
    best_row["algorithm"]
)

best_mean_rank = float(
    best_row["mean_final_value_rank"]
)


# =====================================================================
# 17. Manuscript-ready official COCO Results text
# =====================================================================

results_text = f"""# Final official COCO/BBOB Results text

## Main Results paragraph

We evaluated BasinGraph on the official noiseless COCO/BBOB suite using the COCO observer and the `cocopp` target-runtime post-processing workflow. The final protocol comprised all 24 BBOB functions, five dimensions (2, 3, 5, 10 and 20), 15 instances per function–dimension pair and a budget of 1,000d objective-function evaluations. BasinGraph was compared with CMA-ES, true BIPOP-CMA-ES implemented through the `pycma` BIPOP restart interface, differential evolution, multi-start L-BFGS-B, Latin hypercube sampling and random search. The resulting benchmark contained 12,600 algorithm–problem records over 1,800 official COCO problem instances, together with 168 `.info` files and 840 `.dat` files. All datasets passed the predefined integrity checks, used formal COCO algorithm identifiers and were successfully processed by `cocopp`.

The primary official comparison is based on the `cocopp` empirical cumulative distributions and expected-running-time analyses, rather than raw objective values pooled across functions. As a supplementary final-value sanity check, BasinGraph obtained a task-normalized mean rank of {bg_mean_rank:.3f}, a median rank of {bg_median_rank:.3f} and {bg_wins} final-value wins across the 1,800 official problem instances. Its median function-evaluation budget ratio was {bg_budget_ratio:.3f}. The best mean final-value rank in this secondary analysis was obtained by {best_algorithm} ({best_mean_rank:.3f}). Consequently, the official COCO evidence supports competitiveness and broad external validation of BasinGraph, but does not by itself support a claim of universal dominance over restart-based CMA-ES methods.

## Shorter Results paragraph

On the official COCO/BBOB noiseless suite, we evaluated seven optimizers over 24 functions, five dimensions, 15 instances and a budget of 1,000d evaluations, producing 12,600 algorithm–problem records. The COCO observer generated 168 `.info` and 840 `.dat` files, and all datasets were successfully processed with `cocopp`. The target-runtime ECDF and ERT analyses constitute the primary benchmark evidence. In a supplementary final-value check, BasinGraph achieved a mean rank of {bg_mean_rank:.3f}, a median rank of {bg_median_rank:.3f} and {bg_wins} wins over 1,800 official problem instances.

## Interpretation sentence for Discussion

The official COCO/BBOB analysis indicates that BasinGraph is competitive across heterogeneous black-box landscapes, while restart-based CMA-ES remains particularly strong on selected target precisions and function groups. This result motivates presenting BasinGraph as a complementary basin-structured framework rather than claiming unconditional superiority over all established optimizers.

## Evidence-integrity sentence

The released archive contains the formal COCO observer logs, `cocopp` output, frozen software environment, dependency lock, source-code snapshot, baseline-provenance table and SHA-256 manifest for the final run `{RUN_ID}`.
"""

(
    OUT_DIR
    / "manuscript_text"
    / "FINAL_OFFICIAL_COCO_RESULTS_TEXT.md"
).write_text(results_text)


# =====================================================================
# 18. Suggested figure plan
# =====================================================================

figure_plan = f"""# Suggested Figure 3 plan

## Main Figure 3

### Panel a
Use the aggregate target-runtime ECDF over all noiseless BBOB functions from the final `cocopp` output.

Purpose:
- show fraction of solved targets versus evaluations divided by dimension;
- use formal labels BasinGraph, CMA_ES, BIPOP_CMA_ES, DE, MS_LBFGSB, LHS and Random.

### Panel b
Use a dimension-resolved or function-group ECDF from `cocopp`.

Preferred:
- 20-dimensional aggregate ECDF if text remains readable;
- otherwise use selected function-group ECDF panels.

### Panel c
Create a compact dimension-wise mean-rank plot from:
`source_data/SourceData_Fig3_dimension_summary.csv`.

Label explicitly as:
`Supplementary final-value sanity check`

Do not label panel c as an official ERT result.

## Supplementary figures

- per-function ECDF panels;
- ECDFs by BBOB function group;
- scaling plots by dimension;
- comparison tables produced by `cocopp`;
- final-value rank and pairwise summaries.

## Recommended caption

Figure 3 | Official COCO/BBOB validation of BasinGraph.
a, Target-runtime empirical cumulative distribution generated by `cocopp` over the noiseless BBOB suite.
b, Dimension- or function-group-resolved target-runtime comparison.
c, Supplementary task-normalized final-value ranks across the official COCO problem instances. The final-value panel is provided as a secondary sanity check and does not replace the COCO expected-running-time analysis. The experiment comprised 24 functions, five dimensions, 15 instances and a budget of 1,000d evaluations.
"""

(
    OUT_DIR
    / "manuscript_text"
    / "FIGURE3_PLAN_AND_CAPTION.md"
).write_text(figure_plan)


# =====================================================================
# 19. Methods replacement text
# =====================================================================

methods_text = f"""# Final official COCO/BBOB Methods text

We used the official noiseless COCO/BBOB suite and COCO observer to evaluate BasinGraph and six comparison methods. The suite contained all 24 BBOB functions at dimensions 2, 3, 5, 10 and 20, with instances 1–15 and a budget of 1,000d objective-function evaluations per algorithm–problem pair. The comparison methods were CMA-ES implemented with `pycma`, BIPOP-CMA-ES implemented through `cma.fmin2` with `restarts=9` and `bipop=True`, a self-contained differential-evolution implementation, multi-start L-BFGS-B using SciPy, Latin hypercube sampling and random search. COCO algorithm identifiers were fixed to `BasinGraph`, `CMA_ES`, `BIPOP_CMA_ES`, `DE`, `MS_LBFGSB`, `LHS` and `Random`. The final run produced 12,600 records over 1,800 official COCO problem instances. We used `cocopp` for target-runtime ECDF, expected-running-time and comparison-table generation. The software environment, dependency lock, source snapshot, observer logs, post-processing outputs and SHA-256 manifest were frozen under run identifier `{RUN_ID}`.
"""

(
    OUT_DIR
    / "manuscript_text"
    / "FINAL_OFFICIAL_COCO_METHODS_TEXT.md"
).write_text(methods_text)


# =====================================================================
# 20. README
# =====================================================================

readme = f"""# Final official COCO/BBOB manuscript-input package

Run identifier:
`{RUN_ID}`

This package was generated from the corrected and frozen final COCO/BBOB core validation.

## Key manuscript files

- `manuscript_text/FINAL_OFFICIAL_COCO_RESULTS_TEXT.md`
- `manuscript_text/FINAL_OFFICIAL_COCO_METHODS_TEXT.md`
- `manuscript_text/FIGURE3_PLAN_AND_CAPTION.md`

## Key result tables

- `tables/official_coco_algorithm_summary.csv`
- `tables/official_coco_dimension_summary.csv`
- `tables/official_coco_function_group_summary.csv`
- `tables/official_coco_per_function_summary.csv`
- `tables/official_coco_pairwise_final_value_comparisons.csv`
- `tables/official_coco_budget_usage.csv`

## Figure files

Potential `cocopp` figure and table files are copied under:
- `cocopp_selected_figures/`
- `cocopp_tables/`

Their full inventory is:
- `tables/cocopp_figure_and_table_inventory.csv`

## Interpretation policy

The official COCO/BBOB result must be interpreted primarily through target-runtime ECDF and expected-running-time outputs. Final-value rank tables are supplementary sanity checks only.
"""

(OUT_DIR / "README.md").write_text(readme)


# =====================================================================
# 21. SHA-256 manifest
# =====================================================================

manifest_path = OUT_DIR / "MANIFEST_SHA256.csv"
manifest_rows = []

for file_path in sorted(
    OUT_DIR.rglob("*")
):
    if not file_path.is_file():
        continue

    if file_path == manifest_path:
        continue

    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    manifest_rows.append({
        "relative_path":
            str(file_path.relative_to(OUT_DIR)),
        "sha256":
            digest.hexdigest(),
        "size_bytes":
            file_path.stat().st_size,
    })

with open(
    manifest_path,
    "w",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "relative_path",
            "sha256",
            "size_bytes",
        ],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)


# =====================================================================
# 22. Create upload ZIP
# =====================================================================

zip_path = (
    ROOT
    / "manuscript_inputs"
    / f"{RUN_ID}_FINAL_manuscript_inputs.zip"
)

if zip_path.exists():
    zip_path.unlink()

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=6,
) as archive:
    for file_path in sorted(
        OUT_DIR.rglob("*")
    ):
        if file_path.is_file():
            archive.write(
                file_path,
                file_path.relative_to(
                    OUT_DIR.parent
                ),
            )


print("STEP_12E_OK")
print("RUN_ID:", RUN_ID)
print("Output directory:", OUT_DIR)
print("Upload ZIP:", zip_path)
print("Result rows:", len(df))
print("Unique problem IDs:", df["problem_id"].nunique())
print("BasinGraph mean final-value rank:", bg_mean_rank)
print("BasinGraph median final-value rank:", bg_median_rank)
print("BasinGraph final-value wins:", bg_wins)
print("Best final-value mean rank:", best_algorithm, best_mean_rank)
print("cocopp output:", COCOPP_DIR)
print("Figure/table inventory rows:", len(figure_inventory_df))
