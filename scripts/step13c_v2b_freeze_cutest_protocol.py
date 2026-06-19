"""
Step 13C-v2B: freeze the final pre-registered CUTEst benchmark.

This script uses only the technical inventory produced by Step 13C-v2A.
No optimizer performance is used for inclusion, exclusion, ordering, or
stratum allocation.

Final target:
    20 small problems   (2-20 variables)
    20 medium problems  (21-100 variables)
    10 large problems   (101-500 variables)
    50 total problem instances

Selection policy:
1. Combine the technically valid fixed and scalable inventories.
2. Deduplicate exact instance identifiers.
3. Select within each dimension stratum by deterministic SHA-256 order.
4. Use a scarcity-first order (large, medium, small) and prefer previously
   unused CUTEst base problem names to maximize family diversity.
5. Only if a stratum cannot be filled with unique base names is a second
   pass allowed to select another instance from a previously used base.
6. Freeze the list and SHA-256 manifest before running any smoke test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycutest


# =====================================================================
# 1. Project configuration
# =====================================================================

ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph.optimizer import minimize_basingraph
from baselines.reference_optimizers import (
    optimize_cmaes,
    optimize_multistart_lbfgsb,
)

PROTOCOL_DIR = ROOT / "protocols"
RESULT_DIR = ROOT / "processed_results"

PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FIXED_INVENTORY = PROTOCOL_DIR / "cutest_fixed_valid_inventory_v2.csv"
SCALABLE_INVENTORY = PROTOCOL_DIR / "cutest_scalable_valid_inventory_v2.csv"
INVENTORY_SUMMARY = PROTOCOL_DIR / "CUTEST_TECHNICAL_INVENTORY_SUMMARY_v2.json"
INVENTORY_AUDIT = PROTOCOL_DIR / "CUTEST_TECHNICAL_INVENTORY_AUDIT_v2.md"

for path in [FIXED_INVENTORY, SCALABLE_INVENTORY, INVENTORY_SUMMARY]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Step 13C-v2A inventory artifact: {path}"
        )

SELECTION_SEED = "BasinGraph-CUTEst-v2-final-20260619"

TARGET_COUNTS = {
    "small_2_20": 20,
    "medium_21_100": 20,
    "large_101_500": 10,
}

# Scarce strata are selected first to maximize base-family diversity.
SELECTION_GROUP_ORDER = [
    "large_101_500",
    "medium_21_100",
    "small_2_20",
]

REPORT_GROUP_ORDER = {
    "small_2_20": 1,
    "medium_21_100": 2,
    "large_101_500": 3,
}


# =====================================================================
# 2. Utility functions
# =====================================================================

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def final_selection_hash(instance_id: str) -> str:
    text = f"{SELECTION_SEED}|{instance_id}"
    return sha256_bytes(text.encode("utf-8"))


def safe_destination(problem_name: str, n: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", problem_name)
    return f"BGFINAL_{clean}_N{n}"


def import_selected_problem(row: pd.Series):
    """Import a fixed or scalable CUTEst problem from a protocol row."""
    problem_name = str(row["problem_name"])
    source_type = str(row["source_type"])

    kwargs = {
        "problemName": problem_name,
        "quiet": True,
        "drop_fixed_variables": True,
    }

    if source_type == "scalable":
        sif_n = int(row["sif_N"])
        kwargs["destination"] = safe_destination(problem_name, sif_n)
        kwargs["sifParams"] = {"N": sif_n}

    return pycutest.import_problem(**kwargs)


# =====================================================================
# 3. Load and validate the technical inventory
# =====================================================================

fixed = pd.read_csv(FIXED_INVENTORY)
scalable = pd.read_csv(SCALABLE_INVENTORY)

if len(fixed) != 21:
    raise RuntimeError(
        f"Expected 21 fixed valid instances from v2A, found {len(fixed)}."
    )

if len(scalable) != 63:
    raise RuntimeError(
        f"Expected 63 scalable valid instances from v2A, found {len(scalable)}."
    )

combined = pd.concat([fixed, scalable], ignore_index=True, sort=False)

required_columns = {
    "problem_name",
    "instance_id",
    "source_type",
    "sif_N",
    "dimension",
    "dimension_group",
}

missing = required_columns.difference(combined.columns)
if missing:
    raise ValueError(
        f"Technical inventory is missing required columns: {sorted(missing)}"
    )

# Remove exact duplicate instances while retaining an audit trail.
combined = combined.drop_duplicates(subset=["instance_id"], keep="first").copy()
combined["final_selection_hash"] = combined["instance_id"].map(
    final_selection_hash
)

available_counts = combined["dimension_group"].value_counts().to_dict()

for group, target in TARGET_COUNTS.items():
    available = int(available_counts.get(group, 0))
    if available < target:
        raise RuntimeError(
            f"Insufficient valid instances in {group}: "
            f"target={target}, available={available}."
        )


# =====================================================================
# 4. Deterministic diversity-first selection
# =====================================================================

selected_rows: list[dict] = []
selected_instance_ids: set[str] = set()
selected_base_names: set[str] = set()
selection_audit_rows: list[dict] = []

for group in SELECTION_GROUP_ORDER:
    target = TARGET_COUNTS[group]

    candidates = combined[
        combined["dimension_group"] == group
    ].copy()

    candidates = candidates.sort_values(
        ["final_selection_hash", "instance_id"]
    )

    group_selected: list[dict] = []

    # ------------------------------------------------------------
    # Pass 1: prioritize previously unused CUTEst base names.
    # ------------------------------------------------------------
    for _, row in candidates.iterrows():
        instance_id = str(row["instance_id"])
        base_name = str(row["problem_name"])

        if len(group_selected) >= target:
            break
        if instance_id in selected_instance_ids:
            continue
        if base_name in selected_base_names:
            continue

        record = row.to_dict()
        record["selection_phase"] = "unique_base_first_pass"
        record["selection_group_target"] = target
        group_selected.append(record)
        selected_instance_ids.add(instance_id)
        selected_base_names.add(base_name)

    # ------------------------------------------------------------
    # Pass 2: only if required, allow another instance of a base.
    # ------------------------------------------------------------
    if len(group_selected) < target:
        for _, row in candidates.iterrows():
            instance_id = str(row["instance_id"])

            if len(group_selected) >= target:
                break
            if instance_id in selected_instance_ids:
                continue

            record = row.to_dict()
            record["selection_phase"] = "duplicate_base_fallback"
            record["selection_group_target"] = target
            group_selected.append(record)
            selected_instance_ids.add(instance_id)
            selected_base_names.add(str(row["problem_name"]))

    if len(group_selected) != target:
        raise RuntimeError(
            f"Could not fill {group}: selected={len(group_selected)}, "
            f"target={target}."
        )

    for group_order, record in enumerate(group_selected, start=1):
        record["selection_order_within_group"] = group_order
        selected_rows.append(record)


selected = pd.DataFrame(selected_rows)

# Report order is small, medium, large for readability.
selected["report_group_order"] = selected["dimension_group"].map(
    REPORT_GROUP_ORDER
)
selected = selected.sort_values(
    ["report_group_order", "selection_order_within_group"]
).drop(columns=["report_group_order"])
selected.insert(0, "global_protocol_order", np.arange(1, len(selected) + 1))

if len(selected) != 50:
    raise RuntimeError(f"Expected 50 selected instances, found {len(selected)}.")

# Inventory instances that were technically valid but not selected.
not_selected = combined[
    ~combined["instance_id"].isin(selected["instance_id"])
].copy()
not_selected["nonselection_reason"] = "valid_but_not_selected_after_stratum_target_reached"

# Selection audit across all valid instances.
selected_lookup = set(selected["instance_id"])
for _, row in combined.sort_values(
    ["dimension_group", "final_selection_hash", "instance_id"]
).iterrows():
    selection_audit_rows.append({
        "instance_id": row["instance_id"],
        "problem_name": row["problem_name"],
        "source_type": row["source_type"],
        "sif_N": row["sif_N"],
        "dimension": row["dimension"],
        "dimension_group": row["dimension_group"],
        "final_selection_hash": row["final_selection_hash"],
        "selected": row["instance_id"] in selected_lookup,
    })

selection_audit = pd.DataFrame(selection_audit_rows)


# =====================================================================
# 5. Freeze the pre-registered protocol before optimizer smoke tests
# =====================================================================

SELECTED_PATH = PROTOCOL_DIR / "cutest_pre_registered_problem_list_v2.csv"
NOT_SELECTED_PATH = PROTOCOL_DIR / "cutest_valid_not_selected_v2.csv"
SELECTION_AUDIT_PATH = PROTOCOL_DIR / "cutest_selection_audit_v2.csv"
PROTOCOL_PATH = PROTOCOL_DIR / "CUTEST_PRE_REGISTRATION_PROTOCOL_v2.md"
SUMMARY_PATH = PROTOCOL_DIR / "CUTEST_PRE_REGISTRATION_SUMMARY_v2.json"
MANIFEST_PATH = PROTOCOL_DIR / "CUTEST_PRE_REGISTRATION_MANIFEST_v2.csv"

selected.to_csv(SELECTED_PATH, index=False)
not_selected.to_csv(NOT_SELECTED_PATH, index=False)
selection_audit.to_csv(SELECTION_AUDIT_PATH, index=False)

selected_group_counts = selected["dimension_group"].value_counts().to_dict()
selected_source_counts = selected["source_type"].value_counts().to_dict()
unique_base_count = int(selected["problem_name"].nunique())
duplicate_base_count = int(len(selected) - unique_base_count)

summary = {
    "selection_seed": SELECTION_SEED,
    "optimizer_runs_before_freeze": 0,
    "technical_inventory_instances": int(len(combined)),
    "selected_problem_instances": int(len(selected)),
    "selected_by_dimension_group": {
        key: int(value) for key, value in selected_group_counts.items()
    },
    "selected_by_source_type": {
        key: int(value) for key, value in selected_source_counts.items()
    },
    "unique_base_problem_names": unique_base_count,
    "duplicate_base_instances": duplicate_base_count,
    "valid_but_not_selected": int(len(not_selected)),
    "selection_group_order": SELECTION_GROUP_ORDER,
    "target_counts": TARGET_COUNTS,
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

protocol_text = f"""# BasinGraph CUTEst pre-registration protocol v2

## Timing and audit status

This protocol was frozen after technical inventory and before any
optimizer-performance experiment on the final 50-problem set.

The failed v1 and v1.1 quota attempts are retained as part of the audit trail.
The technical inventory v2 identified 84 valid instances: 41 small,
27 medium and 16 large. No optimizer result was used in the inventory.

## Final benchmark size

- Small, 2-20 variables: 20 instances
- Medium, 21-100 variables: 20 instances
- Large, 101-500 variables: 10 instances
- Total: 50 instances

## Eligibility criteria

- regular CUTEst classification;
- bound constraints only;
- no general equality or inequality constraints;
- continuous variables only;
- fixed or formally parameterized dimension between 2 and 500;
- finite lower and upper bounds;
- strictly positive box widths;
- feasible CUTEst initial point;
- finite objective value at the initial point;
- objective not none, constant or purely linear.

## Deterministic selection

Selection seed:
`{SELECTION_SEED}`

Each technically valid instance is assigned SHA-256 of:
`{SELECTION_SEED}|<instance_id>`

Selection is performed in scarcity-first order:
large, medium, then small.

Within each stratum, instances are ordered by SHA-256 and CUTEst base
problem names not previously used are preferred. A second instance from
a previously used base problem is allowed only if needed to fill a stratum.
No optimizer performance is used.

## Source balance and family diversity

- Selected fixed instances: {selected_source_counts.get('fixed', 0)}
- Selected scalable instances: {selected_source_counts.get('scalable', 0)}
- Unique CUTEst base names: {unique_base_count}
- Additional instances from repeated base names: {duplicate_base_count}

## Frozen artifacts

- `{SELECTED_PATH.name}`
- `{NOT_SELECTED_PATH.name}`
- `{SELECTION_AUDIT_PATH.name}`
- `{SUMMARY_PATH.name}`
- `{MANIFEST_PATH.name}`

The technical inventory artifacts from Step 13C-v2A remain part of the
complete audit trail.
"""

PROTOCOL_PATH.write_text(protocol_text)

# Freeze all protocol-defining files before any smoke-test optimizer run.
manifest_inputs = [
    FIXED_INVENTORY,
    SCALABLE_INVENTORY,
    INVENTORY_SUMMARY,
    SELECTED_PATH,
    NOT_SELECTED_PATH,
    SELECTION_AUDIT_PATH,
    PROTOCOL_PATH,
    SUMMARY_PATH,
]

if INVENTORY_AUDIT.exists():
    manifest_inputs.append(INVENTORY_AUDIT)

manifest_rows = []
for path in manifest_inputs:
    manifest_rows.append({
        "relative_path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    })

with MANIFEST_PATH.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["relative_path", "sha256", "size_bytes"],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)


# =====================================================================
# 6. Unified three-problem smoke test after protocol freeze
# =====================================================================

SMOKE_PATH = RESULT_DIR / "cutest_unified_runner_smoke_v2.csv"

# Deterministically choose the first protocol entry in each stratum.
smoke_problem_rows = []
for group in ["small_2_20", "medium_21_100", "large_101_500"]:
    row = selected[selected["dimension_group"] == group].iloc[0]
    smoke_problem_rows.append(row)

algorithms = [
    ("BasinGraph", minimize_basingraph),
    ("CMA-ES", optimize_cmaes),
    ("Multi-start L-BFGS-B", optimize_multistart_lbfgsb),
]

smoke_rows: list[dict] = []

for problem_index, protocol_row in enumerate(smoke_problem_rows):
    problem = import_selected_problem(protocol_row)

    try:
        n = int(problem.n)
        lb = np.asarray(problem.bl, dtype=float).reshape(-1)
        ub = np.asarray(problem.bu, dtype=float).reshape(-1)
        x0 = np.asarray(problem.x0, dtype=float).reshape(-1)
        f0 = float(problem.obj(x0))

        # Smoke test only: sufficient to verify unified objective, bounds,
        # evaluation accounting and algorithm wrappers.
        budget = int(min(3000, max(300, 10 * n)))

        def objective(x):
            return float(problem.obj(np.asarray(x, dtype=float)))

        for algorithm_index, (algorithm_name, optimizer) in enumerate(algorithms):
            seed = 20260619 + 1000 * problem_index + 100 * algorithm_index
            started = time.perf_counter()

            try:
                result = optimizer(
                    objective=objective,
                    lb=lb,
                    ub=ub,
                    max_evals=budget,
                    seed=seed,
                )
                runner_status = "completed"
                error = ""
            except Exception as exc:
                result = {
                    "fbest": np.nan,
                    "nfe": 0,
                    "message": f"exception:{type(exc).__name__}",
                }
                runner_status = "failed"
                error = str(exc)

            elapsed = time.perf_counter() - started

            smoke_rows.append({
                "global_protocol_order": int(protocol_row["global_protocol_order"]),
                "problem_name": protocol_row["problem_name"],
                "instance_id": protocol_row["instance_id"],
                "source_type": protocol_row["source_type"],
                "sif_N": protocol_row["sif_N"],
                "dimension_group": protocol_row["dimension_group"],
                "dimension": n,
                "algorithm": algorithm_name,
                "seed": seed,
                "budget": budget,
                "f_at_cutest_x0": f0,
                "fbest": result.get("fbest", np.nan),
                "nfe": result.get("nfe", 0),
                "budget_ratio": (
                    result.get("nfe", 0) / budget if budget > 0 else np.nan
                ),
                "algorithm_message": result.get("message", ""),
                "runner_status": runner_status,
                "error": error,
                "wall_time_seconds": elapsed,
            })

    finally:
        if hasattr(problem, "terminate"):
            try:
                problem.terminate()
            except Exception:
                pass

smoke_df = pd.DataFrame(smoke_rows)
smoke_df.to_csv(SMOKE_PATH, index=False)

if len(smoke_df) != 9:
    raise RuntimeError(f"Expected 9 smoke-test rows, found {len(smoke_df)}.")

if (smoke_df["runner_status"] != "completed").any():
    failures = smoke_df[smoke_df["runner_status"] != "completed"]
    raise RuntimeError(
        "Unified CUTEst runner smoke test failed:\n"
        + failures.to_string(index=False)
    )


# =====================================================================
# 7. Final concise summary
# =====================================================================

print("============================================================")
print("STEP_13C_V2B_OK")
print("Optimizer runs before protocol freeze: 0")
print("Technical inventory instances:", len(combined))
print("Pre-registered instances:", len(selected))
print("Small / medium / large:",
      int((selected["dimension_group"] == "small_2_20").sum()),
      int((selected["dimension_group"] == "medium_21_100").sum()),
      int((selected["dimension_group"] == "large_101_500").sum()))
print("Fixed / scalable:",
      int((selected["source_type"] == "fixed").sum()),
      int((selected["source_type"] == "scalable").sum()))
print("Unique base problem names:", unique_base_count)
print("Repeated-base additional instances:", duplicate_base_count)
print("Smoke-test rows:", len(smoke_df))
print("Selected list:", SELECTED_PATH)
print("Protocol:", PROTOCOL_PATH)
print("Manifest:", MANIFEST_PATH)
print("Smoke results:", SMOKE_PATH)
print("============================================================")
