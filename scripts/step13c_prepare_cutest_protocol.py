from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycutest


# =====================================================================
# Configuration
# =====================================================================

ROOT = Path.cwd()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph.optimizer import minimize_basingraph
from baselines.reference_optimizers import (
    optimize_cmaes,
    optimize_multistart_lbfgsb,
)

SELECTION_SEED = "BasinGraph-CUTEst-v1-20260619"

TARGET_COUNTS = {
    "small_2_20": 20,
    "medium_21_100": 20,
    "large_101_500": 10,
}

PROTOCOL_DIR = ROOT / "protocols"
RESULT_DIR = ROOT / "processed_results"

PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Helper functions
# =====================================================================

def stable_hash(problem_name: str) -> str:
    """Performance-independent deterministic ordering key."""
    text = f"{SELECTION_SEED}|{problem_name}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dimension_group(n: int) -> str | None:
    if 2 <= n <= 20:
        return "small_2_20"
    if 21 <= n <= 100:
        return "medium_21_100"
    if 101 <= n <= 500:
        return "large_101_500"
    return None


def is_fixed_integer(value) -> bool:
    return isinstance(value, (int, np.integer))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def technical_validate(problem_name: str, properties: dict) -> tuple[dict | None, str | None]:
    """
    Compile and validate a candidate without running any optimizer.

    Selection/exclusion uses only:
      - compilation success;
      - problem metadata;
      - finite box bounds;
      - continuous variable types;
      - finite objective at CUTEst x0.
    """
    problem = None

    try:
        problem = pycutest.import_problem(
            problem_name,
            quiet=True,
            drop_fixed_variables=True,
        )

        n = int(problem.n)
        m = int(problem.m)

        if m != 0:
            return None, f"general_constraints_present:m={m}"

        x0 = np.asarray(problem.x0, dtype=float).reshape(-1)
        lb = np.asarray(problem.bl, dtype=float).reshape(-1)
        ub = np.asarray(problem.bu, dtype=float).reshape(-1)

        if len(x0) != n or len(lb) != n or len(ub) != n:
            return None, "dimension_or_bound_shape_mismatch"

        # CUTEst often represents infinity by approximately 1e20.
        if (
            np.any(~np.isfinite(lb))
            or np.any(~np.isfinite(ub))
            or np.any(np.abs(lb) >= 1e19)
            or np.any(np.abs(ub) >= 1e19)
        ):
            return None, "nonfinite_or_effectively_infinite_bounds"

        if np.any(ub <= lb):
            return None, "zero_or_negative_bound_width"

        vartype = np.asarray(problem.vartype).reshape(-1)

        if vartype.size != n:
            return None, "variable_type_shape_mismatch"

        if np.any(vartype != 0):
            return None, "noncontinuous_variables_present"

        tolerance = 1e-10 * np.maximum(1.0, np.abs(ub - lb))

        if np.any(x0 < lb - tolerance) or np.any(x0 > ub + tolerance):
            return None, "cutest_initial_point_outside_bounds"

        f0 = float(problem.obj(x0))

        if not np.isfinite(f0):
            return None, "nonfinite_objective_at_x0"

        # Characterization only; it is NOT used for selection.
        hessian_min_eigenvalue = np.nan
        locally_indefinite_at_x0 = False

        if n <= 200:
            try:
                H = np.asarray(problem.hess(x0), dtype=float)
                H = 0.5 * (H + H.T)
                hessian_min_eigenvalue = float(
                    np.linalg.eigvalsh(H).min()
                )
                locally_indefinite_at_x0 = bool(
                    hessian_min_eigenvalue < -1e-8
                )
            except Exception:
                pass

        record = {
            "problem_name": problem_name,
            "dimension": n,
            "constraints": properties.get("constraints"),
            "objective_type": properties.get("objective"),
            "origin": properties.get("origin"),
            "derivative_degree": properties.get("degree"),
            "regular": properties.get("regular"),
            "internal_variables": properties.get("internal"),
            "dimension_group": dimension_group(n),
            "selection_hash": stable_hash(problem_name),
            "f_at_cutest_x0": f0,
            "minimum_bound": float(lb.min()),
            "maximum_bound": float(ub.max()),
            "minimum_box_width": float((ub - lb).min()),
            "maximum_box_width": float((ub - lb).max()),
            "hessian_min_eigenvalue_at_x0": hessian_min_eigenvalue,
            "locally_indefinite_at_x0": locally_indefinite_at_x0,
        }

        return record, None

    except Exception as exc:
        return None, f"{type(exc).__name__}:{exc}"

    finally:
        if problem is not None and hasattr(problem, "terminate"):
            try:
                problem.terminate()
            except Exception:
                pass


# =====================================================================
# 1. Build the performance-blind candidate pool
# =====================================================================

all_names = pycutest.find_problems(
    constraints="bound",
    regular=True,
    internal=False,
    n=[2, 500],
    userN=False,
    m=[0, 0],
)

candidate_rows = []

for name in sorted(set(all_names)):
    try:
        properties = pycutest.problem_properties(name)
    except Exception:
        continue

    n_value = properties.get("n")

    if not is_fixed_integer(n_value):
        continue

    n = int(n_value)
    group = dimension_group(n)

    if group is None:
        continue

    objective_type = str(
        properties.get("objective", "")
    ).strip().lower()

    # Exclude feasibility-only, constant and purely linear objectives.
    if objective_type in {
        "none",
        "constant",
        "linear",
    }:
        continue

    degree = properties.get("degree")

    try:
        degree_value = int(degree)
    except Exception:
        degree_value = -1

    if degree_value < 1:
        continue

    candidate_rows.append({
        "problem_name": name,
        "dimension": n,
        "dimension_group": group,
        "objective_type": objective_type,
        "constraints": properties.get("constraints"),
        "origin": properties.get("origin"),
        "derivative_degree": degree_value,
        "regular": properties.get("regular"),
        "internal_variables": properties.get("internal"),
        "selection_hash": stable_hash(name),
    })

candidate_pool = pd.DataFrame(candidate_rows).sort_values(
    ["dimension_group", "selection_hash", "problem_name"]
)

candidate_pool_path = (
    PROTOCOL_DIR
    / "cutest_candidate_pool_v1.csv"
)

candidate_pool.to_csv(
    candidate_pool_path,
    index=False,
)


# =====================================================================
# 2. Compile-screen candidates before any optimizer is run
# =====================================================================

selected_records = []
excluded_records = []

for group, target_count in TARGET_COUNTS.items():
    group_candidates = candidate_pool[
        candidate_pool["dimension_group"] == group
    ].sort_values(
        ["selection_hash", "problem_name"]
    )

    selected_in_group = 0
    attempted_in_group = 0

    for _, candidate in group_candidates.iterrows():
        if selected_in_group >= target_count:
            break

        name = str(candidate["problem_name"])
        attempted_in_group += 1

        properties = pycutest.problem_properties(name)

        valid_record, exclusion_reason = technical_validate(
            name,
            properties,
        )

        if valid_record is None:
            excluded_records.append({
                "problem_name": name,
                "dimension_group": group,
                "selection_hash": candidate["selection_hash"],
                "exclusion_reason": exclusion_reason,
            })
        else:
            valid_record["selection_order_within_group"] = (
                selected_in_group + 1
            )
            selected_records.append(valid_record)
            selected_in_group += 1

        if attempted_in_group % 10 == 0:
            print(
                f"[{group}] attempted={attempted_in_group}, "
                f"accepted={selected_in_group}/{target_count}"
            )

    if selected_in_group < target_count:
        raise RuntimeError(
            f"Insufficient technically valid problems in {group}: "
            f"needed {target_count}, found {selected_in_group}."
        )


selected = pd.DataFrame(selected_records)

group_order = {
    "small_2_20": 1,
    "medium_21_100": 2,
    "large_101_500": 3,
}

selected["group_order"] = selected[
    "dimension_group"
].map(group_order)

selected = selected.sort_values([
    "group_order",
    "selection_order_within_group",
]).drop(columns=["group_order"])

selected.insert(
    0,
    "global_protocol_order",
    np.arange(1, len(selected) + 1),
)

excluded = pd.DataFrame(excluded_records)

selected_path = (
    PROTOCOL_DIR
    / "cutest_pre_registered_problem_list_v1.csv"
)

excluded_path = (
    PROTOCOL_DIR
    / "cutest_technical_exclusions_v1.csv"
)

selected.to_csv(
    selected_path,
    index=False,
)

excluded.to_csv(
    excluded_path,
    index=False,
)


# =====================================================================
# 3. Freeze protocol before performance smoke test
# =====================================================================

protocol_text = f"""# BasinGraph CUTEst pre-registration protocol v1

## Selection date

Generated automatically before optimizer-performance evaluation.

## Selection seed

`{SELECTION_SEED}`

## Scope

The benchmark is restricted to smooth, regular, fixed-dimensional,
bound-constrained CUTEst problems because the current BasinGraph
formulation addresses box-constrained black-box optimization.

## Performance-independent filters

- CUTEst classification: bound constraints only;
- no general equality or inequality constraints;
- regular problem classification;
- no internal variables;
- fixed dimension between 2 and 500;
- at least first-order analytical derivatives available in CUTEst;
- objective is not none, constant or purely linear;
- all free variables are continuous;
- all variable bounds are finite and have positive width;
- CUTEst initial point is feasible;
- objective at the initial point is finite.

## Dimension strata

- small: 2–20 variables, target 20 problems;
- medium: 21–100 variables, target 20 problems;
- large: 101–500 variables, target 10 problems.

Total target: 50 problems.

## Selection order

Within each dimension stratum, candidate names are ordered by SHA-256
of:

`{SELECTION_SEED}|<CUTEst problem name>`

Candidates are accepted in that fixed order until the target count is
reached. Exclusions use technical validity only and do not use any
optimizer result.

## Important interpretation

The Hessian eigenvalue recorded at the CUTEst initial point is used
only to characterize local nonconvexity. It is not used for selection
or exclusion.

## Frozen files

- `cutest_candidate_pool_v1.csv`
- `cutest_pre_registered_problem_list_v1.csv`
- `cutest_technical_exclusions_v1.csv`
"""

protocol_path = (
    PROTOCOL_DIR
    / "CUTEST_PRE_REGISTRATION_PROTOCOL_v1.md"
)

protocol_path.write_text(protocol_text)

manifest_rows = []

for path in [
    candidate_pool_path,
    selected_path,
    excluded_path,
    protocol_path,
]:
    manifest_rows.append({
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    })

manifest_path = (
    PROTOCOL_DIR
    / "CUTEST_PRE_REGISTRATION_MANIFEST_v1.csv"
)

pd.DataFrame(manifest_rows).to_csv(
    manifest_path,
    index=False,
)


# =====================================================================
# 4. Three-problem smoke test after the list has been frozen
# =====================================================================

smoke_problems = []

for group in TARGET_COUNTS:
    row = selected[
        selected["dimension_group"] == group
    ].iloc[0]
    smoke_problems.append(row)

smoke_rows = []

algorithms = [
    ("BasinGraph", minimize_basingraph),
    ("CMA-ES", optimize_cmaes),
    ("Multi-start L-BFGS-B", optimize_multistart_lbfgsb),
]

for problem_index, selected_row in enumerate(smoke_problems):
    problem_name = str(selected_row["problem_name"])

    problem = pycutest.import_problem(
        problem_name,
        quiet=True,
        drop_fixed_variables=True,
    )

    try:
        lb = np.asarray(problem.bl, dtype=float)
        ub = np.asarray(problem.bu, dtype=float)
        x0 = np.asarray(problem.x0, dtype=float)

        n = int(problem.n)
        budget = int(min(5000, max(200, 20 * n)))
        f0 = float(problem.obj(x0))

        def objective(x):
            return float(
                problem.obj(
                    np.asarray(x, dtype=float)
                )
            )

        for algorithm_index, (algorithm_name, optimizer) in enumerate(algorithms):
            seed = (
                20260619
                + 1000 * problem_index
                + 100 * algorithm_index
            )

            started = time.perf_counter()

            try:
                result = optimizer(
                    objective=objective,
                    lb=lb,
                    ub=ub,
                    max_evals=budget,
                    seed=seed,
                )

                status = "completed"
                error = ""

            except Exception as exc:
                result = {
                    "fbest": np.nan,
                    "nfe": 0,
                    "message": f"exception:{type(exc).__name__}",
                }
                status = "failed"
                error = str(exc)

            elapsed = time.perf_counter() - started

            smoke_rows.append({
                "problem_name": problem_name,
                "dimension_group": selected_row["dimension_group"],
                "dimension": n,
                "algorithm": algorithm_name,
                "seed": seed,
                "budget": budget,
                "f_at_cutest_x0": f0,
                "fbest": result.get("fbest", np.nan),
                "nfe": result.get("nfe", 0),
                "budget_ratio": (
                    result.get("nfe", 0) / budget
                    if budget > 0
                    else np.nan
                ),
                "algorithm_message": result.get("message", ""),
                "runner_status": status,
                "error": error,
                "wall_time_seconds": elapsed,
            })

    finally:
        if hasattr(problem, "terminate"):
            problem.terminate()

smoke_df = pd.DataFrame(smoke_rows)

smoke_path = (
    RESULT_DIR
    / "cutest_unified_runner_smoke_v1.csv"
)

smoke_df.to_csv(
    smoke_path,
    index=False,
)

if len(smoke_df) != 9:
    raise RuntimeError(
        f"Expected 9 smoke-test rows, found {len(smoke_df)}."
    )

if (smoke_df["runner_status"] != "completed").any():
    failures = smoke_df[
        smoke_df["runner_status"] != "completed"
    ]
    raise RuntimeError(
        "CUTEst runner smoke test failed:\n"
        + failures.to_string(index=False)
    )


# =====================================================================
# 5. Final summary
# =====================================================================

summary = {
    "selection_seed": SELECTION_SEED,
    "candidate_pool_size": int(len(candidate_pool)),
    "pre_registered_problem_count": int(len(selected)),
    "technical_exclusion_count": int(len(excluded)),
    "small_problem_count": int(
        (selected["dimension_group"] == "small_2_20").sum()
    ),
    "medium_problem_count": int(
        (selected["dimension_group"] == "medium_21_100").sum()
    ),
    "large_problem_count": int(
        (selected["dimension_group"] == "large_101_500").sum()
    ),
    "smoke_test_rows": int(len(smoke_df)),
    "smoke_test_failures": int(
        (smoke_df["runner_status"] != "completed").sum()
    ),
}

summary_path = (
    PROTOCOL_DIR
    / "CUTEST_PRE_REGISTRATION_SUMMARY_v1.json"
)

summary_path.write_text(
    json.dumps(summary, indent=2)
)

print("============================================================")
print("STEP_13C_OK")
print("Candidate pool:", len(candidate_pool))
print("Pre-registered problems:", len(selected))
print("Technical exclusions:", len(excluded))
print("Small / medium / large:",
      summary["small_problem_count"],
      summary["medium_problem_count"],
      summary["large_problem_count"])
print("Smoke-test rows:", len(smoke_df))
print("Selected list:", selected_path)
print("Technical exclusions:", excluded_path)
print("Protocol:", protocol_path)
print("Manifest:", manifest_path)
print("Smoke results:", smoke_path)
print("============================================================")
