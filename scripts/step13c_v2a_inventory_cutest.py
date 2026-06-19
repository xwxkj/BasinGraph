"""
Step 13C-v2A: performance-blind CUTEst technical inventory.

This script performs NO optimizer runs. It inventories:
1. fixed-dimensional, bound-constrained CUTEst problems;
2. formally parameterized user-scalable CUTEst instances;
3. technical exclusions and their reasons.

It is resumable: existing inventory CSV files are loaded and completed
rather than overwritten from scratch.

Requirements
------------
- Activate conda environment: basingraph-cutest
- Source protocols/cutest_env.sh
- Run from ~/Documents/BasinGraph202606

PyCUTEst API used
-----------------
- pycutest.find_problems(...)
- pycutest.problem_properties(...)
- pycutest.print_available_sif_params(...)
- pycutest.import_problem(..., sifParams={"N": value}, destination=...)
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycutest


# =====================================================================
# Configuration
# =====================================================================

ROOT = Path.cwd()
PROTOCOL_DIR = ROOT / "protocols"
PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)

SELECTION_SEED = "BasinGraph-CUTEst-v2-20260619"

# At most one instance per scalable base problem and dimension stratum.
TARGET_CENTRES = {
    "small_2_20": 10,
    "medium_21_100": 50,
    "large_101_500": 200,
}

FIXED_OUT = PROTOCOL_DIR / "cutest_fixed_valid_inventory_v2.csv"
SCALABLE_OUT = PROTOCOL_DIR / "cutest_scalable_valid_inventory_v2.csv"
EXCLUSION_OUT = PROTOCOL_DIR / "cutest_inventory_exclusions_v2.csv"
PARAM_OUT = PROTOCOL_DIR / "cutest_scalable_parameter_inventory_v2.csv"
SUMMARY_OUT = PROTOCOL_DIR / "CUTEST_TECHNICAL_INVENTORY_SUMMARY_v2.json"
AUDIT_OUT = PROTOCOL_DIR / "CUTEST_TECHNICAL_INVENTORY_AUDIT_v2.md"


# =====================================================================
# Helpers
# =====================================================================

def stable_hash(text: str) -> str:
    payload = f"{SELECTION_SEED}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dimension_group(n: int) -> str | None:
    if 2 <= n <= 20:
        return "small_2_20"
    if 21 <= n <= 100:
        return "medium_21_100"
    if 101 <= n <= 500:
        return "large_101_500"
    return None


def safe_destination(name: str, n: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", name)
    return f"BGV2_{clean}_N{n}"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_record(path: Path, record: dict[str, Any]) -> None:
    frame = pd.DataFrame([record])
    frame.to_csv(
        path,
        mode="a",
        header=not path.exists() or path.stat().st_size == 0,
        index=False,
    )


def capture_sif_parameter_text(problem_name: str) -> tuple[str, str]:
    """
    Capture print_available_sif_params in a subprocess.

    A subprocess is used because SIFDecode may write directly to stdout/stderr,
    which is not always captured by contextlib.redirect_stdout.
    """
    code = (
        "import pycutest; "
        f"pycutest.print_available_sif_params({problem_name!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (completed.stdout or "") + "\n" + (completed.stderr or "")
    status = "ok" if completed.returncode == 0 else f"returncode_{completed.returncode}"
    return text, status


def parse_n_values(parameter_text: str) -> list[int]:
    """
    Parse explicit parameter lines such as:
        N = 10 (int)
        N = 100 (int, .ge. ...)
    """
    values: set[int] = set()
    pattern = re.compile(
        r"^\s*N\s*=\s*([0-9]+)\s*\(",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(parameter_text):
        values.add(int(match.group(1)))
    return sorted(values)


def select_one_n_per_stratum(n_values: list[int]) -> list[int]:
    selected: list[int] = []
    for group, centre in TARGET_CENTRES.items():
        allowed = [n for n in n_values if dimension_group(n) == group]
        if allowed:
            selected.append(min(allowed, key=lambda n: (abs(n - centre), n)))
    return sorted(set(selected))


def objective_type_allowed(properties: dict[str, Any]) -> bool:
    objective = str(properties.get("objective", "")).strip().lower()
    return objective not in {"none", "constant", "linear", ""}


def validate_loaded_problem(
    problem: Any,
    problem_name: str,
    source_type: str,
    sif_n: int | None,
    properties: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    n = int(problem.n)
    m = int(problem.m)

    if m != 0:
        return None, f"general_constraints_present:m={m}"

    group = dimension_group(n)
    if group is None:
        return None, f"dimension_outside_2_500:n={n}"

    x0 = np.asarray(problem.x0, dtype=float).reshape(-1)
    lb = np.asarray(problem.bl, dtype=float).reshape(-1)
    ub = np.asarray(problem.bu, dtype=float).reshape(-1)

    if len(x0) != n or len(lb) != n or len(ub) != n:
        return None, "dimension_or_bound_shape_mismatch"

    # CUTEst commonly encodes infinity using values near 1e20.
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

    instance_id = problem_name if sif_n is None else f"{problem_name}[N={sif_n}]"

    record = {
        "problem_name": problem_name,
        "instance_id": instance_id,
        "source_type": source_type,
        "sif_N": sif_n,
        "dimension": n,
        "dimension_group": group,
        "objective_type": properties.get("objective"),
        "constraints": properties.get("constraints"),
        "origin": properties.get("origin"),
        "derivative_degree": properties.get("degree"),
        "regular": properties.get("regular"),
        "internal_variables": properties.get("internal"),
        "f_at_x0": f0,
        "minimum_bound": float(lb.min()),
        "maximum_bound": float(ub.max()),
        "minimum_box_width": float((ub - lb).min()),
        "maximum_box_width": float((ub - lb).max()),
        "selection_hash": stable_hash(instance_id),
    }
    return record, None


def compile_and_validate(
    problem_name: str,
    properties: dict[str, Any],
    source_type: str,
    sif_n: int | None,
) -> tuple[dict[str, Any] | None, str | None]:
    problem = None
    try:
        kwargs: dict[str, Any] = {
            "problemName": problem_name,
            "quiet": True,
            "drop_fixed_variables": True,
        }
        if sif_n is not None:
            kwargs["destination"] = safe_destination(problem_name, sif_n)
            kwargs["sifParams"] = {"N": int(sif_n)}

        problem = pycutest.import_problem(**kwargs)
        return validate_loaded_problem(
            problem=problem,
            problem_name=problem_name,
            source_type=source_type,
            sif_n=sif_n,
            properties=properties,
        )
    except Exception as exc:
        text = str(exc).replace("\n", " ")[:600]
        return None, f"{type(exc).__name__}:{text}"
    finally:
        if problem is not None and hasattr(problem, "terminate"):
            try:
                problem.terminate()
            except Exception:
                pass


def existing_instance_ids() -> set[str]:
    ids: set[str] = set()
    for path in [FIXED_OUT, SCALABLE_OUT, EXCLUSION_OUT]:
        frame = load_csv(path)
        if "instance_id" in frame.columns:
            ids.update(frame["instance_id"].dropna().astype(str))
    return ids


# =====================================================================
# 1. Fixed-dimensional inventory
# =====================================================================

processed = existing_instance_ids()

fixed_names = sorted(
    set(
        pycutest.find_problems(
            constraints="bound",
            regular=True,
            internal=False,
            n=[2, 500],
            userN=False,
            m=[0, 0],
        )
    )
)

fixed_valid_start = len(load_csv(FIXED_OUT))
fixed_attempted = 0

for name in fixed_names:
    instance_id = name
    if instance_id in processed:
        continue

    fixed_attempted += 1
    properties = pycutest.problem_properties(name)

    n_value = properties.get("n")
    if not isinstance(n_value, (int, np.integer)):
        append_record(
            EXCLUSION_OUT,
            {
                "problem_name": name,
                "instance_id": instance_id,
                "source_type": "fixed",
                "sif_N": None,
                "reason": "classification_dimension_not_fixed_integer",
            },
        )
        processed.add(instance_id)
        continue

    if not objective_type_allowed(properties):
        append_record(
            EXCLUSION_OUT,
            {
                "problem_name": name,
                "instance_id": instance_id,
                "source_type": "fixed",
                "sif_N": None,
                "reason": f"excluded_objective_type:{properties.get('objective')}",
            },
        )
        processed.add(instance_id)
        continue

    record, reason = compile_and_validate(
        problem_name=name,
        properties=properties,
        source_type="fixed",
        sif_n=None,
    )

    if record is None:
        append_record(
            EXCLUSION_OUT,
            {
                "problem_name": name,
                "instance_id": instance_id,
                "source_type": "fixed",
                "sif_N": None,
                "reason": reason,
            },
        )
    else:
        append_record(FIXED_OUT, record)

    processed.add(instance_id)

    if fixed_attempted % 10 == 0:
        current_valid = len(load_csv(FIXED_OUT))
        print(
            f"[fixed] newly_attempted={fixed_attempted}, "
            f"total_valid={current_valid}"
        )


# =====================================================================
# 2. User-scalable inventory
# =====================================================================

processed = existing_instance_ids()

scalable_names = sorted(
    set(
        pycutest.find_problems(
            constraints="bound",
            regular=True,
            internal=False,
            userN=True,
            m=[0, 0],
        )
    )
)

parameter_existing = load_csv(PARAM_OUT)
parameter_done = (
    set(parameter_existing["problem_name"].astype(str))
    if "problem_name" in parameter_existing.columns
    else set()
)

scalable_base_attempted = 0

for name in scalable_names:
    properties = pycutest.problem_properties(name)

    if not objective_type_allowed(properties):
        base_instance_id = f"{name}[scalable-base]"
        if base_instance_id not in processed:
            append_record(
                EXCLUSION_OUT,
                {
                    "problem_name": name,
                    "instance_id": base_instance_id,
                    "source_type": "scalable",
                    "sif_N": None,
                    "reason": f"excluded_objective_type:{properties.get('objective')}",
                },
            )
            processed.add(base_instance_id)
        continue

    parameter_text, parameter_status = capture_sif_parameter_text(name)
    n_values = parse_n_values(parameter_text)
    selected_n_values = select_one_n_per_stratum(n_values)

    if name not in parameter_done:
        append_record(
            PARAM_OUT,
            {
                "problem_name": name,
                "parameter_query_status": parameter_status,
                "available_N_values": ";".join(map(str, n_values)),
                "selected_inventory_N_values": ";".join(map(str, selected_n_values)),
                "parameter_text_sha256": hashlib.sha256(
                    parameter_text.encode("utf-8")
                ).hexdigest(),
            },
        )
        parameter_done.add(name)

    if not n_values:
        base_instance_id = f"{name}[scalable-base]"
        if base_instance_id not in processed:
            append_record(
                EXCLUSION_OUT,
                {
                    "problem_name": name,
                    "instance_id": base_instance_id,
                    "source_type": "scalable",
                    "sif_N": None,
                    "reason": "no_explicit_integer_N_values_detected",
                },
            )
            processed.add(base_instance_id)
        continue

    scalable_base_attempted += 1

    for n_value in selected_n_values:
        instance_id = f"{name}[N={n_value}]"
        if instance_id in processed:
            continue

        record, reason = compile_and_validate(
            problem_name=name,
            properties=properties,
            source_type="scalable",
            sif_n=n_value,
        )

        if record is None:
            append_record(
                EXCLUSION_OUT,
                {
                    "problem_name": name,
                    "instance_id": instance_id,
                    "source_type": "scalable",
                    "sif_N": n_value,
                    "reason": reason,
                },
            )
        else:
            append_record(SCALABLE_OUT, record)

        processed.add(instance_id)

    if scalable_base_attempted % 10 == 0:
        current_valid = len(load_csv(SCALABLE_OUT))
        print(
            f"[scalable] bases_with_explicit_N={scalable_base_attempted}, "
            f"total_valid_instances={current_valid}"
        )


# =====================================================================
# 3. Normalize and summarize output
# =====================================================================

fixed_df = load_csv(FIXED_OUT)
scalable_df = load_csv(SCALABLE_OUT)
exclusion_df = load_csv(EXCLUSION_OUT)
parameter_df = load_csv(PARAM_OUT)

for path, frame in [
    (FIXED_OUT, fixed_df),
    (SCALABLE_OUT, scalable_df),
    (EXCLUSION_OUT, exclusion_df),
    (PARAM_OUT, parameter_df),
]:
    if len(frame):
        sort_columns = [
            column
            for column in ["dimension_group", "selection_hash", "instance_id", "problem_name"]
            if column in frame.columns
        ]
        if sort_columns:
            frame = frame.sort_values(sort_columns)
        frame = frame.drop_duplicates(
            subset=["instance_id"] if "instance_id" in frame.columns else ["problem_name"],
            keep="first",
        )
        frame.to_csv(path, index=False)

combined = pd.concat([fixed_df, scalable_df], ignore_index=True)

if len(combined):
    group_counts = (
        combined["dimension_group"]
        .value_counts()
        .sort_index()
        .to_dict()
    )
else:
    group_counts = {}

summary = {
    "selection_seed": SELECTION_SEED,
    "optimizer_runs_performed": 0,
    "fixed_candidate_names": len(fixed_names),
    "fixed_valid_instances": int(len(fixed_df)),
    "scalable_candidate_names": len(scalable_names),
    "scalable_valid_instances": int(len(scalable_df)),
    "total_valid_instances": int(len(combined)),
    "technical_exclusions": int(len(exclusion_df)),
    "valid_instances_by_group": {
        str(key): int(value)
        for key, value in group_counts.items()
    },
}

SUMMARY_OUT.write_text(json.dumps(summary, indent=2))

AUDIT_OUT.write_text(
    "# CUTEst technical inventory v2\n\n"
    "This inventory was generated before any optimizer-performance run.\n\n"
    "It combines fixed-dimensional problems with formally parameterized "
    "user-scalable CUTEst instances. PyCUTEst `sifParams={\"N\": value}` "
    "was used only where an explicit integer N value was reported by "
    "`print_available_sif_params`.\n\n"
    "At most one small-, one medium- and one large-dimensional instance "
    "was inventoried per scalable base problem. No optimizer result was "
    "used for inclusion, exclusion or ordering.\n\n"
    "## Summary\n\n```json\n"
    + json.dumps(summary, indent=2)
    + "\n```\n"
)

print("============================================================")
print("STEP_13C_V2A_OK")
print("Optimizer runs performed: 0")
print("Fixed valid instances:", len(fixed_df))
print("Scalable valid instances:", len(scalable_df))
print("Total valid instances:", len(combined))
print("Valid instances by group:", group_counts)
print("Technical exclusions:", len(exclusion_df))
print("Summary:", SUMMARY_OUT)
print("Fixed inventory:", FIXED_OUT)
print("Scalable inventory:", SCALABLE_OUT)
print("Exclusions:", EXCLUSION_OUT)
print("============================================================")
