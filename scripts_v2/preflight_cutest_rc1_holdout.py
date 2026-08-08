#!/usr/bin/env python3
"""
Zero-optimizer preflight for the frozen prospective CUTEst holdout.

This script:
- verifies the selected BasinGraph rc1 implementation and semantic contract;
- verifies the frozen 24-instance holdout file and formal-protocol manifest;
- verifies exact non-overlap with the 50-instance development/comparability set;
- imports/compiles all 24 PyCUTEst instances serially;
- checks dimensions, unconstrained status, finite bounds and feasible x0;
- performs no objective evaluation and no optimizer run.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pycutest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION


EXPECTED_BRANCH = "route-b/finalize-rc1-consistent"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
CODE_FREEZE_TAG = "route-b-v2.0.0-rc1-codefreeze"
FINAL_CANDIDATE_TAG = "route-b-v2.0.0-rc1-selected-final-candidate"

FORMAL_DIR = ROOT / "protocols" / "route_b" / "formal_v2_protocol"
HOLDOUT_LIST = FORMAL_DIR / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
FORMAL_MANIFEST = FORMAL_DIR / "V2_FORMAL_PROTOCOL_MANIFEST_SHA256.csv"
DEVELOPMENT_LIST = ROOT / "protocols" / "cutest_pre_registered_problem_list_v2.csv"
OUTPUT = (
    ROOT
    / "protocols"
    / "route_b"
    / "final_rc1"
    / "CUTEST_HOLDOUT_PREFLIGHT.json"
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_destination(instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:16]
    return f"BGHOLD_{digest}"


def sif_params(row: pd.Series) -> dict[str, int] | None:
    if str(row["source_type"]) != "scalable":
        return None
    value = row.get("sif_N")
    if pd.isna(value) or str(value).strip() == "":
        raise RuntimeError(f"Missing sif_N for scalable instance {row['instance_id']}")
    return {"N": int(float(value))}


def import_problem(row: pd.Series):
    kwargs: dict[str, Any] = {
        "quiet": True,
        "drop_fixed_variables": True,
        "destination": safe_destination(str(row["instance_id"])),
    }
    params = sif_params(row)
    if params is not None:
        kwargs["sifParams"] = params
    return pycutest.import_problem(str(row["problem_name"]), **kwargs)


def main() -> None:
    branch = git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong branch: {branch}")

    options = BasinGraphOptions()
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(IMPLEMENTATION_VERSION)
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Selected options hash mismatch.")

    if subprocess.run(
        ["git", "diff", "--quiet", CODE_FREEZE_TAG, "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("basingraph_v2 differs from the selected code freeze.")

    contract_path = (
        ROOT
        / "protocols"
        / "route_b"
        / "final_rc1"
        / "RC1_PAPER_CODE_CONTRACT.json"
    )
    selection_path = (
        ROOT
        / "protocols"
        / "route_b"
        / "final_rc1"
        / "RC1_FINAL_SELECTION_DECISION.json"
    )
    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())

    assert contract["status"] == "RC1_PAPER_CODE_CONTRACT_FROZEN"
    assert contract["holdout_accessed"] is False
    assert selection["status"] == "RC1_SELECTED_AS_FINAL_HOLDOUT_CANDIDATE"
    assert selection["holdout_accessed"] is False

    for path in [HOLDOUT_LIST, FORMAL_MANIFEST, DEVELOPMENT_LIST]:
        if not path.exists():
            raise FileNotFoundError(path)

    manifest = pd.read_csv(FORMAL_MANIFEST)
    manifest_row = manifest[
        manifest["filename"] == HOLDOUT_LIST.name
    ]
    if len(manifest_row) != 1:
        raise RuntimeError(
            f"Holdout list missing/duplicated in formal manifest: {len(manifest_row)}"
        )
    expected_hash = str(manifest_row.iloc[0]["sha256"])
    observed_hash = sha256_file(HOLDOUT_LIST)
    if expected_hash != observed_hash:
        raise RuntimeError(
            "Frozen CUTEst holdout list hash mismatch:\n"
            f"expected={expected_hash}\nobserved={observed_hash}"
        )

    holdout = pd.read_csv(HOLDOUT_LIST)
    development = pd.read_csv(DEVELOPMENT_LIST)

    required_columns = {
        "global_holdout_order",
        "problem_name",
        "instance_id",
        "source_type",
        "sif_N",
        "dimension",
        "dimension_group",
        "objective_type",
        "f_at_x0",
        "minimum_bound",
        "maximum_bound",
    }
    missing = required_columns.difference(holdout.columns)
    if missing:
        raise RuntimeError(f"Holdout list missing columns: {sorted(missing)}")

    assert len(holdout) == 24
    assert holdout["global_holdout_order"].tolist() == list(range(1, 25))
    assert holdout["instance_id"].nunique() == 24
    assert holdout["dimension_group"].value_counts().to_dict() == {
        "small_2_20": 11,
        "medium_21_100": 7,
        "large_101_500": 6,
    }

    overlap = set(holdout["instance_id"].astype(str)).intersection(
        set(development["instance_id"].astype(str))
    )
    if overlap:
        raise RuntimeError(f"Holdout/development instance overlap: {sorted(overlap)}")

    imported_rows = []
    for row in holdout.sort_values("global_holdout_order").itertuples(index=False):
        series = pd.Series(row._asdict())
        problem = import_problem(series)
        try:
            dimension = int(problem.n)
            constraints = int(problem.m)
            lb = np.asarray(problem.bl, dtype=float).reshape(-1)
            ub = np.asarray(problem.bu, dtype=float).reshape(-1)
            x0 = np.asarray(problem.x0, dtype=float).reshape(-1)

            assert dimension == int(float(series["dimension"]))
            assert constraints == 0
            assert len(lb) == dimension
            assert len(ub) == dimension
            assert len(x0) == dimension
            assert np.isfinite(lb).all()
            assert np.isfinite(ub).all()
            assert (ub > lb).all()
            assert np.isfinite(x0).all()
            assert (x0 >= lb - 1e-12).all()
            assert (x0 <= ub + 1e-12).all()

            imported_rows.append(
                {
                    "global_holdout_order": int(series["global_holdout_order"]),
                    "instance_id": str(series["instance_id"]),
                    "problem_name": str(series["problem_name"]),
                    "source_type": str(series["source_type"]),
                    "sif_N": (
                        None
                        if pd.isna(series["sif_N"])
                        else int(float(series["sif_N"]))
                    ),
                    "dimension": dimension,
                    "dimension_group": str(series["dimension_group"]),
                    "destination": safe_destination(str(series["instance_id"])),
                    "constraints": constraints,
                    "finite_bounds": True,
                    "feasible_x0": True,
                }
            )
        finally:
            if hasattr(problem, "terminate"):
                problem.terminate()

    report = {
        "status": "CUTEST_HOLDOUT_PREFLIGHT_OK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": branch,
        "current_commit": git_output("rev-parse", "HEAD"),
        "code_freeze_tag": CODE_FREEZE_TAG,
        "code_freeze_commit": git_output("rev-list", "-n", "1", CODE_FREEZE_TAG),
        "final_candidate_tag": FINAL_CANDIDATE_TAG,
        "final_candidate_commit": git_output(
            "rev-list", "-n", "1", FINAL_CANDIDATE_TAG
        ),
        "implementation_version": IMPLEMENTATION_VERSION,
        "options_hash": options.stable_hash(),
        "holdout_list": str(HOLDOUT_LIST.relative_to(ROOT)),
        "holdout_list_sha256": observed_hash,
        "formal_manifest_sha256": sha256_file(FORMAL_MANIFEST),
        "development_list_sha256": sha256_file(DEVELOPMENT_LIST),
        "holdout_instances": len(holdout),
        "development_instances": len(development),
        "exact_instance_overlap": 0,
        "dimension_groups": holdout["dimension_group"].value_counts().sort_index().to_dict(),
        "unique_base_problems": int(holdout["problem_name"].nunique()),
        "objective_evaluations_performed": 0,
        "optimizer_runs_performed": 0,
        "imports": imported_rows,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")

    print("CUTEST_HOLDOUT_PREFLIGHT_OK")
    print("Holdout instances:", len(holdout))
    print(
        "Groups:",
        holdout["dimension_group"].value_counts().sort_index().to_dict(),
    )
    print("Exact development overlap:", 0)
    print("Objective evaluations:", 0)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
