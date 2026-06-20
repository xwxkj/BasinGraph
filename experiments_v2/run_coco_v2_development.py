#!/usr/bin/env python3
"""
Route B Step B6B: formal COCO/BBOB development run for BasinGraph v2.0.0-rc1.

This script is restricted to the frozen development partition:
- suite: official noiseless BBOB
- functions: 1-24
- dimensions: 2, 5, 10
- instances: 1-3
- budget: 1000 * dimension
- algorithms: BasinGraph_v2, CMA-ES, BIPOP-CMA-ES, DE,
  Multi-start L-BFGS-B, LHS and Random Search

Prospective holdout instances 4-15 are never instantiated by this script.

Every BasinGraph_v2 run saves a compressed, machine-readable record containing
the explicit basin archive, transition graph, diagnostics, phase evaluation
counts and event log.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)
from baselines.reference_optimizers import (
    optimize_bipop_cmaes,
    optimize_cmaes,
    optimize_de,
    optimize_lhs,
    optimize_multistart_lbfgsb,
    optimize_random_search,
)


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
ALGORITHM_FREEZE_TAG = "route-b-v2.0.0-rc1-ablationfreeze"

FUNCTIONS = list(range(1, 25))
DIMENSIONS = [2, 5, 10]
INSTANCES = [1, 2, 3]
BUDGET_MULTIPLIER = 1000

ALGORITHMS = [
    "BasinGraph_v2",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]

DISPLAY_NAMES = {
    "BasinGraph_v2": "BasinGraph v2.0.0-rc1",
    "CMA_ES": "CMA-ES",
    "BIPOP_CMA_ES": "BIPOP-CMA-ES",
    "DE": "Differential Evolution",
    "MS_LBFGSB": "Multi-start L-BFGS-B",
    "LHS": "Latin Hypercube Sampling",
    "Random": "Random Search",
}

BASELINE_FUNCTIONS = {
    "CMA_ES": optimize_cmaes,
    "BIPOP_CMA_ES": optimize_bipop_cmaes,
    "DE": optimize_de,
    "MS_LBFGSB": optimize_multistart_lbfgsb,
    "LHS": optimize_lhs,
    "Random": optimize_random_search,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260619)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_development/coco_rc1",
    )
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_contract() -> dict[str, Any]:
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, found {IMPLEMENTATION_VERSION}"
        )

    options = BasinGraphOptions()
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(
            "Frozen options hash mismatch:\n"
            f"expected={EXPECTED_OPTIONS_HASH}\n"
            f"observed={options.stable_hash()}"
        )

    branch = git_output("branch", "--show-current")
    if branch != "route-b/full-basingraph-v2":
        raise RuntimeError(f"Wrong branch: {branch}")

    # The formal runner may be added after the algorithm freeze, but the
    # algorithm package itself must be byte-identical to the freeze tag.
    diff_status = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            ALGORITHM_FREEZE_TAG,
            "--",
            "basingraph_v2",
        ],
        cwd=PROJECT_ROOT,
    ).returncode
    if diff_status != 0:
        raise RuntimeError(
            "basingraph_v2 differs from the rc1 ablation-freeze tag."
        )

    if subprocess.run(
        ["git", "diff", "--quiet", "--", "basingraph_v2"],
        cwd=PROJECT_ROOT,
    ).returncode != 0:
        raise RuntimeError("Uncommitted changes exist in basingraph_v2.")

    protocol_dir = (
        PROJECT_ROOT / "protocols" / "route_b" / "formal_v2_protocol"
    )
    manifest = protocol_dir / "V2_FORMAL_PROTOCOL_MANIFEST_SHA256.csv"
    lock = protocol_dir / "V2_FORMAL_PROTOCOL_LOCK.json"

    if not manifest.exists() or not lock.exists():
        raise FileNotFoundError("Formal v2 protocol lock is missing.")

    lock_data = json.loads(lock.read_text())
    if lock_data["status"] != "V2_FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Formal protocol is not frozen.")
    if lock_data["coco"]["development_instances"] != INSTANCES:
        raise RuntimeError("Development-instance partition mismatch.")
    if lock_data["coco"]["holdout_instances"] != list(range(4, 16)):
        raise RuntimeError("Prospective holdout partition mismatch.")

    return {
        "branch": branch,
        "git_commit": git_output("rev-parse", "HEAD"),
        "algorithm_freeze_tag": ALGORITHM_FREEZE_TAG,
        "algorithm_freeze_commit": git_output(
            "rev-list",
            "-n",
            "1",
            ALGORITHM_FREEZE_TAG,
        ),
        "implementation_version": IMPLEMENTATION_VERSION,
        "options_hash": options.stable_hash(),
        "protocol_lock_sha256": sha256_file(lock),
        "protocol_manifest_sha256": sha256_file(manifest),
        "basingraph_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
            for path in sorted(
                (PROJECT_ROOT / "basingraph_v2").glob("*.py")
            )
        },
        "baseline_source_sha256": sha256_file(
            PROJECT_ROOT / "baselines" / "reference_optimizers.py"
        ),
    }


def get_bounds(problem, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lb = np.asarray(problem.lower_bounds, dtype=float).reshape(-1)
        ub = np.asarray(problem.upper_bounds, dtype=float).reshape(-1)
    except Exception:
        lb = -5.0 * np.ones(dimension)
        ub = 5.0 * np.ones(dimension)

    if len(lb) != dimension:
        lb = -5.0 * np.ones(dimension)
    if len(ub) != dimension:
        ub = 5.0 * np.ones(dimension)

    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)
    invalid = ub <= lb
    lb[invalid] = -5.0
    ub[invalid] = 5.0
    return lb, ub


def parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(
        r"bbob_f(\d+)_i(\d+)_d(\d+)",
        problem_id,
    )
    if not match:
        raise ValueError(f"Unexpected COCO problem id: {problem_id}")
    return tuple(int(value) for value in match.groups())


def run_basingraph(
    problem,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
    detail_path: Path,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    result = minimize_basingraph_v2(
        objective=problem,
        lb=lb,
        ub=ub,
        max_evals=budget,
        seed=seed,
    )

    payload = {
        "run_metadata": run_metadata,
        "result": result.to_jsonable(),
    }
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(detail_path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return {
        "nfe_internal": int(result.nfe),
        "fbest": float(result.fbest),
        "message": result.message,
        "implementation_version": result.implementation_version,
        "options_hash": result.options_hash,
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "phase_evaluations_json": json.dumps(
            result.phase_evaluations,
            sort_keys=True,
        ),
        "detail_json_gz": str(detail_path.relative_to(PROJECT_ROOT)),
    }


def run_baseline(
    algorithm: str,
    problem,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> dict[str, Any]:
    optimizer = BASELINE_FUNCTIONS[algorithm]
    result = optimizer(
        objective=problem,
        lb=lb,
        ub=ub,
        max_evals=budget,
        seed=seed,
    )
    return {
        "nfe_internal": int(result["nfe"]),
        "fbest": float(result["fbest"]),
        "message": str(result["message"]),
        "implementation_version": "",
        "options_hash": "",
        "archive_nodes": "",
        "graph_edges": "",
        "phase_evaluations_json": "",
        "detail_json_gz": "",
    }


def run_algorithm(
    algorithm: str,
    run_id: str,
    base_seed: int,
    result_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    import cocoex

    suite_options = (
        "dimensions: 2,5,10 "
        "function_indices: 1-24 "
        "instance_indices: 1-3"
    )

    suite = cocoex.Suite("bbob", "", suite_options)

    folder = f"routeb_v2_development/{run_id}/{algorithm}"
    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {folder} "
        f"algorithm_name: {algorithm} "
        f'algorithm_info: "BasinGraph v2 formal development protocol; '
        f'instances 1-3 only"',
    )

    rows: list[dict[str, Any]] = []

    for problem_index, problem in enumerate(suite):
        problem.observe_with(observer)

        problem_id = str(problem.id)
        function_index, instance_index, dimension = parse_problem_id(problem_id)

        if instance_index not in INSTANCES:
            raise RuntimeError(
                f"Holdout leakage detected: {problem_id}"
            )
        if dimension not in DIMENSIONS:
            raise RuntimeError(f"Unexpected dimension: {problem_id}")

        budget = BUDGET_MULTIPLIER * dimension
        seed = base_seed + problem_index
        lb, ub = get_bounds(problem, dimension)

        row = {
            "run_id": run_id,
            "suite": "bbob",
            "partition": "development",
            "problem_id": problem_id,
            "function_index": function_index,
            "instance_index": instance_index,
            "dimension": dimension,
            "algorithm": algorithm,
            "algorithm_display": DISPLAY_NAMES[algorithm],
            "seed": seed,
            "budget": budget,
            "runner_status": "completed",
            "error": "",
        }

        try:
            if algorithm == "BasinGraph_v2":
                detail_path = (
                    result_root
                    / run_id
                    / "details"
                    / f"{problem_id}.json.gz"
                )
                run_metadata = {
                    **contract,
                    "run_id": run_id,
                    "partition": "development",
                    "problem_id": problem_id,
                    "function_index": function_index,
                    "instance_index": instance_index,
                    "dimension": dimension,
                    "seed": seed,
                    "budget": budget,
                }
                row.update(
                    run_basingraph(
                        problem,
                        lb,
                        ub,
                        budget,
                        seed,
                        detail_path,
                        run_metadata,
                    )
                )
            else:
                row.update(
                    run_baseline(
                        algorithm,
                        problem,
                        lb,
                        ub,
                        budget,
                        seed,
                    )
                )

            observed_evaluations = getattr(
                problem,
                "evaluations",
                row["nfe_internal"],
            )
            row["nfe_observer"] = int(observed_evaluations)

        except Exception as exc:
            row.update(
                {
                    "runner_status": "failed",
                    "error": (
                        f"{type(exc).__name__}: {exc}\n"
                        + traceback.format_exc(limit=30)
                    ),
                    "nfe_internal": 0,
                    "nfe_observer": int(
                        getattr(problem, "evaluations", 0)
                    ),
                    "fbest": np.nan,
                    "message": "",
                    "implementation_version": "",
                    "options_hash": "",
                    "archive_nodes": "",
                    "graph_edges": "",
                    "phase_evaluations_json": "",
                    "detail_json_gz": "",
                }
            )

        rows.append(row)

    del observer
    del suite
    return rows


def main() -> None:
    args = parse_args()
    contract = verify_frozen_contract()

    result_root = PROJECT_ROOT / args.result_root
    run_root = result_root / args.run_id

    if run_root.exists():
        raise RuntimeError(
            f"Run directory already exists: {run_root}"
        )
    run_root.mkdir(parents=True)

    metadata = {
        **contract,
        "status": "V2_COCO_DEVELOPMENT_RUN_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "run_id": args.run_id,
        "suite": "bbob",
        "partition": "development",
        "functions": FUNCTIONS,
        "dimensions": DIMENSIONS,
        "instances": INSTANCES,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "algorithms": ALGORITHMS,
        "paired_seed_policy": (
            "base_seed + problem iteration index; identical across algorithms"
        ),
        "base_seed": args.base_seed,
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    all_rows: list[dict[str, Any]] = []
    for algorithm in ALGORITHMS:
        print(f"RUNNING {algorithm}", flush=True)
        rows = run_algorithm(
            algorithm,
            args.run_id,
            args.base_seed,
            result_root,
            contract,
        )
        all_rows.extend(rows)

    raw_path = run_root / "coco_v2_development_raw_results.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    metadata["status"] = "V2_COCO_DEVELOPMENT_RUN_COMPLETE"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["rows"] = len(all_rows)
    metadata["failed_rows"] = sum(
        row["runner_status"] != "completed"
        for row in all_rows
    )
    metadata["raw_results"] = str(
        raw_path.relative_to(PROJECT_ROOT)
    )
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("V2_COCO_DEVELOPMENT_RUN_COMPLETE")
    print("run id:", args.run_id)
    print("rows:", len(all_rows))
    print("failed rows:", metadata["failed_rows"])
    print("raw results:", raw_path)


if __name__ == "__main__":
    main()
