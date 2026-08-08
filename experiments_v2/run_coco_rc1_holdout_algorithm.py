#!/usr/bin/env python3
"""
Run one algorithm on the frozen prospective COCO/BBOB holdout.

Holdout partition:
- functions 1-24;
- dimensions 2, 3, 5, 10 and 20;
- instances 4-15;
- budget 1000d.

This runner refuses development instances 1-3 and verifies the selected
BasinGraph 2.0.0-rc1 paper-code contract before every execution.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference_optimizers import (
    optimize_bipop_cmaes,
    optimize_cmaes,
    optimize_de,
    optimize_lhs,
    optimize_multistart_lbfgsb,
    optimize_random_search,
)
from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


EXPECTED_BRANCH = "route-b/finalize-rc1-consistent"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
CODE_FREEZE_TAG = "route-b-v2.0.0-rc1-codefreeze"
FINAL_CANDIDATE_TAG = "route-b-v2.0.0-rc1-selected-final-candidate"
RUNNER_FREEZE_TAG = "route-b-v2.0.0-rc1-holdout-runner-freeze-v2"

FUNCTIONS = list(range(1, 25))
DIMENSIONS = [2, 3, 5, 10, 20]
INSTANCES = list(range(4, 16))
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
    "BasinGraph_v2": "BasinGraph 2.0.0-rc1",
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
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260620)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_holdout/coco_rc1",
    )
    return parser.parse_args()


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


def verify_contract() -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong branch: {branch}")

    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )

    options = BasinGraphOptions()
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(
            "Selected candidate options hash mismatch:\n"
            f"expected={EXPECTED_OPTIONS_HASH}\n"
            f"observed={options.stable_hash()}"
        )

    if subprocess.run(
        ["git", "diff", "--quiet", CODE_FREEZE_TAG, "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError(
            "basingraph_v2 differs from the selected rc1 code-freeze tag."
        )

    if subprocess.run(
        ["git", "diff", "--quiet", "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("Uncommitted algorithm changes exist.")

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
    formal_lock_path = (
        ROOT
        / "protocols"
        / "route_b"
        / "formal_v2_protocol"
        / "V2_FORMAL_PROTOCOL_LOCK.json"
    )

    contract = json.loads(contract_path.read_text())
    selection = json.loads(selection_path.read_text())
    formal_lock = json.loads(formal_lock_path.read_text())

    assert contract["status"] == "RC1_PAPER_CODE_CONTRACT_FROZEN"
    assert contract["holdout_accessed"] is False
    assert selection["status"] == "RC1_SELECTED_AS_FINAL_HOLDOUT_CANDIDATE"
    assert selection["holdout_accessed"] is False
    assert formal_lock["status"] == "V2_FORMAL_PROTOCOL_FROZEN"
    assert formal_lock["coco"]["development_instances"] == [1, 2, 3]
    assert formal_lock["coco"]["holdout_instances"] == INSTANCES

    code_freeze_commit = git_output("rev-list", "-n", "1", CODE_FREEZE_TAG)
    candidate_commit = git_output(
        "rev-list", "-n", "1", FINAL_CANDIDATE_TAG
    )
    runner_commit = git_output(
        "rev-list", "-n", "1", RUNNER_FREEZE_TAG
    )
    current_commit = git_output("rev-parse", "HEAD")

    if contract["code_freeze_commit"] != code_freeze_commit:
        raise RuntimeError("Contract/code-freeze mismatch.")
    if current_commit != runner_commit:
        raise RuntimeError(
            "Current HEAD does not match the frozen holdout runner tag:\n"
            f"HEAD={current_commit}\nrunner={runner_commit}"
        )

    return {
        "current_commit": current_commit,
        "branch": branch,
        "code_freeze_tag": CODE_FREEZE_TAG,
        "code_freeze_commit": code_freeze_commit,
        "final_candidate_tag": FINAL_CANDIDATE_TAG,
        "final_candidate_commit": candidate_commit,
        "runner_freeze_tag": RUNNER_FREEZE_TAG,
        "runner_freeze_commit": runner_commit,
        "implementation_version": IMPLEMENTATION_VERSION,
        "options_hash": options.stable_hash(),
        "paper_code_contract_sha256": sha256_file(contract_path),
        "selection_decision_sha256": sha256_file(selection_path),
        "formal_protocol_lock_sha256": sha256_file(formal_lock_path),
        "algorithm_source_hashes": contract["source_hashes"],
        "baseline_source_sha256": sha256_file(
            ROOT / "baselines" / "reference_optimizers.py"
        ),
    }


def parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(
        r"bbob_f(\d+)_i(\d+)_d(\d+)",
        problem_id,
    )
    if not match:
        raise ValueError(f"Unexpected COCO problem id: {problem_id}")
    return tuple(int(value) for value in match.groups())


def problem_seed(
    base_seed: int,
    function_index: int,
    dimension: int,
    instance_index: int,
) -> int:
    return int(
        base_seed
        + 100_000 * function_index
        + 1_000 * dimension
        + instance_index
    )


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
    with gzip.open(
        detail_path,
        "wt",
        encoding="utf-8",
        compresslevel=6,
    ) as handle:
        json.dump(payload, handle, separators=(",", ":"))

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
            separators=(",", ":"),
        ),
        "detail_json_gz": str(detail_path.relative_to(ROOT)),
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


def main() -> None:
    args = parse_args()
    contract = verify_contract()

    import cocoex

    result_root = ROOT / args.result_root
    run_root = result_root / args.run_id
    algorithm_root = run_root / args.algorithm
    if algorithm_root.exists():
        raise RuntimeError(f"Output already exists: {algorithm_root}")
    algorithm_root.mkdir(parents=True)

    metadata = {
        **contract,
        "status": "COCO_HOLDOUT_ALGORITHM_RUN_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "run_id": args.run_id,
        "algorithm": args.algorithm,
        "algorithm_display": DISPLAY_NAMES[args.algorithm],
        "suite": "bbob",
        "partition": "prospective_holdout",
        "functions": FUNCTIONS,
        "dimensions": DIMENSIONS,
        "instances": INSTANCES,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "base_seed": args.base_seed,
        "seed_formula": (
            "base_seed + 100000*function + 1000*dimension + instance"
        ),
    }
    (algorithm_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    suite = cocoex.Suite(
        "bbob",
        "instances: 4-15",
        "dimensions: 2,3,5,10,20 function_indices: 1-24",
    )

    # Validate actual IDs before attaching an observer or evaluating.
    suite_ids = list(suite.ids())
    parsed_ids = [parse_problem_id(pid) for pid in suite_ids]
    observed_functions = sorted({f for f, _, _ in parsed_ids})
    observed_instances = sorted({i for _, i, _ in parsed_ids})
    observed_dimensions = sorted({d for _, _, d in parsed_ids})

    suite_preflight = {
        "status": "COCO_HOLDOUT_SUITE_PREFLIGHT_OK",
        "suite_instance": "instances: 4-15",
        "problem_count": len(suite_ids),
        "functions": observed_functions,
        "instances": observed_instances,
        "dimensions": observed_dimensions,
        "objective_evaluations_performed": 0,
    }

    if len(suite_ids) != 24 * 5 * 12:
        raise RuntimeError(
            f"Unexpected holdout problem count: {len(suite_ids)}"
        )
    if observed_functions != list(range(1, 25)):
        raise RuntimeError(
            f"Unexpected holdout functions: {observed_functions}"
        )
    if observed_instances != list(range(4, 16)):
        raise RuntimeError(
            f"Unexpected actual holdout instances: {observed_instances}"
        )
    if observed_dimensions != [2, 3, 5, 10, 20]:
        raise RuntimeError(
            f"Unexpected holdout dimensions: {observed_dimensions}"
        )

    (algorithm_root / "suite_preflight.json").write_text(
        json.dumps(suite_preflight, indent=2)
    )
    result_folder = (
        f"routeb_v2_holdout/{args.run_id}/{args.algorithm}"
    )
    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} "
        f"algorithm_name: {args.algorithm} "
        f'algorithm_info: "Prospective holdout; selected BasinGraph rc1 contract"',
    )

    rows: list[dict[str, Any]] = []

    for problem in suite:
        problem.observe_with(observer)

        problem_id = str(problem.id)
        function_index, instance_index, dimension = parse_problem_id(problem_id)

        if instance_index not in INSTANCES:
            raise RuntimeError(
                f"Development/unknown instance leakage: {problem_id}"
            )
        if dimension not in DIMENSIONS:
            raise RuntimeError(f"Unexpected dimension: {problem_id}")

        seed = problem_seed(
            args.base_seed,
            function_index,
            dimension,
            instance_index,
        )
        budget = BUDGET_MULTIPLIER * dimension
        lb, ub = get_bounds(problem, dimension)

        row: dict[str, Any] = {
            "run_id": args.run_id,
            "suite": "bbob",
            "partition": "prospective_holdout",
            "problem_id": problem_id,
            "function_index": function_index,
            "instance_index": instance_index,
            "dimension": dimension,
            "algorithm": args.algorithm,
            "algorithm_display": DISPLAY_NAMES[args.algorithm],
            "seed": seed,
            "budget": budget,
            "runner_status": "completed",
            "error": "",
        }

        try:
            if args.algorithm == "BasinGraph_v2":
                detail_path = (
                    algorithm_root
                    / "details"
                    / f"{problem_id}.json.gz"
                )
                detail_metadata = {
                    **contract,
                    "run_id": args.run_id,
                    "partition": "prospective_holdout",
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
                        detail_metadata,
                    )
                )
            else:
                row.update(
                    run_baseline(
                        args.algorithm,
                        problem,
                        lb,
                        ub,
                        budget,
                        seed,
                    )
                )

            row["nfe_observer"] = int(
                getattr(problem, "evaluations", row["nfe_internal"])
            )
            row["final_target_hit"] = bool(
                getattr(problem, "final_target_hit", False)
            )

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
                    "final_target_hit": False,
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

    raw_path = algorithm_root / "holdout_results.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata["status"] = "COCO_HOLDOUT_ALGORITHM_RUN_COMPLETE"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["rows"] = len(rows)
    metadata["failed_rows"] = sum(
        row["runner_status"] != "completed"
        for row in rows
    )
    metadata["raw_results_sha256"] = sha256_file(raw_path)
    (algorithm_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("COCO_HOLDOUT_ALGORITHM_RUN_COMPLETE")
    print("algorithm:", args.algorithm)
    print("rows:", len(rows))
    print("failed rows:", metadata["failed_rows"])


if __name__ == "__main__":
    main()
