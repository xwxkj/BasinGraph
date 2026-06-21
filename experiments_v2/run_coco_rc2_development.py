#!/usr/bin/env python3
"""
Route B Step B6D3: paired COCO/BBOB development run for BasinGraph rc2.

This runner instantiates only the frozen COCO development partition:
- functions 1-24;
- dimensions 2, 5 and 10;
- instances 1-3;
- budget 1000d.

It runs BasinGraph v2.0.0-rc2 only. The six unchanged baseline records and the
rc1 BasinGraph records are reused from the already validated rc1 development
run, with identical problem-specific seeds.

Prospective holdout instances 4-15 are never instantiated.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


EXPECTED_IMPLEMENTATION = "2.0.0-rc2"
EXPECTED_OPTIONS_HASH = (
    "15fe9bcbf8e87aabe4767f811524c00f"
    "67b74e3ebfa31fa81cdf6f461cbfeb08"
)
CODE_FREEZE_TAG = "route-b-v2.0.0-rc2-codefreeze"
MACHINE_FREEZE_TAG = "route-b-v2.0.0-rc2-machinefreeze"

FUNCTIONS = list(range(1, 25))
DIMENSIONS = [2, 5, 10]
INSTANCES = [1, 2, 3]
BUDGET_MULTIPLIER = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260619)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_development/coco_rc2",
    )
    parser.add_argument(
        "--rc1-result-root",
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


def verify_machine_contract() -> dict[str, Any]:
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )

    options = BasinGraphOptions()
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(
            "Frozen rc2 options hash mismatch:\n"
            f"expected={EXPECTED_OPTIONS_HASH}\n"
            f"observed={options.stable_hash()}"
        )

    branch = git_output("branch", "--show-current")
    if branch != "route-b/full-basingraph-v2":
        raise RuntimeError(f"Wrong branch: {branch}")

    code_freeze_commit = git_output(
        "rev-list", "-n", "1", CODE_FREEZE_TAG
    )
    machine_freeze_commit = git_output(
        "rev-list", "-n", "1", MACHINE_FREEZE_TAG
    )

    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            CODE_FREEZE_TAG,
            "--",
            "basingraph_v2",
        ],
        cwd=PROJECT_ROOT,
    ).returncode != 0:
        raise RuntimeError(
            "basingraph_v2 differs from the rc2 code-freeze tag."
        )

    if subprocess.run(
        ["git", "diff", "--quiet", "--", "basingraph_v2"],
        cwd=PROJECT_ROOT,
    ).returncode != 0:
        raise RuntimeError("Uncommitted changes exist in basingraph_v2.")

    machine_spec_path = (
        PROJECT_ROOT
        / "protocols"
        / "route_b"
        / "rc2_revision"
        / "RC2_MACHINE_SPEC.json"
    )
    machine_spec = json.loads(machine_spec_path.read_text())

    if machine_spec["status"] != "RC2_MACHINE_SPEC_FROZEN":
        raise RuntimeError("rc2 machine spec is not frozen.")
    if machine_spec["implementation_version"] != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Machine-spec implementation mismatch.")
    if machine_spec["options_hash"] != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Machine-spec options hash mismatch.")
    if machine_spec["git_commit"] != code_freeze_commit:
        raise RuntimeError("Machine spec does not identify the code-freeze commit.")

    for relative, expected in machine_spec["source_hashes"].items():
        source = PROJECT_ROOT / relative
        observed = sha256_file(source)
        if observed != expected:
            raise RuntimeError(
                f"Source hash mismatch: {relative}\n"
                f"expected={expected}\nobserved={observed}"
            )

    formal_lock_path = (
        PROJECT_ROOT
        / "protocols"
        / "route_b"
        / "formal_v2_protocol"
        / "V2_FORMAL_PROTOCOL_LOCK.json"
    )
    formal_lock = json.loads(formal_lock_path.read_text())

    if formal_lock["status"] != "V2_FORMAL_PROTOCOL_FROZEN":
        raise RuntimeError("Formal protocol lock is not frozen.")
    if formal_lock["coco"]["development_instances"] != INSTANCES:
        raise RuntimeError("Development partition mismatch.")
    if formal_lock["coco"]["holdout_instances"] != list(range(4, 16)):
        raise RuntimeError("Prospective holdout partition mismatch.")

    gate_path = (
        PROJECT_ROOT
        / "protocols"
        / "route_b"
        / "rc2_revision"
        / "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json"
    )

    return {
        "branch": branch,
        "current_git_commit": git_output("rev-parse", "HEAD"),
        "code_freeze_tag": CODE_FREEZE_TAG,
        "code_freeze_commit": code_freeze_commit,
        "machine_freeze_tag": MACHINE_FREEZE_TAG,
        "machine_freeze_commit": machine_freeze_commit,
        "implementation_version": IMPLEMENTATION_VERSION,
        "options_hash": options.stable_hash(),
        "machine_spec_sha256": sha256_file(machine_spec_path),
        "formal_protocol_lock_sha256": sha256_file(formal_lock_path),
        "acceptance_gate_sha256": sha256_file(gate_path),
        "source_hashes": machine_spec["source_hashes"],
    }


def parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(
        r"bbob_f(\d+)_i(\d+)_d(\d+)",
        problem_id,
    )
    if not match:
        raise ValueError(f"Unexpected COCO problem id: {problem_id}")
    return tuple(int(value) for value in match.groups())


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


def main() -> None:
    args = parse_args()
    contract = verify_machine_contract()

    rc1_result_root = PROJECT_ROOT / args.rc1_result_root
    rc1_run_id = (rc1_result_root / "LAST_RUN_ID.txt").read_text().strip()
    rc1_raw_path = (
        rc1_result_root
        / rc1_run_id
        / "coco_v2_development_raw_results.csv"
    )
    if not rc1_raw_path.exists():
        raise FileNotFoundError(rc1_raw_path)

    rc1_rows = list(csv.DictReader(rc1_raw_path.open()))
    rc1_seed_by_problem = {}
    for row in rc1_rows:
        problem_id = row["problem_id"]
        seed = int(row["seed"])
        if problem_id in rc1_seed_by_problem:
            if rc1_seed_by_problem[problem_id] != seed:
                raise RuntimeError(
                    f"rc1 seed mismatch across algorithms: {problem_id}"
                )
        else:
            rc1_seed_by_problem[problem_id] = seed

    result_root = PROJECT_ROOT / args.result_root
    run_root = result_root / args.run_id
    if run_root.exists():
        raise RuntimeError(f"Run root already exists: {run_root}")
    run_root.mkdir(parents=True)

    metadata = {
        **contract,
        "status": "V2_RC2_COCO_DEVELOPMENT_RUN_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "run_id": args.run_id,
        "rc1_run_id": rc1_run_id,
        "partition": "development",
        "suite": "bbob",
        "functions": FUNCTIONS,
        "dimensions": DIMENSIONS,
        "instances": INSTANCES,
        "budget_multiplier": BUDGET_MULTIPLIER,
        "algorithm": "BasinGraph_v2_rc2",
        "seed_policy": (
            "exactly reuse each problem-specific seed from the validated rc1 "
            "development run"
        ),
        "rc1_raw_results_sha256": sha256_file(rc1_raw_path),
    }
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    import cocoex

    suite = cocoex.Suite(
        "bbob",
        "",
        (
            "dimensions: 2,5,10 "
            "function_indices: 1-24 "
            "instance_indices: 1-3"
        ),
    )

    result_folder = (
        f"routeb_v2_rc2_development/{args.run_id}/BasinGraph_v2_rc2"
    )
    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} "
        f"algorithm_name: BasinGraph_v2_rc2 "
        f'algorithm_info: "Paired rc2 development run; instances 1-3 only"',
    )

    rows: list[dict[str, Any]] = []

    for problem in suite:
        problem.observe_with(observer)

        problem_id = str(problem.id)
        function_index, instance_index, dimension = parse_problem_id(problem_id)

        if instance_index not in INSTANCES:
            raise RuntimeError(f"Holdout leakage: {problem_id}")
        if problem_id not in rc1_seed_by_problem:
            raise RuntimeError(f"Problem missing from rc1 run: {problem_id}")

        seed = rc1_seed_by_problem[problem_id]
        budget = BUDGET_MULTIPLIER * dimension
        lb, ub = get_bounds(problem, dimension)

        row: dict[str, Any] = {
            "run_id": args.run_id,
            "rc1_run_id": rc1_run_id,
            "suite": "bbob",
            "partition": "development",
            "problem_id": problem_id,
            "function_index": function_index,
            "instance_index": instance_index,
            "dimension": dimension,
            "algorithm": "BasinGraph_v2_rc2",
            "algorithm_display": "BasinGraph v2.0.0-rc2",
            "seed": seed,
            "budget": budget,
            "runner_status": "completed",
            "error": "",
        }

        try:
            result = minimize_basingraph_v2(
                objective=problem,
                lb=lb,
                ub=ub,
                max_evals=budget,
                seed=seed,
            )

            detail_path = (
                run_root
                / "details"
                / f"{problem_id}.json.gz"
            )
            detail_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "run_metadata": {
                    **contract,
                    "run_id": args.run_id,
                    "rc1_run_id": rc1_run_id,
                    "partition": "development",
                    "problem_id": problem_id,
                    "function_index": function_index,
                    "instance_index": instance_index,
                    "dimension": dimension,
                    "seed": seed,
                    "budget": budget,
                },
                "result": result.to_jsonable(),
            }

            with gzip.open(
                detail_path,
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(payload, handle, indent=2)

            degree_out: dict[int, int] = {}
            degree_in: dict[int, int] = {}
            for edge in result.graph_edges:
                degree_out[edge.source_id] = (
                    degree_out.get(edge.source_id, 0) + 1
                )
                degree_in[edge.target_id] = (
                    degree_in.get(edge.target_id, 0) + 1
                )

            controller_payload = {}
            for event in result.event_log:
                if event.event == "controller_decision":
                    controller_payload = event.payload
                    break

            row.update(
                {
                    "nfe_internal": int(result.nfe),
                    "nfe_observer": int(
                        getattr(problem, "evaluations", result.nfe)
                    ),
                    "fbest": float(result.fbest),
                    "final_target_hit": bool(
                        getattr(problem, "final_target_hit", False)
                    ),
                    "message": result.message,
                    "implementation_version": result.implementation_version,
                    "options_hash": result.options_hash,
                    "probe_count_total": int(result.probe_count_total),
                    "retained_probes": len(result.probes),
                    "certified_nodes": len(result.archive),
                    "archive_saturated": (
                        len(result.archive)
                        >= BasinGraphOptions().archive_max_size
                    ),
                    "graph_edges": len(result.graph_edges),
                    "graph_edges_per_node": (
                        len(result.graph_edges)
                        / max(len(result.archive), 1)
                    ),
                    "max_outgoing_degree": max(
                        degree_out.values(),
                        default=0,
                    ),
                    "max_incoming_degree": max(
                        degree_in.values(),
                        default=0,
                    ),
                    "center_local_active": bool(
                        controller_payload.get(
                            "use_center_local",
                            False,
                        )
                    ),
                    "far_basin_active": bool(
                        controller_payload.get(
                            "use_far_basin",
                            False,
                        )
                    ),
                    "curvature_anisotropy": float(
                        result.diagnostics.curvature_anisotropy
                    ),
                    "domain_anisotropy": float(
                        result.diagnostics.domain_anisotropy
                    ),
                    "principal_directions": int(
                        result.direction_diagnostics.retained_directions
                    ),
                    "phase_evaluations_json": json.dumps(
                        result.phase_evaluations,
                        sort_keys=True,
                    ),
                    "detail_json_gz": str(
                        detail_path.relative_to(PROJECT_ROOT)
                    ),
                }
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
                    "probe_count_total": 0,
                    "retained_probes": 0,
                    "certified_nodes": 0,
                    "archive_saturated": False,
                    "graph_edges": 0,
                    "graph_edges_per_node": 0.0,
                    "max_outgoing_degree": 0,
                    "max_incoming_degree": 0,
                    "center_local_active": False,
                    "far_basin_active": False,
                    "curvature_anisotropy": np.nan,
                    "domain_anisotropy": np.nan,
                    "principal_directions": 0,
                    "phase_evaluations_json": "{}",
                    "detail_json_gz": "",
                }
            )

        rows.append(row)

    del observer
    del suite

    raw_path = run_root / "coco_rc2_development_raw_results.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata["status"] = "V2_RC2_COCO_DEVELOPMENT_RUN_COMPLETE"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["rows"] = len(rows)
    metadata["failed_rows"] = sum(
        row["runner_status"] != "completed"
        for row in rows
    )
    metadata["raw_results"] = str(raw_path.relative_to(PROJECT_ROOT))
    (run_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("V2_RC2_COCO_DEVELOPMENT_RUN_COMPLETE")
    print("run id:", args.run_id)
    print("rc1 run id:", rc1_run_id)
    print("rows:", len(rows))
    print("failed rows:", metadata["failed_rows"])
    print("raw results:", raw_path)


if __name__ == "__main__":
    main()
