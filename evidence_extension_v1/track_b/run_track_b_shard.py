#!/usr/bin/env python3
"""Run one algorithm-dimension shard for B21 Track B."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import minimize_basingraph_v2  # noqa: E402
from evidence_extension_v1.track_b.common import (  # noqa: E402
    ALGORITHMS,
    DISPLAY_NAMES,
    FUNCTION_GROUPS,
    MODE_CONFIG,
    OBSERVER_NAMES,
    checkpoint_values,
    parse_problem_id,
    seed_for,
    sha256_file,
    should_store_trace,
    stable_digest,
    suite_options,
    verify_source_identity,
)
from evidence_extension_v1.track_b.modern_baselines import (  # noqa: E402
    OPTIMIZERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--dimension", type=int, choices=(5, 20), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_b")
    return parser.parse_args()


def get_bounds(problem: Any, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lb = np.asarray(problem.lower_bounds, dtype=float).reshape(-1)
        ub = np.asarray(problem.upper_bounds, dtype=float).reshape(-1)
    except Exception:
        lb = -5.0 * np.ones(dimension)
        ub = 5.0 * np.ones(dimension)
    if len(lb) != dimension or len(ub) != dimension:
        lb = -5.0 * np.ones(dimension)
        ub = 5.0 * np.ones(dimension)
    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)
    invalid = ub <= lb
    lb[invalid], ub[invalid] = -5.0, 5.0
    return lb, ub


def monotone(history: list[tuple[int, float]]) -> bool:
    values = [float(value) for _, value in history]
    return all(values[index] <= values[index - 1] for index in range(1, len(values)))


def run_algorithm(
    algorithm: str,
    problem: Any,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if algorithm == "BasinGraph":
        result = minimize_basingraph_v2(
            objective=problem,
            lb=lb,
            ub=ub,
            max_evals=budget,
            seed=seed,
        )
        public = result.to_jsonable()
        record = {
            "implementation": "basingraph_v2.optimizer.minimize_basingraph_v2",
            "implementation_version": result.implementation_version,
            "options_hash": result.options_hash,
            "nfe": int(result.nfe),
            "fbest": float(result.fbest),
            "history": list(result.history),
            "message": result.message,
            "archive_nodes": len(result.archive),
            "graph_edges": len(result.graph_edges),
            "graph_referential_integrity": all(
                edge.source_id in {node.node_id for node in result.archive}
                and edge.target_id in {node.node_id for node in result.archive}
                for edge in result.graph_edges
            ),
            "phase_evaluations": dict(result.phase_evaluations),
            "metadata": {
                "diagnostics": result.diagnostics.to_jsonable(),
                "events": len(result.event_log),
            },
        }
        return record, public

    optimizer = OPTIMIZERS[algorithm]
    result = optimizer(
        objective=problem,
        lb=lb,
        ub=ub,
        max_evals=budget,
        seed=seed,
    )
    record = {
        "implementation": result["implementation"],
        "implementation_version": "",
        "options_hash": "",
        "nfe": int(result["nfe"]),
        "fbest": float(result["fbest"]),
        "history": list(result["history"]),
        "message": result["message"],
        "archive_nodes": 0,
        "graph_edges": 0,
        "graph_referential_integrity": True,
        "phase_evaluations": {},
        "metadata": result.get("metadata", {}),
    }
    public = {
        "algorithm": result["algorithm"],
        "implementation": result["implementation"],
        "xbest": np.asarray(result["xbest"], dtype=float).tolist(),
        "fbest": float(result["fbest"]),
        "nfe": int(result["nfe"]),
        "history": [(int(a), float(b)) for a, b in result["history"]],
        "message": result["message"],
        "metadata": result.get("metadata", {}),
    }
    return record, public


def main() -> None:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    if args.dimension not in config["dimensions"]:
        raise RuntimeError("Dimension is not in the frozen mode configuration.")
    if config["confirmatory"] and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory shard requires --authorize-confirmatory.")

    identity = verify_source_identity(require_clean=True)
    run_root = ROOT / args.result_root / args.run_id
    shard_root = run_root / "shards" / f"{args.algorithm}_d{args.dimension}"
    complete_marker = shard_root / "SHARD_COMPLETE.json"
    if complete_marker.is_file():
        print("TRACK_B_SHARD_ALREADY_COMPLETE")
        print(complete_marker)
        return

    attempt_root = shard_root / f"attempt_{args.attempt:04d}"
    if attempt_root.exists():
        raise RuntimeError(f"Attempt directory already exists: {attempt_root}")
    attempt_root.mkdir(parents=True)

    import cocoex

    suite_instance, suite_options_text = suite_options(args.mode, args.dimension)
    suite = cocoex.Suite("bbob", suite_instance, suite_options_text)
    suite_ids = list(suite.ids())
    parsed = [parse_problem_id(problem_id) for problem_id in suite_ids]
    expected_problem_count = (
        len(config["functions"]) * len(config["instances"])
    )
    if len(suite_ids) != expected_problem_count:
        raise RuntimeError(
            f"Unexpected shard problem count: {len(suite_ids)} != {expected_problem_count}"
        )
    if sorted({f for f, _, _ in parsed}) != config["functions"]:
        raise RuntimeError("Function set mismatch in shard preflight.")
    if sorted({i for _, i, _ in parsed}) != config["instances"]:
        raise RuntimeError("Instance set mismatch in shard preflight.")
    if sorted({d for _, _, d in parsed}) != [args.dimension]:
        raise RuntimeError("Dimension mismatch in shard preflight.")

    observer_relative = (
        Path("exdata")
        / "b21_track_b"
        / args.run_id
        / args.mode
        / args.algorithm
        / f"d{args.dimension}"
        / f"attempt_{args.attempt:04d}"
    )
    observer_absolute = ROOT / observer_relative
    if observer_absolute.exists():
        raise RuntimeError(f"Observer directory already exists: {observer_absolute}")
    result_folder = observer_relative.relative_to("exdata").as_posix()
    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} "
        f"algorithm_name: {OBSERVER_NAMES[args.algorithm]} "
        f'algorithm_info: "B21 Track B {args.mode}; frozen modern baseline comparison"',
    )

    metadata = {
        "status": "TRACK_B_SHARD_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "algorithm": args.algorithm,
        "algorithm_display": DISPLAY_NAMES[args.algorithm],
        "dimension": args.dimension,
        "attempt": args.attempt,
        "partition": config["partition"],
        "functions": config["functions"],
        "instances": config["instances"],
        "budget_multiplier": config["budget_multiplier"],
        "base_seed": args.base_seed,
        "identity": identity,
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "observer_relative": observer_relative.as_posix(),
    }
    (attempt_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for problem in suite:
        problem.observe_with(observer)
        problem_id = str(problem.id)
        function_index, instance_index, dimension = parse_problem_id(problem_id)
        budget = int(config["budget_multiplier"] * dimension)
        seed = seed_for(
            args.base_seed,
            function_index,
            dimension,
            instance_index,
        )
        lb, ub = get_bounds(problem, dimension)
        group_id, group_name = FUNCTION_GROUPS[function_index]
        row: dict[str, Any] = {
            "run_id": args.run_id,
            "mode": args.mode,
            "partition": config["partition"],
            "problem_id": problem_id,
            "function_index": function_index,
            "function_group_id": group_id,
            "function_group": group_name,
            "instance_index": instance_index,
            "dimension": dimension,
            "algorithm": args.algorithm,
            "algorithm_display": DISPLAY_NAMES[args.algorithm],
            "seed": seed,
            "budget": budget,
            "runner_status": "completed",
            "error": "",
        }

        started = time.perf_counter()
        try:
            record, public = run_algorithm(
                args.algorithm,
                problem,
                lb,
                ub,
                budget,
                seed,
            )
            elapsed = time.perf_counter() - started
            history = record["history"]
            history_hash = stable_digest(history)
            phase_evaluations = record["phase_evaluations"]
            checkpoints = checkpoint_values(
                history,
                mode=args.mode,
                dimension=dimension,
                budget=budget,
            )
            row.update(
                {
                    "implementation": record["implementation"],
                    "implementation_version": record["implementation_version"],
                    "options_hash": record["options_hash"],
                    "nfe_internal": record["nfe"],
                    "nfe_observer": int(getattr(problem, "evaluations", 0)),
                    "fbest": record["fbest"],
                    "final_target_hit": bool(
                        getattr(problem, "final_target_hit", False)
                    ),
                    "message": record["message"],
                    "archive_nodes": record["archive_nodes"],
                    "graph_edges": record["graph_edges"],
                    "graph_referential_integrity": record[
                        "graph_referential_integrity"
                    ],
                    "phase_sum": int(sum(phase_evaluations.values())),
                    "phase_evaluations_json": json.dumps(
                        phase_evaluations, sort_keys=True
                    ),
                    "checkpoint_fbest_json": json.dumps(
                        checkpoints, sort_keys=True
                    ),
                    "history_monotone": monotone(history),
                    "history_hash": history_hash,
                    "wall_time_seconds": elapsed,
                    "algorithm_metadata_json": json.dumps(
                        record["metadata"], sort_keys=True, allow_nan=True
                    ),
                    "detail_json_gz": "",
                }
            )

            if should_store_trace(
                args.mode, function_index, instance_index
            ):
                detail_path = attempt_root / "details" / f"{problem_id}.json.gz"
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "run_identity": {
                        "run_id": args.run_id,
                        "mode": args.mode,
                        "algorithm": args.algorithm,
                        "problem_id": problem_id,
                        "seed": seed,
                        "budget": budget,
                    },
                    "result": public,
                }
                with gzip.open(detail_path, "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"), allow_nan=True)
                row["detail_json_gz"] = str(detail_path.relative_to(ROOT))

            if record["nfe"] != budget:
                raise RuntimeError(
                    f"Internal budget mismatch: {record['nfe']} != {budget}"
                )
            if int(getattr(problem, "evaluations", 0)) != budget:
                raise RuntimeError(
                    "COCO observer budget mismatch: "
                    f"{getattr(problem, 'evaluations', 0)} != {budget}"
                )
            if not row["history_monotone"]:
                raise RuntimeError("Best-so-far history is not monotone.")

        except Exception as exc:
            elapsed = time.perf_counter() - started
            row.update(
                {
                    "runner_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=40)}",
                    "implementation": "",
                    "implementation_version": "",
                    "options_hash": "",
                    "nfe_internal": 0,
                    "nfe_observer": int(getattr(problem, "evaluations", 0)),
                    "fbest": np.nan,
                    "final_target_hit": False,
                    "message": "",
                    "archive_nodes": 0,
                    "graph_edges": 0,
                    "graph_referential_integrity": False,
                    "phase_sum": 0,
                    "phase_evaluations_json": "{}",
                    "checkpoint_fbest_json": "{}",
                    "history_monotone": False,
                    "history_hash": "",
                    "wall_time_seconds": elapsed,
                    "algorithm_metadata_json": "{}",
                    "detail_json_gz": "",
                }
            )
        rows.append(row)

    del observer
    del suite

    raw_path = attempt_root / "shard_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = sum(row["runner_status"] != "completed" for row in rows)
    metadata.update(
        {
            "status": (
                "TRACK_B_SHARD_COMPLETE" if failures == 0 else "TRACK_B_SHARD_FAILED"
            ),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "failures": failures,
            "raw_results_sha256": sha256_file(raw_path),
        }
    )
    (attempt_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if failures:
        raise SystemExit(f"Track B shard has {failures} failed rows.")

    marker = {
        "status": "TRACK_B_SHARD_COMPLETE",
        "algorithm": args.algorithm,
        "dimension": args.dimension,
        "attempt": args.attempt,
        "rows": len(rows),
        "attempt_root": str(attempt_root.relative_to(ROOT)),
        "raw_results": str(raw_path.relative_to(ROOT)),
        "raw_results_sha256": sha256_file(raw_path),
        "observer_relative": observer_relative.as_posix(),
        "observer_absolute": str(observer_absolute),
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    print("TRACK_B_SHARD_COMPLETE")
    print(f"algorithm={args.algorithm}")
    print(f"dimension={args.dimension}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
