#!/usr/bin/env python3
"""Run one algorithm-dimension shard for B21 Track D."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import platform
from pathlib import Path
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import minimize_basingraph_v2  # noqa: E402
from evidence_extension_v1.track_b.modern_baselines import (  # noqa: E402
    OPTIMIZERS,
)
from evidence_extension_v1.track_d.common import (  # noqa: E402
    ALGORITHMS,
    DISPLAY_NAMES,
    FUNCTION_GROUPS,
    MODE_CONFIG,
    OBSERVER_NAMES,
    SUITE_NAME,
    checkpoint_values,
    normalize_ru_maxrss_mb,
    parse_problem_id,
    seed_for,
    should_store_trace,
    stable_digest,
    suite_options,
    verify_source_identity,
)


class TimedObjective:
    """Measure cumulative time spent in the observed objective call."""

    def __init__(self, objective: Callable[[np.ndarray], Any]) -> None:
        self.objective = objective
        self.calls = 0
        self.elapsed_seconds = 0.0

    def __call__(self, x: np.ndarray) -> Any:
        started = time.perf_counter()
        try:
            return self.objective(x)
        finally:
            self.elapsed_seconds += time.perf_counter() - started
            self.calls += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "overhead", "confirmatory"), required=True
    )
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument(
        "--dimension", type=int, choices=(40, 80, 160, 320), required=True
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_d")
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
    return all(
        values[index] <= values[index - 1]
        for index in range(1, len(values))
    )


def run_algorithm(
    algorithm: str,
    objective: TimedObjective,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if algorithm == "BasinGraph":
        result = minimize_basingraph_v2(
            objective=objective,
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
        objective=objective,
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


def serialized_sizes(public: dict[str, Any]) -> tuple[int, int, int]:
    history = public.get("history", [])
    core = dict(public)
    core.pop("history", None)
    core_bytes = len(
        json.dumps(core, separators=(",", ":"), allow_nan=True).encode("utf-8")
    )
    history_bytes = len(
        json.dumps(history, separators=(",", ":"), allow_nan=True).encode("utf-8")
    )
    return core_bytes, history_bytes, core_bytes + history_bytes


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
        print("TRACK_D_SHARD_ALREADY_COMPLETE")
        print(complete_marker)
        return

    attempt_root = shard_root / f"attempt_{args.attempt:04d}"
    if attempt_root.exists():
        raise RuntimeError(f"Attempt directory already exists: {attempt_root}")
    attempt_root.mkdir(parents=True)

    import cocoex

    suite_instance, suite_options_text = suite_options(args.mode, args.dimension)
    suite = cocoex.Suite(SUITE_NAME, suite_instance, suite_options_text)
    suite_ids = list(suite.ids())
    parsed = [parse_problem_id(problem_id) for problem_id in suite_ids]
    expected_problem_count = len(config["functions"]) * len(config["instances"])
    if len(suite_ids) != expected_problem_count:
        raise RuntimeError(
            f"Unexpected shard problem count: {len(suite_ids)} != "
            f"{expected_problem_count}"
        )
    if sorted({f for f, _, _ in parsed}) != config["functions"]:
        raise RuntimeError("Function set mismatch in shard preflight.")
    if sorted({i for _, i, _ in parsed}) != config["instances"]:
        raise RuntimeError("Instance set mismatch in shard preflight.")
    if sorted({d for _, _, d in parsed}) != [args.dimension]:
        raise RuntimeError("Dimension mismatch in shard preflight.")

    observer_relative = (
        Path("exdata")
        / "b21_track_d"
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
        f'algorithm_info: "B21 Track D {args.mode}; frozen large-scale evidence"',
    )

    metadata = {
        "status": "TRACK_D_SHARD_STARTED",
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
    failures: list[dict[str, str]] = []
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

        timed_objective = TimedObjective(problem)
        started = time.perf_counter()
        try:
            record, public = run_algorithm(
                args.algorithm,
                timed_objective,
                lb,
                ub,
                budget,
                seed,
            )
            total_wall = time.perf_counter() - started
            objective_wall = float(timed_objective.elapsed_seconds)
            optimizer_overhead = max(0.0, total_wall - objective_wall)
            history = record["history"]
            history_hash = stable_digest(history)
            phase_evaluations = record["phase_evaluations"]
            checkpoints = checkpoint_values(
                history,
                mode=args.mode,
                dimension=dimension,
                budget=budget,
            )
            peak_rss_mb = normalize_ru_maxrss_mb(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                platform.system(),
            )
            core_bytes, history_bytes, estimated_full_bytes = serialized_sizes(public)
            row.update(
                {
                    "implementation": record["implementation"],
                    "implementation_version": record["implementation_version"],
                    "options_hash": record["options_hash"],
                    "nfe_internal": record["nfe"],
                    "nfe_timed_objective": int(timed_objective.calls),
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
                    "total_wall_time_seconds": total_wall,
                    "objective_wall_time_seconds": objective_wall,
                    "optimizer_overhead_seconds": optimizer_overhead,
                    "optimizer_overhead_microseconds_per_evaluation": (
                        optimizer_overhead / budget * 1e6
                    ),
                    "objective_microseconds_per_evaluation": (
                        objective_wall / budget * 1e6
                    ),
                    "process_peak_rss_mb_before_serialization": peak_rss_mb,
                    "serialized_result_core_bytes": core_bytes,
                    "serialized_history_bytes": history_bytes,
                    "estimated_full_result_json_bytes": estimated_full_bytes,
                    "algorithm_metadata_json": json.dumps(
                        record["metadata"], sort_keys=True, allow_nan=True
                    ),
                    "detail_json_gz": "",
                }
            )

            if should_store_trace(args.mode, function_index, instance_index):
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
                    "timing": {
                        "total_wall_time_seconds": total_wall,
                        "objective_wall_time_seconds": objective_wall,
                        "optimizer_overhead_seconds": optimizer_overhead,
                        "process_peak_rss_mb_before_serialization": peak_rss_mb,
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
            if timed_objective.calls != budget:
                raise RuntimeError(
                    f"Timed-objective budget mismatch: {timed_objective.calls} != "
                    f"{budget}"
                )
            if int(getattr(problem, "evaluations", 0)) != budget:
                raise RuntimeError(
                    "COCO observer budget mismatch: "
                    f"{getattr(problem, 'evaluations', 0)} != {budget}"
                )
            if not row["history_monotone"]:
                raise RuntimeError("Best-so-far history is not monotone.")

        except Exception as exc:
            row.update(
                {
                    "runner_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "total_wall_time_seconds": time.perf_counter() - started,
                    "objective_wall_time_seconds": timed_objective.elapsed_seconds,
                    "nfe_timed_objective": timed_objective.calls,
                    "nfe_observer": int(getattr(problem, "evaluations", 0)),
                }
            )
            failures.append(
                {
                    "problem_id": problem_id,
                    "error": row["error"],
                    "traceback": traceback.format_exc(),
                }
            )
        rows.append(row)
        if hasattr(problem, "free"):
            problem.free()

    raw_path = attempt_root / "raw_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        (attempt_root / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Track D shard had {len(failures)} failed runs.")

    final_peak_rss_mb = normalize_ru_maxrss_mb(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform.system(),
    )
    marker = {
        "status": "TRACK_D_SHARD_COMPLETE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "algorithm": args.algorithm,
        "dimension": args.dimension,
        "attempt": args.attempt,
        "rows": len(rows),
        "raw_results": str(raw_path.relative_to(ROOT)),
        "observer_relative": observer_relative.as_posix(),
        "observer_absolute": str(observer_absolute),
        "process_peak_rss_mb": final_peak_rss_mb,
        "source_identity_sha256": identity["source_identity_sha256"],
    }
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")

    print("TRACK_D_SHARD_COMPLETE")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
