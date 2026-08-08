#!/usr/bin/env python3
"""Run one variant-dimension shard for B21 Track A."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import json
import os
from pathlib import Path
import platform
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
from evidence_extension_v1.track_a.common import (  # noqa: E402
    FUNCTION_GROUPS,
    MODE_CONFIG,
    VARIANT_OBSERVER_NAMES,
    VARIANT_ORDER,
    checkpoint_values,
    options_for_variant,
    parse_problem_id,
    seed_for,
    set_single_thread_environment,
    sha256_file,
    should_store_trace,
    stable_digest,
    suite_options,
    verify_source_identity,
)

ROW_FIELDS = [
    "run_id", "mode", "partition", "problem_id", "function_index",
    "function_group_id", "function_group", "instance_index", "dimension",
    "variant", "observer_algorithm_name", "seed", "budget", "runner_status",
    "error", "wall_time_seconds", "implementation_version", "options_hash",
    "nfe_internal", "nfe_observer", "phase_sum", "fbest",
    "final_target_hit", "archive_nodes", "graph_edges",
    "graph_referential_integrity", "history_monotone",
    "phase_evaluations_json", "checkpoint_fbest_json", "diagnostics_json",
    "event_counts_json", "history_hash", "graph_hash", "event_hash",
    "trajectory_hash", "trace_json_gz",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--variant", choices=VARIANT_ORDER, required=True)
    parser.add_argument("--dimension", type=int, choices=(5, 20), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--result-root", default="results_b21/track_a")
    parser.add_argument("--authorize-confirmatory", action="store_true")
    return parser.parse_args()


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_row(path: Path, row: dict[str, Any]) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def history_monotone(history: list[tuple[int, float]]) -> bool:
    values = [float(value) for _, value in history]
    return all(
        later <= earlier + 1e-14
        for earlier, later in zip(values[:-1], values[1:])
    )


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory shard requires --authorize-confirmatory.")
    set_single_thread_environment()
    identity = verify_source_identity(require_clean=True)
    config = MODE_CONFIG[args.mode]
    if args.dimension not in config["dimensions"]:
        raise RuntimeError("Unexpected Track A dimension.")

    options = options_for_variant(args.variant)
    variant_hash = options.stable_hash()
    observer_name = VARIANT_OBSERVER_NAMES[args.variant]

    run_root = ROOT / args.result_root / args.run_id
    shard_id = f"{args.variant}_d{args.dimension}"
    shard_root = run_root / "shards" / shard_id
    if shard_root.exists():
        raise RuntimeError(f"Shard output already exists: {shard_root}")
    shard_root.mkdir(parents=True)

    observer_relative = (
        f"b21_track_a/{args.run_id}/{args.mode}/"
        f"{args.variant}/d{args.dimension}/attempt_{args.attempt:02d}"
    )
    observer_absolute = ROOT / "exdata" / observer_relative
    if observer_absolute.exists():
        raise RuntimeError(f"Observer folder already exists: {observer_absolute}")

    suite_instance, suite_options_text = suite_options(args.mode, args.dimension)
    metadata = {
        "status": "TRACK_A_SHARD_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "mode": args.mode,
        "partition": config["partition"],
        "confirmatory": bool(config["confirmatory"]),
        "variant": args.variant,
        "variant_hash": variant_hash,
        "dimension": args.dimension,
        "attempt": args.attempt,
        "base_seed": args.base_seed,
        "budget_multiplier": config["budget_multiplier"],
        "suite_instance": suite_instance,
        "suite_options": suite_options_text,
        "observer_algorithm_name": observer_name,
        "observer_relative": observer_relative,
        "identity": identity,
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
    }
    metadata_path = shard_root / "shard_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    import cocoex

    suite = cocoex.Suite("bbob", suite_instance, suite_options_text)
    ids = list(suite.ids())
    expected_count = len(config["functions"]) * len(config["instances"])
    if len(ids) != expected_count:
        raise RuntimeError(
            f"Unexpected shard problem count: {len(ids)} != {expected_count}"
        )

    observer = cocoex.Observer(
        "bbob",
        (
            f"result_folder: {observer_relative} "
            f"algorithm_name: {observer_name} "
            f'algorithm_info: "B21 Track A {args.mode}; '
            f'variant={args.variant}; immutable BasinGraph 2.0.0-rc1"'
        ),
    )

    partial_path = shard_root / "shard_results.partial.csv"
    rows = []
    shard_started = time.perf_counter()

    try:
        for problem in suite:
            problem.observe_with(observer)
            problem_id = str(problem.id)
            function_index, instance_index, dimension = parse_problem_id(problem_id)
            if function_index not in config["functions"]:
                raise RuntimeError(f"Unexpected function: {problem_id}")
            if instance_index not in config["instances"]:
                raise RuntimeError(f"Unexpected instance: {problem_id}")
            if dimension != args.dimension:
                raise RuntimeError(f"Unexpected dimension: {problem_id}")

            group_id, group_name = FUNCTION_GROUPS[function_index]
            seed = seed_for(
                args.base_seed, function_index, dimension, instance_index
            )
            budget = int(config["budget_multiplier"] * dimension)
            started = time.perf_counter()
            row = {
                "run_id": args.run_id,
                "mode": args.mode,
                "partition": config["partition"],
                "problem_id": problem_id,
                "function_index": function_index,
                "function_group_id": group_id,
                "function_group": group_name,
                "instance_index": instance_index,
                "dimension": dimension,
                "variant": args.variant,
                "observer_algorithm_name": observer_name,
                "seed": seed,
                "budget": budget,
                "runner_status": "completed",
                "error": "",
            }

            try:
                result = minimize_basingraph_v2(
                    objective=problem,
                    lb=np.asarray(problem.lower_bounds, dtype=float),
                    ub=np.asarray(problem.upper_bounds, dtype=float),
                    max_evals=budget,
                    seed=seed,
                    options=options,
                )
                payload = result.to_jsonable()
                active_ids = {
                    int(node["node_id"]) for node in payload["archive"]
                }
                graph_valid = all(
                    int(edge["source_id"]) in active_ids
                    and int(edge["target_id"]) in active_ids
                    for edge in payload["graph_edges"]
                )
                event_counts = Counter(
                    str(event["event"]) for event in payload["event_log"]
                )
                event_projection = [
                    {
                        "nfe": event["nfe"],
                        "phase": event["phase"],
                        "event": event["event"],
                    }
                    for event in payload["event_log"]
                ]
                history_hash = stable_digest(payload["history"])
                graph_hash = stable_digest(payload["graph_edges"])
                event_hash = stable_digest(event_projection)
                trajectory_hash = stable_digest({
                    "history_hash": history_hash,
                    "graph_hash": graph_hash,
                    "event_hash": event_hash,
                    "phase_evaluations": payload["phase_evaluations"],
                })
                trace_relative = ""
                if should_store_trace(args.mode, function_index, instance_index):
                    trace_dir = shard_root / "traces"
                    trace_dir.mkdir(exist_ok=True)
                    trace_path = trace_dir / f"{problem_id}.json.gz"
                    with gzip.open(
                        trace_path,
                        "wt",
                        encoding="utf-8",
                        compresslevel=6,
                    ) as handle:
                        json.dump(payload, handle, separators=(",", ":"))
                    trace_relative = str(trace_path.relative_to(ROOT))

                row.update({
                    "wall_time_seconds": time.perf_counter() - started,
                    "implementation_version": result.implementation_version,
                    "options_hash": result.options_hash,
                    "nfe_internal": int(result.nfe),
                    "nfe_observer": int(problem.evaluations),
                    "phase_sum": int(sum(result.phase_evaluations.values())),
                    "fbest": float(result.fbest),
                    "final_target_hit": bool(
                        getattr(problem, "final_target_hit", False)
                    ),
                    "archive_nodes": len(result.archive),
                    "graph_edges": len(result.graph_edges),
                    "graph_referential_integrity": graph_valid,
                    "history_monotone": history_monotone(result.history),
                    "phase_evaluations_json": json.dumps(
                        result.phase_evaluations,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "checkpoint_fbest_json": json.dumps(
                        checkpoint_values(
                            result.history,
                            mode=args.mode,
                            dimension=dimension,
                            budget=budget,
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "diagnostics_json": json.dumps(
                        payload["diagnostics"],
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "event_counts_json": json.dumps(
                        event_counts,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "history_hash": history_hash,
                    "graph_hash": graph_hash,
                    "event_hash": event_hash,
                    "trajectory_hash": trajectory_hash,
                    "trace_json_gz": trace_relative,
                })
            except Exception as exc:
                row.update({
                    "wall_time_seconds": time.perf_counter() - started,
                    "runner_status": "failed",
                    "error": (
                        f"{type(exc).__name__}: {exc}\n"
                        + traceback.format_exc(limit=40)
                    ),
                    "implementation_version": "",
                    "options_hash": variant_hash,
                    "nfe_internal": 0,
                    "nfe_observer": int(getattr(problem, "evaluations", 0)),
                    "phase_sum": 0,
                    "fbest": np.nan,
                    "final_target_hit": False,
                    "archive_nodes": 0,
                    "graph_edges": 0,
                    "graph_referential_integrity": False,
                    "history_monotone": False,
                    "phase_evaluations_json": "{}",
                    "checkpoint_fbest_json": "{}",
                    "diagnostics_json": "{}",
                    "event_counts_json": "{}",
                    "history_hash": "",
                    "graph_hash": "",
                    "event_hash": "",
                    "trajectory_hash": "",
                    "trace_json_gz": "",
                })

            append_row(partial_path, row)
            rows.append(row)
            if row["runner_status"] != "completed":
                raise RuntimeError(f"Track A problem failed: {problem_id}")
    finally:
        del observer
        del suite

    final_path = shard_root / "shard_results.csv"
    partial_path.replace(final_path)
    checks = {
        "expected_rows": len(rows) == expected_count,
        "all_completed": all(
            row["runner_status"] == "completed" for row in rows
        ),
        "implementation_version": all(
            row["implementation_version"] == "2.0.0-rc1" for row in rows
        ),
        "options_hash": all(
            row["options_hash"] == variant_hash for row in rows
        ),
        "budget_accounting": all(
            int(row["nfe_internal"]) == int(row["budget"])
            and int(row["nfe_observer"]) == int(row["budget"])
            for row in rows
        ),
        "phase_accounting": all(
            int(row["phase_sum"]) == int(row["budget"]) for row in rows
        ),
        "archive_capacity": all(
            1 <= int(row["archive_nodes"]) <= 80 for row in rows
        ),
        "graph_integrity": all(
            bool(row["graph_referential_integrity"]) for row in rows
        ),
        "history_monotone": all(
            bool(row["history_monotone"]) for row in rows
        ),
    }
    validation = {
        "status": (
            "TRACK_A_SHARD_VALIDATION_OK"
            if all(checks.values())
            else "TRACK_A_SHARD_VALIDATION_FAILED"
        ),
        "checks": checks,
        "rows": len(rows),
        "expected_rows": expected_count,
        "raw_results_sha256": sha256_file(final_path),
    }
    (shard_root / "shard_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    metadata["status"] = "TRACK_A_SHARD_COMPLETE"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = time.perf_counter() - shard_started
    metadata["rows"] = len(rows)
    metadata["observer_absolute"] = str(observer_absolute)
    metadata["raw_results_sha256"] = sha256_file(final_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    complete = {
        "status": validation["status"],
        "run_id": args.run_id,
        "shard_id": shard_id,
        "variant": args.variant,
        "dimension": args.dimension,
        "attempt": args.attempt,
        "observer_relative": observer_relative,
        "observer_absolute": str(observer_absolute),
        "rows": len(rows),
        "raw_results_sha256": sha256_file(final_path),
    }
    (shard_root / "SHARD_COMPLETE.json").write_text(
        json.dumps(complete, indent=2), encoding="utf-8"
    )
    write_manifest(shard_root)

    print(validation["status"])
    print(f"Shard: {shard_id}")
    print(f"Rows: {len(rows)}")
    print(f"Observer: {observer_absolute}")
    print(f"Output: {shard_root}")
    if validation["status"].endswith("FAILED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
