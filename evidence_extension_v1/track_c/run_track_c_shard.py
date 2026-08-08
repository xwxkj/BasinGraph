#!/usr/bin/env python3
"""Run one task–algorithm shard for B21 Track C."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
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

from basingraph_v2.optimizer import minimize_basingraph_v2
from evidence_extension_v1.track_b.modern_baselines import OPTIMIZERS
from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    C1_INSTANCES,
    DEVELOPMENT_INSTANCE,
    DISPLAY_NAMES,
    NIST_DATASETS,
    PAIRED_SEEDS,
    SMOKE_TASKS,
    TARGET_RATIOS,
    normalize_gap,
    seed_for_c1,
    seed_for_nist,
    stable_digest,
    target_values,
    verify_source_identity,
)
from evidence_extension_v1.track_c.nist import make_nist_task
from evidence_extension_v1.track_c.tasks import ScientificTask, make_c1_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--domain", choices=("c1", "c2"), required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_c")
    return parser.parse_args()


def monotone(history: list[tuple[int, float]]) -> bool:
    values = [float(value) for _, value in history]
    return all(values[i] <= values[i - 1] for i in range(1, len(values)))


def first_hit(history: list[tuple[int, float]], target: float, budget: int) -> int:
    for nfe, value in history:
        if float(value) <= float(target):
            return int(nfe)
    return int(budget)


def checkpoint_gaps(task: ScientificTask, history: list[tuple[int, float]]) -> dict[str, float]:
    ordered = sorted((int(n), float(v)) for n, v in history)
    points = sorted(set([1, 3, 10, 30, 100, 300, task.budget_multiplier]))
    out: dict[str, float] = {}
    cursor = 0
    best = float("inf")
    for multiplier in points:
        nfe = min(task.budget, int(multiplier * task.dimension))
        while cursor < len(ordered) and ordered[cursor][0] <= nfe:
            best = min(best, ordered[cursor][1])
            cursor += 1
        out[str(multiplier)] = normalize_gap(best, task.f_ref, task.f_base)
    return out


def run_algorithm(task: ScientificTask, algorithm: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if algorithm == "BasinGraph":
        result = minimize_basingraph_v2(
            objective=task.objective,
            lb=task.lb,
            ub=task.ub,
            max_evals=task.budget,
            seed=seed,
        )
        public = result.to_jsonable()
        record = {
            "implementation": "basingraph_v2.optimizer.minimize_basingraph_v2",
            "implementation_version": result.implementation_version,
            "options_hash": result.options_hash,
            "nfe": int(result.nfe),
            "fbest": float(result.fbest),
            "xbest": np.asarray(result.xbest, dtype=float),
            "history": [(int(a), float(b)) for a, b in result.history],
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

    result = OPTIMIZERS[algorithm](
        objective=task.objective,
        lb=task.lb,
        ub=task.ub,
        max_evals=task.budget,
        seed=seed,
    )
    record = {
        "implementation": result["implementation"],
        "implementation_version": "",
        "options_hash": "",
        "nfe": int(result["nfe"]),
        "fbest": float(result["fbest"]),
        "xbest": np.asarray(result["xbest"], dtype=float),
        "history": [(int(a), float(b)) for a, b in result["history"]],
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
        "xbest": record["xbest"].tolist(),
        "fbest": record["fbest"],
        "nfe": record["nfe"],
        "history": record["history"],
        "message": record["message"],
        "metadata": record["metadata"],
    }
    return record, public


def task_runs(mode: str, domain: str, task_name: str) -> list[tuple[ScientificTask, int, int]]:
    if domain == "c1":
        if task_name not in C1_FAMILIES:
            raise RuntimeError(f"Unknown C1 family: {task_name}")
        if mode == "smoke":
            if (domain, task_name) not in SMOKE_TASKS:
                raise RuntimeError("Task is not registered for smoke.")
            task = make_c1_task(task_name, DEVELOPMENT_INSTANCE)
            return [(task, 0, seed_for_c1(task_name, DEVELOPMENT_INSTANCE, 0))]
        output = []
        for instance in C1_INSTANCES:
            task = make_c1_task(task_name, instance)
            for paired_seed in PAIRED_SEEDS:
                output.append((task, paired_seed, seed_for_c1(task_name, instance, paired_seed)))
        return output

    if task_name not in NIST_DATASETS:
        raise RuntimeError(f"Unknown NIST dataset: {task_name}")
    if mode == "smoke":
        if (domain, task_name) not in SMOKE_TASKS:
            raise RuntimeError("Dataset is not registered for smoke.")
        return [(make_nist_task(task_name), 0, seed_for_nist(task_name, 0))]
    task = make_nist_task(task_name)
    return [(task, paired_seed, seed_for_nist(task_name, paired_seed)) for paired_seed in PAIRED_SEEDS]


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory shard requires --authorize-confirmatory.")
    identity = verify_source_identity(require_clean=True)
    run_root = ROOT / args.result_root / args.run_id
    shard_name = f"{args.domain}_{args.task_name}_{args.algorithm}"
    shard_root = run_root / "shards" / shard_name
    complete_marker = shard_root / "SHARD_COMPLETE.json"
    if complete_marker.is_file():
        print("TRACK_C_SHARD_ALREADY_COMPLETE")
        return
    attempt_root = shard_root / f"attempt_{args.attempt:04d}"
    if attempt_root.exists():
        raise RuntimeError(f"Attempt directory already exists: {attempt_root}")
    attempt_root.mkdir(parents=True)

    metadata = {
        "status": "TRACK_C_SHARD_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "domain": args.domain,
        "task_name": args.task_name,
        "algorithm": args.algorithm,
        "attempt": args.attempt,
        "identity": identity,
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
    }
    (attempt_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for task, paired_seed, seed in task_runs(args.mode, args.domain, args.task_name):
        row: dict[str, Any] = {
            "run_id": args.run_id,
            "mode": args.mode,
            "domain": args.domain,
            "task_name": args.task_name,
            "task_id": task.task_id,
            "instance": task.instance,
            "dimension": task.dimension,
            "algorithm": args.algorithm,
            "algorithm_display": DISPLAY_NAMES[args.algorithm],
            "paired_seed": paired_seed,
            "seed": seed,
            "budget_multiplier": task.budget_multiplier,
            "budget": task.budget,
            "f_ref": task.f_ref,
            "f_base": task.f_base,
            "runner_status": "completed",
            "error": "",
        }
        started = time.perf_counter()
        try:
            record, public = run_algorithm(task, args.algorithm, seed)
            wall_time = time.perf_counter() - started
            history = record["history"]
            targets = target_values(task.f_ref, task.f_base)
            hit_evals = [first_hit(history, target, task.budget) for target in targets]
            final_metrics = task.metrics(record["xbest"])
            normalized = normalize_gap(record["fbest"], task.f_ref, task.f_base)
            phase = record["phase_evaluations"]
            row.update(
                {
                    "implementation": record["implementation"],
                    "implementation_version": record["implementation_version"],
                    "options_hash": record["options_hash"],
                    "nfe": record["nfe"],
                    "fbest": record["fbest"],
                    "normalized_gap": normalized,
                    "message": record["message"],
                    "wall_time_seconds": wall_time,
                    "history_monotone": monotone(history),
                    "history_hash": stable_digest(history),
                    "target_hit_evaluations_json": json.dumps(hit_evals),
                    "checkpoint_normalized_gap_json": json.dumps(checkpoint_gaps(task, history), sort_keys=True),
                    "secondary_metrics_json": json.dumps(final_metrics, sort_keys=True),
                    "task_metadata_json": json.dumps(task.metadata, sort_keys=True),
                    "algorithm_metadata_json": json.dumps(record["metadata"], sort_keys=True, allow_nan=True),
                    "archive_nodes": record["archive_nodes"],
                    "graph_edges": record["graph_edges"],
                    "graph_referential_integrity": record["graph_referential_integrity"],
                    "phase_sum": int(sum(phase.values())),
                    "phase_evaluations_json": json.dumps(phase, sort_keys=True),
                    "detail_json_gz": "",
                }
            )
            if paired_seed == 0:
                detail = attempt_root / "details" / f"{task.task_id}.json.gz"
                detail.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "run_identity": {
                        "run_id": args.run_id,
                        "domain": args.domain,
                        "task_name": args.task_name,
                        "task_id": task.task_id,
                        "algorithm": args.algorithm,
                        "seed": seed,
                        "budget": task.budget,
                    },
                    "reference": {
                        "f_ref": task.f_ref,
                        "f_base": task.f_base,
                        "reference_x": task.reference_x.tolist(),
                        "metadata": task.metadata,
                    },
                    "result": public,
                    "secondary_metrics": final_metrics,
                }
                with gzip.open(detail, "wt", encoding="utf-8") as handle:
                    json.dump(payload, handle, separators=(",", ":"), allow_nan=True)
                row["detail_json_gz"] = str(detail.relative_to(ROOT))
            if record["nfe"] != task.budget:
                raise RuntimeError(f"Budget mismatch: {record['nfe']} != {task.budget}")
            if not row["history_monotone"]:
                raise RuntimeError("Best-so-far history is not monotone.")
            if args.algorithm == "BasinGraph":
                if row["phase_sum"] != task.budget:
                    raise RuntimeError("BasinGraph phase accounting mismatch.")
                if not row["graph_referential_integrity"]:
                    raise RuntimeError("BasinGraph graph integrity failed.")
                if not 1 <= row["archive_nodes"] <= 80:
                    raise RuntimeError("BasinGraph archive capacity failed.")
        except Exception as exc:
            row.update(
                {
                    "runner_status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_time_seconds": time.perf_counter() - started,
                }
            )
            (attempt_root / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        rows.append(row)

    raw_path = attempt_root / "raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    failures = [row for row in rows if row["runner_status"] != "completed"]
    marker = {
        "status": "TRACK_C_SHARD_COMPLETE" if not failures else "TRACK_C_SHARD_FAILED",
        "run_id": args.run_id,
        "mode": args.mode,
        "domain": args.domain,
        "task_name": args.task_name,
        "algorithm": args.algorithm,
        "attempt": args.attempt,
        "rows": len(rows),
        "failures": len(failures),
        "raw_results": str(raw_path.relative_to(ROOT)),
    }
    shard_root.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Track C shard has {len(failures)} failed runs.")
    print("TRACK_C_SHARD_COMPLETE")
    print(json.dumps(marker, indent=2))


if __name__ == "__main__":
    main()
