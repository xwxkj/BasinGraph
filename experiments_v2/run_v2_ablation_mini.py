#!/usr/bin/env python3
"""
Route B Step B5B: BasinGraph v2.0.0-rc1 ablation mini benchmark.

Purpose
-------
Verify that every manuscript-level algorithm module:

1. is controlled by a frozen option;
2. changes actual search behaviour when disabled;
3. preserves strict FE accounting and graph/archive integrity;
4. produces machine-readable evidence tied to the rc1 options hash.

This is engineering validation only. It is not a final manuscript experiment.

Design
------
- 7 deterministic diagnostic tasks;
- 7 variants;
- 5 paired seeds;
- 245 runs;
- no post-hoc task selection.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_FULL_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)

VARIANT_BUILDERS = {
    "Full": lambda base: base,
    "NoGraphGuidance": lambda base: replace(
        base,
        enable_graph_guidance=False,
    ),
    "SingleBracket": lambda base: replace(
        base,
        enable_multibracket=False,
    ),
    "NoFarBasin": lambda base: replace(
        base,
        enable_far_basin=False,
    ),
    "NoGeometryController": lambda base: replace(
        base,
        enable_geometry_controller=False,
    ),
    "NoArchiveFallback": lambda base: replace(
        base,
        enable_archive_fallback=False,
    ),
    "NoFinalPolish": lambda base: replace(
        base,
        enable_final_polish=False,
    ),
}

VARIANT_ORDER = list(VARIANT_BUILDERS)


def _rotation(dimension: int) -> np.ndarray:
    rng = np.random.default_rng(14401 + dimension)
    matrix = rng.standard_normal((dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def task_specifications() -> list[dict[str, Any]]:
    return [
        {
            "task": "ShiftedSphere_D5",
            "dimension": 5,
            "lb": [-5.0] * 5,
            "ub": [5.0] * 5,
            "known_optimum": 0.0,
            "budget": 1200,
        },
        {
            "task": "Rosenbrock_D5",
            "dimension": 5,
            "lb": [-3.0] * 5,
            "ub": [3.0] * 5,
            "known_optimum": 0.0,
            "budget": 1200,
        },
        {
            "task": "ShiftedRastrigin_D5",
            "dimension": 5,
            "lb": [-5.12] * 5,
            "ub": [5.12] * 5,
            "known_optimum": 0.0,
            "budget": 1400,
        },
        {
            "task": "ShiftedAckley_D5",
            "dimension": 5,
            "lb": [-32.768] * 5,
            "ub": [32.768] * 5,
            "known_optimum": 0.0,
            "budget": 1400,
        },
        {
            "task": "RotatedEllipsoid_D10",
            "dimension": 10,
            "lb": [-5.0] * 10,
            "ub": [5.0] * 10,
            "known_optimum": 0.0,
            "budget": 2200,
        },
        {
            "task": "FarBasinDoubleWell_D5",
            "dimension": 5,
            "lb": [-100.0] * 5,
            "ub": [100.0] * 5,
            "known_optimum": 0.0,
            "budget": 1600,
        },
        {
            "task": "BoundaryQuartic_D8",
            "dimension": 8,
            "lb": [-10.0] * 8,
            "ub": [10.0] * 8,
            "known_optimum": 0.0,
            "budget": 1800,
        },
    ]


def evaluate_task(task: str, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    dimension = x.size

    if task == "ShiftedSphere_D5":
        shift = np.asarray([1.2, -0.8, 0.5, -1.3, 1.7])
        return float(np.sum((x - shift) ** 2))

    if task == "Rosenbrock_D5":
        return float(
            np.sum(
                100.0 * (x[1:] - x[:-1] ** 2) ** 2
                + (1.0 - x[:-1]) ** 2
            )
        )

    if task == "ShiftedRastrigin_D5":
        shift = np.asarray([0.7, -1.1, 1.3, -0.4, 0.9])
        z = x - shift
        return float(
            10.0 * dimension
            + np.sum(z * z - 10.0 * np.cos(2.0 * np.pi * z))
        )

    if task == "ShiftedAckley_D5":
        shift = np.asarray([3.0, -2.0, 1.0, -4.0, 2.5])
        z = x - shift
        first = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z * z)))
        second = -np.exp(np.mean(np.cos(2.0 * np.pi * z)))
        return float(first + second + 20.0 + math.e)

    if task == "RotatedEllipsoid_D10":
        shift = np.linspace(-1.0, 1.0, dimension)
        z = _rotation(dimension) @ (x - shift)
        weights = np.logspace(0.0, 6.0, dimension)
        return float(np.sum(weights * z * z) / np.sum(weights))

    if task == "FarBasinDoubleWell_D5":
        a = -60.0
        b = 70.0
        scaled_product = ((x - a) * (x - b) / (b - a)) ** 2
        bias = 1.0e-4 * (x - b) ** 2
        return float(np.sum(scaled_product + bias))

    if task == "BoundaryQuartic_D8":
        target = 9.5 * np.ones(dimension)
        return float(np.sum(((x - target) / 20.0) ** 4))

    raise ValueError(f"Unknown task: {task}")


def stable_digest(payload: Any) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"

    spec = payload["task_spec"]
    variant = payload["variant"]
    seed_index = int(payload["seed_index"])
    seed = int(payload["seed"])
    output_root = Path(payload["output_root"])

    base = BasinGraphOptions()
    options = VARIANT_BUILDERS[variant](base)

    task = str(spec["task"])
    dimension = int(spec["dimension"])
    lb = np.asarray(spec["lb"], dtype=float)
    ub = np.asarray(spec["ub"], dtype=float)
    budget = int(spec["budget"])
    known_optimum = float(spec["known_optimum"])

    center = 0.5 * (lb + ub)
    initial_value = evaluate_task(task, center)

    started = time.perf_counter()
    try:
        result = minimize_basingraph_v2(
            objective=lambda x: evaluate_task(task, x),
            lb=lb,
            ub=ub,
            max_evals=budget,
            seed=seed,
            options=options,
        )
        status = "completed"
        error = ""
    except Exception as exc:
        result = None
        status = "failed"
        error = (
            f"{type(exc).__name__}: {exc}\n"
            + traceback.format_exc(limit=30)
        )

    elapsed = time.perf_counter() - started

    if result is None:
        return {
            "task": task,
            "dimension": dimension,
            "variant": variant,
            "seed_index": seed_index,
            "seed": seed,
            "budget": budget,
            "known_optimum": known_optimum,
            "initial_value": initial_value,
            "implementation_version": "",
            "options_hash": options.stable_hash(),
            "nfe": 0,
            "fbest": np.nan,
            "normalized_gap": np.nan,
            "archive_nodes": 0,
            "graph_edges": 0,
            "phase_evaluations_json": "{}",
            "history_hash": "",
            "graph_hash": "",
            "event_hash": "",
            "trajectory_hash": "",
            "runner_status": status,
            "error": error,
            "wall_time_seconds": elapsed,
            "detail_json_gz": "",
        }

    jsonable = result.to_jsonable()
    active_ids = {node["node_id"] for node in jsonable["archive"]}
    graph_valid = all(
        edge["source_id"] in active_ids and edge["target_id"] in active_ids
        for edge in jsonable["graph_edges"]
    )

    history_hash = stable_digest(jsonable["history"])
    graph_hash = stable_digest(jsonable["graph_edges"])
    event_projection = [
        {
            "nfe": event["nfe"],
            "phase": event["phase"],
            "event": event["event"],
        }
        for event in jsonable["event_log"]
    ]
    event_hash = stable_digest(event_projection)
    trajectory_hash = stable_digest(
        {
            "history_hash": history_hash,
            "graph_hash": graph_hash,
            "event_hash": event_hash,
            "phase_evaluations": jsonable["phase_evaluations"],
        }
    )

    scale = max(abs(initial_value - known_optimum), 1.0e-15)
    normalized_gap = max(result.fbest - known_optimum, 0.0) / scale

    detail_relative = ""
    if seed_index == 0:
        detail_dir = output_root / "details" / task
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_path = detail_dir / f"{variant}.json.gz"
        with gzip.open(detail_path, "wt", encoding="utf-8") as handle:
            json.dump(jsonable, handle, indent=2)
        detail_relative = str(detail_path.relative_to(PROJECT_ROOT))

    return {
        "task": task,
        "dimension": dimension,
        "variant": variant,
        "seed_index": seed_index,
        "seed": seed,
        "budget": budget,
        "known_optimum": known_optimum,
        "initial_value": initial_value,
        "implementation_version": result.implementation_version,
        "options_hash": result.options_hash,
        "nfe": result.nfe,
        "fbest": result.fbest,
        "normalized_gap": normalized_gap,
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "graph_referential_integrity": graph_valid,
        "phase_evaluations_json": json.dumps(
            result.phase_evaluations,
            sort_keys=True,
        ),
        "history_hash": history_hash,
        "graph_hash": graph_hash,
        "event_hash": event_hash,
        "trajectory_hash": trajectory_hash,
        "runner_status": status,
        "error": error,
        "wall_time_seconds": elapsed,
        "detail_json_gz": detail_relative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260619)
    parser.add_argument(
        "--output",
        default="results_v2/ablation_mini_rc1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )

    full_hash = BasinGraphOptions().stable_hash()
    if full_hash != EXPECTED_FULL_HASH:
        raise RuntimeError(
            "Frozen full-options hash mismatch:\n"
            f"expected={EXPECTED_FULL_HASH}\nobserved={full_hash}"
        )

    output_root = PROJECT_ROOT / args.output
    if output_root.exists():
        if not args.force:
            raise RuntimeError(
                f"Output exists: {output_root}; use --force to rebuild."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    variant_specs = {}
    for name, builder in VARIANT_BUILDERS.items():
        options = builder(BasinGraphOptions())
        variant_specs[name] = {
            "options": options.to_jsonable(),
            "options_hash": options.stable_hash(),
        }

    (output_root / "variant_specifications.json").write_text(
        json.dumps(variant_specs, indent=2, sort_keys=True)
    )
    (output_root / "task_specifications.json").write_text(
        json.dumps(task_specifications(), indent=2)
    )

    jobs = []
    for spec in task_specifications():
        for variant in VARIANT_ORDER:
            for seed_index in range(args.seed_count):
                jobs.append(
                    {
                        "task_spec": spec,
                        "variant": variant,
                        "seed_index": seed_index,
                        "seed": args.base_seed + seed_index,
                        "output_root": str(output_root),
                    }
                )

    records = []
    started = time.perf_counter()

    if args.workers <= 1:
        for index, job in enumerate(jobs, start=1):
            records.append(run_one(job))
            if index % 25 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                records.append(future.result())
                if index % 25 == 0 or index == len(futures):
                    print(f"[{index}/{len(futures)}]", flush=True)

    records.sort(
        key=lambda row: (
            row["task"],
            VARIANT_ORDER.index(row["variant"]),
            row["seed_index"],
        )
    )

    raw_path = output_root / "ablation_mini_raw_results.csv"
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "status": "V2_RC1_ABLATION_MINI_RUN_COMPLETE",
        "implementation_version": IMPLEMENTATION_VERSION,
        "full_options_hash": full_hash,
        "tasks": len(task_specifications()),
        "variants": len(VARIANT_ORDER),
        "seeds": args.seed_count,
        "rows": len(records),
        "workers": args.workers,
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "elapsed_seconds": time.perf_counter() - started,
        "raw_results": str(raw_path.relative_to(PROJECT_ROOT)),
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("V2_RC1_ABLATION_MINI_RUN_COMPLETE")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
