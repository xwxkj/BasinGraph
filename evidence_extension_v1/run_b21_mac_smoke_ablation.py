#!/usr/bin/env python3
"""Run the first B21 macOS smoke/ablation engineering batch.

This runner is deliberately separate from the original prospective COCO/BBOB
and CUTEst evidence. It uses the immutable result-bearing BasinGraph source and
writes a self-auditing, resumable result directory.
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
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import (  # noqa: E402
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_FULL_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
PROTOCOL = (
    "protocols/evidence_extension_v1/"
    "B21_MAC_SMOKE_ABLATION_PROTOCOL.md"
)
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
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
    "NoCenterLocal": lambda base: replace(
        base,
        center_local_max_dim=-1,
        local_mode_min_score=float("inf"),
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
            "stratum": "smooth_local",
        },
        {
            "task": "Rosenbrock_D5",
            "dimension": 5,
            "lb": [-3.0] * 5,
            "ub": [3.0] * 5,
            "known_optimum": 0.0,
            "budget": 1200,
            "stratum": "curved_valley",
        },
        {
            "task": "ShiftedRastrigin_D5",
            "dimension": 5,
            "lb": [-5.12] * 5,
            "ub": [5.12] * 5,
            "known_optimum": 0.0,
            "budget": 1400,
            "stratum": "multimodal",
        },
        {
            "task": "ShiftedAckley_D5",
            "dimension": 5,
            "lb": [-32.768] * 5,
            "ub": [32.768] * 5,
            "known_optimum": 0.0,
            "budget": 1400,
            "stratum": "multimodal",
        },
        {
            "task": "RotatedEllipsoid_D10",
            "dimension": 10,
            "lb": [-5.0] * 10,
            "ub": [5.0] * 10,
            "known_optimum": 0.0,
            "budget": 2200,
            "stratum": "anisotropic",
        },
        {
            "task": "FarBasinDoubleWell_D5",
            "dimension": 5,
            "lb": [-100.0] * 5,
            "ub": [100.0] * 5,
            "known_optimum": 0.0,
            "budget": 1600,
            "stratum": "far_basin",
        },
        {
            "task": "BoundaryQuartic_D8",
            "dimension": 8,
            "lb": [-10.0] * 8,
            "ub": [10.0] * 8,
            "known_optimum": 0.0,
            "budget": 1800,
            "stratum": "boundary",
        },
        {
            "task": "HighDimSphere_D25",
            "dimension": 25,
            "lb": [-5.0] * 25,
            "ub": [5.0] * 25,
            "known_optimum": 0.0,
            "budget": 1800,
            "stratum": "controller_activation",
            "smoke_only": True,
        },
    ]


def task_specs_for_mode(mode: str) -> list[dict[str, Any]]:
    specs = task_specifications()
    if mode == "smoke":
        selected = {
            "ShiftedSphere_D5",
            "ShiftedRastrigin_D5",
            "FarBasinDoubleWell_D5",
            "HighDimSphere_D25",
        }
        return [spec for spec in specs if spec["task"] in selected]
    return [spec for spec in specs if not spec.get("smoke_only", False)]


def evaluate_task(task: str, x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    dimension = x.size

    if task == "ShiftedSphere_D5":
        shift = np.asarray([1.2, -0.8, 0.5, -1.3, 1.7])
        return float(np.sum((x - shift) ** 2))

    if task == "HighDimSphere_D25":
        shift = np.linspace(-1.5, 1.5, dimension)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_single_thread_environment() -> None:
    for key in THREAD_ENV:
        os.environ[key] = "1"


def run_one(payload: dict[str, Any]) -> dict[str, Any]:
    set_single_thread_environment()

    spec = payload["task_spec"]
    variant = str(payload["variant"])
    seed_index = int(payload["seed_index"])
    seed = int(payload["seed"])
    output_root = Path(payload["output_root"])

    options = VARIANT_BUILDERS[variant](BasinGraphOptions())
    task = str(spec["task"])
    dimension = int(spec["dimension"])
    lb = np.asarray(spec["lb"], dtype=float)
    ub = np.asarray(spec["ub"], dtype=float)
    budget = int(spec["budget"])
    known_optimum = float(spec["known_optimum"])
    initial_value = evaluate_task(task, 0.5 * (lb + ub))

    started = time.perf_counter()
    try:
        result = minimize_basingraph_v2(
            objective=lambda point: evaluate_task(task, point),
            lb=lb,
            ub=ub,
            max_evals=budget,
            seed=seed,
            options=options,
        )
        status = "completed"
        error = ""
    except Exception as exc:  # pragma: no cover - retained for audit logs
        result = None
        status = "failed"
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=40)}"
    elapsed = time.perf_counter() - started

    base_row = {
        "task": task,
        "stratum": str(spec["stratum"]),
        "dimension": dimension,
        "variant": variant,
        "seed_index": seed_index,
        "seed": seed,
        "budget": budget,
        "known_optimum": known_optimum,
        "initial_value": initial_value,
        "options_hash": options.stable_hash(),
        "runner_status": status,
        "error": error,
        "wall_time_seconds": elapsed,
    }

    if result is None:
        return {
            **base_row,
            "implementation_version": "",
            "nfe": 0,
            "fbest": np.nan,
            "normalized_gap": np.nan,
            "archive_nodes": 0,
            "graph_edges": 0,
            "graph_referential_integrity": False,
            "phase_sum": 0,
            "phase_evaluations_json": "{}",
            "history_hash": "",
            "graph_hash": "",
            "event_hash": "",
            "trajectory_hash": "",
            "detail_json_gz": "",
        }

    jsonable = result.to_jsonable()
    active_ids = {node["node_id"] for node in jsonable["archive"]}
    graph_valid = all(
        edge["source_id"] in active_ids and edge["target_id"] in active_ids
        for edge in jsonable["graph_edges"]
    )
    phase_sum = int(sum(jsonable["phase_evaluations"].values()))
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
    normalized_gap = max(float(result.fbest) - known_optimum, 0.0) / scale

    detail_relative = ""
    if seed_index == 0:
        detail_dir = output_root / "details" / task
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_path = detail_dir / f"{variant}.json.gz"
        with gzip.open(detail_path, "wt", encoding="utf-8") as handle:
            json.dump(jsonable, handle, indent=2)
        detail_relative = str(detail_path.relative_to(PROJECT_ROOT))

    return {
        **base_row,
        "implementation_version": result.implementation_version,
        "nfe": int(result.nfe),
        "fbest": float(result.fbest),
        "normalized_gap": float(normalized_gap),
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "graph_referential_integrity": bool(graph_valid),
        "phase_sum": phase_sum,
        "phase_evaluations_json": json.dumps(
            result.phase_evaluations,
            sort_keys=True,
        ),
        "history_hash": history_hash,
        "graph_hash": graph_hash,
        "event_hash": event_hash,
        "trajectory_hash": trajectory_hash,
        "detail_json_gz": detail_relative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "full-mini"),
        default="smoke",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260807)
    parser.add_argument("--output", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def default_seed_count(mode: str) -> int:
    return {"smoke": 1, "pilot": 2, "full-mini": 5}[mode]


def default_workers() -> int:
    cpus = os.cpu_count() or 1
    return max(1, min(4, cpus - 1 if cpus > 1 else 1))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("No records to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def phase_count(row: pd.Series, phase: str) -> int:
    mapping = json.loads(row["phase_evaluations_json"])
    return int(mapping.get(phase, 0))


def make_summaries(df: pd.DataFrame, output_root: Path) -> dict[str, Any]:
    completed = df[df["runner_status"] == "completed"].copy()
    completed["rank"] = completed.groupby(
        ["task", "seed_index"]
    )["normalized_gap"].rank(method="average", ascending=True)
    completed["win"] = completed.groupby(
        ["task", "seed_index"]
    )["normalized_gap"].transform(
        lambda values: np.isclose(
            values,
            values.min(),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    variant_summary = (
        completed.groupby("variant")
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            wins=("win", "sum"),
            median_normalized_gap=("normalized_gap", "median"),
            mean_archive_nodes=("archive_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            runs=("task", "count"),
        )
        .reset_index()
    )
    variant_summary["variant_order"] = variant_summary["variant"].map(
        {name: index for index, name in enumerate(VARIANT_ORDER)}
    )
    variant_summary.sort_values(
        ["mean_rank", "variant_order"],
        inplace=True,
    )
    variant_summary.drop(columns=["variant_order"], inplace=True)
    variant_summary.to_csv(output_root / "variant_summary.csv", index=False)

    full = completed[completed["variant"] == "Full"][
        [
            "task",
            "seed_index",
            "fbest",
            "normalized_gap",
            "trajectory_hash",
            "phase_evaluations_json",
        ]
    ].rename(
        columns={
            "fbest": "full_fbest",
            "normalized_gap": "full_normalized_gap",
            "trajectory_hash": "full_trajectory_hash",
            "phase_evaluations_json": "full_phase_evaluations_json",
        }
    )

    difference_rows = []
    for variant in VARIANT_ORDER[1:]:
        paired = completed[completed["variant"] == variant].merge(
            full,
            on=["task", "seed_index"],
            how="inner",
            validate="one_to_one",
        )
        trajectory_different = paired["trajectory_hash"] != paired["full_trajectory_hash"]
        final_different = ~np.isclose(
            paired["fbest"],
            paired["full_fbest"],
            rtol=1e-12,
            atol=1e-14,
        )
        phase_different = (
            paired["phase_evaluations_json"]
            != paired["full_phase_evaluations_json"]
        )
        difference_rows.append(
            {
                "variant": variant,
                "paired_runs": len(paired),
                "trajectory_different_runs": int(trajectory_different.sum()),
                "phase_allocation_different_runs": int(phase_different.sum()),
                "final_value_different_runs": int(final_different.sum()),
                "median_gap_difference_vs_full": float(
                    np.median(
                        paired["normalized_gap"]
                        - paired["full_normalized_gap"]
                    )
                ),
            }
        )
    differences = pd.DataFrame(difference_rows)
    differences.to_csv(output_root / "paired_differences_vs_full.csv", index=False)

    return {
        "variant_summary": variant_summary,
        "differences": differences,
    }


def validate(
    df: pd.DataFrame,
    *,
    expected_rows: int,
    seed_count: int,
    summaries: dict[str, Any],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any = None) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), "details": details}
        )

    record("expected_rows", len(df) == expected_rows, {"expected": expected_rows, "observed": len(df)})
    record("all_runs_completed", bool((df["runner_status"] == "completed").all()))
    record("implementation_version", bool((df["implementation_version"] == EXPECTED_IMPLEMENTATION).all()))
    record("all_budgets_exhausted", bool((df["nfe"] == df["budget"]).all()))
    record("phase_accounting_exact", bool((df["phase_sum"] == df["nfe"]).all()))
    record("graph_referential_integrity", bool(df["graph_referential_integrity"].astype(bool).all()))
    record("archive_capacity", bool(((df["archive_nodes"] >= 1) & (df["archive_nodes"] <= 80)).all()))
    record("seed_count", df["seed_index"].nunique() == seed_count)
    record("variant_set", set(df["variant"].unique()) == set(VARIANT_ORDER))

    full_hashes = df.loc[df["variant"] == "Full", "options_hash"].unique()
    record(
        "full_options_hash",
        len(full_hashes) == 1 and full_hashes[0] == EXPECTED_FULL_HASH,
        full_hashes.tolist(),
    )
    variant_hash_counts = df.groupby("variant")["options_hash"].nunique()
    record("one_hash_per_variant", bool((variant_hash_counts == 1).all()), variant_hash_counts.to_dict())
    record(
        "unique_variant_hashes",
        df.groupby("variant")["options_hash"].first().nunique() == len(VARIANT_ORDER),
    )

    disabled = {
        "NoFarBasin": "far_basin",
        "NoArchiveFallback": "archive_fallback",
        "NoFinalPolish": "final_polish",
        "NoCenterLocal": "center_local",
    }
    for variant, phase in disabled.items():
        subset = df[df["variant"] == variant]
        values = [phase_count(row, phase) for _, row in subset.iterrows()]
        record(
            f"disabled_phase_zero::{variant}",
            all(value == 0 for value in values),
            {"phase": phase, "values": values},
        )

    differences: pd.DataFrame = summaries["differences"]
    for _, row in differences.iterrows():
        record(
            f"trajectory_changed::{row['variant']}",
            int(row["trajectory_different_runs"]) > 0,
            row.to_dict(),
        )

    passed = all(item["passed"] for item in checks)
    variant_summary: pd.DataFrame = summaries["variant_summary"]
    return {
        "status": (
            "B21_MAC_SMOKE_ABLATION_VALIDATION_OK"
            if passed
            else "B21_MAC_SMOKE_ABLATION_VALIDATION_FAILED"
        ),
        "engineering_only": True,
        "confirmatory_evidence": False,
        "checks_total": len(checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_failed": sum(not item["passed"] for item in checks),
        "best_mean_rank_variant": str(variant_summary.iloc[0]["variant"]),
        "best_mean_rank": float(variant_summary.iloc[0]["mean_rank"]),
        "checks": checks,
    }


def write_manifest(output_root: Path) -> None:
    manifest_path = output_root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_single_thread_environment()

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

    seed_count = args.seed_count or default_seed_count(args.mode)
    workers = args.workers or default_workers()
    specs = task_specs_for_mode(args.mode)
    output_relative = args.output or f"results_b21/mac_{args.mode}_ablation"
    output_root = PROJECT_ROOT / output_relative
    raw_path = output_root / "raw_results.csv"

    if output_root.exists() and args.force:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variant_specs = {}
    for name, builder in VARIANT_BUILDERS.items():
        options = builder(BasinGraphOptions())
        variant_specs[name] = {
            "options": options.to_jsonable(),
            "options_hash": options.stable_hash(),
        }
    (output_root / "variant_specifications.json").write_text(
        json.dumps(variant_specs, indent=2, sort_keys=True, allow_nan=True),
        encoding="utf-8",
    )
    (output_root / "task_specifications.json").write_text(
        json.dumps(specs, indent=2),
        encoding="utf-8",
    )

    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if raw_path.is_file() and not args.no_resume:
        for row in pd.read_csv(raw_path).to_dict(orient="records"):
            key = (str(row["task"]), str(row["variant"]), int(row["seed_index"]))
            existing[key] = row

    jobs = []
    for spec in specs:
        for variant in VARIANT_ORDER:
            for seed_index in range(seed_count):
                key = (str(spec["task"]), variant, seed_index)
                previous = existing.get(key)
                if previous and previous.get("runner_status") == "completed":
                    continue
                jobs.append(
                    {
                        "task_spec": spec,
                        "variant": variant,
                        "seed_index": seed_index,
                        "seed": args.base_seed + seed_index,
                        "output_root": str(output_root),
                    }
                )

    run_id = (
        f"b21_{args.mode}_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    print(f"B21 mode: {args.mode}")
    print(f"Output: {output_root}")
    print(f"Workers: {workers}")
    print(f"Pending jobs: {len(jobs)}")

    new_records = []
    if workers <= 1:
        for index, job in enumerate(jobs, start=1):
            new_records.append(run_one(job))
            if index % 10 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}]", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), start=1):
                new_records.append(future.result())
                if index % 10 == 0 or index == len(futures):
                    print(f"[{index}/{len(futures)}]", flush=True)

    for row in new_records:
        key = (str(row["task"]), str(row["variant"]), int(row["seed_index"]))
        existing[key] = row

    records = list(existing.values())
    records.sort(
        key=lambda row: (
            str(row["task"]),
            VARIANT_ORDER.index(str(row["variant"])),
            int(row["seed_index"]),
        )
    )
    write_csv(raw_path, records)

    df = pd.DataFrame(records)
    summaries = make_summaries(df, output_root)
    expected_rows = len(specs) * len(VARIANT_ORDER) * seed_count
    validation = validate(
        df,
        expected_rows=expected_rows,
        seed_count=seed_count,
        summaries=summaries,
    )
    (output_root / "validation_report.json").write_text(
        json.dumps(validation, indent=2),
        encoding="utf-8",
    )

    metadata = {
        "status": validation["status"],
        "run_id": run_id,
        "mode": args.mode,
        "engineering_only": True,
        "confirmatory_evidence": False,
        "protocol": PROTOCOL,
        "repository_commit": git_commit(),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "full_options_hash": EXPECTED_FULL_HASH,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "workers": workers,
        "seed_count": seed_count,
        "base_seed": args.base_seed,
        "tasks": [spec["task"] for spec in specs],
        "variants": VARIANT_ORDER,
        "expected_rows": expected_rows,
        "observed_rows": len(df),
        "thread_environment": {key: os.environ.get(key) for key in THREAD_ENV},
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_manifest(output_root)

    print(validation["status"])
    print(f"Rows: {len(df)} / {expected_rows}")
    print(f"Best diagnostic mean-rank variant: {validation['best_mean_rank_variant']}")
    print(f"Validation report: {output_root / 'validation_report.json'}")
    print(f"Output manifest: {output_root / 'MANIFEST_SHA256.csv'}")

    if validation["status"].endswith("FAILED"):
        for check in validation["checks"]:
            if not check["passed"]:
                print(
                    f"FAILED: {check['name']} :: {check['details']}",
                    file=sys.stderr,
                )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
