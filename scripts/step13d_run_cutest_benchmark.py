#!/usr/bin/env python3
"""
Step 13D: Formal CUTEst benchmark runner for BasinGraph.

Protocol
--------
- Reads the frozen pre-registered CUTEst v2 list.
- Runs seven optimizers under the same objective-evaluation budget.
- Uses deterministic, paired seeds across algorithms.
- Supports resumable, parallel execution on macOS/Linux.
- Writes one job CSV per problem/seed and compressed convergence histories.
- Does not select or exclude problems using optimizer performance.

Recommended execution
---------------------
Run three resumable batches of ten seeds:
    bash scripts/run_step13d_cutest_batch.sh 0 10 4
    bash scripts/run_step13d_cutest_batch.sh 10 10 4
    bash scripts/run_step13d_cutest_batch.sh 20 10 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ALGORITHM_IDS = [
    "basingraph",
    "cmaes",
    "bipop-cmaes",
    "de",
    "ms-lbfgsb",
    "lhs",
    "random",
]

ALGORITHM_NAMES = {
    "basingraph": "BasinGraph",
    "cmaes": "CMA-ES",
    "bipop-cmaes": "BIPOP-CMA-ES",
    "de": "Differential Evolution",
    "ms-lbfgsb": "Multi-start L-BFGS-B",
    "lhs": "Latin Hypercube Sampling",
    "random": "Random Search",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_") or "unnamed"


def scalable_destination(problem_name: str, n_value: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", problem_name)
    return f"BGV2_{clean}_N{int(n_value)}"


def budget_for_dimension(
    dimension: int,
    multiplier: int,
    minimum: int,
    maximum: int,
) -> int:
    return int(min(maximum, max(minimum, multiplier * int(dimension))))


def verify_protocol_manifest(
    protocol_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not protocol_path.exists():
        raise FileNotFoundError(f"Missing protocol list: {protocol_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing protocol manifest: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    target_name = str(protocol_path.relative_to(PROJECT_ROOT))

    matched = manifest[manifest["relative_path"] == target_name]
    if len(matched) != 1:
        raise RuntimeError(
            f"Manifest does not contain exactly one entry for {target_name}"
        )

    expected = str(matched.iloc[0]["sha256"]).strip().lower()
    observed = sha256_file(protocol_path).lower()
    if observed != expected:
        raise RuntimeError(
            "Frozen CUTEst protocol hash mismatch.\n"
            f"Expected: {expected}\nObserved: {observed}"
        )

    return {
        "protocol_relative_path": target_name,
        "protocol_sha256": observed,
        "manifest_relative_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
    }


def load_problem(row: dict[str, Any]):
    import pycutest

    problem_name = str(row["problem_name"])
    source_type = str(row["source_type"])

    kwargs: dict[str, Any] = {
        "problemName": problem_name,
        "quiet": True,
        "drop_fixed_variables": True,
    }

    if source_type == "scalable":
        raw_n = row.get("sif_N")
        if raw_n is None or pd.isna(raw_n):
            raise ValueError(f"Missing sif_N for scalable problem {problem_name}")
        n_value = int(float(raw_n))
        kwargs["sifParams"] = {"N": n_value}
        kwargs["destination"] = scalable_destination(problem_name, n_value)

    return pycutest.import_problem(**kwargs)


def get_optimizers():
    from basingraph.optimizer import minimize_basingraph
    from baselines.reference_optimizers import (
        optimize_bipop_cmaes,
        optimize_cmaes,
        optimize_de,
        optimize_lhs,
        optimize_multistart_lbfgsb,
        optimize_random_search,
    )

    return {
        "basingraph": minimize_basingraph,
        "cmaes": optimize_cmaes,
        "bipop-cmaes": optimize_bipop_cmaes,
        "de": optimize_de,
        "ms-lbfgsb": optimize_multistart_lbfgsb,
        "lhs": optimize_lhs,
        "random": optimize_random_search,
    }


def normalize_history(history: Any, nfe: int, fbest: float) -> tuple[np.ndarray, np.ndarray]:
    if history is None:
        return (
            np.asarray([int(nfe)], dtype=np.int64),
            np.asarray([float(fbest)], dtype=np.float64),
        )

    try:
        arr = np.asarray(history, dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 2 and arr.shape[0] > 0:
            evals = arr[:, 0].astype(np.int64, copy=False)
            bests = arr[:, 1].astype(np.float64, copy=False)
            return evals, bests
    except Exception:
        pass

    return (
        np.asarray([int(nfe)], dtype=np.int64),
        np.asarray([float(fbest)], dtype=np.float64),
    )


def worker_run_problem_seed(payload: dict[str, Any]) -> dict[str, Any]:
    # Keep each worker single-threaded to avoid BLAS oversubscription.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"

    row = payload["problem_row"]
    protocol_seed_index = int(payload["protocol_seed_index"])
    protocol_seed = int(payload["protocol_seed"])
    budget_multiplier = int(payload["budget_multiplier"])
    min_budget = int(payload["min_budget"])
    max_budget = int(payload["max_budget"])
    algorithms = list(payload["algorithms"])
    result_root = Path(payload["result_root"])

    problem = None
    job_started = time.perf_counter()
    records: list[dict[str, Any]] = []

    instance_id = str(row["instance_id"])
    order = int(row["global_protocol_order"])
    dimension_group = str(row["dimension_group"])
    source_type = str(row["source_type"])
    problem_name = str(row["problem_name"])

    job_stem = (
        f"{order:03d}_{safe_component(instance_id)}"
        f"_seed{protocol_seed_index + 1:02d}"
    )
    job_csv = result_root / "job_records" / f"{job_stem}.csv"

    if job_csv.exists():
        try:
            existing = pd.read_csv(job_csv)
            if set(existing["algorithm_id"]) == set(algorithms) and len(existing) == len(algorithms):
                return {
                    "status": "skipped_existing",
                    "job_csv": str(job_csv),
                    "records": len(existing),
                    "order": order,
                    "seed_index": protocol_seed_index,
                }
        except Exception:
            pass

    try:
        problem = load_problem(row)

        dimension = int(problem.n)
        if dimension != int(row["dimension"]):
            raise RuntimeError(
                f"Dimension mismatch for {instance_id}: "
                f"protocol={row['dimension']}, imported={dimension}"
            )

        lb = np.asarray(problem.bl, dtype=float).reshape(-1)
        ub = np.asarray(problem.bu, dtype=float).reshape(-1)
        x0 = np.asarray(problem.x0, dtype=float).reshape(-1)
        f0 = float(problem.obj(x0))

        budget = budget_for_dimension(
            dimension,
            budget_multiplier,
            min_budget,
            max_budget,
        )

        def objective(x):
            return float(problem.obj(np.asarray(x, dtype=float)))

        optimizers = get_optimizers()

        for algorithm_id in algorithms:
            algorithm_name = ALGORITHM_NAMES[algorithm_id]
            optimizer = optimizers[algorithm_id]

            started = time.perf_counter()
            result: dict[str, Any]
            error = ""
            runner_status = "completed"

            try:
                result = optimizer(
                    objective=objective,
                    lb=lb,
                    ub=ub,
                    max_evals=budget,
                    seed=protocol_seed,
                )
            except Exception as exc:
                result = {
                    "xbest": None,
                    "fbest": np.nan,
                    "nfe": 0,
                    "history": [],
                    "message": f"exception:{type(exc).__name__}",
                }
                error = (
                    f"{type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc(limit=20)}"
                )
                runner_status = "failed"

            elapsed = time.perf_counter() - started

            fbest = float(result.get("fbest", np.nan))
            nfe = int(result.get("nfe", 0))
            algorithm_message = str(result.get("message", ""))

            hist_evals, hist_best = normalize_history(
                result.get("history"),
                nfe=nfe,
                fbest=fbest,
            )

            history_dir = (
                result_root
                / "histories"
                / f"{order:03d}_{safe_component(instance_id)}"
                / safe_component(algorithm_id)
            )
            history_dir.mkdir(parents=True, exist_ok=True)
            history_path = history_dir / f"seed{protocol_seed_index + 1:02d}.npz"
            np.savez_compressed(
                history_path,
                nfe=hist_evals,
                fbest=hist_best,
            )

            records.append({
                "global_protocol_order": order,
                "problem_name": problem_name,
                "instance_id": instance_id,
                "source_type": source_type,
                "sif_N": row.get("sif_N"),
                "dimension_group": dimension_group,
                "dimension": dimension,
                "algorithm": algorithm_name,
                "algorithm_id": algorithm_id,
                "protocol_seed_index": protocol_seed_index,
                "protocol_seed": protocol_seed,
                "budget": budget,
                "f_at_cutest_x0": f0,
                "fbest": fbest,
                "nfe": nfe,
                "budget_ratio": (nfe / budget if budget > 0 else np.nan),
                "algorithm_message": algorithm_message,
                "runner_status": runner_status,
                "error": error,
                "wall_time_seconds": elapsed,
                "history_relative_path": str(history_path.relative_to(result_root)),
            })

        job_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(job_csv, index=False)

        return {
            "status": "completed",
            "job_csv": str(job_csv),
            "records": len(records),
            "order": order,
            "seed_index": protocol_seed_index,
            "wall_time_seconds": time.perf_counter() - job_started,
        }

    except Exception as exc:
        failure_dir = result_root / "job_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_path = failure_dir / f"{job_stem}.txt"
        failure_path.write_text(
            f"{type(exc).__name__}: {exc}\n\n"
            f"{traceback.format_exc()}"
        )
        return {
            "status": "job_failed",
            "job_csv": "",
            "records": 0,
            "order": order,
            "seed_index": protocol_seed_index,
            "error": f"{type(exc).__name__}: {exc}",
            "failure_log": str(failure_path),
        }

    finally:
        if problem is not None and hasattr(problem, "terminate"):
            try:
                problem.terminate()
            except Exception:
                pass


def merge_available_job_records(result_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((result_root / "job_records").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
            if len(frame):
                frames.append(frame)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values(
        [
            "global_protocol_order",
            "protocol_seed_index",
            "algorithm",
        ],
        inplace=True,
    )
    merged.to_csv(
        result_root / "cutest_raw_results_all_available.csv",
        index=False,
    )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        default="protocols/cutest_pre_registered_problem_list_v2.csv",
    )
    parser.add_argument(
        "--manifest",
        default="protocols/CUTEST_PRE_REGISTRATION_MANIFEST_v2.csv",
    )
    parser.add_argument(
        "--result-root",
        default="cutest_results/protocol_v2",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260619)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-multiplier", type=int, default=50)
    parser.add_argument("--min-budget", type=int, default=1000)
    parser.add_argument("--max-budget", type=int, default=20000)
    parser.add_argument(
        "--algorithms",
        default=",".join(ALGORITHM_IDS),
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    protocol_path = PROJECT_ROOT / args.protocol
    manifest_path = PROJECT_ROOT / args.manifest
    result_root = PROJECT_ROOT / args.result_root
    result_root.mkdir(parents=True, exist_ok=True)

    protocol_integrity = verify_protocol_manifest(
        protocol_path,
        manifest_path,
    )

    protocol = pd.read_csv(protocol_path)
    if len(protocol) != 50:
        raise RuntimeError(f"Expected 50 pre-registered instances, found {len(protocol)}")

    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    unknown = sorted(set(algorithms).difference(ALGORITHM_IDS))
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}")

    seed_indices = list(
        range(args.seed_start, args.seed_start + args.seed_count)
    )

    jobs: list[dict[str, Any]] = []
    for _, row in protocol.iterrows():
        row_dict = {}
        for key, value in row.to_dict().items():
            if isinstance(value, np.generic):
                value = value.item()
            row_dict[key] = value

        for seed_index in seed_indices:
            jobs.append({
                "problem_row": row_dict,
                "protocol_seed_index": seed_index,
                "protocol_seed": args.base_seed + seed_index,
                "budget_multiplier": args.budget_multiplier,
                "min_budget": args.min_budget,
                "max_budget": args.max_budget,
                "algorithms": algorithms,
                "result_root": str(result_root),
            })

    batch_id = (
        f"seed{args.seed_start + 1:02d}"
        f"_to_{args.seed_start + args.seed_count:02d}"
        f"_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )

    metadata = {
        "batch_id": batch_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "workers": int(args.workers),
        "seed_start_zero_based": int(args.seed_start),
        "seed_count": int(args.seed_count),
        "base_seed": int(args.base_seed),
        "protocol_seeds": [
            int(args.base_seed + i) for i in seed_indices
        ],
        "problem_instances": int(len(protocol)),
        "algorithms": algorithms,
        "algorithm_names": [ALGORITHM_NAMES[a] for a in algorithms],
        "budget_multiplier": int(args.budget_multiplier),
        "min_budget": int(args.min_budget),
        "max_budget": int(args.max_budget),
        "planned_problem_seed_jobs": int(len(jobs)),
        **protocol_integrity,
        "source_hashes": {
            "basingraph/optimizer.py": sha256_file(
                PROJECT_ROOT / "basingraph" / "optimizer.py"
            ),
            "baselines/reference_optimizers.py": sha256_file(
                PROJECT_ROOT / "baselines" / "reference_optimizers.py"
            ),
            "scripts/step13d_run_cutest_benchmark.py": sha256_file(
                Path(__file__).resolve()
            ),
        },
    }

    metadata_dir = result_root / "batch_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / f"{batch_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    progress_path = result_root / "progress.json"
    started = time.perf_counter()
    counts = {
        "completed": 0,
        "skipped_existing": 0,
        "job_failed": 0,
    }

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        mp_context=context,
    ) as executor:
        futures = [
            executor.submit(worker_run_problem_seed, payload)
            for payload in jobs
        ]

        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            status = str(result.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1

            if args.verbose:
                print(
                    f"[{index}/{len(futures)}] "
                    f"status={status} "
                    f"problem_order={result.get('order')} "
                    f"seed_index={result.get('seed_index')}"
                )

            if index % 10 == 0 or index == len(futures):
                progress = {
                    "batch_id": batch_id,
                    "completed_futures": index,
                    "total_futures": len(futures),
                    "counts": counts,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                progress_path.write_text(json.dumps(progress, indent=2))

    merged = merge_available_job_records(result_root)

    expected_batch_rows = (
        len(protocol) * len(seed_indices) * len(algorithms)
    )

    batch_rows = merged[
        merged["protocol_seed_index"].isin(seed_indices)
    ] if len(merged) else pd.DataFrame()

    batch_validation = {
        "batch_id": batch_id,
        "expected_problem_seed_jobs": len(jobs),
        "status_counts": counts,
        "expected_batch_rows": expected_batch_rows,
        "observed_batch_rows": int(len(batch_rows)),
        "all_available_rows": int(len(merged)),
        "job_failure_files": len(list((result_root / "job_failures").glob("*.txt"))),
        "elapsed_seconds": time.perf_counter() - started,
    }

    validation_path = (
        result_root
        / "batch_metadata"
        / f"{batch_id}_validation.json"
    )
    validation_path.write_text(json.dumps(batch_validation, indent=2))

    print("STEP_13D_BATCH_OK")
    print("Batch ID:", batch_id)
    print("Problem-seed jobs:", len(jobs))
    print("Expected batch rows:", expected_batch_rows)
    print("Observed batch rows:", len(batch_rows))
    print("All available rows:", len(merged))
    print("Status counts:", counts)
    print("Result root:", result_root)
    print("Validation:", validation_path)


if __name__ == "__main__":
    main()
