#!/usr/bin/env python3
"""
One-click parallel launcher for the prospective COCO holdout.

Each algorithm runs in an isolated subprocess and COCO observer directory.
No performance summary is printed until all seven algorithms and the official
cocopp post-processing have completed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = [
    "BasinGraph_v2",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--base-seed", type=int, default=20260620)
    return parser.parse_args()


def run_algorithm(
    algorithm: str,
    run_id: str,
    base_seed: int,
    log_root: Path,
) -> tuple[str, int]:
    log_path = log_root / f"{algorithm}.log"
    command = [
        sys.executable,
        "experiments_v2/run_coco_rc1_holdout_algorithm.py",
        "--algorithm",
        algorithm,
        "--run-id",
        run_id,
        "--base-seed",
        str(base_seed),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        environment[key] = "1"

    with log_path.open("w") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return algorithm, process.returncode


def main() -> None:
    args = parse_args()
    run_id = args.run_id or (
        "v2rc1_coco_holdout_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )

    result_root = ROOT / "results_v2/formal_holdout/coco_rc1"
    run_root = result_root / run_id
    log_root = ROOT / "logs_v2" / run_id

    if run_root.exists():
        raise RuntimeError(f"Run already exists: {run_root}")

    run_root.mkdir(parents=True)
    log_root.mkdir(parents=True)
    (result_root / "LAST_RUN_ID.txt").write_text(run_id + "\n")

    launch_metadata = {
        "status": "COCO_HOLDOUT_LAUNCH_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "workers": args.workers,
        "base_seed": args.base_seed,
        "algorithms": ALGORITHMS,
    }
    (run_root / "launch_metadata.json").write_text(
        json.dumps(launch_metadata, indent=2)
    )

    failures = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.workers, len(ALGORITHMS)))
    ) as executor:
        futures = {
            executor.submit(
                run_algorithm,
                algorithm,
                run_id,
                args.base_seed,
                log_root,
            ): algorithm
            for algorithm in ALGORITHMS
        }

        for future in as_completed(futures):
            algorithm, returncode = future.result()
            if returncode != 0:
                failures[algorithm] = returncode

    if failures:
        raise RuntimeError(
            "Holdout algorithm failures: "
            + json.dumps(failures, sort_keys=True)
        )

    all_rows = []
    metadata = []
    for algorithm in ALGORITHMS:
        algorithm_root = run_root / algorithm
        with (algorithm_root / "holdout_results.csv").open() as handle:
            all_rows.extend(csv.DictReader(handle))
        metadata.append(
            json.loads((algorithm_root / "run_metadata.json").read_text())
        )

    combined_path = run_root / "coco_rc1_holdout_raw_results.csv"
    with combined_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    launch_metadata["status"] = "COCO_HOLDOUT_ALGORITHMS_COMPLETE"
    launch_metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    launch_metadata["rows"] = len(all_rows)
    launch_metadata["algorithm_metadata"] = metadata
    (run_root / "launch_metadata.json").write_text(
        json.dumps(launch_metadata, indent=2)
    )

    print("COCO_HOLDOUT_ALGORITHMS_COMPLETE")
    print("RUN_ID=" + run_id)
    print("ROWS=" + str(len(all_rows)))


if __name__ == "__main__":
    main()
