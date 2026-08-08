#!/usr/bin/env python3
"""Launch all 16 algorithm-dimension shards for B21 Track B."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_b.common import (  # noqa: E402
    ALGORITHMS,
    MODE_CONFIG,
    THREAD_ENV,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_b")
    return parser.parse_args()


def next_attempt(shard_root: Path) -> int:
    values = []
    if shard_root.is_dir():
        for path in shard_root.glob("attempt_*"):
            try:
                values.append(int(path.name.split("_")[-1]))
            except ValueError:
                continue
    return max(values, default=0) + 1


def run_shard(
    *,
    mode: str,
    algorithm: str,
    dimension: int,
    run_id: str,
    base_seed: int,
    attempt: int,
    authorize: bool,
    log_path: Path,
    result_root: str,
) -> tuple[str, int, int]:
    command = [
        sys.executable,
        "evidence_extension_v1/track_b/run_track_b_shard.py",
        "--mode",
        mode,
        "--algorithm",
        algorithm,
        "--dimension",
        str(dimension),
        "--run-id",
        run_id,
        "--base-seed",
        str(base_seed),
        "--attempt",
        str(attempt),
        "--result-root",
        result_root,
    ]
    if authorize:
        command.append("--authorize-confirmatory")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    for key in THREAD_ENV:
        environment[key] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return f"{algorithm}_d{dimension}", attempt, process.returncode


def main() -> None:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    if config["confirmatory"] and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory launch requires --authorize-confirmatory.")
    identity = verify_source_identity(require_clean=True)

    result_root = ROOT / args.result_root
    run_root = result_root / args.run_id
    log_root = ROOT / "logs_b21" / "track_b" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    launch_metadata_path = run_root / "launch_metadata.json"
    metadata = {
        "status": "TRACK_B_LAUNCH_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "workers": args.workers,
        "base_seed": args.base_seed,
        "algorithms": ALGORITHMS,
        "dimensions": config["dimensions"],
        "expected_rows": config["expected_rows"],
        "identity": identity,
    }
    launch_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    jobs = []
    for algorithm in ALGORITHMS:
        for dimension in config["dimensions"]:
            shard_root = run_root / "shards" / f"{algorithm}_d{dimension}"
            marker = shard_root / "SHARD_COMPLETE.json"
            if marker.is_file():
                continue
            attempt = next_attempt(shard_root)
            jobs.append(
                {
                    "mode": args.mode,
                    "algorithm": algorithm,
                    "dimension": dimension,
                    "run_id": args.run_id,
                    "base_seed": args.base_seed,
                    "attempt": attempt,
                    "authorize": args.authorize_confirmatory,
                    "log_path": log_root / f"{algorithm}_d{dimension}_attempt{attempt:04d}.log",
                    "result_root": args.result_root,
                }
            )

    print(f"Track B mode: {args.mode}")
    print(f"Run ID: {args.run_id}")
    print(f"Pending shards: {len(jobs)}")
    print(f"Workers: {args.workers}")

    failures: dict[str, int] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as executor:
            futures = {executor.submit(run_shard, **job): job for job in jobs}
            for future in as_completed(futures):
                name, attempt, returncode = future.result()
                print(f"{name}: attempt={attempt} returncode={returncode}", flush=True)
                if returncode != 0:
                    failures[name] = returncode
    if failures:
        metadata["status"] = "TRACK_B_LAUNCH_FAILED"
        metadata["failures"] = failures
        launch_metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        raise RuntimeError("Track B shard failures: " + json.dumps(failures, sort_keys=True))

    rows: list[dict[str, str]] = []
    observer_paths: list[str] = []
    shard_markers = []
    for algorithm in ALGORITHMS:
        for dimension in config["dimensions"]:
            shard_root = run_root / "shards" / f"{algorithm}_d{dimension}"
            marker_path = shard_root / "SHARD_COMPLETE.json"
            if not marker_path.is_file():
                raise RuntimeError(f"Missing shard marker: {marker_path}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            shard_markers.append(marker)
            observer_paths.append(marker["observer_relative"])
            raw_path = ROOT / marker["raw_results"]
            with raw_path.open(encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))

    if len(rows) != config["expected_rows"]:
        raise RuntimeError(
            f"Combined row count mismatch: {len(rows)} != {config['expected_rows']}"
        )
    rows.sort(
        key=lambda row: (
            ALGORITHMS.index(row["algorithm"]),
            int(row["dimension"]),
            int(row["function_index"]),
            int(row["instance_index"]),
        )
    )
    raw_path = run_root / "track_b_raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (run_root / "complete_observer_paths.txt").write_text(
        "\n".join(observer_paths) + "\n", encoding="utf-8"
    )
    marker = {
        "status": "TRACK_B_ALL_SHARDS_COMPLETE",
        "run_id": args.run_id,
        "mode": args.mode,
        "rows": len(rows),
        "expected_rows": config["expected_rows"],
        "raw_results": str(raw_path.relative_to(ROOT)),
        "shards": shard_markers,
    }
    (run_root / "ALL_SHARDS_COMPLETE.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )

    metadata.update(
        {
            "status": "TRACK_B_ALL_SHARDS_COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "shards": len(shard_markers),
        }
    )
    launch_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("TRACK_B_ALL_SHARDS_COMPLETE")
    print(f"Rows: {len(rows)}")
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()
