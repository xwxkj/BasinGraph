#!/usr/bin/env python3
"""Resumable launcher for all B21 Track A shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_a.common import (  # noqa: E402
    MODE_CONFIG,
    VARIANT_ORDER,
    set_single_thread_environment,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--result-root", default="results_b21/track_a")
    parser.add_argument("--authorize-confirmatory", action="store_true")
    return parser.parse_args()


def archived_attempts(run_root: Path, shard_id: str) -> list[int]:
    attempts: list[int] = []
    for path in (run_root / "incomplete_attempts").glob(
        f"**/{shard_id}_attempt_*"
    ):
        try:
            attempts.append(int(path.name.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return attempts


def archive_incomplete_shard(
    run_root: Path,
    output: Path,
    shard_id: str,
    archive_stamp: str,
) -> int:
    archived = archived_attempts(run_root, shard_id)
    if not output.exists():
        return max(archived, default=0) + 1

    metadata_path = output / "shard_metadata.json"
    prior_attempt = max(archived, default=0) + 1
    if metadata_path.is_file():
        try:
            prior_attempt = int(
                json.loads(metadata_path.read_text(encoding="utf-8"))["attempt"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    archive_root = run_root / "incomplete_attempts" / archive_stamp
    archive_root.mkdir(parents=True, exist_ok=True)
    destination = archive_root / f"{shard_id}_attempt_{prior_attempt:02d}"
    if destination.exists():
        raise RuntimeError(
            f"Incomplete-attempt archive already exists: {destination}"
        )
    shutil.move(str(output), str(destination))
    return max([*archived, prior_attempt], default=0) + 1


def complete_shard(path: Path) -> bool:
    marker = path / "SHARD_COMPLETE.json"
    if not marker.is_file():
        return False
    payload = json.loads(marker.read_text(encoding="utf-8"))
    return payload.get("status") == "TRACK_A_SHARD_VALIDATION_OK"


def run_shard(job: dict, log_root: Path) -> tuple[str, int]:
    shard_id = job["shard_id"]
    log_path = log_root / f"{shard_id}_attempt_{job['attempt']:02d}.log"
    command = [
        sys.executable,
        "evidence_extension_v1/track_a/run_track_a_shard.py",
        "--mode", job["mode"],
        "--variant", job["variant"],
        "--dimension", str(job["dimension"]),
        "--run-id", job["run_id"],
        "--attempt", str(job["attempt"]),
        "--base-seed", str(job["base_seed"]),
        "--result-root", job["result_root"],
    ]
    if job["mode"] == "confirmatory":
        command.append("--authorize-confirmatory")

    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = "1"
    environment["PYTHONPATH"] = str(ROOT)

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    return shard_id, process.returncode


def combine_results(run_root: Path, *, mode: str) -> Path:
    config = MODE_CONFIG[mode]
    rows = []
    for variant in VARIANT_ORDER:
        for dimension in config["dimensions"]:
            shard = run_root / "shards" / f"{variant}_d{dimension}"
            if not complete_shard(shard):
                raise RuntimeError(f"Incomplete shard: {shard}")
            with (shard / "shard_results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows.extend(csv.DictReader(handle))

    output = run_root / "track_a_raw_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    frame = pd.DataFrame(rows)
    if frame.duplicated(["variant", "problem_id"]).any():
        raise RuntimeError("Duplicate Track A variant/problem records.")
    if len(frame) != config["expected_rows"]:
        raise RuntimeError(f"Unexpected combined row count: {len(frame)}")
    return output


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError(
            "Confirmatory launcher requires --authorize-confirmatory."
        )
    set_single_thread_environment()
    identity = verify_source_identity(require_clean=True)
    config = MODE_CONFIG[args.mode]

    result_root = ROOT / args.result_root
    run_root = result_root / args.run_id
    shard_root = run_root / "shards"
    log_root = run_root / "logs"
    shard_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    metadata_path = run_root / "launch_metadata.json"
    metadata = {
        "status": "TRACK_A_LAUNCH_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "mode": args.mode,
        "confirmatory": bool(config["confirmatory"]),
        "workers": args.workers,
        "base_seed": args.base_seed,
        "identity": identity,
        "variants": VARIANT_ORDER,
        "dimensions": config["dimensions"],
        "expected_rows": config["expected_rows"],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    jobs = []
    archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for variant in VARIANT_ORDER:
        for dimension in config["dimensions"]:
            shard_id = f"{variant}_d{dimension}"
            output = shard_root / shard_id
            if complete_shard(output):
                continue
            attempt = archive_incomplete_shard(
                run_root, output, shard_id, archive_stamp
            )
            jobs.append({
                "shard_id": shard_id,
                "mode": args.mode,
                "variant": variant,
                "dimension": dimension,
                "run_id": args.run_id,
                "attempt": attempt,
                "base_seed": args.base_seed,
                "result_root": args.result_root,
            })

    print(f"Track A mode: {args.mode}")
    print(f"Run ID: {args.run_id}")
    print(f"Pending shards: {len(jobs)}")
    print(f"Workers: {args.workers}")

    started = time.perf_counter()
    failures = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(args.workers, len(jobs) or 1))
    ) as executor:
        futures = {
            executor.submit(run_shard, job, log_root): job["shard_id"]
            for job in jobs
        }
        for future in as_completed(futures):
            shard_id, returncode = future.result()
            print(f"{shard_id}: returncode={returncode}", flush=True)
            if returncode != 0:
                failures[shard_id] = returncode

    if failures:
        metadata["status"] = "TRACK_A_LAUNCH_FAILED"
        metadata["failures"] = failures
        metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Track A shard failures: " + json.dumps(failures, sort_keys=True)
        )

    raw_path = combine_results(run_root, mode=args.mode)
    metadata["status"] = "TRACK_A_ALL_SHARDS_COMPLETE"
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["raw_results"] = str(raw_path.relative_to(ROOT))
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    marker = {
        "status": "TRACK_A_ALL_SHARDS_COMPLETE",
        "run_id": args.run_id,
        "mode": args.mode,
        "rows": config["expected_rows"],
        "raw_results": str(raw_path.relative_to(ROOT)),
    }
    (run_root / "ALL_SHARDS_COMPLETE.json").write_text(
        json.dumps(marker, indent=2), encoding="utf-8"
    )

    print("TRACK_A_ALL_SHARDS_COMPLETE")
    print(f"Rows: {config['expected_rows']}")
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()
