#!/usr/bin/env python3
"""Launch registered B21 Track C task–algorithm shards."""

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

from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    NIST_DATASETS,
    SMOKE_TASKS,
    THREAD_ENV,
    verify_source_identity,
)

EXPECTED_ROWS = {"smoke": len(SMOKE_TASKS) * len(ALGORITHMS), "confirmatory": 2310}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_c")
    return parser.parse_args()


def next_attempt(shard_root: Path) -> int:
    attempts = []
    if shard_root.is_dir():
        for path in shard_root.glob("attempt_*"):
            try:
                attempts.append(int(path.name.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return max(attempts, default=0) + 1


def job_matrix(mode: str) -> list[tuple[str, str, str]]:
    tasks = SMOKE_TASKS if mode == "smoke" else (
        [("c1", name) for name in C1_FAMILIES]
        + [("c2", name) for name in NIST_DATASETS]
    )
    return [(domain, task_name, algorithm) for domain, task_name in tasks for algorithm in ALGORITHMS]


def run_shard(job: dict[str, object]) -> tuple[str, int, int]:
    command = [
        sys.executable,
        "evidence_extension_v1/track_c/run_track_c_shard.py",
        "--mode", str(job["mode"]),
        "--domain", str(job["domain"]),
        "--task-name", str(job["task_name"]),
        "--algorithm", str(job["algorithm"]),
        "--run-id", str(job["run_id"]),
        "--attempt", str(job["attempt"]),
        "--result-root", str(job["result_root"]),
    ]
    if bool(job["authorize"]):
        command.append("--authorize-confirmatory")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT)
    for key in THREAD_ENV:
        environment[key] = "1"
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    name = f"{job['domain']}_{job['task_name']}_{job['algorithm']}"
    return name, int(job["attempt"]), process.returncode


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory launch requires authorization.")
    identity = verify_source_identity(require_clean=True)
    result_root = ROOT / args.result_root
    run_root = result_root / args.run_id
    log_root = ROOT / "logs_b21" / "track_c" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    metadata_path = run_root / "launch_metadata.json"
    metadata = {
        "status": "TRACK_C_LAUNCH_STARTED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "workers": args.workers,
        "expected_rows": EXPECTED_ROWS[args.mode],
        "identity": identity,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    jobs = []
    for domain, task_name, algorithm in job_matrix(args.mode):
        shard_name = f"{domain}_{task_name}_{algorithm}"
        shard_root = run_root / "shards" / shard_name
        marker = shard_root / "SHARD_COMPLETE.json"
        if marker.is_file():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if existing.get("status") == "TRACK_C_SHARD_COMPLETE":
                continue
        attempt = next_attempt(shard_root)
        jobs.append(
            {
                "mode": args.mode,
                "domain": domain,
                "task_name": task_name,
                "algorithm": algorithm,
                "run_id": args.run_id,
                "attempt": attempt,
                "authorize": args.authorize_confirmatory,
                "result_root": args.result_root,
                "log_path": log_root / f"{shard_name}_attempt{attempt:04d}.log",
            }
        )

    print(f"Track C mode: {args.mode}")
    print(f"Run ID: {args.run_id}")
    print(f"Pending shards: {len(jobs)}")
    print(f"Workers: {args.workers}")
    failures: dict[str, int] = {}
    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, len(jobs)))) as executor:
            futures = {executor.submit(run_shard, job): job for job in jobs}
            for future in as_completed(futures):
                name, attempt, returncode = future.result()
                print(f"{name}: attempt={attempt} returncode={returncode}", flush=True)
                if returncode != 0:
                    failures[name] = returncode
    if failures:
        metadata["status"] = "TRACK_C_LAUNCH_FAILED"
        metadata["failures"] = failures
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        raise RuntimeError("Track C shard failures: " + json.dumps(failures, sort_keys=True))

    rows: list[dict[str, str]] = []
    shard_markers = []
    for domain, task_name, algorithm in job_matrix(args.mode):
        shard_name = f"{domain}_{task_name}_{algorithm}"
        marker_path = run_root / "shards" / shard_name / "SHARD_COMPLETE.json"
        if not marker_path.is_file():
            raise RuntimeError(f"Missing shard marker: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") != "TRACK_C_SHARD_COMPLETE":
            raise RuntimeError(f"Incomplete shard: {shard_name}")
        shard_markers.append(marker)
        with (ROOT / marker["raw_results"]).open(encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS[args.mode]:
        raise RuntimeError(f"Combined row count mismatch: {len(rows)} != {EXPECTED_ROWS[args.mode]}")
    rows.sort(
        key=lambda row: (
            row["domain"],
            row["task_name"],
            int(row["instance"]),
            int(row["paired_seed"]),
            ALGORITHMS.index(row["algorithm"]),
        )
    )
    raw_path = run_root / "track_c_raw_results.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    complete = {
        "status": "TRACK_C_ALL_SHARDS_COMPLETE",
        "run_id": args.run_id,
        "mode": args.mode,
        "rows": len(rows),
        "expected_rows": EXPECTED_ROWS[args.mode],
        "raw_results": str(raw_path.relative_to(ROOT)),
        "shards": shard_markers,
    }
    (run_root / "ALL_SHARDS_COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    metadata.update(
        {
            "status": "TRACK_C_ALL_SHARDS_COMPLETE",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "rows": len(rows),
            "shards": len(shard_markers),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("TRACK_C_ALL_SHARDS_COMPLETE")
    print(f"Rows: {len(rows)}")
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    main()
