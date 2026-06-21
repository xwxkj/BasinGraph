#!/usr/bin/env python3
"""
Build a compact review package for CUTEst holdout performance analysis.

The package preserves all seven rows and all improvement histories from every
atomic job, plus a compact BasinGraph mechanism summary. It does not calculate
performance metrics, ranks, targets, tests or conclusions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_v2" / "formal_holdout" / "cutest_rc1"
PROTOCOL_ROOT = ROOT / "protocols" / "route_b" / "final_rc1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to CUTEST_HOLDOUT_FINAL_RUN_ID.txt.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_id = args.run_id or (
        PROTOCOL_ROOT / "CUTEST_HOLDOUT_FINAL_RUN_ID.txt"
    ).read_text().strip()
    run_root = RESULT_ROOT / run_id

    lock_path = PROTOCOL_ROOT / "CUTEST_HOLDOUT_EVIDENCE_LOCK.json"
    lock = json.loads(lock_path.read_text())
    if lock["status"] != "V2_RC1_CUTEST_HOLDOUT_EVIDENCE_FROZEN":
        raise RuntimeError("CUTEst evidence is not frozen.")
    if lock["run_id"] != run_id:
        raise RuntimeError("Run ID differs from frozen evidence lock.")

    package_root = (
        ROOT
        / "results_v2"
        / "cutest_holdout_review_package"
        / run_id
    )
    if package_root.exists():
        shutil.rmtree(package_root)

    core_root = package_root / "core_results"
    protocol_dest = package_root / "protocol"
    core_root.mkdir(parents=True)
    protocol_dest.mkdir(parents=True)

    core_files = [
        "cutest_holdout_raw_results_all_available.csv",
        "validation_report.json",
        "run_metadata.json",
        "CUTEST_HOLDOUT_JOB_MANIFEST_SHA256.csv",
    ]
    for name in core_files:
        shutil.copy2(run_root / name, core_root / name)

    batch_root = run_root / "batch_metadata"
    shutil.copytree(
        batch_root,
        core_root / "batch_metadata",
    )

    protocol_files = [
        "CUTEST_HOLDOUT_PREFLIGHT.json",
        "CUTEST_HOLDOUT_EXECUTION_PROTOCOL.md",
        "CUTEST_HOLDOUT_ANALYSIS_PLAN.json",
        "CUTEST_HOLDOUT_ANALYSIS_PLAN.md",
        "CUTEST_HOLDOUT_FINAL_RUN_ID.txt",
        "CUTEST_HOLDOUT_EVIDENCE_MANIFEST_SHA256.csv",
        "CUTEST_HOLDOUT_EVIDENCE_LOCK.json",
        "RC1_PAPER_CODE_CONTRACT.json",
        "RC1_EXACT_ALGORITHM_SEMANTICS.md",
        "RC1_FINAL_SELECTION_DECISION.json",
    ]
    for name in protocol_files:
        shutil.copy2(PROTOCOL_ROOT / name, protocol_dest / name)

    holdout_list = (
        ROOT
        / "protocols"
        / "route_b"
        / "formal_v2_protocol"
        / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
    )
    shutil.copy2(holdout_list, protocol_dest / holdout_list.name)

    compact_path = (
        core_root / "cutest_holdout_analysis_records.jsonl.gz"
    )
    job_files = sorted((run_root / "jobs").rglob("*.json.gz"))

    if len(job_files) != 720:
        raise RuntimeError(f"Expected 720 job files, found {len(job_files)}")

    with gzip.open(
        compact_path,
        "wt",
        encoding="utf-8",
        compresslevel=6,
    ) as output:
        for job_path in job_files:
            with gzip.open(
                job_path,
                "rt",
                encoding="utf-8",
            ) as handle:
                payload = json.load(handle)

            bg = payload["basingraph_result"]
            compact = {
                "job_metadata": payload["job_metadata"],
                "rows": payload["rows"],
                "improvement_histories": payload[
                    "improvement_histories"
                ],
                "basingraph_mechanism": {
                    "nfe": bg["nfe"],
                    "archive_nodes": len(bg["archive"]),
                    "graph_edges": len(bg["graph_edges"]),
                    "phase_evaluations": bg["phase_evaluations"],
                    "diagnostics": bg["diagnostics"],
                    "event_log_length": len(bg["event_log"]),
                    "message": bg["message"],
                },
                "source_job_relative_path": str(
                    job_path.relative_to(ROOT)
                ),
                "source_job_sha256": sha256_file(job_path),
            }
            output.write(
                json.dumps(
                    compact,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )

    readme = f"""# BasinGraph prospective CUTEst holdout review

Run ID: `{run_id}`

This package contains the completed, frozen prospective CUTEst holdout:

- 24 problem instances;
- 30 paired seeds;
- seven algorithms;
- 720 atomic problem-seed jobs;
- 5,040 raw algorithm records;
- full improvement histories in compact JSONL form;
- no performance summary generated before the frozen analysis plan.

The 24-instance holdout has zero exact instance overlap with the frozen
50-instance development/comparability suite.

Primary interpretation must follow
`protocol/CUTEST_HOLDOUT_ANALYSIS_PLAN.md`.
"""
    (package_root / "README_REVIEW.md").write_text(readme)

    manifest_path = package_root / "MANIFEST_SHA256.csv"
    manifest_rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        manifest_rows.append(
            {
                "relative_path": str(path.relative_to(package_root)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    write_csv(manifest_path, manifest_rows)

    archive_path = (
        ROOT
        / "results_v2"
        / f"BasinGraph_CUTEST_holdout_review_{run_id}.tar.gz"
    )
    if archive_path.exists():
        archive_path.unlink()

    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(package_root, arcname=run_id)

    print("CUTEST_HOLDOUT_REVIEW_PACKAGE_OK")
    print("Run ID:", run_id)
    print("Job records:", len(job_files))
    print("Compact analysis input:", compact_path)
    print("Package files:", len(manifest_rows))
    print("Archive:", archive_path)
    print("Archive bytes:", archive_path.stat().st_size)


if __name__ == "__main__":
    main()
