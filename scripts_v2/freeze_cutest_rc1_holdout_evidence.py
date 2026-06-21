#!/usr/bin/env python3
"""
Freeze the completed prospective CUTEst holdout before performance analysis.

This script verifies integrity, writes a predeclared analysis plan, creates a
full local job archive, and writes machine-readable evidence locks. It does not
compute ranks, targets, statistical tests or performance summaries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_v2" / "formal_holdout" / "cutest_rc1"
PROTOCOL_ROOT = ROOT / "protocols" / "route_b" / "final_rc1"

EXPECTED_RUNNER_TAG = (
    "route-b-v2.0.0-rc1-cutest-holdout-runner-freeze"
)
EXPECTED_CODE_TAG = "route-b-v2.0.0-rc1-codefreeze"
EXPECTED_CANDIDATE_TAG = (
    "route-b-v2.0.0-rc1-selected-final-candidate"
)
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960"
    "b96a0239173ce791b4ed1ee63ed0f69"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to LAST_RUN_ID.txt.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows supplied for {path}")
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
        RESULT_ROOT / "LAST_RUN_ID.txt"
    ).read_text().strip()
    run_root = RESULT_ROOT / run_id

    raw_path = run_root / "cutest_holdout_raw_results_all_available.csv"
    validation_path = run_root / "validation_report.json"
    metadata_path = run_root / "run_metadata.json"
    job_manifest_path = run_root / "CUTEST_HOLDOUT_JOB_MANIFEST_SHA256.csv"
    jobs_root = run_root / "jobs"

    holdout_list = (
        ROOT
        / "protocols"
        / "route_b"
        / "formal_v2_protocol"
        / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
    )
    preflight_path = PROTOCOL_ROOT / "CUTEST_HOLDOUT_PREFLIGHT.json"
    execution_protocol = (
        PROTOCOL_ROOT / "CUTEST_HOLDOUT_EXECUTION_PROTOCOL.md"
    )
    paper_code_contract = (
        PROTOCOL_ROOT / "RC1_PAPER_CODE_CONTRACT.json"
    )
    selection_decision = (
        PROTOCOL_ROOT / "RC1_FINAL_SELECTION_DECISION.json"
    )

    required = [
        raw_path,
        validation_path,
        metadata_path,
        job_manifest_path,
        jobs_root,
        holdout_list,
        preflight_path,
        execution_protocol,
        paper_code_contract,
        selection_decision,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    validation = json.loads(validation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    contract = json.loads(paper_code_contract.read_text())
    selection = json.loads(selection_decision.read_text())

    assert validation["status"] == "V2_RC1_CUTEST_HOLDOUT_VALIDATION_OK"
    assert validation["run_id"] == run_id
    assert validation["rows"] == 5040
    assert validation["problem_seed_jobs"] == 720
    assert validation["problems"] == 24
    assert validation["algorithms"] == 7
    assert validation["paired_seeds"] == 30
    assert validation["development_instance_overlap"] == 0
    assert validation["all_runs_completed"] is True
    assert validation["performance_analysis_performed"] is False

    assert metadata["status"] == "CUTEST_HOLDOUT_RUN_STARTED"
    assert metadata["performance_analysis_performed"] is False
    assert metadata["options_hash"] == EXPECTED_OPTIONS_HASH

    assert preflight["status"] == "CUTEST_HOLDOUT_PREFLIGHT_OK"
    assert preflight["objective_evaluations_performed"] == 0
    assert preflight["optimizer_runs_performed"] == 0

    assert contract["status"] == "RC1_PAPER_CODE_CONTRACT_FROZEN"
    assert contract["holdout_accessed"] is False
    assert selection["status"] == "RC1_SELECTED_AS_FINAL_HOLDOUT_CANDIDATE"
    assert selection["holdout_accessed"] is False

    assert sha256_file(raw_path) == validation["raw_results_sha256"]
    assert sha256_file(metadata_path) == validation["run_metadata_sha256"]
    assert (
        sha256_file(job_manifest_path)
        == validation["job_manifest_sha256"]
    )
    assert sha256_file(holdout_list) == validation["holdout_list_sha256"]

    job_manifest = pd.read_csv(job_manifest_path)
    assert len(job_manifest) == 720
    assert job_manifest["relative_path"].nunique() == 720

    manifest_failures: list[str] = []
    total_job_bytes = 0

    for row in job_manifest.itertuples(index=False):
        path = ROOT / row.relative_path
        if not path.exists():
            manifest_failures.append(f"missing:{row.relative_path}")
            continue
        observed_hash = sha256_file(path)
        observed_size = path.stat().st_size
        total_job_bytes += observed_size
        if observed_hash != row.sha256:
            manifest_failures.append(f"sha256:{row.relative_path}")
        if observed_size != int(row.size_bytes):
            manifest_failures.append(f"size:{row.relative_path}")

    if manifest_failures:
        raise RuntimeError(
            "Job-manifest verification failed: "
            + "; ".join(manifest_failures[:20])
        )

    # Freeze the analysis plan before opening performance values.
    analysis_plan = {
        "status": "CUTEST_HOLDOUT_ANALYSIS_PLAN_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "performance_values_inspected_before_freeze": False,
        "primary_analysis": {
            "name": "target-runtime ECDF and ERT",
            "problem_reference_value": (
                "For each problem, f_ref is the minimum final fbest across "
                "all 7 algorithms and 30 paired seeds. This symmetric "
                "best-observed reference is defined before reading values."
            ),
            "normalization_scale": (
                "s_p = max(abs(f_at_cutest_x0 - f_ref), "
                "1e-12*(1+abs(f_ref)))"
            ),
            "target_definition": (
                "f_target(p,tau) = f_ref + tau*s_p"
            ),
            "relative_targets": [
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
            ],
            "runtime": (
                "First objective-evaluation index in the improvement history "
                "whose best-so-far value is <= f_target."
            ),
            "failure_cost": (
                "The full prescribed problem budget is charged to an "
                "unsuccessful run."
            ),
            "ert": (
                "For each problem, algorithm and target: sum(success "
                "runtimes + full budgets for failures) / number of successes; "
                "ERT is infinite when no run succeeds."
            ),
            "data_profile_x": (
                "Function evaluations divided by problem dimension."
            ),
            "aggregation_unit": (
                "Problem-seed-target triples for ECDF; problem-target blocks "
                "for ERT comparison."
            ),
        },
        "secondary_analysis": {
            "final_normalized_gap": (
                "max(fbest-f_ref,0)/s_p"
            ),
            "task_rank": (
                "Rank final fbest within each problem-seed block; average ties."
            ),
            "problem_summary": (
                "Median final normalized gap and mean task rank over 30 seeds."
            ),
            "pairwise_counts": (
                "BasinGraph better/worse/tied against each baseline on "
                "problem-seed final values with a scale-aware tolerance."
            ),
        },
        "confirmatory_statistics": {
            "block_unit": "24 holdout problems",
            "algorithm_score": (
                "Per-problem median final normalized gap over 30 seeds."
            ),
            "omnibus_test": (
                "Friedman test across seven algorithms."
            ),
            "post_hoc": (
                "Paired two-sided Wilcoxon signed-rank tests comparing "
                "BasinGraph with each baseline."
            ),
            "multiplicity": "Holm correction over six comparisons.",
            "effect_size": (
                "Paired rank-biserial correlation, reported with direction."
            ),
            "alpha": 0.05,
        },
        "stratified_reporting": {
            "groups": [
                "small_2_20",
                "medium_21_100",
                "large_101_500",
            ],
            "status": (
                "Descriptive ECDF, ERT, final-gap and rank summaries; no "
                "separate confirmatory claim because group problem counts are "
                "11, 7 and 6."
            ),
        },
        "interpretation_rules": [
            "Primary claims rely on target-runtime ECDF/ERT.",
            "Final-value ranks are secondary.",
            "Report the prospective holdout separately from the 50-problem "
            "development/comparability set.",
            "Do not modify any algorithm or parameter after this holdout.",
            "Use operational basin-state terminology from the frozen contract.",
        ],
    }

    plan_json = PROTOCOL_ROOT / "CUTEST_HOLDOUT_ANALYSIS_PLAN.json"
    plan_json.write_text(json.dumps(analysis_plan, indent=2) + "\n")

    plan_md = PROTOCOL_ROOT / "CUTEST_HOLDOUT_ANALYSIS_PLAN.md"
    plan_md.write_text(
        f"""# Prospective CUTEst holdout analysis plan

Run ID: `{run_id}`

This plan was frozen after integrity validation and before performance values
were inspected.

## Primary analysis

For each problem, define `f_ref` as the minimum final objective observed across
all seven algorithms and 30 paired seeds. Define

`s_p = max(|f(x0)-f_ref|, 1e-12*(1+|f_ref|))`.

Targets are `f_ref + tau*s_p` for

`tau in {{1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}}`.

The first best-so-far history entry reaching each target defines runtime.
Unsuccessful runs are charged the full prescribed budget. ERT is the sum of
successful runtimes and failed-run budgets divided by the number of successes.
The primary ECDF uses function evaluations divided by dimension.

## Secondary analysis

- final normalized gaps;
- task-normalized final ranks;
- pairwise BasinGraph better/worse/tied counts;
- dimension-group descriptive summaries.

## Confirmatory statistics

Use 24 problems as blocks. Apply a Friedman test to per-problem median final
normalized gaps. Compare BasinGraph with each baseline using paired,
two-sided Wilcoxon signed-rank tests, Holm correction and paired rank-biserial
effect sizes.

No algorithm or parameter change is permitted after this holdout.
"""
    )

    final_run_id_path = PROTOCOL_ROOT / "CUTEST_HOLDOUT_FINAL_RUN_ID.txt"
    final_run_id_path.write_text(run_id + "\n")

    # Preserve all complete atomic jobs in one local evidence archive.
    archive_path = run_root / f"CUTEST_holdout_jobs_{run_id}.tar.gz"
    if not archive_path.exists():
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(jobs_root, arcname="jobs")

    locked_paths = [
        raw_path,
        validation_path,
        metadata_path,
        job_manifest_path,
        archive_path,
        holdout_list,
        preflight_path,
        execution_protocol,
        paper_code_contract,
        selection_decision,
        plan_json,
        plan_md,
        final_run_id_path,
    ]

    evidence_manifest_rows = [
        {
            "relative_path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in locked_paths
    ]
    evidence_manifest_path = (
        PROTOCOL_ROOT / "CUTEST_HOLDOUT_EVIDENCE_MANIFEST_SHA256.csv"
    )
    write_csv(evidence_manifest_path, evidence_manifest_rows)

    lock = {
        "status": "V2_RC1_CUTEST_HOLDOUT_EVIDENCE_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "implementation_version": "2.0.0-rc1",
        "options_hash": EXPECTED_OPTIONS_HASH,
        "problems": 24,
        "paired_seeds": 30,
        "algorithms": 7,
        "problem_seed_jobs": 720,
        "rows": 5040,
        "dimension_groups": validation["dimension_groups"],
        "development_instance_overlap": 0,
        "all_runs_completed": True,
        "performance_analysis_performed_before_freeze": False,
        "analysis_plan_sha256": sha256_file(plan_json),
        "code_freeze_tag": EXPECTED_CODE_TAG,
        "code_freeze_commit": git_output(
            "rev-list", "-n", "1", EXPECTED_CODE_TAG
        ),
        "final_candidate_tag": EXPECTED_CANDIDATE_TAG,
        "final_candidate_commit": git_output(
            "rev-list", "-n", "1", EXPECTED_CANDIDATE_TAG
        ),
        "runner_freeze_tag": EXPECTED_RUNNER_TAG,
        "runner_freeze_commit": git_output(
            "rev-list", "-n", "1", EXPECTED_RUNNER_TAG
        ),
        "job_manifest_entries": len(job_manifest),
        "job_manifest_verified": True,
        "total_job_bytes": total_job_bytes,
        "full_jobs_archive": str(archive_path.relative_to(ROOT)),
        "full_jobs_archive_sha256": sha256_file(archive_path),
        "evidence_manifest": str(
            evidence_manifest_path.relative_to(ROOT)
        ),
        "files": {
            row["relative_path"]: row["sha256"]
            for row in evidence_manifest_rows
        },
    }

    lock_path = PROTOCOL_ROOT / "CUTEST_HOLDOUT_EVIDENCE_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")

    print("CUTEST_HOLDOUT_EVIDENCE_FREEZE_OK")
    print("Run ID:", run_id)
    print("Job files verified:", len(job_manifest))
    print("Full jobs archive:", archive_path)
    print("Analysis plan:", plan_json)
    print("Evidence lock:", lock_path)


if __name__ == "__main__":
    main()
