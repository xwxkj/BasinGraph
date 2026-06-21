#!/usr/bin/env python3
"""Integrity validation for the complete prospective CUTEst holdout."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
ALGORITHMS = [
    "BasinGraph_v2",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]
HOLDOUT_LIST = (
    ROOT
    / "protocols"
    / "route_b"
    / "formal_v2_protocol"
    / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
)
DEVELOPMENT_LIST = ROOT / "protocols" / "cutest_pre_registered_problem_list_v2.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_holdout/cutest_rc1",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_root = ROOT / args.result_root / args.run_id
    raw_path = run_root / "cutest_holdout_raw_results_all_available.csv"
    metadata_path = run_root / "run_metadata.json"

    raw = pd.read_csv(raw_path)
    metadata = json.loads(metadata_path.read_text())
    holdout = pd.read_csv(HOLDOUT_LIST)
    development = pd.read_csv(DEVELOPMENT_LIST)

    expected_jobs = 24 * 30
    expected_rows = expected_jobs * 7

    assert len(raw) == expected_rows, (len(raw), expected_rows)
    assert raw["instance_id"].nunique() == 24
    assert raw["algorithm"].nunique() == 7
    assert set(raw["algorithm"].unique()) == set(ALGORITHMS)
    assert raw["protocol_seed_index"].nunique() == 30
    assert sorted(raw["protocol_seed_index"].unique()) == list(range(30))
    assert (raw["runner_status"] == "completed").all()
    assert not raw["error"].fillna("").astype(str).str.len().gt(0).any()

    overlap = set(holdout["instance_id"].astype(str)).intersection(
        set(development["instance_id"].astype(str))
    )
    assert not overlap

    counts = raw.groupby(["instance_id", "protocol_seed_index"]).size()
    assert (counts == 7).all()
    seed_pairing = raw.groupby(
        ["instance_id", "protocol_seed_index"]
    )["seed"].nunique()
    assert (seed_pairing == 1).all()

    expected_budget = raw["dimension"].map(
        lambda dimension: min(20_000, max(1_000, 50 * int(dimension)))
    )
    assert np.array_equal(
        raw["budget"].to_numpy(dtype=int),
        expected_budget.to_numpy(dtype=int),
    )
    assert (raw["nfe"] <= raw["budget"]).all()
    assert np.allclose(
        raw["budget_ratio"],
        raw["nfe"] / raw["budget"],
        rtol=0,
        atol=1e-12,
    )

    bg = raw[raw["algorithm"] == "BasinGraph_v2"].copy()
    assert len(bg) == expected_jobs
    assert (bg["implementation_version"] == "2.0.0-rc1").all()
    assert (bg["options_hash"] == EXPECTED_OPTIONS_HASH).all()
    assert (bg["nfe"] == bg["budget"]).all()
    assert (bg["archive_nodes"] > 0).all()

    expected_group_rows = {
        "small_2_20": 11 * 30 * 7,
        "medium_21_100": 7 * 30 * 7,
        "large_101_500": 6 * 30 * 7,
    }
    assert raw["dimension_group"].value_counts().to_dict() == expected_group_rows

    job_files = sorted((run_root / "jobs").rglob("*.json.gz"))
    assert len(job_files) == expected_jobs, len(job_files)

    manifest_rows = []
    total_archive_nodes = 0
    total_graph_edges = 0

    for job_path in job_files:
        with gzip.open(job_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert len(payload["rows"]) == 7
        assert set(row["algorithm"] for row in payload["rows"]) == set(ALGORITHMS)
        assert all(row["runner_status"] == "completed" for row in payload["rows"])

        detail = payload["basingraph_result"]
        assert detail is not None
        assert detail["implementation_version"] == "2.0.0-rc1"
        assert detail["options_hash"] == EXPECTED_OPTIONS_HASH
        assert sum(detail["phase_evaluations"].values()) == detail["nfe"]
        assert len(detail["archive"]) > 0

        active = {
            int(node["node_id"])
            for node in detail["archive"]
        }
        assert all(
            int(edge["source_id"]) in active
            and int(edge["target_id"]) in active
            for edge in detail["graph_edges"]
        )

        total_archive_nodes += len(detail["archive"])
        total_graph_edges += len(detail["graph_edges"])
        manifest_rows.append(
            {
                "relative_path": str(job_path.relative_to(ROOT)),
                "sha256": sha256_file(job_path),
                "size_bytes": job_path.stat().st_size,
                "instance_id": payload["job_metadata"]["instance_id"],
                "protocol_seed_index": payload["job_metadata"][
                    "protocol_seed_index"
                ],
                "archive_nodes": len(detail["archive"]),
                "graph_edges": len(detail["graph_edges"]),
            }
        )

    manifest_path = run_root / "CUTEST_HOLDOUT_JOB_MANIFEST_SHA256.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    failures = list((run_root / "failures").glob("*.json")) if (
        run_root / "failures"
    ).exists() else []
    assert not failures, failures

    report = {
        "status": "V2_RC1_CUTEST_HOLDOUT_VALIDATION_OK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "partition": "prospective_holdout",
        "rows": len(raw),
        "problem_seed_jobs": expected_jobs,
        "problems": raw["instance_id"].nunique(),
        "algorithms": raw["algorithm"].nunique(),
        "paired_seeds": raw["protocol_seed_index"].nunique(),
        "seed_indices": list(range(30)),
        "dimension_groups": raw["dimension_group"].value_counts().sort_index().to_dict(),
        "development_instance_overlap": 0,
        "all_runs_completed": True,
        "paired_seeds_verified": True,
        "budget_formula_verified": True,
        "basingraph_full_budget": True,
        "phase_accounting_verified": True,
        "graph_referential_integrity_verified": True,
        "job_files": len(job_files),
        "basingraph_detail_records": len(bg),
        "total_archive_nodes": total_archive_nodes,
        "total_graph_edges": total_graph_edges,
        "performance_analysis_performed": False,
        "holdout_list_sha256": sha256_file(HOLDOUT_LIST),
        "raw_results_sha256": sha256_file(raw_path),
        "run_metadata_sha256": sha256_file(metadata_path),
        "job_manifest_sha256": sha256_file(manifest_path),
    }
    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("V2_RC1_CUTEST_HOLDOUT_VALIDATION_OK")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
