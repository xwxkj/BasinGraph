#!/usr/bin/env python3
"""Validate the completed prospective COCO holdout before interpretation."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    run_root = (
        ROOT
        / "results_v2"
        / "formal_holdout"
        / "coco_rc1"
        / args.run_id
    )
    raw_path = run_root / "coco_rc1_holdout_raw_results.csv"
    launch_path = run_root / "launch_metadata.json"

    df = pd.read_csv(raw_path)
    launch = json.loads(launch_path.read_text())

    expected_problems = 24 * 5 * 12
    expected_rows = expected_problems * 7

    assert launch["status"] == "COCO_HOLDOUT_ALGORITHMS_COMPLETE"
    assert len(df) == expected_rows, (len(df), expected_rows)
    assert df["problem_id"].nunique() == expected_problems
    assert set(df["algorithm"].unique()) == set(ALGORITHMS)
    assert set(df["instance_index"].unique()) == set(range(4, 16))
    assert set(df["dimension"].unique()) == {2, 3, 5, 10, 20}
    assert (df["partition"] == "prospective_holdout").all()
    assert (df["runner_status"] == "completed").all()
    assert not df["error"].fillna("").astype(str).str.len().gt(0).any()
    assert (df["nfe_internal"] <= df["budget"]).all()
    assert (df["nfe_observer"] <= df["budget"]).all()
    assert (df["nfe_internal"] == df["nfe_observer"]).all()

    counts = df.groupby("algorithm").size()
    assert (counts == expected_problems).all()

    paired_seeds = df.groupby("problem_id")["seed"].nunique()
    assert (paired_seeds == 1).all()

    bg = df[df["algorithm"] == "BasinGraph_v2"].copy()
    assert len(bg) == expected_problems
    assert (bg["implementation_version"] == "2.0.0-rc1").all()
    assert (bg["options_hash"] == EXPECTED_OPTIONS_HASH).all()
    assert (bg["nfe_internal"] == bg["budget"]).all()
    assert (bg["archive_nodes"] > 0).all()

    detail_rows = []
    total_nodes = 0
    total_edges = 0

    for row in bg.itertuples(index=False):
        detail_path = ROOT / row.detail_json_gz
        assert detail_path.exists(), detail_path

        with gzip.open(
            detail_path,
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        result = payload["result"]
        metadata = payload["run_metadata"]

        assert result["implementation_version"] == "2.0.0-rc1"
        assert result["options_hash"] == EXPECTED_OPTIONS_HASH
        assert result["nfe"] == int(row.budget)
        assert sum(result["phase_evaluations"].values()) == result["nfe"]
        assert len(result["archive"]) == int(row.archive_nodes)
        assert len(result["graph_edges"]) == int(row.graph_edges)
        assert metadata["instance_index"] in range(4, 16)
        assert metadata["problem_id"] == row.problem_id

        active = {
            int(node["node_id"])
            for node in result["archive"]
        }
        assert all(
            int(edge["source_id"]) in active
            and int(edge["target_id"]) in active
            for edge in result["graph_edges"]
        )

        total_nodes += len(result["archive"])
        total_edges += len(result["graph_edges"])
        detail_rows.append(
            {
                "problem_id": row.problem_id,
                "relative_path": row.detail_json_gz,
                "sha256": sha256_file(detail_path),
                "archive_nodes": len(result["archive"]),
                "graph_edges": len(result["graph_edges"]),
            }
        )

    manifest_path = run_root / "BasinGraph_holdout_detail_manifest.csv"
    pd.DataFrame(detail_rows).to_csv(manifest_path, index=False)

    exdata_root = (
        ROOT
        / "exdata"
        / "routeb_v2_holdout"
        / args.run_id
    )
    info_files = list(exdata_root.rglob("*.info"))
    dat_files = list(exdata_root.rglob("*.dat"))

    assert len(info_files) == 168, len(info_files)
    assert len(dat_files) == 840, len(dat_files)

    report = {
        "status": "V2_RC1_COCO_HOLDOUT_VALIDATION_OK",
        "run_id": args.run_id,
        "rows": len(df),
        "problems": df["problem_id"].nunique(),
        "algorithms": df["algorithm"].nunique(),
        "instances": list(range(4, 16)),
        "dimensions": [2, 3, 5, 10, 20],
        "development_instance_leakage": False,
        "all_runs_completed": True,
        "paired_seeds_verified": True,
        "internal_observer_nfe_match": True,
        "basingraph_full_budget": True,
        "phase_accounting_verified": True,
        "graph_referential_integrity_verified": True,
        "basingraph_detail_files": len(detail_rows),
        "total_archive_nodes": total_nodes,
        "total_graph_edges": total_edges,
        "coco_info_files": len(info_files),
        "coco_dat_files": len(dat_files),
        "prospective_holdout": True,
        "raw_results_sha256": sha256_file(raw_path),
        "launch_metadata_sha256": sha256_file(launch_path),
        "detail_manifest_sha256": sha256_file(manifest_path),
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("V2_RC1_COCO_HOLDOUT_VALIDATION_OK")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
