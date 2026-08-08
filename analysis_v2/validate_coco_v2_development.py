#!/usr/bin/env python3
"""Validate and summarize the formal COCO v2 development run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
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
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_development/coco_rc1",
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
    run_root = PROJECT_ROOT / args.result_root / args.run_id
    raw_path = run_root / "coco_v2_development_raw_results.csv"
    metadata_path = run_root / "run_metadata.json"

    df = pd.read_csv(raw_path)
    metadata = json.loads(metadata_path.read_text())

    expected_rows = 24 * 3 * 3 * 7
    expected_problems = 24 * 3 * 3

    assert len(df) == expected_rows, (len(df), expected_rows)
    assert df["problem_id"].nunique() == expected_problems
    assert set(df["algorithm"].unique()) == set(ALGORITHMS)
    assert set(df["instance_index"].unique()) == {1, 2, 3}
    assert set(df["dimension"].unique()) == {2, 5, 10}
    assert (df["partition"] == "development").all()
    assert (df["runner_status"] == "completed").all()
    assert not df["error"].fillna("").astype(str).str.len().gt(0).any()
    assert (df["nfe_internal"] <= df["budget"]).all()
    assert (df["nfe_observer"] <= df["budget"]).all()
    assert (df["nfe_internal"] == df["nfe_observer"]).all()

    # Paired seed policy: one identical seed across all algorithms per problem.
    paired_seeds = df.groupby("problem_id")["seed"].nunique()
    assert (paired_seeds == 1).all()

    counts = df.groupby("algorithm").size()
    assert (counts == expected_problems).all()

    bg = df[df["algorithm"] == "BasinGraph_v2"].copy()
    assert len(bg) == expected_problems
    assert (bg["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
    assert (bg["options_hash"] == EXPECTED_OPTIONS_HASH).all()
    assert (bg["nfe_internal"] == bg["budget"]).all()
    assert (bg["archive_nodes"] > 0).all()
    assert bg["graph_edges"].sum() > 0

    total_nodes = 0
    total_edges = 0
    detail_hash_rows = []

    for row in bg.itertuples(index=False):
        path = PROJECT_ROOT / row.detail_json_gz
        assert path.exists(), path

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        result = payload["result"]
        run_meta = payload["run_metadata"]

        assert result["implementation_version"] == EXPECTED_IMPLEMENTATION
        assert result["options_hash"] == EXPECTED_OPTIONS_HASH
        assert result["nfe"] == int(row.budget)
        assert sum(result["phase_evaluations"].values()) == result["nfe"]
        assert len(result["archive"]) == int(row.archive_nodes)
        assert len(result["graph_edges"]) == int(row.graph_edges)
        assert run_meta["problem_id"] == row.problem_id
        assert run_meta["instance_index"] in {1, 2, 3}

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
        detail_hash_rows.append(
            {
                "problem_id": row.problem_id,
                "relative_path": row.detail_json_gz,
                "sha256": sha256_file(path),
                "archive_nodes": len(result["archive"]),
                "graph_edges": len(result["graph_edges"]),
            }
        )

    # Development-only final-value summaries.
    df["final_value_rank"] = df.groupby("problem_id")["fbest"].rank(
        method="average",
        ascending=True,
    )
    df["final_value_win"] = df.groupby("problem_id")["fbest"].transform(
        lambda values: np.isclose(
            values,
            values.min(),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    summary = (
        df.groupby(["algorithm", "algorithm_display"])
        .agg(
            mean_final_value_rank=("final_value_rank", "mean"),
            median_final_value_rank=("final_value_rank", "median"),
            final_value_wins=("final_value_win", "sum"),
            median_budget_ratio=(
                "nfe_internal",
                lambda values: np.nan,
            ),
            rows=("problem_id", "count"),
        )
        .reset_index()
    )

    budget_ratio = (
        df.assign(
            budget_ratio=df["nfe_internal"] / df["budget"]
        )
        .groupby("algorithm")["budget_ratio"]
        .median()
    )
    summary["median_budget_ratio"] = summary["algorithm"].map(
        budget_ratio
    )
    summary.sort_values(
        ["mean_final_value_rank", "algorithm"],
        inplace=True,
    )

    dimension_summary = (
        df.groupby(["dimension", "algorithm"])
        .agg(
            mean_final_value_rank=("final_value_rank", "mean"),
            final_value_wins=("final_value_win", "sum"),
            rows=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(
            ["dimension", "mean_final_value_rank", "algorithm"]
        )
    )

    summary_path = run_root / "coco_v2_development_algorithm_summary.csv"
    dimension_path = run_root / "coco_v2_development_dimension_summary.csv"
    detail_manifest = run_root / "BasinGraph_v2_detail_manifest.csv"

    summary.to_csv(summary_path, index=False)
    dimension_summary.to_csv(dimension_path, index=False)
    pd.DataFrame(detail_hash_rows).to_csv(
        detail_manifest,
        index=False,
    )

    exdata_root = (
        PROJECT_ROOT
        / "exdata"
        / "routeb_v2_development"
        / args.run_id
    )
    info_files = list(exdata_root.rglob("*.info"))
    dat_files = list(exdata_root.rglob("*.dat"))

    assert len(info_files) == 24 * 7, len(info_files)
    assert len(dat_files) == 24 * 3 * 7, len(dat_files)

    report = {
        "status": "V2_COCO_DEVELOPMENT_VALIDATION_OK",
        "run_id": args.run_id,
        "rows": len(df),
        "problems": df["problem_id"].nunique(),
        "algorithms": df["algorithm"].nunique(),
        "instances": sorted(
            int(value)
            for value in df["instance_index"].unique()
        ),
        "dimensions": sorted(
            int(value)
            for value in df["dimension"].unique()
        ),
        "holdout_leakage": False,
        "all_runs_completed": True,
        "paired_seeds_verified": True,
        "internal_observer_nfe_match": True,
        "basingraph_full_budget": True,
        "phase_accounting_verified": True,
        "graph_referential_integrity_verified": True,
        "basingraph_detail_files": len(detail_hash_rows),
        "total_archive_nodes": total_nodes,
        "total_graph_edges": total_edges,
        "coco_info_files": len(info_files),
        "coco_dat_files": len(dat_files),
        "development_only": True,
        "best_final_value_mean_rank_algorithm": str(
            summary.iloc[0]["algorithm"]
        ),
        "best_final_value_mean_rank": float(
            summary.iloc[0]["mean_final_value_rank"]
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip(),
        "raw_results_sha256": sha256_file(raw_path),
        "metadata_sha256": sha256_file(metadata_path),
    }

    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("V2_COCO_DEVELOPMENT_VALIDATION_OK")
    print(json.dumps(report, indent=2))
    print("\nDevelopment-only algorithm summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
