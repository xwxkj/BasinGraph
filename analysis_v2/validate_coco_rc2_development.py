#!/usr/bin/env python3
"""Validate rc2 development records and machine-level invariants."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IMPLEMENTATION = "2.0.0-rc2"
EXPECTED_OPTIONS_HASH = (
    "15fe9bcbf8e87aabe4767f811524c00f"
    "67b74e3ebfa31fa81cdf6f461cbfeb08"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_development/coco_rc2",
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
    raw_path = run_root / "coco_rc2_development_raw_results.csv"
    metadata_path = run_root / "run_metadata.json"

    df = pd.read_csv(raw_path)
    metadata = json.loads(metadata_path.read_text())

    assert len(df) == 216
    assert df["problem_id"].nunique() == 216
    assert set(df["instance_index"].unique()) == {1, 2, 3}
    assert set(df["dimension"].unique()) == {2, 5, 10}
    assert (df["partition"] == "development").all()
    assert (df["runner_status"] == "completed").all()
    assert not df["error"].fillna("").astype(str).str.len().gt(0).any()
    assert (df["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
    assert (df["options_hash"] == EXPECTED_OPTIONS_HASH).all()
    assert (df["nfe_internal"] == df["budget"]).all()
    assert (df["nfe_observer"] == df["budget"]).all()
    assert (df["nfe_internal"] == df["nfe_observer"]).all()
    assert (df["certified_nodes"] > 0).all()
    assert (df["probe_count_total"] > 0).all()
    assert (df["principal_directions"] > 0).all()
    assert (df["max_outgoing_degree"] <= 3).all()
    assert (df["max_incoming_degree"] <= 3).all()

    rc1_raw_path = (
        PROJECT_ROOT
        / "results_v2"
        / "formal_development"
        / "coco_rc1"
        / metadata["rc1_run_id"]
        / "coco_v2_development_raw_results.csv"
    )
    rc1 = pd.read_csv(rc1_raw_path)
    rc1_seeds = (
        rc1.groupby("problem_id")["seed"]
        .first()
        .rename("rc1_seed")
        .reset_index()
    )
    paired = df.merge(
        rc1_seeds,
        on="problem_id",
        validate="one_to_one",
    )
    assert (paired["seed"] == paired["rc1_seed"]).all()

    detail_rows = []
    total_nodes = 0
    total_edges = 0
    curvature_values = []

    for row in df.itertuples(index=False):
        detail_path = PROJECT_ROOT / row.detail_json_gz
        assert detail_path.exists(), detail_path

        with gzip.open(
            detail_path,
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        result = payload["result"]
        assert result["implementation_version"] == EXPECTED_IMPLEMENTATION
        assert result["options_hash"] == EXPECTED_OPTIONS_HASH
        assert result["nfe"] == int(row.budget)
        assert sum(result["phase_evaluations"].values()) == result["nfe"]
        assert result["probe_count_total"] >= len(result["probes"])
        assert len(result["archive"]) == int(row.certified_nodes)
        assert len(result["graph_edges"]) == int(row.graph_edges)
        assert all(node["certified"] for node in result["archive"])
        assert all(
            node["certification_mode"]
            not in {"initial_design", "far_basin", "budget_completion"}
            for node in result["archive"]
        )

        active = {
            int(node["node_id"])
            for node in result["archive"]
        }
        assert all(
            int(edge["source_id"]) in active
            and int(edge["target_id"]) in active
            for edge in result["graph_edges"]
        )

        outgoing: dict[int, int] = {}
        incoming: dict[int, int] = {}
        for edge in result["graph_edges"]:
            source = int(edge["source_id"])
            target = int(edge["target_id"])
            outgoing[source] = outgoing.get(source, 0) + 1
            incoming[target] = incoming.get(target, 0) + 1
        assert max(outgoing.values(), default=0) <= 3
        assert max(incoming.values(), default=0) <= 3

        principal_nfe = int(
            result["phase_evaluations"].get("principal_direction", 0)
        )
        assert principal_nfe > 0
        assert (
            result["direction_diagnostics"]["retained_directions"] > 0
        )

        total_nodes += len(result["archive"])
        total_edges += len(result["graph_edges"])
        curvature_values.append(
            float(result["diagnostics"]["curvature_anisotropy"])
        )

        detail_rows.append(
            {
                "problem_id": row.problem_id,
                "relative_path": row.detail_json_gz,
                "sha256": sha256_file(detail_path),
                "probe_count_total": result["probe_count_total"],
                "certified_nodes": len(result["archive"]),
                "graph_edges": len(result["graph_edges"]),
                "principal_direction_nfe": principal_nfe,
                "curvature_anisotropy": result["diagnostics"][
                    "curvature_anisotropy"
                ],
            }
        )

    curvature_array = np.asarray(curvature_values, dtype=float)
    finite_curvature = curvature_array[np.isfinite(curvature_array)]
    assert len(np.unique(np.round(finite_curvature, 12))) > 1
    assert np.max(finite_curvature) > 1.0

    manifest_path = run_root / "BasinGraph_rc2_detail_manifest.csv"
    pd.DataFrame(detail_rows).to_csv(manifest_path, index=False)

    exdata_root = (
        PROJECT_ROOT
        / "exdata"
        / "routeb_v2_rc2_development"
        / args.run_id
        / "BasinGraph_v2_rc2"
    )
    info_files = list(exdata_root.rglob("*.info"))
    dat_files = list(exdata_root.rglob("*.dat"))
    assert len(info_files) == 24, len(info_files)
    assert len(dat_files) == 72, len(dat_files)

    report = {
        "status": "V2_RC2_COCO_DEVELOPMENT_VALIDATION_OK",
        "run_id": args.run_id,
        "rc1_run_id": metadata["rc1_run_id"],
        "rows": len(df),
        "problems": df["problem_id"].nunique(),
        "instances": [1, 2, 3],
        "dimensions": [2, 5, 10],
        "holdout_leakage": False,
        "all_runs_completed": True,
        "paired_rc1_seeds_verified": True,
        "exact_nfe_accounting": True,
        "all_archive_nodes_certified": True,
        "raw_probe_nodes_in_archive": 0,
        "graph_referential_integrity_verified": True,
        "graph_degree_caps_verified": True,
        "landscape_anisotropy_nonconstant": True,
        "principal_direction_phase_exercised": True,
        "detail_files": len(detail_rows),
        "total_certified_nodes": total_nodes,
        "total_graph_edges": total_edges,
        "hardest_target_successes": int(
            df["final_target_hit"].astype(bool).sum()
        ),
        "archive_saturation_fraction": float(
            df["archive_saturated"].astype(bool).mean()
        ),
        "center_local_activation_fraction": float(
            df["center_local_active"].astype(bool).mean()
        ),
        "max_group_dimension_mean_edges_per_node": None,
        "coco_info_files": len(info_files),
        "coco_dat_files": len(dat_files),
        "raw_results_sha256": sha256_file(raw_path),
        "metadata_sha256": sha256_file(metadata_path),
        "detail_manifest_sha256": sha256_file(manifest_path),
    }

    report_path = run_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("V2_RC2_COCO_DEVELOPMENT_VALIDATION_OK")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
