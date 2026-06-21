#!/usr/bin/env python3
"""
Compare rc1 and rc2 on the identical frozen COCO development partition.

This script applies all frozen acceptance-gate criteria that can be evaluated
from run records. Official cocopp ECDF gates remain pending until the combined
rc1/rc2 cocopp output is reviewed.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASELINES = [
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2-run-id", required=True)
    parser.add_argument(
        "--rc1-result-root",
        default="results_v2/formal_development/coco_rc1",
    )
    parser.add_argument(
        "--rc2-result-root",
        default="results_v2/formal_development/coco_rc2",
    )
    return parser.parse_args()


def bbob_group(function_index: int) -> str:
    if 1 <= function_index <= 5:
        return "Separable (f1-f5)"
    if 6 <= function_index <= 9:
        return "Low/moderate conditioning (f6-f9)"
    if 10 <= function_index <= 14:
        return "High conditioning / unimodal (f10-f14)"
    if 15 <= function_index <= 19:
        return "Multimodal with global structure (f15-f19)"
    if 20 <= function_index <= 24:
        return "Multimodal with weak structure (f20-f24)"
    raise ValueError(function_index)


def main() -> None:
    args = parse_args()
    rc1_root = PROJECT_ROOT / args.rc1_result_root
    rc1_run_id = (rc1_root / "LAST_RUN_ID.txt").read_text().strip()
    rc1_run_root = rc1_root / rc1_run_id
    rc2_run_root = (
        PROJECT_ROOT / args.rc2_result_root / args.rc2_run_id
    )

    rc1 = pd.read_csv(
        rc1_run_root / "coco_v2_development_raw_results.csv"
    )
    rc2 = pd.read_csv(
        rc2_run_root / "coco_rc2_development_raw_results.csv"
    )
    rc2_validation = json.loads(
        (rc2_run_root / "validation_report.json").read_text()
    )
    gate = json.loads(
        (
            PROJECT_ROOT
            / "protocols"
            / "route_b"
            / "rc2_revision"
            / "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json"
        ).read_text()
    )

    rc1_bg = rc1[
        rc1["algorithm"] == "BasinGraph_v2"
    ].copy()
    rc1_bg["algorithm"] = "BasinGraph_rc1"

    rc2_for_rank = rc2.copy()
    rc2_for_rank["algorithm"] = "BasinGraph_rc2"

    baselines = rc1[rc1["algorithm"].isin(BASELINES)].copy()
    ranking = pd.concat(
        [
            rc2_for_rank,
            baselines,
        ],
        ignore_index=True,
        sort=False,
    )

    assert ranking["problem_id"].nunique() == 216
    assert len(ranking) == 216 * 7

    ranking["function_group"] = ranking["function_index"].map(
        bbob_group
    )
    ranking["final_value_rank"] = ranking.groupby(
        "problem_id"
    )["fbest"].rank(
        method="average",
        ascending=True,
    )
    ranking["final_value_win"] = ranking.groupby(
        "problem_id"
    )["fbest"].transform(
        lambda values: np.isclose(
            values,
            values.min(),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    overall = (
        ranking.groupby("algorithm")
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["mean_rank", "algorithm"])
    )

    by_group = (
        ranking.groupby(["function_group", "algorithm"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["function_group", "mean_rank", "algorithm"])
    )

    by_dimension = (
        ranking.groupby(["dimension", "algorithm"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["dimension", "mean_rank", "algorithm"])
    )

    paired = rc2[
        ["problem_id", "function_index", "dimension", "fbest"]
    ].rename(columns={"fbest": "rc2_fbest"}).merge(
        rc1_bg[
            ["problem_id", "fbest"]
        ].rename(columns={"fbest": "rc1_fbest"}),
        on="problem_id",
        validate="one_to_one",
    )
    tolerance = 1e-12 * (
        1.0
        + np.maximum(
            np.abs(paired["rc1_fbest"]),
            np.abs(paired["rc2_fbest"]),
        )
    )
    difference = paired["rc2_fbest"] - paired["rc1_fbest"]
    paired["rc2_better"] = difference < -tolerance
    paired["rc2_worse"] = difference > tolerance
    paired["tie"] = ~(
        paired["rc2_better"] | paired["rc2_worse"]
    )
    paired["rc2_minus_rc1"] = difference

    # Reconstruct group-dimension graph statistics from rc2 details.
    detail_rows = []
    for row in rc2.itertuples(index=False):
        with gzip.open(
            PROJECT_ROOT / row.detail_json_gz,
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)
        result = payload["result"]
        detail_rows.append(
            {
                "problem_id": row.problem_id,
                "function_group": bbob_group(int(row.function_index)),
                "dimension": int(row.dimension),
                "certified_nodes": len(result["archive"]),
                "graph_edges": len(result["graph_edges"]),
                "graph_edges_per_node": (
                    len(result["graph_edges"])
                    / max(len(result["archive"]), 1)
                ),
                "archive_saturated": (
                    len(result["archive"])
                    >= int(result["options"]["archive_max_size"])
                ),
                "center_local_active": bool(row.center_local_active),
                "curvature_anisotropy": result["diagnostics"][
                    "curvature_anisotropy"
                ],
                "principal_direction_nfe": int(
                    result["phase_evaluations"].get(
                        "principal_direction", 0
                    )
                ),
            }
        )
    details = pd.DataFrame(detail_rows)

    mechanism_summary = (
        details.groupby(["function_group", "dimension"])
        .agg(
            runs=("problem_id", "count"),
            archive_saturation_fraction=(
                "archive_saturated", "mean"
            ),
            mean_certified_nodes=("certified_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
            mean_edges_per_node=("graph_edges_per_node", "mean"),
            center_local_activation_fraction=(
                "center_local_active", "mean"
            ),
            mean_curvature_anisotropy=(
                "curvature_anisotropy", "mean"
            ),
            mean_principal_direction_nfe=(
                "principal_direction_nfe", "mean"
            ),
        )
        .reset_index()
    )

    rc2_overall_row = overall[
        overall["algorithm"] == "BasinGraph_rc2"
    ].iloc[0]
    weak_row = by_group[
        (by_group["algorithm"] == "BasinGraph_rc2")
        & (
            by_group["function_group"]
            == "Multimodal with weak structure (f20-f24)"
        )
    ].iloc[0]

    mechanism_results = {
        "archive_saturation_fraction": float(
            details["archive_saturated"].mean()
        ),
        "max_group_dimension_mean_edges_per_node": float(
            mechanism_summary["mean_edges_per_node"].max()
        ),
        "center_local_activation_fraction": float(
            details["center_local_active"].mean()
        ),
    }

    partial_gates = {
        "integrity": {
            "all_pass": all(
                [
                    rc2_validation["all_runs_completed"],
                    rc2_validation["exact_nfe_accounting"],
                    rc2_validation["all_archive_nodes_certified"],
                    (
                        rc2_validation["raw_probe_nodes_in_archive"]
                        == 0
                    ),
                    rc2_validation[
                        "graph_referential_integrity_verified"
                    ],
                    rc2_validation["landscape_anisotropy_nonconstant"],
                    rc2_validation[
                        "principal_direction_phase_exercised"
                    ],
                ]
            ),
            "observed": rc2_validation,
        },
        "mechanism": {
            "archive_saturation_pass": (
                mechanism_results["archive_saturation_fraction"]
                <= gate["mechanism_gates_all_required"][
                    "archive_saturation_fraction_max"
                ]
            ),
            "graph_density_pass": (
                mechanism_results[
                    "max_group_dimension_mean_edges_per_node"
                ]
                <= gate["mechanism_gates_all_required"][
                    "maximum_group_dimension_mean_edges_per_node"
                ]
            ),
            "center_local_pass": (
                mechanism_results[
                    "center_local_activation_fraction"
                ]
                <= gate["mechanism_gates_all_required"][
                    "center_local_activation_fraction_max"
                ]
            ),
            "observed": mechanism_results,
        },
        "noninferiority": {
            "fixed_budget_mean_rank_pass": (
                float(rc2_overall_row["mean_rank"])
                <= gate["noninferiority_gates_all_required"][
                    "fixed_budget_mean_rank_max"
                ]
            ),
            "weak_structure_mean_rank_pass": (
                float(weak_row["mean_rank"])
                <= gate["noninferiority_gates_all_required"][
                    "weak_structure_multimodal_mean_rank_max"
                ]
            ),
            "aggregate_ecdf_100d_pass": None,
            "observed": {
                "fixed_budget_mean_rank": float(
                    rc2_overall_row["mean_rank"]
                ),
                "weak_structure_mean_rank": float(
                    weak_row["mean_rank"]
                ),
            },
        },
        "improvement": {
            "hardest_target_successes_pass": (
                int(rc2_validation["hardest_target_successes"])
                >= gate["improvement_gates"][
                    "hardest_target_successes_min"
                ]
            ),
            "fixed_budget_mean_rank_target_pass": (
                float(rc2_overall_row["mean_rank"])
                <= gate["improvement_gates"][
                    "fixed_budget_mean_rank_target"
                ]
            ),
            "aggregate_ecdf_1000d_pass": None,
            "high_conditioning_ecdf_pass": None,
            "observed": {
                "hardest_target_successes": int(
                    rc2_validation["hardest_target_successes"]
                ),
                "fixed_budget_mean_rank": float(
                    rc2_overall_row["mean_rank"]
                ),
            },
        },
    }

    known_improvement_passes = sum(
        value is True
        for key, value in partial_gates["improvement"].items()
        if key.endswith("_pass")
    )

    partial_status = (
        "PENDING_COCOPP_REVIEW"
        if (
            partial_gates["integrity"]["all_pass"]
            and all(
                value
                for key, value in partial_gates["mechanism"].items()
                if key.endswith("_pass")
            )
            and all(
                value
                for key, value in partial_gates["noninferiority"].items()
                if key.endswith("_pass")
                and value is not None
            )
        )
        else "RC2_REJECTED_BEFORE_COCOPP"
    )

    report = {
        "status": partial_status,
        "rc1_run_id": rc1_run_id,
        "rc2_run_id": args.rc2_run_id,
        "development_only": True,
        "holdout_accessed": False,
        "paired_problems": len(paired),
        "rc2_better": int(paired["rc2_better"].sum()),
        "rc2_worse": int(paired["rc2_worse"].sum()),
        "ties": int(paired["tie"].sum()),
        "known_improvement_gates_passed": known_improvement_passes,
        "required_improvement_gates": gate[
            "improvement_gates_require_at_least"
        ],
        "gates": partial_gates,
        "final_decision": (
            "Pending official combined cocopp ECDF review."
            if partial_status == "PENDING_COCOPP_REVIEW"
            else "Reject rc2 without opening holdout."
        ),
    }

    out = rc2_run_root / "paired_rc1_rc2_comparison"
    out.mkdir(parents=True, exist_ok=True)

    overall.to_csv(out / "rc2_with_baselines_overall.csv", index=False)
    by_group.to_csv(out / "rc2_with_baselines_by_group.csv", index=False)
    by_dimension.to_csv(
        out / "rc2_with_baselines_by_dimension.csv", index=False
    )
    paired.to_csv(out / "paired_rc1_rc2_final_values.csv", index=False)
    mechanism_summary.to_csv(
        out / "rc2_mechanism_by_group_dimension.csv", index=False
    )
    (out / "partial_acceptance_gate.json").write_text(
        json.dumps(report, indent=2)
    )

    # Update validation report with the frozen mechanism statistic.
    rc2_validation[
        "max_group_dimension_mean_edges_per_node"
    ] = mechanism_results[
        "max_group_dimension_mean_edges_per_node"
    ]
    (rc2_run_root / "validation_report.json").write_text(
        json.dumps(rc2_validation, indent=2)
    )

    print("V2_RC1_RC2_DEVELOPMENT_COMPARISON_OK")
    print(json.dumps(report, indent=2))
    print("\nrc2 with baselines:")
    print(overall.to_string(index=False))
    print("\nMechanism summary:")
    print(mechanism_summary.to_string(index=False))


if __name__ == "__main__":
    main()
