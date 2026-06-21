#!/usr/bin/env python3
"""
Route B Step B6C: diagnostic audit of the frozen COCO development partition.

This script reads only:
- COCO instances 1-3 development results;
- BasinGraph v2 compressed detail records;
- the existing cocopp output path.

It does not instantiate COCO, does not run an optimizer, and does not access
holdout instances 4-15.

The goal is to decide whether v2.0.0-rc1 needs one final algorithm revision
before the final code tag and prospective holdout evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALGORITHM_ORDER = [
    "BIPOP_CMA_ES",
    "CMA_ES",
    "DE",
    "BasinGraph_v2",
    "MS_LBFGSB",
    "Random",
    "LHS",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id",
        default=None,
        help="Defaults to LAST_RUN_ID.txt.",
    )
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


def event_payload(
    events: list[dict[str, Any]],
    event_name: str,
) -> dict[str, Any]:
    for event in events:
        if event.get("event") == event_name:
            return dict(event.get("payload", {}))
    return {}


def safe_spearman(
    x: pd.Series,
    y: pd.Series,
) -> tuple[float, float]:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    if mask.sum() < 5:
        return np.nan, np.nan
    x_values = x.to_numpy(dtype=float)[mask]
    y_values = y.to_numpy(dtype=float)[mask]
    if np.allclose(x_values, x_values[0]) or np.allclose(
        y_values, y_values[0]
    ):
        return np.nan, np.nan
    result = spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    args = parse_args()
    result_root = PROJECT_ROOT / args.result_root

    if args.run_id is None:
        run_id = (result_root / "LAST_RUN_ID.txt").read_text().strip()
    else:
        run_id = args.run_id

    run_root = result_root / run_id
    raw_path = run_root / "coco_v2_development_raw_results.csv"
    validation_path = run_root / "validation_report.json"
    cocopp_path_file = run_root / "cocopp_output_path.txt"

    for path in [raw_path, validation_path, cocopp_path_file]:
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path)
    validation = json.loads(validation_path.read_text())

    assert validation["status"] == "V2_COCO_DEVELOPMENT_VALIDATION_OK"
    assert set(raw["instance_index"].unique()) == {1, 2, 3}
    assert raw["problem_id"].nunique() == 216
    assert len(raw) == 1512

    out = run_root / "development_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Development-only final-value summaries
    # --------------------------------------------------------------
    raw["function_group"] = raw["function_index"].map(bbob_group)
    raw["final_value_rank"] = raw.groupby("problem_id")["fbest"].rank(
        method="average",
        ascending=True,
    )
    raw["final_value_win"] = raw.groupby("problem_id")["fbest"].transform(
        lambda values: np.isclose(
            values,
            values.min(),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    overall = (
        raw.groupby(["algorithm", "algorithm_display"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            median_budget_ratio=(
                "nfe_internal",
                lambda values: np.nan,
            ),
            problems=("problem_id", "count"),
        )
        .reset_index()
    )
    budget_ratios = (
        raw.assign(
            budget_ratio=raw["nfe_internal"] / raw["budget"]
        )
        .groupby("algorithm")["budget_ratio"]
        .median()
    )
    overall["median_budget_ratio"] = overall["algorithm"].map(
        budget_ratios
    )
    overall["algorithm_order"] = overall["algorithm"].map(
        {name: i for i, name in enumerate(ALGORITHM_ORDER)}
    )
    overall.sort_values(
        ["mean_rank", "algorithm_order"],
        inplace=True,
    )
    overall.drop(columns=["algorithm_order"], inplace=True)

    by_dimension = (
        raw.groupby(["dimension", "algorithm"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["dimension", "mean_rank", "algorithm"])
    )

    by_group = (
        raw.groupby(["function_group", "algorithm"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["function_group", "mean_rank", "algorithm"])
    )

    by_function = (
        raw.groupby(["function_index", "algorithm"])
        .agg(
            mean_rank=("final_value_rank", "mean"),
            median_rank=("final_value_rank", "median"),
            wins=("final_value_win", "sum"),
            problems=("problem_id", "count"),
        )
        .reset_index()
        .sort_values(["function_index", "mean_rank", "algorithm"])
    )

    bg_values = raw[
        raw["algorithm"] == "BasinGraph_v2"
    ][["problem_id", "fbest"]].rename(
        columns={"fbest": "basingraph_fbest"}
    )

    pairwise_rows = []
    for baseline in [
        name for name in ALGORITHM_ORDER if name != "BasinGraph_v2"
    ]:
        baseline_values = raw[
            raw["algorithm"] == baseline
        ][["problem_id", "fbest"]].rename(
            columns={"fbest": "baseline_fbest"}
        )
        paired = bg_values.merge(
            baseline_values,
            on="problem_id",
            validate="one_to_one",
        )
        tolerance = 1e-12 * (
            1.0
            + np.maximum(
                np.abs(paired["basingraph_fbest"]),
                np.abs(paired["baseline_fbest"]),
            )
        )
        difference = (
            paired["basingraph_fbest"] - paired["baseline_fbest"]
        )
        better = difference < -tolerance
        worse = difference > tolerance
        tie = ~(better | worse)

        pairwise_rows.append(
            {
                "baseline": baseline,
                "paired_problems": len(paired),
                "basingraph_better": int(better.sum()),
                "basingraph_worse": int(worse.sum()),
                "ties": int(tie.sum()),
                "paired_win_probability": float(
                    (better.sum() + 0.5 * tie.sum()) / len(paired)
                ),
                "median_bg_minus_baseline": float(
                    np.median(difference)
                ),
            }
        )
    pairwise = pd.DataFrame(pairwise_rows)

    # --------------------------------------------------------------
    # 2. BasinGraph internal diagnostics
    # --------------------------------------------------------------
    bg_raw = raw[raw["algorithm"] == "BasinGraph_v2"].copy()
    detail_rows = []

    for row in bg_raw.itertuples(index=False):
        detail_path = PROJECT_ROOT / row.detail_json_gz
        with gzip.open(
            detail_path,
            "rt",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)

        result = payload["result"]
        events = result["event_log"]
        controller = event_payload(events, "controller_decision")
        options = result["options"]
        archive = result["archive"]
        edges = result["graph_edges"]
        phases = result["phase_evaluations"]
        diagnostics = result["diagnostics"]

        visits = np.asarray(
            [node["visits"] for node in archive],
            dtype=float,
        )
        novelty = np.asarray(
            [node["novelty"] for node in archive],
            dtype=float,
        )
        access = np.asarray(
            [edge["accessibility"] for edge in edges],
            dtype=float,
        )
        improvements = np.asarray(
            [edge["best_improvement"] for edge in edges],
            dtype=float,
        )

        record = {
            "problem_id": row.problem_id,
            "function_index": int(row.function_index),
            "function_group": bbob_group(int(row.function_index)),
            "dimension": int(row.dimension),
            "instance_index": int(row.instance_index),
            "fbest": float(row.fbest),
            "final_value_rank": float(row.final_value_rank),
            "budget": int(row.budget),
            "nfe": int(result["nfe"]),
            "archive_nodes": len(archive),
            "archive_capacity": int(options["archive_max_size"]),
            "archive_saturated": (
                len(archive) >= int(options["archive_max_size"])
            ),
            "graph_edges": len(edges),
            "graph_edges_per_node": (
                len(edges) / max(len(archive), 1)
            ),
            "mean_node_visits": float(visits.mean())
            if len(visits) else np.nan,
            "max_node_visits": float(visits.max())
            if len(visits) else np.nan,
            "mean_node_novelty": float(novelty.mean())
            if len(novelty) else np.nan,
            "mean_edge_accessibility": float(access.mean())
            if len(access) else np.nan,
            "max_edge_accessibility": float(access.max())
            if len(access) else np.nan,
            "positive_improvement_edges": int(
                (improvements > 0).sum()
            ) if len(improvements) else 0,
            "positive_improvement_edge_fraction": float(
                (improvements > 0).mean()
            ) if len(improvements) else 0.0,
            "use_center_local": bool(
                controller.get("use_center_local", False)
            ),
            "use_far_basin": bool(
                controller.get("use_far_basin", False)
            ),
            **{
                f"diag_{key}": value
                for key, value in diagnostics.items()
            },
        }

        for phase_name, count in phases.items():
            record[f"phase_{phase_name}_nfe"] = int(count)
            record[f"phase_{phase_name}_fraction"] = (
                int(count) / max(int(result["nfe"]), 1)
            )

        detail_rows.append(record)

    details = pd.DataFrame(detail_rows)

    phase_fraction_columns = sorted(
        column
        for column in details.columns
        if column.startswith("phase_")
        and column.endswith("_fraction")
    )

    internal_overall = {
        "runs": len(details),
        "archive_saturation_fraction": float(
            details["archive_saturated"].mean()
        ),
        "mean_archive_nodes": float(
            details["archive_nodes"].mean()
        ),
        "mean_graph_edges": float(
            details["graph_edges"].mean()
        ),
        "mean_graph_edges_per_node": float(
            details["graph_edges_per_node"].mean()
        ),
        "mean_positive_improvement_edge_fraction": float(
            details["positive_improvement_edge_fraction"].mean()
        ),
        "center_local_activation_fraction": float(
            details["use_center_local"].mean()
        ),
        "far_basin_activation_fraction": float(
            details["use_far_basin"].mean()
        ),
        "mean_phase_fractions": {
            column: float(details[column].mean())
            for column in phase_fraction_columns
        },
    }

    internal_by_group = (
        details.groupby(["function_group", "dimension"])
        .agg(
            runs=("problem_id", "count"),
            mean_final_rank=("final_value_rank", "mean"),
            archive_saturation_fraction=(
                "archive_saturated",
                "mean",
            ),
            mean_archive_nodes=("archive_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
            mean_graph_edges_per_node=(
                "graph_edges_per_node",
                "mean",
            ),
            mean_positive_improvement_edge_fraction=(
                "positive_improvement_edge_fraction",
                "mean",
            ),
            center_local_activation_fraction=(
                "use_center_local",
                "mean",
            ),
            far_basin_activation_fraction=(
                "use_far_basin",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------------
    # 3. Correlations and development hypotheses
    # --------------------------------------------------------------
    correlation_features = [
        "archive_nodes",
        "graph_edges",
        "graph_edges_per_node",
        "mean_node_visits",
        "mean_node_novelty",
        "mean_edge_accessibility",
        "positive_improvement_edge_fraction",
        "diag_anisotropy",
        "diag_boundary_signal",
        "diag_ruggedness_score",
        "diag_sign_change_rate",
        *phase_fraction_columns,
    ]

    correlation_rows = []
    for feature in correlation_features:
        if feature not in details.columns:
            continue
        rho, p_value = safe_spearman(
            details[feature],
            details["final_value_rank"],
        )
        correlation_rows.append(
            {
                "feature": feature,
                "spearman_rho_with_rank": rho,
                "p_value": p_value,
                "interpretation": (
                    "positive rho means larger feature is associated "
                    "with worse rank"
                ),
            }
        )
    correlations = pd.DataFrame(correlation_rows).sort_values(
        "spearman_rho_with_rank",
        ascending=False,
        na_position="last",
    )

    bg_overall = overall[
        overall["algorithm"] == "BasinGraph_v2"
    ].iloc[0]
    bg_by_dimension = (
        by_dimension[
            by_dimension["algorithm"] == "BasinGraph_v2"
        ][["dimension", "mean_rank"]]
        .sort_values("dimension")
    )
    bg_by_group = (
        by_group[
            by_group["algorithm"] == "BasinGraph_v2"
        ][["function_group", "mean_rank", "wins"]]
        .sort_values("mean_rank", ascending=False)
    )

    hypotheses = []

    if internal_overall["archive_saturation_fraction"] >= 0.80:
        hypotheses.append(
            {
                "priority": "high",
                "finding": (
                    "The active archive reaches capacity in at least "
                    "80% of development runs."
                ),
                "candidate_revision": (
                    "Replace objective-only archive eviction with a "
                    "quality-diversity-accessibility retention rule, or "
                    "increase merge adaptivity. Validate only on the "
                    "development partition."
                ),
            }
        )

    if float(bg_overall["mean_rank"]) > 3.0:
        hypotheses.append(
            {
                "priority": "high",
                "finding": (
                    "BasinGraph v2 mean fixed-budget final-value rank "
                    f"is {float(bg_overall['mean_rank']):.3f} on the "
                    "development partition."
                ),
                "candidate_revision": (
                    "Inspect function-group and cocopp target-runtime "
                    "results before final freeze; do not open holdout."
                ),
            }
        )

    if (
        len(bg_by_dimension) >= 2
        and bg_by_dimension.iloc[-1]["mean_rank"]
        < bg_by_dimension.iloc[0]["mean_rank"]
    ):
        hypotheses.append(
            {
                "priority": "positive",
                "finding": (
                    "Mean rank improves with dimension across the "
                    "observed 2D-to-10D development range."
                ),
                "candidate_revision": (
                    "Preserve the graph/far-basin mechanisms while "
                    "targeting low-dimensional and smooth-function "
                    "weaknesses."
                ),
            }
        )

    weakest_groups = bg_by_group.head(2)
    hypotheses.append(
        {
            "priority": "diagnostic",
            "finding": (
                "Weakest BasinGraph development groups: "
                + "; ".join(
                    f"{row.function_group} (mean rank "
                    f"{row.mean_rank:.3f})"
                    for row in weakest_groups.itertuples()
                )
            ),
            "candidate_revision": (
                "Use only development instances and the existing "
                "development-only ablation protocol to test targeted "
                "changes."
            ),
        }
    )

    # --------------------------------------------------------------
    # 4. Save outputs
    # --------------------------------------------------------------
    overall.to_csv(out / "algorithm_overall_summary.csv", index=False)
    by_dimension.to_csv(out / "algorithm_dimension_summary.csv", index=False)
    by_group.to_csv(out / "algorithm_function_group_summary.csv", index=False)
    by_function.to_csv(out / "algorithm_per_function_summary.csv", index=False)
    pairwise.to_csv(out / "pairwise_vs_basingraph.csv", index=False)
    details.to_csv(out / "basingraph_run_diagnostics.csv", index=False)
    internal_by_group.to_csv(
        out / "basingraph_internal_by_group_dimension.csv",
        index=False,
    )
    correlations.to_csv(
        out / "basingraph_rank_correlations.csv",
        index=False,
    )

    recommendation = {
        "status": "V2_COCO_DEVELOPMENT_DIAGNOSTIC_OK",
        "run_id": run_id,
        "development_only": True,
        "holdout_accessed": False,
        "raw_results_sha256": sha256_file(raw_path),
        "validation_report_sha256": sha256_file(validation_path),
        "cocopp_output_path": cocopp_path_file.read_text().strip(),
        "basingraph_overall_mean_rank": float(
            bg_overall["mean_rank"]
        ),
        "basingraph_overall_wins": int(bg_overall["wins"]),
        "internal_summary": internal_overall,
        "hypotheses": hypotheses,
        "decision_rule": (
            "No algorithm revision may be accepted solely from final-value "
            "ranks. A revision requires concordant evidence from development "
            "cocopp target-runtime output, function-group diagnostics and "
            "development-only ablation."
        ),
    }
    (out / "development_diagnostic_report.json").write_text(
        json.dumps(recommendation, indent=2)
    )

    review_template = f"""# COCO development cocopp review

Run ID: `{run_id}`

Open the following local file in a browser:

`{cocopp_path_file.read_text().strip()}/index.html`

## Record before any algorithm revision

For each dimension 2, 5 and 10, record:

1. aggregate ECDF ordering at low budgets;
2. aggregate ECDF ordering near 1000d;
3. relative performance on separable functions;
4. relative performance on high-conditioning/unimodal functions;
5. relative performance on multimodal functions with global structure;
6. relative performance on multimodal functions with weak structure;
7. whether BasinGraph's target-runtime behavior agrees with or contradicts
   the fixed-budget final-value ranks.

Do not inspect instances 4-15.
"""
    (out / "COCOPP_MANUAL_REVIEW_TEMPLATE.md").write_text(
        review_template
    )

    print("V2_COCO_DEVELOPMENT_DIAGNOSTIC_OK")
    print("run id:", run_id)
    print("BasinGraph mean rank:", float(bg_overall["mean_rank"]))
    print("BasinGraph wins:", int(bg_overall["wins"]))
    print(
        "archive saturation fraction:",
        internal_overall["archive_saturation_fraction"],
    )
    print(
        "mean graph edges/run:",
        internal_overall["mean_graph_edges"],
    )
    print(
        "center-local activation:",
        internal_overall["center_local_activation_fraction"],
    )
    print(
        "far-basin activation:",
        internal_overall["far_basin_activation_fraction"],
    )
    print("\nBasinGraph rank by dimension:")
    print(bg_by_dimension.to_string(index=False))
    print("\nBasinGraph rank by function group:")
    print(bg_by_group.to_string(index=False))
    print("\nDevelopment hypotheses:")
    for item in hypotheses:
        print(f"- [{item['priority']}] {item['finding']}")
        print(f"  {item['candidate_revision']}")
    print("\nOutput:", out)


if __name__ == "__main__":
    main()
