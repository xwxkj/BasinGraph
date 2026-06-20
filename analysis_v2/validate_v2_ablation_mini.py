#!/usr/bin/env python3
"""Validate and summarize the Route B v2.0.0-rc1 ablation mini benchmark."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_FULL_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)

VARIANTS = [
    "Full",
    "NoGraphGuidance",
    "SingleBracket",
    "NoFarBasin",
    "NoGeometryController",
    "NoArchiveFallback",
    "NoFinalPolish",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results_v2/ablation_mini_rc1",
    )
    return parser.parse_args()


def phase_count(row: pd.Series, phase: str) -> int:
    mapping = json.loads(row["phase_evaluations_json"])
    return int(mapping.get(phase, 0))


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output
    raw_path = output_root / "ablation_mini_raw_results.csv"

    df = pd.read_csv(raw_path)
    variant_specs = json.loads(
        (output_root / "variant_specifications.json").read_text()
    )

    expected_rows = 7 * 7 * 5
    assert len(df) == expected_rows, (len(df), expected_rows)
    assert df["task"].nunique() == 7
    assert set(df["variant"].unique()) == set(VARIANTS)
    assert df["seed_index"].nunique() == 5
    assert (df["runner_status"] == "completed").all()
    assert (df["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
    assert (df["nfe"] == df["budget"]).all()
    assert df["graph_referential_integrity"].astype(bool).all()
    assert (df["archive_nodes"] > 0).all()

    assert variant_specs["Full"]["options_hash"] == EXPECTED_FULL_HASH
    assert df.loc[df["variant"] == "Full", "options_hash"].nunique() == 1
    assert (
        df.loc[df["variant"] == "Full", "options_hash"].iloc[0]
        == EXPECTED_FULL_HASH
    )

    unique_hashes = {
        variant_specs[name]["options_hash"]
        for name in VARIANTS
    }
    assert len(unique_hashes) == len(VARIANTS)

    # Verify every run's phase counts sum to its nfe.
    phase_sum = df["phase_evaluations_json"].map(
        lambda value: sum(json.loads(value).values())
    )
    assert np.array_equal(
        phase_sum.to_numpy(dtype=int),
        df["nfe"].to_numpy(dtype=int),
    )

    # Verify the intended disabled phases.
    no_far = df[df["variant"] == "NoFarBasin"]
    assert all(phase_count(row, "far_basin") == 0 for _, row in no_far.iterrows())

    no_archive = df[df["variant"] == "NoArchiveFallback"]
    assert all(
        phase_count(row, "archive_fallback") == 0
        for _, row in no_archive.iterrows()
    )

    no_final = df[df["variant"] == "NoFinalPolish"]
    assert all(
        phase_count(row, "final_polish") == 0
        for _, row in no_final.iterrows()
    )

    # Graph-guidance ablation still constructs the graph but does not use it.
    no_graph = df[df["variant"] == "NoGraphGuidance"]
    assert no_graph["graph_edges"].sum() > 0

    # Compare every ablation with Full on exactly paired task/seed runs.
    full = df[df["variant"] == "Full"][
        [
            "task",
            "seed_index",
            "fbest",
            "normalized_gap",
            "trajectory_hash",
            "phase_evaluations_json",
        ]
    ].rename(
        columns={
            "fbest": "full_fbest",
            "normalized_gap": "full_normalized_gap",
            "trajectory_hash": "full_trajectory_hash",
            "phase_evaluations_json": "full_phase_evaluations_json",
        }
    )

    difference_rows = []

    for variant in VARIANTS[1:]:
        sub = df[df["variant"] == variant].merge(
            full,
            on=["task", "seed_index"],
            how="inner",
            validate="one_to_one",
        )

        trajectory_different = (
            sub["trajectory_hash"] != sub["full_trajectory_hash"]
        )
        final_different = ~np.isclose(
            sub["fbest"],
            sub["full_fbest"],
            rtol=1e-12,
            atol=1e-14,
        )
        phase_different = (
            sub["phase_evaluations_json"]
            != sub["full_phase_evaluations_json"]
        )

        assert trajectory_different.any(), (
            f"{variant} did not alter any trajectory."
        )

        difference_rows.append(
            {
                "variant": variant,
                "paired_runs": len(sub),
                "trajectory_different_runs": int(trajectory_different.sum()),
                "phase_allocation_different_runs": int(phase_different.sum()),
                "final_value_different_runs": int(final_different.sum()),
                "median_gap_difference_vs_full": float(
                    np.median(
                        sub["normalized_gap"]
                        - sub["full_normalized_gap"]
                    )
                ),
            }
        )

    differences = pd.DataFrame(difference_rows)

    # Rank variants within each task/seed by normalized final gap.
    df["rank"] = df.groupby(
        ["task", "seed_index"]
    )["normalized_gap"].rank(
        method="average",
        ascending=True,
    )
    df["win"] = df.groupby(
        ["task", "seed_index"]
    )["normalized_gap"].transform(
        lambda values: np.isclose(
            values,
            values.min(),
            rtol=1e-12,
            atol=1e-14,
        )
    )

    summary = (
        df.groupby("variant")
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            wins=("win", "sum"),
            median_normalized_gap=("normalized_gap", "median"),
            mean_archive_nodes=("archive_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            runs=("task", "count"),
        )
        .reset_index()
    )

    summary["variant_order"] = summary["variant"].map(
        {name: index for index, name in enumerate(VARIANTS)}
    )
    summary.sort_values(
        ["mean_rank", "variant_order"],
        inplace=True,
    )
    summary.drop(columns=["variant_order"], inplace=True)

    task_summary = (
        df.groupby(["task", "variant"])
        .agg(
            median_normalized_gap=("normalized_gap", "median"),
            mean_rank=("rank", "mean"),
            wins=("win", "sum"),
            mean_archive_nodes=("archive_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        output_root / "ablation_mini_variant_summary.csv",
        index=False,
    )
    task_summary.to_csv(
        output_root / "ablation_mini_task_summary.csv",
        index=False,
    )
    differences.to_csv(
        output_root / "ablation_mini_behavior_differences.csv",
        index=False,
    )

    report = {
        "status": "V2_RC1_ABLATION_MINI_VALIDATION_OK",
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "full_options_hash": EXPECTED_FULL_HASH,
        "rows": len(df),
        "tasks": df["task"].nunique(),
        "variants": df["variant"].nunique(),
        "seeds": df["seed_index"].nunique(),
        "all_runs_exhausted_budget": True,
        "phase_accounting_verified": True,
        "graph_referential_integrity_verified": True,
        "each_ablation_changed_trajectory": True,
        "best_mean_rank_variant": str(summary.iloc[0]["variant"]),
        "best_mean_rank": float(summary.iloc[0]["mean_rank"]),
    }
    (output_root / "validation_report.json").write_text(
        json.dumps(report, indent=2)
    )

    print("V2_RC1_ABLATION_MINI_VALIDATION_OK")
    print(json.dumps(report, indent=2))
    print("\nVariant summary:")
    print(summary.to_string(index=False))
    print("\nBehaviour differences:")
    print(differences.to_string(index=False))


if __name__ == "__main__":
    main()
