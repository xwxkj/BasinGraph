#!/usr/bin/env python3
"""
Step 13E: Analysis and manuscript package for the frozen CUTEst benchmark.

Inputs
------
- protocols/cutest_pre_registered_problem_list_v2.csv
- protocols/CUTEST_PRE_REGISTRATION_MANIFEST_v2.csv
- cutest_results/protocol_v2/cutest_raw_results_all_available.csv
- cutest_results/protocol_v2/histories/**/*.npz

Outputs
-------
- integrity and reference-value tables
- final-value ranks and dimension-group summaries
- target-attainment, ERT, performance-profile and data-profile tables
- Friedman, Holm-corrected Wilcoxon and effect-size tables
- failure-mode analysis
- publication figures (PNG/PDF/SVG)
- manuscript-ready Methods, Results and Discussion text
- Source Data and SHA-256 manifest
- a ZIP package suitable for upload and manuscript integration

Analysis policy
---------------
1. The primary CUTEst reference value for each problem is the best objective
   value observed across all seven algorithms and 30 paired seeds.
2. Progress is normalized relative to the CUTEst initial objective f0:
       residual = max(f - f_ref, 0) / max(|f0 - f_ref|, numerical floor).
3. Predefined targets are residual <= 1e-1, 1e-3 and 1e-5.
4. The main target-runtime analysis uses residual <= 1e-3.
5. Failed target runs are charged the full prescribed evaluation budget in ERT.
   Therefore early termination by L-BFGS-B or CMA-ES is not treated as an
   error, but unsuccessful early termination receives the full-budget penalty.
6. Inferential tests use one median score per problem and algorithm, avoiding
   pseudo-replication across the 30 seeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALGORITHM_ORDER = [
    "BasinGraph",
    "BIPOP-CMA-ES",
    "CMA-ES",
    "Differential Evolution",
    "Multi-start L-BFGS-B",
    "Latin Hypercube Sampling",
    "Random Search",
]

ALGORITHM_SHORT = {
    "BasinGraph": "BasinGraph",
    "BIPOP-CMA-ES": "BIPOP-CMA-ES",
    "CMA-ES": "CMA-ES",
    "Differential Evolution": "DE",
    "Multi-start L-BFGS-B": "MS-L-BFGS-B",
    "Latin Hypercube Sampling": "LHS",
    "Random Search": "Random",
}

COLORS = {
    "BasinGraph": "#005F8C",
    "BIPOP-CMA-ES": "#E69F00",
    "CMA-ES": "#56B4E9",
    "Differential Evolution": "#009E73",
    "Multi-start L-BFGS-B": "#D55E00",
    "Latin Hypercube Sampling": "#8C8C8C",
    "Random Search": "#C7C7C7",
}

TARGETS = [1e-1, 1e-3, 1e-5]
MAIN_TARGET = 1e-3
EPS = np.finfo(float).eps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_protocol(protocol_path: Path, manifest_path: Path) -> dict[str, str]:
    manifest = pd.read_csv(manifest_path)
    rel = str(protocol_path.relative_to(PROJECT_ROOT))
    row = manifest[manifest["relative_path"] == rel]
    if len(row) != 1:
        raise RuntimeError(f"Manifest entry missing or duplicated for {rel}")
    expected = str(row.iloc[0]["sha256"]).strip().lower()
    observed = sha256_file(protocol_path).lower()
    if observed != expected:
        raise RuntimeError(
            f"Protocol hash mismatch for {rel}: expected {expected}, observed {observed}"
        )
    return {
        "protocol_path": rel,
        "protocol_sha256": observed,
        "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": sha256_file(manifest_path),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite_idx = np.where(np.isfinite(p))[0]
    if len(finite_idx) == 0:
        return adjusted.tolist()

    order = finite_idx[np.argsort(p[finite_idx])]
    m = len(order)
    running = 0.0
    for i, idx in enumerate(order):
        value = min(1.0, (m - i) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def vargha_delaney_lower_better(x: np.ndarray, y: np.ndarray) -> float:
    """A12 interpreted as probability that x is better (smaller) than y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    comparisons = x[:, None] - y[None, :]
    return float(
        (np.sum(comparisons < 0) + 0.5 * np.sum(comparisons == 0))
        / comparisons.size
    )


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    samples = rng.choice(values, size=(n_boot, n), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_history(result_root: Path, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
    path = result_root / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    nfe = np.asarray(data["nfe"], dtype=int).reshape(-1)
    fbest = np.asarray(data["fbest"], dtype=float).reshape(-1)
    if len(nfe) != len(fbest) or len(nfe) == 0:
        raise ValueError(f"Invalid history arrays in {path}")
    order = np.argsort(nfe, kind="stable")
    nfe = nfe[order]
    fbest = np.minimum.accumulate(fbest[order])
    return nfe, fbest


def first_target_evaluation(
    nfe: np.ndarray,
    fbest: np.ndarray,
    target_value: float,
) -> float:
    reached = np.flatnonzero(fbest <= target_value)
    if len(reached) == 0:
        return np.nan
    return float(nfe[reached[0]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        default="cutest_results/protocol_v2/cutest_raw_results_all_available.csv",
    )
    parser.add_argument(
        "--result-root",
        default="cutest_results/protocol_v2",
    )
    parser.add_argument(
        "--protocol",
        default="protocols/cutest_pre_registered_problem_list_v2.csv",
    )
    parser.add_argument(
        "--manifest",
        default="protocols/CUTEST_PRE_REGISTRATION_MANIFEST_v2.csv",
    )
    parser.add_argument(
        "--out",
        default="cutest_results/protocol_v2/analysis_final_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path = PROJECT_ROOT / args.raw
    result_root = PROJECT_ROOT / args.result_root
    protocol_path = PROJECT_ROOT / args.protocol
    manifest_path = PROJECT_ROOT / args.manifest
    out_dir = PROJECT_ROOT / args.out

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for sub in [
        "tables",
        "source_data",
        "figures",
        "manuscript_text",
        "protocols",
    ]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    protocol_integrity = verify_protocol(protocol_path, manifest_path)
    raw = pd.read_csv(raw_path)
    protocol = pd.read_csv(protocol_path)

    # ------------------------------------------------------------------
    # 1. Integrity validation
    # ------------------------------------------------------------------
    expected_algorithms = set(ALGORITHM_ORDER)
    assert len(raw) == 10500, f"Expected 10500 rows, found {len(raw)}"
    assert raw["instance_id"].nunique() == 50
    assert set(raw["algorithm"].unique()) == expected_algorithms
    assert raw["protocol_seed_index"].nunique() == 30
    assert raw["protocol_seed_index"].min() == 0
    assert raw["protocol_seed_index"].max() == 29
    assert (raw["runner_status"] == "completed").all()
    assert not raw["algorithm_message"].astype(str).str.contains(
        "exception", case=False, na=False
    ).any()

    # Confirm exact balanced design.
    balance = raw.groupby(["instance_id", "algorithm"]).size()
    assert (balance == 30).all()

    # Attach protocol metadata from the frozen list.
    protocol_cols = [
        "instance_id",
        "objective_type",
        "origin",
        "dimension_group",
        "source_type",
        "dimension",
        "problem_name",
    ]
    protocol_meta = protocol[protocol_cols].drop_duplicates("instance_id")
    raw = raw.drop(
        columns=[
            c for c in ["objective_type", "origin"]
            if c in raw.columns
        ],
        errors="ignore",
    ).merge(
        protocol_meta[
            ["instance_id", "objective_type", "origin"]
        ],
        on="instance_id",
        how="left",
        validate="many_to_one",
    )

    # ------------------------------------------------------------------
    # 2. Pooled reference values and normalized residuals
    # ------------------------------------------------------------------
    reference = (
        raw.groupby("instance_id")
        .agg(
            f_reference=("fbest", "min"),
            f_initial=("f_at_cutest_x0", "median"),
            dimension=("dimension", "first"),
            dimension_group=("dimension_group", "first"),
            objective_type=("objective_type", "first"),
            origin=("origin", "first"),
            problem_name=("problem_name", "first"),
            source_type=("source_type", "first"),
            budget=("budget", "first"),
        )
        .reset_index()
    )

    numerical_floor = 1e-12 * (
        1.0 + np.abs(reference["f_reference"].to_numpy())
    )
    improvement_scale = np.abs(
        reference["f_initial"].to_numpy()
        - reference["f_reference"].to_numpy()
    )
    reference["normalization_scale"] = np.maximum(
        improvement_scale,
        numerical_floor,
    )
    reference["degenerate_reference_scale"] = improvement_scale <= numerical_floor

    raw = raw.merge(
        reference[
            ["instance_id", "f_reference", "normalization_scale"]
        ],
        on="instance_id",
        how="left",
        validate="many_to_one",
    )

    raw["normalized_residual"] = np.maximum(
        raw["fbest"] - raw["f_reference"],
        0.0,
    ) / raw["normalization_scale"]

    # A bounded log score is useful for summaries without infinite values.
    raw["log10_residual_score"] = np.log10(
        np.maximum(raw["normalized_residual"], 1e-16)
    )

    for tau in TARGETS:
        raw[f"success_{tau:.0e}"] = raw["normalized_residual"] <= tau

    # ------------------------------------------------------------------
    # 3. Read convergence histories and compute target attainment
    # ------------------------------------------------------------------
    target_rows: list[dict[str, Any]] = []
    missing_histories: list[str] = []

    target_lookup = reference.set_index("instance_id")

    for row in raw.itertuples(index=False):
        f_ref = float(target_lookup.loc[row.instance_id, "f_reference"])
        scale = float(target_lookup.loc[row.instance_id, "normalization_scale"])

        try:
            hist_nfe, hist_fbest = load_history(
                result_root,
                str(row.history_relative_path),
            )
        except Exception as exc:
            missing_histories.append(
                f"{row.instance_id}|{row.algorithm}|{row.protocol_seed_index}: {exc}"
            )
            hist_nfe = np.asarray([int(row.nfe)])
            hist_fbest = np.asarray([float(row.fbest)])

        base = {
            "instance_id": row.instance_id,
            "algorithm": row.algorithm,
            "algorithm_id": row.algorithm_id,
            "protocol_seed_index": int(row.protocol_seed_index),
            "protocol_seed": int(row.protocol_seed),
            "dimension": int(row.dimension),
            "dimension_group": row.dimension_group,
            "objective_type": row.objective_type,
            "budget": int(row.budget),
            "actual_nfe": int(row.nfe),
        }

        for tau in TARGETS:
            target_value = f_ref + tau * scale
            first_nfe = first_target_evaluation(
                hist_nfe,
                hist_fbest,
                target_value,
            )
            success = np.isfinite(first_nfe)
            target_rows.append({
                **base,
                "target_residual": tau,
                "target_value": target_value,
                "success": bool(success),
                "nfe_to_target": first_nfe,
                "nfe_per_dimension_to_target": (
                    first_nfe / row.dimension if success else np.nan
                ),
                # Fixed-budget penalty for unsuccessful early termination.
                "ert_cost_contribution": (
                    first_nfe if success else float(row.budget)
                ),
            })

    targets = pd.DataFrame(target_rows)

    # ------------------------------------------------------------------
    # 4. Per-problem algorithm summaries and final-value ranks
    # ------------------------------------------------------------------
    problem_algorithm = (
        raw.groupby(
            [
                "instance_id",
                "algorithm",
                "dimension_group",
                "objective_type",
            ]
        )
        .agg(
            median_normalized_residual=("normalized_residual", "median"),
            mean_normalized_residual=("normalized_residual", "mean"),
            median_log10_residual=("log10_residual_score", "median"),
            q25_normalized_residual=("normalized_residual", lambda x: x.quantile(0.25)),
            q75_normalized_residual=("normalized_residual", lambda x: x.quantile(0.75)),
            median_fbest=("fbest", "median"),
            mean_fbest=("fbest", "mean"),
            median_budget_ratio=("budget_ratio", "median"),
            median_wall_time_seconds=("wall_time_seconds", "median"),
            success_rate_1e_1=("success_1e-01", "mean"),
            success_rate_1e_3=("success_1e-03", "mean"),
            success_rate_1e_5=("success_1e-05", "mean"),
        )
        .reset_index()
    )

    problem_algorithm["final_value_rank"] = (
        problem_algorithm.groupby("instance_id")["median_normalized_residual"]
        .rank(method="average", ascending=True)
    )
    problem_algorithm["final_value_win"] = (
        problem_algorithm.groupby("instance_id")["median_normalized_residual"]
        .transform(lambda x: np.isclose(x, x.min(), rtol=1e-12, atol=1e-16))
    )

    # ------------------------------------------------------------------
    # 5. Target-runtime ERT per problem and algorithm
    # ------------------------------------------------------------------
    ert_rows = []

    for (instance_id, algorithm, tau), group in targets.groupby(
        ["instance_id", "algorithm", "target_residual"]
    ):
        success_count = int(group["success"].sum())
        runs = len(group)
        total_cost = float(group["ert_cost_contribution"].sum())
        ert = total_cost / success_count if success_count > 0 else np.inf

        ert_rows.append({
            "instance_id": instance_id,
            "algorithm": algorithm,
            "target_residual": tau,
            "runs": runs,
            "successes": success_count,
            "success_rate": success_count / runs,
            "ert_function_evaluations": ert,
            "median_nfe_to_target_successes": (
                float(group.loc[group["success"], "nfe_to_target"].median())
                if success_count > 0 else np.nan
            ),
            "dimension": int(group["dimension"].iloc[0]),
            "dimension_group": group["dimension_group"].iloc[0],
        })

    ert = pd.DataFrame(ert_rows)

    main_ert = ert[ert["target_residual"] == MAIN_TARGET].copy()
    main_ert["best_ert_problem"] = (
        main_ert.groupby("instance_id")["ert_function_evaluations"]
        .transform("min")
    )
    main_ert["performance_ratio"] = (
        main_ert["ert_function_evaluations"]
        / main_ert["best_ert_problem"]
    )
    main_ert.loc[
        ~np.isfinite(main_ert["ert_function_evaluations"]),
        "performance_ratio",
    ] = np.inf

    # ------------------------------------------------------------------
    # 6. Overall summaries with bootstrap CIs
    # ------------------------------------------------------------------
    rng = np.random.default_rng(20260619)
    overall_rows = []

    for algorithm in ALGORITHM_ORDER:
        sub = problem_algorithm[problem_algorithm["algorithm"] == algorithm]
        ci_low, ci_high = bootstrap_mean_ci(
            sub["final_value_rank"].to_numpy(),
            rng=rng,
            n_boot=args.bootstrap,
        )
        overall_rows.append({
            "algorithm": algorithm,
            "mean_final_value_rank": sub["final_value_rank"].mean(),
            "mean_rank_ci95_low": ci_low,
            "mean_rank_ci95_high": ci_high,
            "median_final_value_rank": sub["final_value_rank"].median(),
            "final_value_wins": int(sub["final_value_win"].sum()),
            "median_normalized_residual": sub["median_normalized_residual"].median(),
            "mean_success_rate_1e_1": sub["success_rate_1e_1"].mean(),
            "mean_success_rate_1e_3": sub["success_rate_1e_3"].mean(),
            "mean_success_rate_1e_5": sub["success_rate_1e_5"].mean(),
            "median_wall_time_seconds": sub["median_wall_time_seconds"].median(),
            "median_budget_ratio": sub["median_budget_ratio"].median(),
            "problems": len(sub),
        })

    overall = pd.DataFrame(overall_rows).sort_values(
        ["mean_final_value_rank", "algorithm"]
    )

    group_summary = (
        problem_algorithm.groupby(["dimension_group", "algorithm"])
        .agg(
            mean_final_value_rank=("final_value_rank", "mean"),
            median_final_value_rank=("final_value_rank", "median"),
            final_value_wins=("final_value_win", "sum"),
            median_normalized_residual=("median_normalized_residual", "median"),
            mean_success_rate_1e_3=("success_rate_1e_3", "mean"),
            median_wall_time_seconds=("median_wall_time_seconds", "median"),
            problems=("instance_id", "count"),
        )
        .reset_index()
    )

    objective_summary = (
        problem_algorithm.groupby(["objective_type", "algorithm"])
        .agg(
            mean_final_value_rank=("final_value_rank", "mean"),
            final_value_wins=("final_value_win", "sum"),
            median_normalized_residual=("median_normalized_residual", "median"),
            mean_success_rate_1e_3=("success_rate_1e_3", "mean"),
            problems=("instance_id", "count"),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 7. Friedman and post-hoc comparisons
    # ------------------------------------------------------------------
    rank_matrix = (
        problem_algorithm.pivot(
            index="instance_id",
            columns="algorithm",
            values="final_value_rank",
        )[ALGORITHM_ORDER]
    )

    friedman = friedmanchisquare(
        *[rank_matrix[a].to_numpy() for a in ALGORITHM_ORDER]
    )

    friedman_table = pd.DataFrame([{
        "problems": len(rank_matrix),
        "algorithms": len(ALGORITHM_ORDER),
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
        "metric": "per-problem rank of median normalized residual",
    }])

    score_matrix = (
        problem_algorithm.pivot(
            index="instance_id",
            columns="algorithm",
            values="median_normalized_residual",
        )[ALGORITHM_ORDER]
    )

    bg_scores = score_matrix["BasinGraph"].to_numpy()
    pairwise_rows = []

    for baseline in ALGORITHM_ORDER:
        if baseline == "BasinGraph":
            continue
        base_scores = score_matrix[baseline].to_numpy()
        diff = bg_scores - base_scores

        try:
            test = wilcoxon(
                bg_scores,
                base_scores,
                alternative="two-sided",
                zero_method="wilcox",
            )
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        except Exception:
            p_value = np.nan
            statistic = np.nan

        wins = int(np.sum(bg_scores < base_scores))
        losses = int(np.sum(bg_scores > base_scores))
        ties = int(np.sum(np.isclose(bg_scores, base_scores, rtol=1e-12, atol=1e-16)))

        pairwise_rows.append({
            "baseline": baseline,
            "problems": len(bg_scores),
            "basingraph_better": wins,
            "basingraph_worse": losses,
            "ties": ties,
            "paired_win_probability": (wins + 0.5 * ties) / len(bg_scores),
            "vargha_delaney_A12_basingraph_better": vargha_delaney_lower_better(
                bg_scores, base_scores
            ),
            "wilcoxon_statistic": statistic,
            "wilcoxon_p_value_two_sided": p_value,
            "median_score_difference_bg_minus_baseline": float(np.median(diff)),
        })

    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["holm_adjusted_p_value"] = holm_adjust(
        pairwise["wilcoxon_p_value_two_sided"].tolist()
    )
    pairwise["significant_after_holm_0_05"] = (
        pairwise["holm_adjusted_p_value"] < 0.05
    )

    # ------------------------------------------------------------------
    # 8. Performance-profile and data-profile source data
    # ------------------------------------------------------------------
    finite_ratios = main_ert.loc[
        np.isfinite(main_ert["performance_ratio"]),
        "performance_ratio",
    ]
    max_ratio = max(2.0, float(finite_ratios.quantile(0.98)) if len(finite_ratios) else 10.0)
    ratio_grid = np.geomspace(1.0, max_ratio, 250)

    performance_rows = []
    for algorithm in ALGORITHM_ORDER:
        values = main_ert.loc[
            main_ert["algorithm"] == algorithm,
            "performance_ratio",
        ].to_numpy()
        for alpha in ratio_grid:
            performance_rows.append({
                "algorithm": algorithm,
                "performance_ratio_threshold": alpha,
                "fraction_of_problems": float(np.mean(values <= alpha)),
                "target_residual": MAIN_TARGET,
            })
    performance_profile = pd.DataFrame(performance_rows)

    main_targets = targets[targets["target_residual"] == MAIN_TARGET].copy()
    finite_nd = main_targets.loc[
        main_targets["success"],
        "nfe_per_dimension_to_target",
    ]
    max_nd = max(10.0, float(finite_nd.quantile(0.99)) if len(finite_nd) else 100.0)
    budget_grid = np.geomspace(0.5, max_nd, 250)

    data_rows = []
    total_pairs_per_algorithm = (
        main_targets.groupby("algorithm").size().to_dict()
    )
    for algorithm in ALGORITHM_ORDER:
        sub = main_targets[main_targets["algorithm"] == algorithm]
        values = sub["nfe_per_dimension_to_target"].to_numpy()
        for budget_per_dimension in budget_grid:
            solved = np.isfinite(values) & (values <= budget_per_dimension)
            data_rows.append({
                "algorithm": algorithm,
                "evaluations_per_dimension": budget_per_dimension,
                "fraction_problem_seed_pairs_solved": float(np.mean(solved)),
                "target_residual": MAIN_TARGET,
                "problem_seed_pairs": total_pairs_per_algorithm[algorithm],
            })
    data_profile = pd.DataFrame(data_rows)

    target_success_summary = (
        targets.groupby(["algorithm", "target_residual"])
        .agg(
            success_rate=("success", "mean"),
            successes=("success", "sum"),
            problem_seed_pairs=("success", "count"),
            median_nfe_to_target_successes=(
                "nfe_to_target",
                lambda x: x.dropna().median(),
            ),
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 9. Failure-mode analysis
    # ------------------------------------------------------------------
    pa = problem_algorithm.copy()
    best_per_problem = (
        pa.sort_values(
            ["instance_id", "median_normalized_residual", "algorithm"]
        )
        .groupby("instance_id")
        .first()
        .reset_index()
        .rename(columns={
            "algorithm": "best_algorithm",
            "median_normalized_residual": "best_median_normalized_residual",
        })
    )

    bg_problem = pa[pa["algorithm"] == "BasinGraph"].copy().rename(columns={
        "final_value_rank": "basingraph_rank",
        "median_normalized_residual": "basingraph_median_normalized_residual",
        "success_rate_1e_3": "basingraph_success_rate_1e_3",
    })

    failures = bg_problem.merge(
        best_per_problem[
            [
                "instance_id",
                "best_algorithm",
                "best_median_normalized_residual",
            ]
        ],
        on="instance_id",
        how="left",
    )
    failures = failures[failures["basingraph_rank"] > 1.0].copy()
    failures["residual_ratio_to_best"] = (
        (failures["basingraph_median_normalized_residual"] + 1e-16)
        / (failures["best_median_normalized_residual"] + 1e-16)
    )
    failures.sort_values(
        ["basingraph_rank", "residual_ratio_to_best"],
        ascending=[False, False],
        inplace=True,
    )

    # ------------------------------------------------------------------
    # 10. Save tables
    # ------------------------------------------------------------------
    reference.to_csv(out_dir / "tables" / "cutest_reference_values.csv", index=False)
    raw.to_csv(out_dir / "tables" / "cutest_run_level_metrics.csv", index=False)
    targets.to_csv(out_dir / "tables" / "cutest_target_attainment_run_level.csv", index=False)
    ert.to_csv(out_dir / "tables" / "cutest_ert_by_problem_algorithm_target.csv", index=False)
    main_ert.to_csv(out_dir / "tables" / "cutest_main_target_performance_ratios.csv", index=False)
    problem_algorithm.to_csv(out_dir / "tables" / "cutest_problem_algorithm_summary.csv", index=False)
    overall.to_csv(out_dir / "tables" / "cutest_algorithm_overall_summary.csv", index=False)
    group_summary.to_csv(out_dir / "tables" / "cutest_dimension_group_summary.csv", index=False)
    objective_summary.to_csv(out_dir / "tables" / "cutest_objective_type_summary.csv", index=False)
    friedman_table.to_csv(out_dir / "tables" / "cutest_friedman_test.csv", index=False)
    pairwise.to_csv(out_dir / "tables" / "cutest_pairwise_statistics_vs_basingraph.csv", index=False)
    performance_profile.to_csv(out_dir / "source_data" / "SourceData_CUTEst_performance_profile.csv", index=False)
    data_profile.to_csv(out_dir / "source_data" / "SourceData_CUTEst_data_profile.csv", index=False)
    target_success_summary.to_csv(out_dir / "source_data" / "SourceData_CUTEst_target_success.csv", index=False)
    group_summary.to_csv(out_dir / "source_data" / "SourceData_CUTEst_dimension_groups.csv", index=False)
    failures.to_csv(out_dir / "tables" / "cutest_basingraph_failure_modes.csv", index=False)

    if missing_histories:
        (out_dir / "protocols" / "missing_or_invalid_histories.txt").write_text(
            "\n".join(missing_histories)
        )

    # ------------------------------------------------------------------
    # 11. Publication figure
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    # a. Performance profile
    ax = fig.add_subplot(grid[0, 0])
    for algorithm in ALGORITHM_ORDER:
        sub = performance_profile[performance_profile["algorithm"] == algorithm]
        ax.plot(
            sub["performance_ratio_threshold"],
            sub["fraction_of_problems"],
            label=ALGORITHM_SHORT[algorithm],
            color=COLORS[algorithm],
            linewidth=2.4 if algorithm == "BasinGraph" else 1.4,
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0, ratio_grid.max())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Performance ratio (ERT / best ERT)")
    ax.set_ylabel("Fraction of problems")
    ax.set_title("a  CUTEst performance profile, residual ≤ 10⁻³", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.22, linewidth=0.5)

    # b. Data profile
    ax = fig.add_subplot(grid[0, 1])
    for algorithm in ALGORITHM_ORDER:
        sub = data_profile[data_profile["algorithm"] == algorithm]
        ax.plot(
            sub["evaluations_per_dimension"],
            sub["fraction_problem_seed_pairs_solved"],
            label=ALGORITHM_SHORT[algorithm],
            color=COLORS[algorithm],
            linewidth=2.4 if algorithm == "BasinGraph" else 1.4,
        )
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Objective evaluations / dimension")
    ax.set_ylabel("Fraction of problem–seed pairs")
    ax.set_title("b  CUTEst data profile, residual ≤ 10⁻³", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.22, linewidth=0.5)

    # c. Mean rank by dimension group
    ax = fig.add_subplot(grid[1, 0])
    group_order = ["small_2_20", "medium_21_100", "large_101_500"]
    group_labels = ["Small\n2–20", "Medium\n21–100", "Large\n101–500"]
    x = np.arange(len(group_order))
    width = 0.11
    for i, algorithm in enumerate(ALGORITHM_ORDER):
        values = []
        for group in group_order:
            row = group_summary[
                (group_summary["dimension_group"] == group)
                & (group_summary["algorithm"] == algorithm)
            ]
            values.append(float(row.iloc[0]["mean_final_value_rank"]))
        ax.bar(
            x + (i - 3) * width,
            values,
            width=width,
            label=ALGORITHM_SHORT[algorithm],
            color=COLORS[algorithm],
            edgecolor="none",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Mean rank (lower is better)")
    ax.set_title("c  Final-value rank by dimension group", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)

    # d. Success rate by target
    ax = fig.add_subplot(grid[1, 1])
    target_positions = np.arange(len(TARGETS))
    for algorithm in ALGORITHM_ORDER:
        sub = target_success_summary[
            target_success_summary["algorithm"] == algorithm
        ].set_index("target_residual")
        values = [float(sub.loc[t, "success_rate"]) for t in TARGETS]
        ax.plot(
            target_positions,
            values,
            marker="o",
            markersize=4,
            label=ALGORITHM_SHORT[algorithm],
            color=COLORS[algorithm],
            linewidth=2.4 if algorithm == "BasinGraph" else 1.4,
        )
    ax.set_xticks(target_positions)
    ax.set_xticklabels(["10⁻¹", "10⁻³", "10⁻⁵"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Normalized residual target")
    ax.set_ylabel("Success rate")
    ax.set_title("d  Target-attainment success", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.22, linewidth=0.5)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )

    figure_base = out_dir / "figures" / "figure_cutest_validation"
    fig.savefig(figure_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 12. Manuscript-ready text
    # ------------------------------------------------------------------
    top = overall.iloc[0]
    bg = overall[overall["algorithm"] == "BasinGraph"].iloc[0]

    friedman_p = float(friedman_table.iloc[0]["friedman_p_value"])
    sig_count = int(pairwise["significant_after_holm_0_05"].sum())

    methods_text = f"""# CUTEst Methods text

We evaluated BasinGraph on a performance-independent, pre-registered subset of 50 bound-constrained CUTEst instances. The subset comprised 20 small (2–20 variables), 20 medium (21–100 variables) and 10 large (101–500 variables) instances, including 21 fixed-dimensional and 29 formally parameterized scalable instances. The frozen list contained 48 distinct CUTEst base problems and was selected from a technical inventory of 84 valid instances before optimizer-performance evaluation. Each method was evaluated over 30 paired random seeds. The comparison included BasinGraph, CMA-ES, BIPOP-CMA-ES, differential evolution, multi-start L-BFGS-B, Latin hypercube sampling and random search. The objective-evaluation budget was min(20,000, max(1,000, 50n)), where n is the number of variables.

For each problem, the reference value was the best objective observed across all algorithms and seeds. We normalized final and intermediate objective gaps by max(|f0 − fref|, numerical floor), where f0 is the CUTEst initial objective. Predefined residual targets were 10⁻¹, 10⁻³ and 10⁻⁵. Expected running time for the 10⁻³ target charged unsuccessful runs the full prescribed budget, including algorithms that terminated early. Statistical inference used one median normalized residual per problem and algorithm. We applied a Friedman test across all seven algorithms, followed by two-sided paired Wilcoxon tests against BasinGraph with Holm correction. We also report paired win probabilities, Vargha–Delaney A12 effect sizes and bootstrap 95% confidence intervals for mean ranks.
"""

    results_text = f"""# CUTEst Results text

The final CUTEst experiment comprised 10,500 completed optimization records (50 problems × 7 algorithms × 30 paired seeds), with no runner failures or exception-marked runs. The pre-registered benchmark retained balanced small-, medium- and large-scale strata and included both fixed-dimensional and scalable CUTEst instances. Across the 50 problems, {top['algorithm']} obtained the lowest mean rank of {top['mean_final_value_rank']:.3f} based on the median normalized residual, whereas BasinGraph achieved a mean rank of {bg['mean_final_value_rank']:.3f} (95% bootstrap interval {bg['mean_rank_ci95_low']:.3f}–{bg['mean_rank_ci95_high']:.3f}), a median rank of {bg['median_final_value_rank']:.3f} and {int(bg['final_value_wins'])} final-value wins.

The omnibus Friedman test detected differences among the seven algorithms (P = {friedman_p:.3g}). In Holm-corrected paired comparisons, {sig_count} of the six contrasts with BasinGraph remained significant at α = 0.05. Target-runtime performance was evaluated using a normalized residual threshold of 10⁻³. The performance profile reports expected running time relative to the best method on each problem, while the data profile reports the fraction of problem–seed pairs solved as a function of objective evaluations per dimension. These target-runtime views complement the final-value ranks and avoid interpreting raw objective values across heterogeneous CUTEst problems.
"""

    discussion_text = """# CUTEst Discussion text

The CUTEst results provide an external nonlinear-optimization test beyond the BBOB suite and the internally constructed diagnostic functions. They also clarify the role of early termination. Multi-start L-BFGS-B and CMA-ES occasionally stopped before exhausting the nominal budget; this was not treated as a computational failure. For target-runtime calculations, an unsuccessful early termination was charged the full prescribed budget, whereas a successful early termination retained its observed time to target. This policy preserves the benefit of fast convergence without rewarding premature unsuccessful termination.

The dimension-stratified analysis should be used to identify where BasinGraph's basin-discovery mechanisms remain advantageous and where local or covariance-adaptive solvers dominate. The failure-mode table reports every problem on which BasinGraph did not attain the best median normalized residual, rather than suppressing unfavourable cases.
"""

    figure_caption = """# Suggested CUTEst figure caption

Figure X | Pre-registered CUTEst validation. a, Dolan–Moré-style performance profile based on expected objective evaluations required to reach a normalized residual of 10⁻³. b, Data profile showing the fraction of problem–seed pairs reaching the same target as a function of objective evaluations per dimension. c, Mean final-value rank for small (2–20 variables), medium (21–100) and large (101–500) CUTEst instances. d, Success rate for normalized residual targets of 10⁻¹, 10⁻³ and 10⁻⁵. The experiment comprised 50 pre-registered problem instances, seven algorithms and 30 paired seeds.
"""

    (out_dir / "manuscript_text" / "CUTEST_METHODS_TEXT.md").write_text(methods_text)
    (out_dir / "manuscript_text" / "CUTEST_RESULTS_TEXT.md").write_text(results_text)
    (out_dir / "manuscript_text" / "CUTEST_DISCUSSION_TEXT.md").write_text(discussion_text)
    (out_dir / "manuscript_text" / "CUTEST_FIGURE_CAPTION.md").write_text(figure_caption)

    # ------------------------------------------------------------------
    # 13. Protocol and integrity files
    # ------------------------------------------------------------------
    shutil.copy2(protocol_path, out_dir / "protocols" / protocol_path.name)
    shutil.copy2(manifest_path, out_dir / "protocols" / manifest_path.name)

    integrity = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(raw),
        "problems": raw["instance_id"].nunique(),
        "algorithms": raw["algorithm"].nunique(),
        "seeds": raw["protocol_seed_index"].nunique(),
        "runner_failures": int((raw["runner_status"] != "completed").sum()),
        "exception_rows": int(raw["algorithm_message"].astype(str).str.contains(
            "exception", case=False, na=False
        ).sum()),
        "missing_or_invalid_history_count": len(missing_histories),
        "targets": TARGETS,
        "main_target": MAIN_TARGET,
        "reference_policy": "best observed value across all algorithms and 30 seeds",
        "failure_cost_policy": "full prescribed budget for unsuccessful target runs",
        "protocol_integrity": protocol_integrity,
        "friedman_statistic": float(friedman.statistic),
        "friedman_p_value": float(friedman.pvalue),
    }
    (out_dir / "protocols" / "CUTEST_ANALYSIS_INTEGRITY.json").write_text(
        json.dumps(integrity, indent=2)
    )

    analysis_policy = """# CUTEst final analysis policy

- Frozen protocol: CUTEST_PRE_REGISTRATION_PROTOCOL_v2.
- No problem was added or removed after observing optimizer performance.
- One median normalized residual per problem and algorithm is used for ranks and inferential statistics.
- Reference value: best observed objective over all seven algorithms and 30 seeds.
- Normalization: max(f − fref, 0) / max(|f0 − fref|, 10⁻¹²(1 + |fref|)).
- Targets: 10⁻¹, 10⁻³ and 10⁻⁵.
- Main target-runtime threshold: 10⁻³.
- ERT failure cost: full prescribed budget, even if an unsuccessful solver terminated early.
- Primary statistical test: Friedman across seven algorithms.
- Post-hoc tests: paired, two-sided Wilcoxon against BasinGraph with Holm correction.
- Effect sizes: paired win probability and Vargha–Delaney A12.
- Confidence intervals: problem-level bootstrap for mean ranks.
"""
    (out_dir / "protocols" / "CUTEST_FINAL_ANALYSIS_POLICY.md").write_text(
        analysis_policy
    )

    # ------------------------------------------------------------------
    # 14. README and manifest
    # ------------------------------------------------------------------
    readme = f"""# BasinGraph final CUTEst analysis package

This package analyses the frozen 50-instance CUTEst benchmark with seven algorithms and 30 paired seeds.

## Main outputs

- `figures/figure_cutest_validation.png|pdf|svg`
- `tables/cutest_algorithm_overall_summary.csv`
- `tables/cutest_dimension_group_summary.csv`
- `tables/cutest_pairwise_statistics_vs_basingraph.csv`
- `tables/cutest_basingraph_failure_modes.csv`
- `manuscript_text/CUTEST_METHODS_TEXT.md`
- `manuscript_text/CUTEST_RESULTS_TEXT.md`
- `manuscript_text/CUTEST_DISCUSSION_TEXT.md`
- `source_data/`

## Important interpretation

Multi-start L-BFGS-B and CMA-ES may terminate before exhausting the nominal budget. This is not a failure. In ERT calculations, unsuccessful early termination is charged the full prescribed budget.

Headline final-value result:
- best mean rank: {top['algorithm']} ({top['mean_final_value_rank']:.3f});
- BasinGraph mean rank: {bg['mean_final_value_rank']:.3f};
- BasinGraph final-value wins: {int(bg['final_value_wins'])}.
"""
    (out_dir / "README.md").write_text(readme)

    manifest_path_out = out_dir / "MANIFEST_SHA256.csv"
    manifest_rows = []
    for path in sorted(out_dir.rglob("*")):
        if path.is_file() and path != manifest_path_out:
            manifest_rows.append({
                "relative_path": str(path.relative_to(out_dir)),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            })
    pd.DataFrame(manifest_rows).to_csv(manifest_path_out, index=False)

    zip_path = out_dir.parent / "BasinGraph_CUTEst_FINAL_analysis_and_manuscript_inputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir.parent))

    print("STEP_13E_OK")
    print("Rows:", len(raw))
    print("Problems:", raw["instance_id"].nunique())
    print("Algorithms:", raw["algorithm"].nunique())
    print("Seeds:", raw["protocol_seed_index"].nunique())
    print("Missing/invalid histories:", len(missing_histories))
    print("Best mean-rank algorithm:", top["algorithm"], top["mean_final_value_rank"])
    print("BasinGraph mean rank:", bg["mean_final_value_rank"])
    print("BasinGraph wins:", int(bg["final_value_wins"]))
    print("Friedman p-value:", friedman.pvalue)
    print("Holm-significant comparisons:", sig_count, "of 6")
    print("Output directory:", out_dir)
    print("Upload ZIP:", zip_path)


if __name__ == "__main__":
    main()
