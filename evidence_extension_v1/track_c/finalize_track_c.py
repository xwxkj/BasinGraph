#!/usr/bin/env python3
"""Validate and summarize B21 Track C scientific applications."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    DISPLAY_NAMES,
    EXPECTED_OPTIONS_HASH,
    NIST_DATASETS,
    TARGET_RATIOS,
    checkpoint_multipliers,
    normalize_gap,
    sha256_file,
    verify_source_identity,
)

EXPECTED_ROWS = {"smoke": 35, "confirmatory": 2310}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_c")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        out[index] = running
    return out.tolist()


def pairwise_final(frame: pd.DataFrame) -> pd.DataFrame:
    key = ["domain", "task_id", "paired_seed"]
    basin = frame[frame.algorithm == "BasinGraph"][key + ["normalized_gap"]].rename(
        columns={"normalized_gap": "basingraph_gap"}
    )
    rows = []
    raw_p = []
    for algorithm in ALGORITHMS[1:]:
        paired = frame[frame.algorithm == algorithm][key + ["normalized_gap"]].merge(
            basin, on=key, validate="one_to_one"
        )
        delta = paired.normalized_gap.to_numpy(float) - paired.basingraph_gap.to_numpy(float)
        nonzero = delta[~np.isclose(delta, 0.0, rtol=1e-10, atol=1e-12)]
        if len(nonzero):
            test = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        else:
            statistic, p_value = 0.0, 1.0
        worse = int(np.sum(delta > 1e-12))
        better = int(np.sum(delta < -1e-12))
        ties = int(len(delta) - worse - better)
        effect = (worse - better) / max(worse + better, 1)
        rows.append(
            {
                "baseline": algorithm,
                "baseline_display": DISPLAY_NAMES[algorithm],
                "paired_blocks": len(delta),
                "baseline_worse_than_basingraph": worse,
                "baseline_better_than_basingraph": better,
                "ties": ties,
                "median_gap_difference_baseline_minus_basingraph": float(np.median(delta)),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
                "rank_biserial_positive_means_basingraph_better": effect,
            }
        )
        raw_p.append(p_value)
    for row, adjusted in zip(rows, holm_adjust(raw_p)):
        row["holm_p"] = adjusted
    return pd.DataFrame(rows)


def primary_target_table(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in frame.itertuples(index=False):
        hit_evals = json.loads(row.target_hit_evaluations_json)
        targets = [float(row.f_ref) + ratio * max(float(row.f_base) - float(row.f_ref), 1e-15) for ratio in TARGET_RATIOS]
        final_value = float(row.fbest)
        checkpoints = checkpoint_multipliers(int(row.budget_multiplier))
        for target_index, (ratio, target, hit) in enumerate(zip(TARGET_RATIOS, targets, hit_evals)):
            success = final_value <= target
            for checkpoint in checkpoints:
                threshold = min(int(row.budget), int(checkpoint * row.dimension))
                reached = bool(success and int(hit) <= threshold)
                records.append(
                    {
                        "domain": row.domain,
                        "task_name": row.task_name,
                        "task_id": row.task_id,
                        "paired_seed": int(row.paired_seed),
                        "algorithm": row.algorithm,
                        "dimension": int(row.dimension),
                        "target_index": target_index,
                        "target_ratio": ratio,
                        "checkpoint_evaluations_per_dimension": checkpoint,
                        "is_final_checkpoint": checkpoint == int(row.budget_multiplier),
                        "reached": reached,
                        "evaluations": int(hit) if success else int(row.budget),
                        "budget": int(row.budget),
                    }
                )
    return pd.DataFrame(records)


def paired_target_statistics(primary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    endpoint_specs: list[tuple[str, pd.DataFrame]] = [
        ("final_registered_budget", primary[primary.is_final_checkpoint])
    ]
    for checkpoint in sorted(
        primary.checkpoint_evaluations_per_dimension.unique()
    ):
        endpoint_specs.append(
            (
                f"checkpoint_{int(checkpoint)}d",
                primary[
                    primary.checkpoint_evaluations_per_dimension == checkpoint
                ],
            )
        )

    rng = np.random.default_rng(20260808)
    for endpoint, subset in endpoint_specs:
        block = (
            subset.groupby(
                ["domain", "task_id", "paired_seed", "algorithm"],
                as_index=False,
            )
            .agg(target_fraction=("reached", "mean"))
        )
        basin = block[block.algorithm == "BasinGraph"][
            ["domain", "task_id", "paired_seed", "target_fraction"]
        ].rename(columns={"target_fraction": "basingraph_fraction"})
        endpoint_rows: list[dict[str, object]] = []
        raw_p: list[float] = []
        for algorithm in ALGORITHMS[1:]:
            paired = block[block.algorithm == algorithm][
                ["domain", "task_id", "paired_seed", "target_fraction"]
            ].merge(
                basin,
                on=["domain", "task_id", "paired_seed"],
                validate="one_to_one",
            )
            difference = (
                paired.basingraph_fraction.to_numpy(float)
                - paired.target_fraction.to_numpy(float)
            )
            nonzero = difference[
                ~np.isclose(difference, 0.0, rtol=1e-12, atol=1e-14)
            ]
            if len(nonzero):
                test = wilcoxon(
                    nonzero,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            else:
                statistic = 0.0
                p_value = 1.0

            if len(difference):
                samples = rng.integers(
                    0,
                    len(difference),
                    size=(2000, len(difference)),
                )
                bootstrap = difference[samples].mean(axis=1)
                low, high = np.quantile(bootstrap, [0.025, 0.975])
            else:
                low = high = 0.0

            bg_better = int(np.sum(difference > 1e-14))
            baseline_better = int(np.sum(difference < -1e-14))
            ties = int(len(difference) - bg_better - baseline_better)
            endpoint_rows.append(
                {
                    "endpoint": endpoint,
                    "baseline": algorithm,
                    "baseline_display": DISPLAY_NAMES[algorithm],
                    "paired_blocks": len(difference),
                    "basingraph_better_blocks": bg_better,
                    "baseline_better_blocks": baseline_better,
                    "ties": ties,
                    "mean_fraction_difference_basingraph_minus_baseline": (
                        float(np.mean(difference)) if len(difference) else 0.0
                    ),
                    "bootstrap_95_low": float(low),
                    "bootstrap_95_high": float(high),
                    "wilcoxon_statistic": statistic,
                    "raw_p": p_value,
                    "rank_biserial_positive_means_basingraph_better": (
                        (bg_better - baseline_better)
                        / max(bg_better + baseline_better, 1)
                    ),
                }
            )
            raw_p.append(p_value)
        for row, adjusted in zip(
            endpoint_rows,
            holm_adjust(raw_p),
        ):
            row["holm_p"] = adjusted
        rows.extend(endpoint_rows)
    return pd.DataFrame(rows)


def write_manifest(root: Path) -> None:
    path = root / "MANIFEST_SHA256.csv"
    rows = []
    for file in sorted(root.rglob("*")):
        if file.is_file() and file != path:
            rows.append(
                {
                    "relative_path": file.relative_to(root).as_posix(),
                    "sha256": sha256_file(file),
                    "size_bytes": file.stat().st_size,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory finalization requires authorization.")
    identity = verify_source_identity(require_clean=True)
    run_root = ROOT / args.result_root / args.run_id
    raw_path = run_root / "track_c_raw_results.csv"
    marker_path = run_root / "ALL_SHARDS_COMPLETE.json"
    if not raw_path.is_file() or not marker_path.is_file():
        raise RuntimeError("Track C shards are incomplete.")
    frame = pd.read_csv(raw_path)
    frame["history_monotone"] = bool_series(frame["history_monotone"])
    frame["graph_referential_integrity"] = bool_series(frame["graph_referential_integrity"])
    key = ["domain", "task_id", "paired_seed", "algorithm"]
    reference_path = run_root / "task_specific_reference_results.csv"
    expected_reference_rows = 3 if args.mode == "smoke" else 18
    reference_rows = (
        len(pd.read_csv(reference_path)) if reference_path.is_file() else 0
    )
    checks: dict[str, bool] = {
        "task_specific_reference_rows": (
            reference_rows == expected_reference_rows
        ),
        "expected_rows": len(frame) == EXPECTED_ROWS[args.mode],
        "all_completed": bool((frame.runner_status == "completed").all()),
        "unique_runs": not frame.duplicated(key).any(),
        "algorithm_set": set(frame.algorithm.unique()) == set(ALGORITHMS),
        "internal_budget": bool((frame.nfe == frame.budget).all()),
        "history_monotone": bool(frame.history_monotone.all()),
        "finite_fbest": bool(np.isfinite(frame.fbest.to_numpy(float)).all()),
        "finite_normalized_gap": bool(np.isfinite(frame.normalized_gap.to_numpy(float)).all()),
        "implementation_recorded": bool(frame.implementation.astype(str).str.len().gt(0).all()),
    }
    if args.mode == "confirmatory":
        checks.update(
            {
                "c1_rows": int((frame.domain == "c1").sum()) == 1890,
                "c2_rows": int((frame.domain == "c2").sum()) == 420,
                "c1_families": set(frame.loc[frame.domain == "c1", "task_name"]) == set(C1_FAMILIES),
                "nist_datasets": set(frame.loc[frame.domain == "c2", "task_name"]) == set(NIST_DATASETS),
                "paired_seeds": set(frame.paired_seed.unique()) == set(range(10)),
            }
        )
    basin = frame[frame.algorithm == "BasinGraph"]
    checks.update(
        {
            "basingraph_options_hash": set(basin.options_hash.unique()) == {EXPECTED_OPTIONS_HASH},
            "basingraph_phase_accounting": bool((basin.phase_sum == basin.budget).all()),
            "basingraph_archive_capacity": bool(((basin.archive_nodes >= 1) & (basin.archive_nodes <= 80)).all()),
            "basingraph_graph_integrity": bool(basin.graph_referential_integrity.all()),
        }
    )

    frame["final_rank"] = frame.groupby(["domain", "task_id", "paired_seed"])["normalized_gap"].rank(
        method="average", ascending=True
    )
    frame["final_win"] = frame.groupby(["domain", "task_id", "paired_seed"])["normalized_gap"].transform(
        lambda values: np.isclose(values, values.min(), rtol=1e-10, atol=1e-12)
    )
    summary = (
        frame.groupby("algorithm")
        .agg(
            mean_final_rank=("final_rank", "mean"),
            median_final_rank=("final_rank", "median"),
            final_wins=("final_win", "sum"),
            median_normalized_gap=("normalized_gap", "median"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            runs=("task_id", "count"),
        )
        .reset_index()
    )
    summary["algorithm_display"] = summary.algorithm.map(DISPLAY_NAMES)
    summary.sort_values(["mean_final_rank", "algorithm"], inplace=True)
    summary.to_csv(run_root / "algorithm_summary.csv", index=False)

    family_summary = (
        frame.groupby(["domain", "task_name", "algorithm"])
        .agg(
            mean_final_rank=("final_rank", "mean"),
            median_normalized_gap=("normalized_gap", "median"),
            final_wins=("final_win", "sum"),
            runs=("task_id", "count"),
        )
        .reset_index()
    )
    family_summary.to_csv(run_root / "family_algorithm_summary.csv", index=False)

    primary = primary_target_table(frame)
    primary.to_csv(run_root / "target_runtime_records.csv", index=False)
    checkpoint_summary = (
        primary.groupby(
            ["algorithm", "domain", "checkpoint_evaluations_per_dimension"]
        )
        .agg(
            target_fraction=("reached", "mean"),
            records=("reached", "size"),
        )
        .reset_index()
    )
    checkpoint_summary.to_csv(
        run_root / "target_fraction_summary.csv",
        index=False,
    )

    final_target_summary = (
        primary[primary.is_final_checkpoint]
        .groupby(["algorithm", "domain"])
        .agg(
            target_fraction=("reached", "mean"),
            records=("reached", "size"),
        )
        .reset_index()
    )
    final_target_summary.to_csv(
        run_root / "final_registered_budget_target_fraction.csv",
        index=False,
    )

    primary_pairwise = paired_target_statistics(primary)
    primary_pairwise.to_csv(
        run_root / "pairwise_target_fraction.csv",
        index=False,
    )

    family_target = (
        primary.groupby(
            ["algorithm", "domain", "task_name", "checkpoint_evaluations_per_dimension"]
        )
        .agg(target_fraction=("reached", "mean"), records=("reached", "size"))
        .reset_index()
    )
    family_target.to_csv(run_root / "family_target_fraction_summary.csv", index=False)

    pairwise = pairwise_final(frame)
    pairwise.to_csv(run_root / "pairwise_final_normalized_gap.csv", index=False)
    pivot = frame.pivot(
        index=["domain", "task_id", "paired_seed"], columns="algorithm", values="normalized_gap"
    )
    friedman = friedmanchisquare(*[pivot[name].to_numpy() for name in ALGORITHMS])
    (run_root / "friedman_final_normalized_gap.json").write_text(
        json.dumps(
            {
                "statistic": float(friedman.statistic),
                "p_value": float(friedman.pvalue),
                "blocks": len(pivot),
                "algorithms": ALGORITHMS,
                "endpoint": "final normalized gap",
                "secondary_endpoint": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    passed = all(checks.values())
    report = {
        "status": "TRACK_C_FINAL_VALIDATION_OK" if passed else "TRACK_C_FINAL_VALIDATION_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "confirmatory_evidence": args.mode == "confirmatory",
        "identity": identity,
        "rows": len(frame),
        "blocks": frame[["domain", "task_id", "paired_seed"]].drop_duplicates().shape[0],
        "checks_total": len(checks),
        "checks_passed": sum(checks.values()),
        "checks_failed": sum(not value for value in checks.values()),
        "checks": checks,
        "best_mean_final_rank_algorithm": str(summary.iloc[0].algorithm),
    }
    (run_root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.mode == "smoke" and passed:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        identity_path = ROOT / "protocols/evidence_extension_v1/track_c/TRACK_C_SOURCE_IDENTITY.json"
        gate = {
            "status": "TRACK_C_LOCAL_SMOKE_GATE_OK",
            "run_id": args.run_id,
            "source_commit": head,
            "source_identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
            "confirmatory_objective_evaluations": 0,
            "rows": len(frame),
        }
        gate_path = ROOT / "results_b21/track_c/LOCAL_SMOKE_GATE.json"
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate, indent=2), encoding="utf-8")
    write_manifest(run_root)
    if not passed:
        raise RuntimeError("Track C final validation failed.")
    print("TRACK_C_FINAL_VALIDATION_OK")
    print(f"Rows: {len(frame)}")
    print(f"Blocks: {report['blocks']}")
    print(f"Output: {run_root}")


if __name__ == "__main__":
    main()
