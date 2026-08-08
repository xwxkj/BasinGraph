#!/usr/bin/env python3
"""Validate, summarize and run official cocopp for B21 Track B."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_b.common import (  # noqa: E402
    ALGORITHMS,
    DISPLAY_NAMES,
    EXPECTED_FULL_HASH,
    MODE_CONFIG,
    seed_for,
    sha256_file,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--result-root", default="results_b21/track_b")
    parser.add_argument("--skip-cocopp", action="store_true")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def paired_final_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    basin = frame[frame["algorithm"] == "BasinGraph"][
        ["problem_id", "fbest"]
    ].rename(columns={"fbest": "basingraph_fbest"})
    rows = []
    raw_p = []
    for algorithm in ALGORITHMS[1:]:
        paired = frame[frame["algorithm"] == algorithm][
            ["problem_id", "fbest"]
        ].merge(basin, on="problem_id", validate="one_to_one")
        delta = (
            paired["fbest"].to_numpy(dtype=float)
            - paired["basingraph_fbest"].to_numpy(dtype=float)
        )
        nonzero = delta[~np.isclose(delta, 0.0, rtol=1e-12, atol=1e-14)]
        if len(nonzero) == 0:
            statistic = 0.0
            p_value = 1.0
        else:
            test = wilcoxon(
                nonzero,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        baseline_worse = int(np.sum(delta > 1e-14))
        baseline_better = int(np.sum(delta < -1e-14))
        ties = int(len(delta) - baseline_worse - baseline_better)
        rank_biserial = (
            float((baseline_worse - baseline_better) / (baseline_worse + baseline_better))
            if baseline_worse + baseline_better
            else 0.0
        )
        rows.append(
            {
                "baseline": algorithm,
                "baseline_display": DISPLAY_NAMES[algorithm],
                "paired_problems": len(delta),
                "baseline_worse_than_basingraph": baseline_worse,
                "baseline_better_than_basingraph": baseline_better,
                "ties": ties,
                "median_f_difference_baseline_minus_basingraph": float(np.median(delta)),
                "wilcoxon_statistic": statistic,
                "raw_p": p_value,
                "rank_biserial_positive_means_basingraph_better": rank_biserial,
            }
        )
        raw_p.append(p_value)
    adjusted = holm_adjust(raw_p)
    for row, value in zip(rows, adjusted):
        row["holm_p"] = value
    return pd.DataFrame(rows)


def checkpoint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for row in frame.itertuples(index=False):
        checkpoints = json.loads(row.checkpoint_fbest_json)
        for checkpoint, value in checkpoints.items():
            expanded.append(
                {
                    "problem_id": row.problem_id,
                    "algorithm": row.algorithm,
                    "dimension": int(row.dimension),
                    "function_group_id": row.function_group_id,
                    "checkpoint_evaluations_per_dimension": int(checkpoint),
                    "fbest": float(value),
                }
            )
    data = pd.DataFrame(expanded)
    data["rank"] = data.groupby(
        ["problem_id", "checkpoint_evaluations_per_dimension"]
    )["fbest"].rank(method="average", ascending=True)
    return (
        data.groupby(
            [
                "algorithm",
                "dimension",
                "function_group_id",
                "checkpoint_evaluations_per_dimension",
            ]
        )
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            problems=("problem_id", "count"),
        )
        .reset_index()
    )


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    if config["confirmatory"] and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory finalization requires authorization.")
    identity = verify_source_identity(require_clean=True)
    run_root = ROOT / args.result_root / args.run_id
    raw_path = run_root / "track_b_raw_results.csv"
    marker_path = run_root / "ALL_SHARDS_COMPLETE.json"
    if not raw_path.is_file() or not marker_path.is_file():
        raise RuntimeError("Track B shards are not complete.")

    frame = pd.read_csv(raw_path)
    expected_problems = (
        len(config["functions"])
        * len(config["dimensions"])
        * len(config["instances"])
    )
    checks: dict[str, bool] = {
        "expected_rows": len(frame) == config["expected_rows"],
        "all_completed": bool((frame["runner_status"] == "completed").all()),
        "unique_algorithm_problem": not frame.duplicated(
            ["algorithm", "problem_id"]
        ).any(),
        "problem_count": frame["problem_id"].nunique() == expected_problems,
        "algorithm_set": set(frame["algorithm"].unique()) == set(ALGORITHMS),
        "functions": sorted(frame["function_index"].unique().tolist())
        == config["functions"],
        "dimensions": sorted(frame["dimension"].unique().tolist())
        == config["dimensions"],
        "instances": sorted(frame["instance_index"].unique().tolist())
        == config["instances"],
        "internal_budget": bool((frame["nfe_internal"] == frame["budget"]).all()),
        "observer_budget": bool((frame["nfe_observer"] == frame["budget"]).all()),
        "history_monotone": bool(bool_series(frame["history_monotone"]).all()),
        "seed_formula": all(
            int(row.seed)
            == seed_for(
                args.base_seed,
                int(row.function_index),
                int(row.dimension),
                int(row.instance_index),
            )
            for row in frame.itertuples(index=False)
        ),
        "implementation_recorded": bool(frame["implementation"].astype(str).str.len().gt(0).all()),
    }

    basin = frame[frame["algorithm"] == "BasinGraph"]
    checks.update(
        {
            "basingraph_options_hash": set(basin["options_hash"].unique())
            == {EXPECTED_FULL_HASH},
            "basingraph_phase_accounting": bool(
                (basin["phase_sum"] == basin["budget"]).all()
            ),
            "basingraph_archive_capacity": bool(
                ((basin["archive_nodes"] >= 1) & (basin["archive_nodes"] <= 80)).all()
            ),
            "basingraph_graph_integrity": bool(
                bool_series(basin["graph_referential_integrity"]).all()
            ),
        }
    )

    frame["final_target_hit"] = bool_series(frame["final_target_hit"])
    frame["rank"] = frame.groupby("problem_id")["fbest"].rank(
        method="average", ascending=True
    )
    frame["win"] = frame.groupby("problem_id")["fbest"].transform(
        lambda values: np.isclose(
            values, values.min(), rtol=1e-12, atol=1e-14
        )
    )

    summary = (
        frame.groupby("algorithm")
        .agg(
            mean_final_rank=("rank", "mean"),
            median_final_rank=("rank", "median"),
            final_wins=("win", "sum"),
            final_target_successes=("final_target_hit", "sum"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            runs=("problem_id", "count"),
        )
        .reset_index()
    )
    summary["algorithm_display"] = summary["algorithm"].map(DISPLAY_NAMES)
    summary["algorithm_order"] = summary["algorithm"].map(
        {name: index for index, name in enumerate(ALGORITHMS)}
    )
    summary.sort_values(
        ["mean_final_rank", "algorithm_order"], inplace=True
    )
    summary.drop(columns=["algorithm_order"], inplace=True)
    summary.to_csv(run_root / "algorithm_summary.csv", index=False)

    group_summary = (
        frame.groupby(
            ["algorithm", "dimension", "function_group_id", "function_group"]
        )
        .agg(
            mean_final_rank=("rank", "mean"),
            median_final_rank=("rank", "median"),
            final_wins=("win", "sum"),
            final_target_successes=("final_target_hit", "sum"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            runs=("problem_id", "count"),
        )
        .reset_index()
    )
    group_summary.to_csv(
        run_root / "function_group_dimension_summary.csv", index=False
    )

    checkpoint = checkpoint_summary(frame)
    checkpoint.to_csv(run_root / "checkpoint_rank_summary.csv", index=False)

    pairwise = paired_final_statistics(frame)
    pairwise.to_csv(run_root / "pairwise_final_values.csv", index=False)

    pivot = frame.pivot(index="problem_id", columns="algorithm", values="fbest")
    friedman_values = [pivot[name].to_numpy() for name in ALGORITHMS]
    friedman = friedmanchisquare(*friedman_values)
    friedman_report = {
        "statistic": float(friedman.statistic),
        "p_value": float(friedman.pvalue),
        "blocks": int(len(pivot)),
        "algorithms": ALGORITHMS,
        "endpoint": "final objective values",
        "secondary_endpoint": True,
    }
    (run_root / "friedman_final_values.json").write_text(
        json.dumps(friedman_report, indent=2), encoding="utf-8"
    )

    implementation_summary = (
        frame.groupby(["algorithm", "implementation"])
        .agg(runs=("problem_id", "count"), messages=("message", "nunique"))
        .reset_index()
    )
    implementation_summary.to_csv(
        run_root / "implementation_summary.csv", index=False
    )

    complete_observers: dict[str, dict[int, Path]] = {
        algorithm: {} for algorithm in ALGORITHMS
    }
    observer_paths: list[Path] = []
    for algorithm in ALGORITHMS:
        for dimension in config["dimensions"]:
            shard_root = run_root / "shards" / f"{algorithm}_d{dimension}"
            marker = json.loads(
                (shard_root / "SHARD_COMPLETE.json").read_text(encoding="utf-8")
            )
            observer = Path(marker["observer_absolute"])
            exists = observer.is_dir()
            checks[f"observer_exists::{algorithm}_d{dimension}"] = exists
            if exists:
                complete_observers[algorithm][dimension] = observer
                observer_paths.append(observer)

    staging_root = (
        ROOT / "exdata" / "b21_track_b_cocopp_inputs" / args.run_id / args.mode
    )
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    def link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    cocopp_inputs = []
    for algorithm in ALGORITHMS:
        algorithm_root = staging_root / algorithm
        algorithm_root.mkdir(parents=True)
        for dimension in config["dimensions"]:
            source = complete_observers[algorithm].get(dimension)
            if source is None:
                continue
            shutil.copytree(
                source,
                algorithm_root / f"d{dimension}",
                copy_function=link_or_copy,
            )
        cocopp_inputs.append(algorithm_root)

    checks["cocopp_input_count"] = len(cocopp_inputs) == len(ALGORITHMS)
    pp_root = run_root / "cocopp"
    cocopp_log = run_root / "cocopp.log"
    cocopp_status = "skipped"
    if not args.skip_cocopp:
        command = [
            sys.executable,
            "-m",
            "cocopp",
            "-o",
            str(pp_root),
            *[str(path) for path in cocopp_inputs],
        ]
        with cocopp_log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        cocopp_status = "completed" if process.returncode == 0 else "failed"
        checks["cocopp_completed"] = process.returncode == 0

    passed = all(checks.values())
    report = {
        "status": "TRACK_B_FINAL_VALIDATION_OK" if passed else "TRACK_B_FINAL_VALIDATION_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "run_id": args.run_id,
        "confirmatory_evidence": bool(config["confirmatory"]),
        "identity": identity,
        "rows": len(frame),
        "problems": frame["problem_id"].nunique(),
        "algorithms": ALGORITHMS,
        "cocopp_status": cocopp_status,
        "checks_total": len(checks),
        "checks_passed": sum(checks.values()),
        "checks_failed": sum(not value for value in checks.values()),
        "checks": checks,
        "best_mean_final_rank_algorithm": str(summary.iloc[0]["algorithm"]),
        "friedman_final_values": friedman_report,
        "raw_results_sha256": sha256_file(raw_path),
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    write_manifest(run_root)

    print(report["status"])
    print(f"Rows: {len(frame)}")
    print(f"Problems: {frame['problem_id'].nunique()}")
    print(f"COCO postprocessing: {cocopp_status}")
    print(f"Output: {run_root}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        print("Failed checks: " + ", ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
