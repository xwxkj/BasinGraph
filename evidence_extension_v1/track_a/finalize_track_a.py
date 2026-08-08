#!/usr/bin/env python3
"""Validate, analyze and postprocess B21 Track A."""

from __future__ import annotations

import argparse
import csv
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

from evidence_extension_v1.track_a.common import (  # noqa: E402
    MODE_CONFIG,
    VARIANT_ORDER,
    expected_variant_hashes,
    seed_for,
    sha256_file,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--result-root", default="results_b21/track_a")
    parser.add_argument("--skip-cocopp", action="store_true")
    parser.add_argument("--authorize-confirmatory", action="store_true")
    return parser.parse_args()


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


def write_manifest(root: Path) -> None:
    manifest = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest:
            continue
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False)


def paired_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    full = frame[frame["variant"] == "Full"][["problem_id", "fbest"]].rename(
        columns={"fbest": "full_fbest"}
    )
    rows = []
    raw_p = []
    for variant in VARIANT_ORDER[1:]:
        paired = frame[frame["variant"] == variant][
            ["problem_id", "fbest"]
        ].merge(full, on="problem_id", validate="one_to_one")
        delta = (
            paired["fbest"].to_numpy(dtype=float)
            - paired["full_fbest"].to_numpy(dtype=float)
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
        worse = int(np.sum(delta > 1e-14))
        better = int(np.sum(delta < -1e-14))
        ties = int(len(delta) - worse - better)
        rank_biserial = (
            float((worse - better) / (worse + better))
            if worse + better
            else 0.0
        )
        rows.append({
            "ablation": variant,
            "paired_problems": len(delta),
            "ablation_worse_than_full": worse,
            "ablation_better_than_full": better,
            "ties": ties,
            "median_f_difference_ablation_minus_full": float(np.median(delta)),
            "wilcoxon_statistic": statistic,
            "raw_p": p_value,
            "rank_biserial_positive_means_full_better": rank_biserial,
        })
        raw_p.append(p_value)
    for row, value in zip(rows, holm_adjust(raw_p)):
        row["holm_p"] = value
    return pd.DataFrame(rows)


def checkpoint_summary(frame: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for row in frame.itertuples(index=False):
        checkpoints = json.loads(row.checkpoint_fbest_json)
        for checkpoint, value in checkpoints.items():
            expanded.append({
                "problem_id": row.problem_id,
                "variant": row.variant,
                "dimension": int(row.dimension),
                "function_group_id": row.function_group_id,
                "checkpoint_evaluations_per_dimension": int(checkpoint),
                "fbest": float(value),
            })
    data = pd.DataFrame(expanded)
    data["rank"] = data.groupby(
        ["problem_id", "checkpoint_evaluations_per_dimension"]
    )["fbest"].rank(method="average", ascending=True)
    return (
        data.groupby([
            "variant", "dimension", "function_group_id",
            "checkpoint_evaluations_per_dimension",
        ])
        .agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            problems=("problem_id", "count"),
        )
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError(
            "Confirmatory finalization requires --authorize-confirmatory."
        )
    identity = verify_source_identity(require_clean=True)
    config = MODE_CONFIG[args.mode]
    run_root = ROOT / args.result_root / args.run_id
    raw_path = run_root / "track_a_raw_results.csv"
    marker = run_root / "ALL_SHARDS_COMPLETE.json"
    if not raw_path.is_file() or not marker.is_file():
        raise RuntimeError("Track A shards are not complete.")

    frame = pd.read_csv(raw_path)
    hashes = expected_variant_hashes()
    expected_problems = (
        len(config["functions"])
        * len(config["dimensions"])
        * len(config["instances"])
    )

    checks = {
        "expected_rows": len(frame) == config["expected_rows"],
        "all_completed": bool((frame["runner_status"] == "completed").all()),
        "unique_variant_problem": not frame.duplicated(
            ["variant", "problem_id"]
        ).any(),
        "problem_count": frame["problem_id"].nunique() == expected_problems,
        "variant_set": set(frame["variant"].unique()) == set(VARIANT_ORDER),
        "functions": sorted(frame["function_index"].unique().tolist())
        == config["functions"],
        "dimensions": sorted(frame["dimension"].unique().tolist())
        == config["dimensions"],
        "instances": sorted(frame["instance_index"].unique().tolist())
        == config["instances"],
        "internal_budget": bool(
            (frame["nfe_internal"] == frame["budget"]).all()
        ),
        "observer_budget": bool(
            (frame["nfe_observer"] == frame["budget"]).all()
        ),
        "phase_accounting": bool(
            (frame["phase_sum"] == frame["budget"]).all()
        ),
        "archive_capacity": bool((
            (frame["archive_nodes"] >= 1)
            & (frame["archive_nodes"] <= 80)
        ).all()),
        "graph_integrity": bool(
            bool_series(frame["graph_referential_integrity"]).all()
        ),
        "history_monotone": bool(
            bool_series(frame["history_monotone"]).all()
        ),
        "seed_formula": all(
            int(row.seed) == seed_for(
                args.base_seed,
                int(row.function_index),
                int(row.dimension),
                int(row.instance_index),
            )
            for row in frame.itertuples(index=False)
        ),
        "variant_hashes": all(
            set(frame.loc[
                frame["variant"] == variant,
                "options_hash",
            ].unique()) == {expected_hash}
            for variant, expected_hash in hashes.items()
        ),
    }

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
        frame.groupby("variant")
        .agg(
            mean_final_rank=("rank", "mean"),
            median_final_rank=("rank", "median"),
            final_wins=("win", "sum"),
            final_target_successes=("final_target_hit", "sum"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            mean_archive_nodes=("archive_nodes", "mean"),
            mean_graph_edges=("graph_edges", "mean"),
            runs=("problem_id", "count"),
        )
        .reset_index()
    )
    summary["variant_order"] = summary["variant"].map(
        {name: index for index, name in enumerate(VARIANT_ORDER)}
    )
    summary.sort_values(["mean_final_rank", "variant_order"], inplace=True)
    summary.drop(columns=["variant_order"], inplace=True)
    summary.to_csv(run_root / "variant_summary.csv", index=False)

    group_summary = (
        frame.groupby([
            "variant", "dimension", "function_group_id", "function_group"
        ])
        .agg(
            mean_final_rank=("rank", "mean"),
            median_final_rank=("rank", "median"),
            final_wins=("win", "sum"),
            final_target_successes=("final_target_hit", "sum"),
            runs=("problem_id", "count"),
        )
        .reset_index()
    )
    group_summary.to_csv(
        run_root / "function_group_dimension_summary.csv", index=False
    )

    phase_rows = []
    for row in frame.itertuples(index=False):
        mapping = json.loads(row.phase_evaluations_json)
        for phase, evaluations in mapping.items():
            phase_rows.append({
                "variant": row.variant,
                "dimension": int(row.dimension),
                "function_group_id": row.function_group_id,
                "phase": phase,
                "evaluations": int(evaluations),
                "fraction": int(evaluations) / int(row.budget),
            })
    phase_summary = (
        pd.DataFrame(phase_rows)
        .groupby(["variant", "dimension", "function_group_id", "phase"])
        .agg(
            mean_evaluations=("evaluations", "mean"),
            mean_fraction=("fraction", "mean"),
            median_fraction=("fraction", "median"),
            runs=("fraction", "count"),
        )
        .reset_index()
    )
    phase_summary.to_csv(run_root / "phase_summary.csv", index=False)

    paired_statistics(frame).to_csv(
        run_root / "pairwise_final_values.csv", index=False
    )
    checkpoint_summary(frame).to_csv(
        run_root / "checkpoint_rank_summary.csv", index=False
    )

    pivot = frame.pivot(index="problem_id", columns="variant", values="fbest")
    friedman = friedmanchisquare(
        *[pivot[name].to_numpy() for name in VARIANT_ORDER]
    )
    friedman_report = {
        "statistic": float(friedman.statistic),
        "p_value": float(friedman.pvalue),
        "blocks": int(len(pivot)),
        "variants": VARIANT_ORDER,
        "endpoint": "final objective value ranks",
        "secondary_descriptive_test": True,
    }
    (run_root / "friedman_final_values.json").write_text(
        json.dumps(friedman_report, indent=2), encoding="utf-8"
    )

    observer_paths = []
    for variant in VARIANT_ORDER:
        for dimension in config["dimensions"]:
            shard = run_root / "shards" / f"{variant}_d{dimension}"
            complete = json.loads(
                (shard / "SHARD_COMPLETE.json").read_text(encoding="utf-8")
            )
            observer = Path(complete["observer_absolute"])
            key = f"observer_exists::{variant}_d{dimension}"
            checks[key] = observer.is_dir()
            if observer.is_dir():
                observer_paths.append(observer)

    pp_root = run_root / "cocopp"
    cocopp_log = run_root / "cocopp.log"
    cocopp_status = "skipped"
    if not args.skip_cocopp:
        command = [sys.executable, "-m", "cocopp", "-o", str(pp_root)]
        if args.mode == "smoke":
            command.extend(["--in-a-hurry", "1000", "--no-svg"])
        command.extend(str(path) for path in observer_paths)
        with cocopp_log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        cocopp_status = "completed" if process.returncode == 0 else "failed"
        checks["cocopp_completed"] = process.returncode == 0
        checks["cocopp_index"] = (pp_root / "index.html").is_file()

    (run_root / "complete_observer_paths.txt").write_text(
        "\n".join(str(path.relative_to(ROOT)) for path in observer_paths)
        + "\n",
        encoding="utf-8",
    )

    report = {
        "status": (
            "TRACK_A_FINAL_VALIDATION_OK"
            if all(checks.values())
            else "TRACK_A_FINAL_VALIDATION_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "confirmatory": bool(config["confirmatory"]),
        "identity": identity,
        "run_id": args.run_id,
        "rows": len(frame),
        "problems": expected_problems,
        "variants": len(VARIANT_ORDER),
        "raw_results_sha256": sha256_file(raw_path),
        "cocopp_status": cocopp_status,
        "checks": checks,
        "friedman_final_values": friedman_report,
    }
    (run_root / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_manifest(run_root)

    print(report["status"])
    print(f"Rows: {len(frame)}")
    print(f"Problems: {expected_problems}")
    print(f"COCO postprocessing: {cocopp_status}")
    print(f"Output: {run_root}")

    if report["status"].endswith("FAILED"):
        failed = {name: value for name, value in checks.items() if not value}
        print(json.dumps(failed, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
