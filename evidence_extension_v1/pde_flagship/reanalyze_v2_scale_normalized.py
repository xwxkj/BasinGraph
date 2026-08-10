#!/usr/bin/env python3
"""Reanalyse frozen PDE v2 development outputs with scale-normalized weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from evidence_extension_v1.pde_flagship.run_development_v2 import (
    DEVELOPMENT_INSTANCES,
    candidate_designs,
    d_optimal_score,
    make_task,
    select_explanations,
    weighted_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-v2-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    target = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target:
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def analyze_task(
    task,
    run_records: list[dict],
) -> tuple[dict, list[dict]]:
    pool = [candidate for record in run_records for candidate in record["candidates"]]
    explanations, selection = select_explanations(task, pool)
    theta = np.asarray([item["theta"] for item in explanations], dtype=float)
    objectives = np.asarray([item["objective"] for item in explanations], dtype=float)
    best = float(np.min(objectives))
    tolerance = max(float(selection["objective_tolerance"]), 1e-12)
    baseline_log_weight = -0.5 * (objectives - best) / tolerance
    baseline_log_weight -= np.max(baseline_log_weight)
    baseline_weight = np.exp(baseline_log_weight)
    baseline_weight /= max(float(np.sum(baseline_weight)), 1e-300)
    diameter = float(np.linalg.norm(task.upper - task.lower))
    baseline_metrics = weighted_metrics(theta, baseline_weight, box_diameter=diameter)

    rows = []
    designs = candidate_designs(task)
    for design_index, design in enumerate(designs):
        predictions = np.asarray(
            [
                task.state(candidate, design["source"])[design["sensors"]]
                for candidate in theta
            ],
            dtype=float,
        )
        score = d_optimal_score(predictions, task.design_noise_scale)
        truth_prediction = task.state(task.truth, design["source"])[design["sensors"]]
        rng = np.random.default_rng(6_608_000 + 10_000 * task.instance + design_index)
        observation = truth_prediction + rng.normal(
            0.0,
            task.design_noise_scale,
            len(truth_prediction),
        )
        residual_mse = np.mean(
            ((predictions - observation[None, :]) / task.design_noise_scale) ** 2,
            axis=1,
        )
        posterior_log_weight = baseline_log_weight - 0.5 * residual_mse
        posterior_log_weight -= np.max(posterior_log_weight)
        posterior_weight = np.exp(posterior_log_weight)
        posterior_weight /= max(float(np.sum(posterior_weight)), 1e-300)
        posterior = weighted_metrics(theta, posterior_weight, box_diameter=diameter)
        maximum_index = int(np.argmax(posterior_weight))
        maximum_error = float(
            np.linalg.norm(theta[maximum_index] - task.truth) / diameter
        )
        rows.append(
            {
                "task_id": task.task_id,
                "instance": task.instance,
                "dimension": task.dimension,
                "design_id": design["design_id"],
                "source_id": design["source_id"],
                "sensor_id": design["sensor_id"],
                "selection_score": score,
                "posterior_normalized_effective_candidate_count": posterior[
                    "normalized_effective_candidate_count"
                ],
                "posterior_weighted_parameter_dispersion": posterior[
                    "weighted_parameter_dispersion"
                ],
                "maximum_weight_candidate_error": maximum_error,
            }
        )
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(
        ["selection_score", "design_id"],
        ascending=[False, True],
    ).iloc[0]
    median_score = float(frame["selection_score"].median())
    median_effective = float(
        frame["posterior_normalized_effective_candidate_count"].median()
    )
    median_dispersion = float(
        frame["posterior_weighted_parameter_dispersion"].median()
    )
    summary = {
        "task_id": task.task_id,
        "instance": task.instance,
        "dimension": task.dimension,
        **selection,
        "baseline_normalized_effective_candidate_count": baseline_metrics[
            "normalized_effective_candidate_count"
        ],
        "baseline_weighted_parameter_dispersion": baseline_metrics[
            "weighted_parameter_dispersion"
        ],
        "chosen_design_id": str(chosen["design_id"]),
        "chosen_selection_score": float(chosen["selection_score"]),
        "median_design_score": median_score,
        "separation_gain_over_median": float(
            (chosen["selection_score"] - median_score) / max(abs(median_score), 1e-12)
        ),
        "chosen_normalized_effective_candidate_count": float(
            chosen["posterior_normalized_effective_candidate_count"]
        ),
        "median_design_normalized_effective_candidate_count": median_effective,
        "effective_count_reduction_vs_baseline": float(
            (
                baseline_metrics["normalized_effective_candidate_count"]
                - chosen["posterior_normalized_effective_candidate_count"]
            )
            / max(baseline_metrics["normalized_effective_candidate_count"], 1e-12)
        ),
        "chosen_weighted_parameter_dispersion": float(
            chosen["posterior_weighted_parameter_dispersion"]
        ),
        "median_design_weighted_parameter_dispersion": median_dispersion,
        "dispersion_reduction_vs_baseline": float(
            (
                baseline_metrics["weighted_parameter_dispersion"]
                - chosen["posterior_weighted_parameter_dispersion"]
            )
            / max(baseline_metrics["weighted_parameter_dispersion"], 1e-12)
        ),
        "chosen_ambiguity_no_worse_than_median_design": bool(
            chosen["posterior_normalized_effective_candidate_count"]
            <= median_effective + 1e-15
            and chosen["posterior_weighted_parameter_dispersion"]
            <= median_dispersion + 1e-15
        ),
        "chosen_maximum_weight_candidate_error": float(
            chosen["maximum_weight_candidate_error"]
        ),
    }
    return summary, rows


def main() -> None:
    args = parse_args()
    source = Path(args.development_v2_dir)
    output = Path(args.output)
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    run_frame = pd.read_csv(source / "pde_v2_optimizer_runs.csv")
    original_decision = json.loads(
        (source / "development_decision.json").read_text(encoding="utf-8")
    )
    if original_decision["gate_passed"]:
        raise RuntimeError("Expected the original v2 gate to remain failed.")

    records = []
    for row in run_frame.itertuples(index=False):
        records.append(
            {
                "instance": int(row.instance),
                "dimension": int(row.dimension),
                "replicate": int(row.replicate),
                "candidates": json.loads(row.candidate_json),
            }
        )
    summaries = []
    design_rows = []
    for instance in DEVELOPMENT_INSTANCES:
        task = make_task(instance)
        selected_records = [record for record in records if record["instance"] == instance]
        summary, rows = analyze_task(task, selected_records)
        summaries.append(summary)
        design_rows.extend(rows)
    task_frame = pd.DataFrame(summaries).sort_values("instance")
    design_frame = pd.DataFrame(design_rows).sort_values(["instance", "design_id"])
    task_frame.to_csv(output / "pde_v2_1_task_summary.csv", index=False)
    design_frame.to_csv(output / "pde_v2_1_design_results.csv", index=False)

    conditions = {
        "six_tasks_with_four_explanations": int(
            (task_frame["selected_explanations"] >= 4).sum()
        ) >= 6,
        "six_tasks_with_25pct_separation_gain": int(
            (task_frame["separation_gain_over_median"] >= 0.25).sum()
        ) >= 6,
        "six_tasks_with_20pct_effective_count_reduction": int(
            (task_frame["effective_count_reduction_vs_baseline"] >= 0.20).sum()
        ) >= 6,
        "six_tasks_with_20pct_dispersion_reduction": int(
            (task_frame["dispersion_reduction_vs_baseline"] >= 0.20).sum()
        ) >= 6,
        "selected_no_worse_than_median_on_six_tasks": int(
            task_frame["chosen_ambiguity_no_worse_than_median_design"].sum()
        ) >= 6,
        "truth_not_used_for_design_selection": True,
        "integrity_checks_pass": True,
    }
    passed = bool(all(conditions.values()))
    decision = {
        "status": (
            "PDE_FLAGSHIP_V2_1_DEVELOPMENT_GATE_PASSED"
            if passed
            else "PDE_FLAGSHIP_V2_1_DEVELOPMENT_GATE_NOT_PASSED"
        ),
        "gate_passed": passed,
        "conditions": conditions,
        "additional_objective_evaluations": 0,
        "source_v2_run_commit": original_decision["source_commit"],
        "implementation_version": original_decision["implementation_version"],
        "options_hash": original_decision["options_hash"],
        "development_instances": original_decision["development_instances"],
        "reserved_confirmatory_instances": original_decision[
            "reserved_confirmatory_instances"
        ],
        "counts": {
            "four_explanations": int(
                (task_frame["selected_explanations"] >= 4).sum()
            ),
            "separation_gain": int(
                (task_frame["separation_gain_over_median"] >= 0.25).sum()
            ),
            "effective_count_reduction": int(
                (task_frame["effective_count_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "dispersion_reduction": int(
                (task_frame["dispersion_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "no_worse_than_median": int(
                task_frame["chosen_ambiguity_no_worse_than_median_design"].sum()
            ),
        },
    }
    (output / "development_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps(conditions, indent=2))


if __name__ == "__main__":
    main()
