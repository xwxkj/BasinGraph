#!/usr/bin/env python3
"""Run PDE non-identifiability flagship development cycle v2."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import (  # noqa: E402
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)
from evidence_extension_v1.pde_flagship.run_development import (  # noqa: E402
    sensor_indices,
    thomas,
)


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
EXPECTED_OPTIMIZER_BLOB = "4433d2b92075dcd858529f7f13342631847def4c"
DEVELOPMENT_INSTANCES = tuple(range(21, 29))
CONFIRMATORY_INSTANCES = tuple(range(201, 209))
SEEDS_PER_INSTANCE = 10
BUDGET_MULTIPLIER = 400
MAX_EXPLANATIONS = 16
MIN_PARAMETER_SEPARATION = 0.08
BASELINE_WEIGHT_TEMPERATURE = 5.0
BASE_SEED = 202608700
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass
class PDETaskV2:
    instance: int
    dimension: int
    lower: np.ndarray
    upper: np.ndarray
    truth: np.ndarray
    basis: np.ndarray
    interior_x: np.ndarray
    baseline_source: np.ndarray
    baseline_sensor_indices: np.ndarray
    observations: np.ndarray
    baseline_noise_scale: float
    design_noise_scale: float
    f_zero: float

    @property
    def task_id(self) -> str:
        return f"elliptic_pde_flagship_v2_i{self.instance}_d{self.dimension}"

    def state(self, theta: np.ndarray, source: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        log_k = np.clip(self.basis @ theta, -3.0, 3.0)
        k_nodes = np.exp(log_k)
        k_half = 0.5 * (k_nodes[:-1] + k_nodes[1:])
        left = k_half[:-1]
        right = k_half[1:]
        h = 1.0 / (len(source) + 1)
        diagonal = (left + right) / (h * h)
        lower = -left[1:] / (h * h)
        upper = -right[:-1] / (h * h)
        return thomas(lower, diagonal, upper, source)

    def objective(self, theta: np.ndarray) -> float:
        prediction = self.state(theta, self.baseline_source)[
            self.baseline_sensor_indices
        ]
        residual = (prediction - self.observations) / self.baseline_noise_scale
        return float(
            np.mean(residual * residual)
            + 1e-8 * np.mean(np.asarray(theta, dtype=float) ** 2)
        )


def weighted_metrics(
    candidates: np.ndarray,
    weights: np.ndarray,
    *,
    box_diameter: float,
) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    weights = weights / max(float(np.sum(weights)), 1e-300)
    mean = np.sum(weights[:, None] * candidates, axis=0)
    dispersion = float(
        np.sum(weights * np.sum((candidates - mean[None, :]) ** 2, axis=1))
        / max(box_diameter * box_diameter, 1e-300)
    )
    entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-300, 1.0))))
    normalized_effective_count = float(
        np.exp(entropy) / max(1, len(candidates))
    )
    return {
        "normalized_effective_candidate_count": normalized_effective_count,
        "weighted_parameter_dispersion": dispersion,
    }


def make_task(instance: int) -> PDETaskV2:
    dimension = (12, 16, 20)[(instance - 21) % 3]
    rng = np.random.default_rng(5_608_000 + instance)
    n = 128 + 4 * dimension
    x = np.linspace(0.0, 1.0, n + 2)
    interior_x = x[1:-1]
    basis = np.stack(
        [
            np.sin(math.pi * (index + 1) * x) / (index + 1)
            for index in range(dimension)
        ],
        axis=1,
    )
    truth = rng.normal(
        0.0,
        0.55 / np.sqrt(np.arange(1, dimension + 1)),
    )
    truth[6:] *= 0.55
    baseline_source = (
        1.0
        + 0.30 * np.sin(2.0 * math.pi * interior_x)
        + 0.12 * np.cos(3.0 * math.pi * interior_x)
    )
    baseline_sensors = sensor_indices(n, 0.15, count=5)
    provisional = PDETaskV2(
        instance=instance,
        dimension=dimension,
        lower=-2.0 * np.ones(dimension),
        upper=2.0 * np.ones(dimension),
        truth=truth,
        basis=basis,
        interior_x=interior_x,
        baseline_source=baseline_source,
        baseline_sensor_indices=baseline_sensors,
        observations=np.empty(len(baseline_sensors)),
        baseline_noise_scale=1.0,
        design_noise_scale=1.0,
        f_zero=0.0,
    )
    truth_state = provisional.state(truth, baseline_source)
    clean = truth_state[baseline_sensors]
    state_scale = max(float(np.std(truth_state)), 1e-6)
    baseline_noise = max(0.01 * float(np.std(clean)), 1e-6)
    design_noise = max(0.02 * state_scale, 1e-6)
    observations = clean + rng.normal(0.0, baseline_noise, len(clean))
    provisional.observations = observations
    provisional.baseline_noise_scale = baseline_noise
    provisional.design_noise_scale = design_noise
    provisional.f_zero = provisional.objective(np.zeros(dimension))
    return provisional


def seed_for(instance: int, replicate: int, dimension: int) -> int:
    return int(BASE_SEED + 100_000 * instance + 1_000 * dimension + replicate)


def run_one(instance: int, replicate: int) -> dict[str, Any]:
    for key in THREAD_ENV:
        os.environ[key] = "1"
    task = make_task(instance)
    seed = seed_for(instance, replicate, task.dimension)
    budget = BUDGET_MULTIPLIER * task.dimension
    started = time.perf_counter()
    result = minimize_basingraph_v2(
        task.objective,
        task.lower,
        task.upper,
        max_evals=budget,
        seed=seed,
        options=BasinGraphOptions(),
    )
    active_ids = {int(node.node_id) for node in result.archive}
    graph_valid = all(
        int(edge.source_id) in active_ids and int(edge.target_id) in active_ids
        for edge in result.graph_edges
    )
    candidates = [
        {
            "theta": np.asarray(node.center, dtype=float).tolist(),
            "objective": float(node.f_center),
            "source": str(node.source),
        }
        for node in result.archive
    ]
    candidates.append(
        {
            "theta": np.asarray(result.xbest, dtype=float).tolist(),
            "objective": float(result.fbest),
            "source": "incumbent",
        }
    )
    return {
        "task_id": task.task_id,
        "instance": instance,
        "dimension": task.dimension,
        "replicate": replicate,
        "seed": seed,
        "budget": budget,
        "nfe": int(result.nfe),
        "phase_sum": int(sum(result.phase_evaluations.values())),
        "implementation_version": result.implementation_version,
        "options_hash": result.options_hash,
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "graph_referential_integrity": bool(graph_valid),
        "fbest": float(result.fbest),
        "wall_time_seconds": float(time.perf_counter() - started),
        "candidates": candidates,
    }


def deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: float(item["objective"]))
    output: list[dict[str, Any]] = []
    for item in ordered:
        theta = np.asarray(item["theta"], dtype=float)
        if any(
            np.linalg.norm(theta - np.asarray(other["theta"], dtype=float))
            <= 1e-8
            for other in output
        ):
            continue
        output.append(item)
    return output


def select_explanations(
    task: PDETaskV2,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    candidates = deduplicate(candidates)
    best = min(float(item["objective"]) for item in candidates)
    tolerance = max(5.0, 0.15 * max(task.f_zero - best, 0.0))
    eligible = [
        item for item in candidates
        if float(item["objective"]) <= best + tolerance
    ]
    if not eligible:
        eligible = [min(candidates, key=lambda item: float(item["objective"]))]
    diameter = float(np.linalg.norm(task.upper - task.lower))
    selected = [min(eligible, key=lambda item: float(item["objective"]))]
    remaining = [item for item in eligible if item is not selected[0]]
    while remaining and len(selected) < MAX_EXPLANATIONS:
        distances = [
            min(
                np.linalg.norm(
                    np.asarray(item["theta"], dtype=float)
                    - np.asarray(chosen["theta"], dtype=float)
                ) / diameter
                for chosen in selected
            )
            for item in remaining
        ]
        index = int(np.argmax(distances))
        if distances[index] < MIN_PARAMETER_SEPARATION:
            break
        selected.append(remaining.pop(index))
    pairwise = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairwise.append(
                float(
                    np.linalg.norm(
                        np.asarray(selected[i]["theta"], dtype=float)
                        - np.asarray(selected[j]["theta"], dtype=float)
                    ) / diameter
                )
            )
    return selected, {
        "candidate_pool": float(len(candidates)),
        "eligible_candidates": float(len(eligible)),
        "selected_explanations": float(len(selected)),
        "minimum_selected_distance": float(min(pairwise)) if pairwise else 0.0,
        "median_selected_distance": float(np.median(pairwise)) if pairwise else 0.0,
        "best_objective": best,
        "zero_objective": float(task.f_zero),
        "objective_tolerance": float(tolerance),
    }


def weighted_sensor_indices(n: int, mode: str, count: int = 8) -> np.ndarray:
    u = np.linspace(0.08, 0.92, count)
    if mode == "left":
        positions = u ** 1.65
    elif mode == "right":
        positions = 1.0 - (1.0 - u) ** 1.65
    elif mode == "centre":
        positions = 0.5 + 0.46 * np.sign(u - 0.5) * np.abs(2.0 * u - 1.0) ** 1.6
    else:
        raise KeyError(mode)
    return np.unique(np.clip(np.rint(positions * (n - 1)).astype(int), 1, n - 2))


def candidate_designs(task: PDETaskV2) -> list[dict[str, Any]]:
    source_designs = []
    for frequency in range(1, 9):
        for amplitude in (0.25, 0.50):
            source_designs.append(
                (
                    f"sin_f{frequency}_a{amplitude:.2f}",
                    task.baseline_source
                    + amplitude
                    * np.sin(2.0 * math.pi * frequency * task.interior_x),
                )
            )
    for centre in (0.15, 0.30, 0.50, 0.70, 0.85):
        for width in (0.06, 0.12):
            bump = np.exp(
                -0.5 * ((task.interior_x - centre) / width) ** 2
            )
            source_designs.append(
                (
                    f"gauss_c{centre:.2f}_w{width:.2f}",
                    task.baseline_source + 0.50 * bump,
                )
            )
    n = len(task.interior_x)
    sensor_designs = [
        ("uniform_o0.0", sensor_indices(n, 0.0, count=12)),
        ("uniform_o0.5", sensor_indices(n, 0.5, count=12)),
        ("weighted_left", weighted_sensor_indices(n, "left")),
        ("weighted_centre", weighted_sensor_indices(n, "centre")),
        ("weighted_right", weighted_sensor_indices(n, "right")),
    ]
    return [
        {
            "design_id": f"{source_id}__{sensor_id}",
            "source_id": source_id,
            "sensor_id": sensor_id,
            "source": source,
            "sensors": sensors,
        }
        for source_id, source in source_designs
        for sensor_id, sensors in sensor_designs
    ]


def d_optimal_score(predictions: np.ndarray, noise_scale: float) -> float:
    if len(predictions) < 2:
        return 0.0
    centred = predictions - np.mean(predictions, axis=0, keepdims=True)
    covariance = centred.T @ centred / max(1, len(predictions) - 1)
    matrix = np.eye(covariance.shape[0]) + covariance / max(
        noise_scale * noise_scale,
        1e-24,
    )
    sign, logdet = np.linalg.slogdet(matrix)
    return float(logdet) if sign > 0 else float("-inf")


def analyse_task(
    task: PDETaskV2,
    run_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = [item for record in run_records for item in record["candidates"]]
    explanations, selection = select_explanations(task, pool)
    theta = np.asarray([item["theta"] for item in explanations], dtype=float)
    objectives = np.asarray([item["objective"] for item in explanations], dtype=float)
    best = float(np.min(objectives))
    baseline_log_weight = -0.5 * (objectives - best) / BASELINE_WEIGHT_TEMPERATURE
    baseline_log_weight -= np.max(baseline_log_weight)
    baseline_weight = np.exp(baseline_log_weight)
    baseline_weight /= max(float(np.sum(baseline_weight)), 1e-300)
    diameter = float(np.linalg.norm(task.upper - task.lower))
    baseline_metrics = weighted_metrics(theta, baseline_weight, box_diameter=diameter)

    designs = candidate_designs(task)
    design_rows = []
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
        rng = np.random.default_rng(
            6_608_000 + 10_000 * task.instance + design_index
        )
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
        posterior = weighted_metrics(
            theta,
            posterior_weight,
            box_diameter=diameter,
        )
        maximum_index = int(np.argmax(posterior_weight))
        maximum_error = float(
            np.linalg.norm(theta[maximum_index] - task.truth) / diameter
        )
        design_rows.append(
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
    frame = pd.DataFrame(design_rows)
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
        "baseline_noise_scale": float(task.baseline_noise_scale),
        "design_noise_scale": float(task.design_noise_scale),
    }
    return summary, design_rows


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


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results_b21/pde_flagship/development_v2",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for key in THREAD_ENV:
        os.environ[key] = "1"
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected implementation version.")
    if BasinGraphOptions().stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected options hash.")
    if git_output("rev-parse", "HEAD:basingraph_v2/optimizer.py") != EXPECTED_OPTIMIZER_BLOB:
        raise RuntimeError("Frozen optimizer source changed.")

    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    jobs = [
        (instance, replicate)
        for instance in DEVELOPMENT_INSTANCES
        for replicate in range(1, SEEDS_PER_INSTANCE + 1)
    ]
    records = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_one, instance, replicate): (instance, replicate)
            for instance, replicate in jobs
        }
        for future in as_completed(futures):
            instance, replicate = futures[future]
            try:
                records.append(future.result())
                print("PDE_FLAGSHIP_V2_RUN_OK", instance, replicate, flush=True)
            except Exception as exc:
                failures.append(
                    {
                        "instance": instance,
                        "replicate": replicate,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=40),
                    }
                )
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"PDE flagship v2 run failures: {len(failures)}")

    run_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "candidates"}
        row["candidate_json"] = json.dumps(record["candidates"], separators=(",", ":"))
        run_rows.append(row)
    run_frame = pd.DataFrame(run_rows).sort_values(["instance", "replicate"])
    if len(run_frame) != len(jobs):
        raise RuntimeError("Unexpected PDE v2 run count.")
    if not run_frame["graph_referential_integrity"].all():
        raise RuntimeError("Returned graph integrity failed.")
    if not (run_frame["nfe"] == run_frame["budget"]).all():
        raise RuntimeError("Evaluation budget mismatch.")
    if not (run_frame["phase_sum"] == run_frame["budget"]).all():
        raise RuntimeError("Phase accounting mismatch.")
    run_frame.to_csv(output / "pde_v2_optimizer_runs.csv", index=False)

    task_summaries = []
    design_rows = []
    for instance in DEVELOPMENT_INSTANCES:
        task = make_task(instance)
        instance_records = [record for record in records if record["instance"] == instance]
        summary, designs = analyse_task(task, instance_records)
        task_summaries.append(summary)
        design_rows.extend(designs)
    task_frame = pd.DataFrame(task_summaries).sort_values("instance")
    design_frame = pd.DataFrame(design_rows).sort_values(["instance", "design_id"])
    task_frame.to_csv(output / "pde_v2_task_summary.csv", index=False)
    design_frame.to_csv(output / "pde_v2_design_results.csv", index=False)

    gate_conditions = {
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
    gate_passed = bool(all(gate_conditions.values()))
    decision = {
        "status": (
            "PDE_FLAGSHIP_V2_DEVELOPMENT_GATE_PASSED"
            if gate_passed
            else "PDE_FLAGSHIP_V2_DEVELOPMENT_GATE_NOT_PASSED"
        ),
        "gate_passed": gate_passed,
        "gate_conditions": gate_conditions,
        "source_commit": git_output("rev-parse", "HEAD"),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "optimizer_blob": EXPECTED_OPTIMIZER_BLOB,
        "development_instances": list(DEVELOPMENT_INSTANCES),
        "reserved_confirmatory_instances": list(CONFIRMATORY_INSTANCES),
        "runs": len(run_frame),
        "tasks": len(task_frame),
        "counts": {
            "tasks_with_four_explanations": int(
                (task_frame["selected_explanations"] >= 4).sum()
            ),
            "tasks_with_25pct_separation_gain": int(
                (task_frame["separation_gain_over_median"] >= 0.25).sum()
            ),
            "tasks_with_20pct_effective_count_reduction": int(
                (task_frame["effective_count_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "tasks_with_20pct_dispersion_reduction": int(
                (task_frame["dispersion_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "tasks_selected_no_worse_than_median": int(
                task_frame["chosen_ambiguity_no_worse_than_median_design"].sum()
            ),
        },
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    (output / "development_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps(gate_conditions, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
