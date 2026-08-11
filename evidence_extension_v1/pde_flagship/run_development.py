#!/usr/bin/env python3
"""Run the frozen elliptic-PDE non-identifiability development study."""

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


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
EXPECTED_OPTIMIZER_BLOB = "4433d2b92075dcd858529f7f13342631847def4c"
DEVELOPMENT_INSTANCES = tuple(range(1, 7))
CONFIRMATORY_INSTANCES = tuple(range(101, 107))
SEEDS_PER_INSTANCE = 8
BUDGET_MULTIPLIER = 300
NORMALIZED_OBJECTIVE_THRESHOLD = 0.10
MIN_PARAMETER_SEPARATION = 0.12
MAX_EXPLANATIONS = 12
BASE_SEED = 202608500
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass
class PDETask:
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
    noise_scale: float
    f_zero: float

    @property
    def task_id(self) -> str:
        return f"elliptic_pde_flagship_i{self.instance}_d{self.dimension}"

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
        scale = max(float(np.var(self.observations)), 1e-12)
        return float(
            np.mean((prediction - self.observations) ** 2) / scale
            + 1e-6 * np.mean(np.asarray(theta, dtype=float) ** 2)
        )


def thomas(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    n = len(diagonal)
    c = np.array(upper, dtype=float, copy=True)
    d = np.array(rhs, dtype=float, copy=True)
    b = np.array(diagonal, dtype=float, copy=True)
    for index in range(1, n):
        factor = lower[index - 1] / b[index - 1]
        b[index] -= factor * c[index - 1]
        d[index] -= factor * d[index - 1]
    solution = np.empty(n, dtype=float)
    solution[-1] = d[-1] / b[-1]
    for index in range(n - 2, -1, -1):
        solution[index] = (
            d[index] - c[index] * solution[index + 1]
        ) / b[index]
    return solution


def sensor_indices(n: int, offset: float, count: int = 12) -> np.ndarray:
    positions = (np.arange(count, dtype=float) + 1.0 + offset) / (
        count + 1.0 + 2.0 * offset
    )
    indices = np.clip(np.rint(positions * (n - 1)).astype(int), 1, n - 2)
    return np.unique(indices)


def make_task(instance: int) -> PDETask:
    dimension = (8, 10, 12)[(instance - 1) % 3]
    rng = np.random.default_rng(3_608_000 + instance)
    n = 96 + 4 * dimension
    x = np.linspace(0.0, 1.0, n + 2)
    interior_x = x[1:-1]
    basis = np.stack(
        [
            np.sin(math.pi * (index + 1) * x) / (index + 1)
            for index in range(dimension)
        ],
        axis=1,
    )
    truth = rng.normal(0.0, 0.45 / np.sqrt(np.arange(1, dimension + 1)))
    truth[4:] *= 0.55
    baseline_source = (
        1.0
        + 0.35 * np.sin(2.0 * math.pi * interior_x)
        + 0.15 * np.cos(3.0 * math.pi * interior_x)
    )
    provisional = PDETask(
        instance,
        dimension,
        -2.0 * np.ones(dimension),
        2.0 * np.ones(dimension),
        truth,
        basis,
        interior_x,
        baseline_source,
        sensor_indices(n, 0.0, count=8),
        np.empty(8),
        0.0,
        0.0,
    )
    truth_state = provisional.state(truth, baseline_source)
    observations_clean = truth_state[provisional.baseline_sensor_indices]
    noise_scale = 0.003 * max(float(np.std(observations_clean)), 1e-7)
    observations = observations_clean + rng.normal(
        0.0,
        noise_scale,
        len(observations_clean),
    )
    provisional.observations = observations
    provisional.noise_scale = noise_scale
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


def deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: float(item["objective"]))
    output: list[dict[str, Any]] = []
    for item in ordered:
        theta = np.asarray(item["theta"], dtype=float)
        if any(np.linalg.norm(theta - np.asarray(other["theta"])) <= 1e-9 for other in output):
            continue
        output.append(item)
    return output


def farthest_first_explanations(
    task: PDETask,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    candidates = deduplicate_candidates(candidates)
    best = min(float(item["objective"]) for item in candidates)
    denominator = max(task.f_zero - best, 1e-12)
    eligible = [
        item
        for item in candidates
        if (float(item["objective"]) - best) / denominator
        <= NORMALIZED_OBJECTIVE_THRESHOLD
    ]
    if not eligible:
        eligible = [min(candidates, key=lambda item: float(item["objective"]))]
    box_diameter = float(np.linalg.norm(task.upper - task.lower))
    selected = [min(eligible, key=lambda item: float(item["objective"]))]
    remaining = [item for item in eligible if item is not selected[0]]
    while remaining and len(selected) < MAX_EXPLANATIONS:
        distances = []
        for item in remaining:
            theta = np.asarray(item["theta"], dtype=float)
            minimum = min(
                np.linalg.norm(theta - np.asarray(chosen["theta"], dtype=float))
                / box_diameter
                for chosen in selected
            )
            distances.append(minimum)
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
                        np.asarray(selected[i]["theta"])
                        - np.asarray(selected[j]["theta"])
                    )
                    / box_diameter
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
    }


def candidate_designs(task: PDETask) -> list[dict[str, Any]]:
    designs = []
    n = len(task.interior_x)
    for frequency in range(1, 9):
        for amplitude in (0.25, 0.50):
            for offset in (0.0, 0.5):
                source = task.baseline_source + amplitude * np.sin(
                    2.0 * math.pi * frequency * task.interior_x
                )
                designs.append(
                    {
                        "design_id": f"f{frequency}_a{amplitude:.2f}_o{offset:.1f}",
                        "frequency": frequency,
                        "amplitude": amplitude,
                        "offset": offset,
                        "source": source,
                        "sensors": sensor_indices(n, offset, count=12),
                    }
                )
    return designs


def pairwise_prediction_score(predictions: np.ndarray, noise_scale: float) -> float:
    if len(predictions) < 2:
        return 0.0
    values = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            values.append(
                float(
                    np.sqrt(np.mean((predictions[i] - predictions[j]) ** 2))
                    / max(noise_scale, 1e-12)
                )
            )
    return float(np.quantile(values, 0.25))


def posterior_metrics(
    candidates: np.ndarray,
    predictions: np.ndarray,
    observation: np.ndarray,
    noise_scale: float,
    truth: np.ndarray,
    box_diameter: float,
) -> dict[str, float]:
    residual_mse = np.mean((predictions - observation[None, :]) ** 2, axis=1)
    log_weight = -0.5 * residual_mse / max(noise_scale * noise_scale, 1e-24)
    log_weight -= np.max(log_weight)
    weights = np.exp(log_weight)
    weights /= max(float(np.sum(weights)), 1e-300)
    mean = np.sum(weights[:, None] * candidates, axis=0)
    dispersion = float(
        np.sum(weights * np.sum((candidates - mean[None, :]) ** 2, axis=1))
        / max(box_diameter * box_diameter, 1e-300)
    )
    effective_count = float(1.0 / max(np.sum(weights * weights), 1e-300))
    best_index = int(np.argmax(weights))
    best_error = float(
        np.linalg.norm(candidates[best_index] - truth) / max(box_diameter, 1e-300)
    )
    return {
        "weighted_parameter_dispersion": dispersion,
        "effective_candidate_count": effective_count,
        "maximum_weight_candidate_error": best_error,
    }


def analyze_task(task: PDETask, run_records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pool = [item for record in run_records for item in record["candidates"]]
    explanations, selection = farthest_first_explanations(task, pool)
    theta = np.asarray([item["theta"] for item in explanations], dtype=float)
    designs = candidate_designs(task)
    design_rows = []
    truth_noise_rng = np.random.default_rng(4_608_000 + task.instance)
    box_diameter = float(np.linalg.norm(task.upper - task.lower))
    for design in designs:
        predictions = np.asarray(
            [
                task.state(candidate, design["source"])[design["sensors"]]
                for candidate in theta
            ],
            dtype=float,
        )
        score = pairwise_prediction_score(predictions, task.noise_scale)
        truth_prediction = task.state(task.truth, design["source"])[design["sensors"]]
        observation = truth_prediction + truth_noise_rng.normal(
            0.0,
            task.noise_scale,
            len(truth_prediction),
        )
        posterior = posterior_metrics(
            theta,
            predictions,
            observation,
            task.noise_scale,
            task.truth,
            box_diameter,
        )
        design_rows.append(
            {
                "task_id": task.task_id,
                "instance": task.instance,
                "dimension": task.dimension,
                "design_id": design["design_id"],
                "frequency": design["frequency"],
                "amplitude": design["amplitude"],
                "offset": design["offset"],
                "selection_score": score,
                **posterior,
            }
        )
    design_frame = pd.DataFrame(design_rows)
    chosen = design_frame.sort_values(
        ["selection_score", "design_id"],
        ascending=[False, True],
    ).iloc[0]
    median_score = float(design_frame["selection_score"].median())
    median_dispersion = float(
        design_frame["weighted_parameter_dispersion"].median()
    )
    summary = {
        "task_id": task.task_id,
        "instance": task.instance,
        "dimension": task.dimension,
        **selection,
        "chosen_design_id": str(chosen["design_id"]),
        "chosen_selection_score": float(chosen["selection_score"]),
        "median_design_score": median_score,
        "separation_gain_over_median": float(
            (chosen["selection_score"] - median_score) / max(median_score, 1e-12)
        ),
        "chosen_weighted_parameter_dispersion": float(
            chosen["weighted_parameter_dispersion"]
        ),
        "median_design_weighted_parameter_dispersion": median_dispersion,
        "dispersion_reduction_vs_median": float(
            (median_dispersion - chosen["weighted_parameter_dispersion"])
            / max(median_dispersion, 1e-12)
        ),
        "chosen_effective_candidate_count": float(
            chosen["effective_candidate_count"]
        ),
        "chosen_maximum_weight_candidate_error": float(
            chosen["maximum_weight_candidate_error"]
        ),
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
        default="results_b21/pde_flagship/development",
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
                record = future.result()
                records.append(record)
                print("PDE_FLAGSHIP_RUN_OK", instance, replicate, flush=True)
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
        raise RuntimeError(f"PDE flagship run failures: {len(failures)}")

    run_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "candidates"}
        row["candidate_json"] = json.dumps(record["candidates"], separators=(",", ":"))
        run_rows.append(row)
    run_frame = pd.DataFrame(run_rows).sort_values(["instance", "replicate"])
    if len(run_frame) != len(jobs):
        raise RuntimeError("Unexpected PDE run count.")
    if not run_frame["graph_referential_integrity"].all():
        raise RuntimeError("Returned graph integrity failed.")
    if not (run_frame["nfe"] == run_frame["budget"]).all():
        raise RuntimeError("Evaluation budget mismatch.")
    if not (run_frame["phase_sum"] == run_frame["budget"]).all():
        raise RuntimeError("Phase accounting mismatch.")
    run_frame.to_csv(output / "pde_optimizer_runs.csv", index=False)

    task_summaries = []
    design_rows = []
    for instance in DEVELOPMENT_INSTANCES:
        task = make_task(instance)
        instance_records = [record for record in records if record["instance"] == instance]
        summary, designs = analyze_task(task, instance_records)
        task_summaries.append(summary)
        design_rows.extend(designs)
    task_frame = pd.DataFrame(task_summaries).sort_values("instance")
    design_frame = pd.DataFrame(design_rows).sort_values(["instance", "design_id"])
    task_frame.to_csv(output / "pde_task_summary.csv", index=False)
    design_frame.to_csv(output / "pde_design_results.csv", index=False)

    gate_conditions = {
        "four_tasks_with_three_explanations": int(
            (task_frame["selected_explanations"] >= 3).sum()
        ) >= 4,
        "four_tasks_with_25pct_separation_gain": int(
            (task_frame["separation_gain_over_median"] >= 0.25).sum()
        ) >= 4,
        "four_tasks_with_20pct_dispersion_reduction": int(
            (task_frame["dispersion_reduction_vs_median"] >= 0.20).sum()
        ) >= 4,
        "truth_not_used_for_design_selection": True,
        "integrity_checks_pass": True,
    }
    gate_passed = bool(all(gate_conditions.values()))
    decision = {
        "status": (
            "PDE_FLAGSHIP_DEVELOPMENT_GATE_PASSED"
            if gate_passed
            else "PDE_FLAGSHIP_DEVELOPMENT_GATE_NOT_PASSED"
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
        "tasks_with_three_explanations": int(
            (task_frame["selected_explanations"] >= 3).sum()
        ),
        "tasks_with_25pct_separation_gain": int(
            (task_frame["separation_gain_over_median"] >= 0.25).sum()
        ),
        "tasks_with_20pct_dispersion_reduction": int(
            (task_frame["dispersion_reduction_vs_median"] >= 0.20).sum()
        ),
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
