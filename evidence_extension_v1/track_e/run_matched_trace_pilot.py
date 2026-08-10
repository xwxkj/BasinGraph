#!/usr/bin/env python3
"""Run the registered Track E matched-trace operational-state pilot."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import platform
import sys
import time
from typing import Callable, Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.archive import BasinArchive  # noqa: E402
from basingraph_v2.evaluation import (  # noqa: E402
    BudgetExhausted,
    EvaluationLedger,
)
from basingraph_v2.graph import BasinTransitionGraph  # noqa: E402
from basingraph_v2.optimizer import (  # noqa: E402
    BasinGraphOptions,
    _local_polish,
    minimize_basingraph_v2,
)
from basingraph_v2.types import BasinNode, TransitionEdge  # noqa: E402


FAMILIES = (
    "rotated_rastrigin",
    "rotated_ackley",
    "rotated_griewank",
    "gallagher_mixture",
    "double_funnel",
    "lunacek_birstrigin",
)
DIMENSIONS = (10, 20)
INSTANCES = (1, 2, 3, 4)
ARMS = (
    "FullState",
    "ArchiveOnly",
    "EdgeShuffled",
    "TraceOnly",
    "ColdRestart",
)
TARGET_RATIOS = (0.3, 0.1, 0.03, 0.01, 0.003)
CHECKPOINT_MULTIPLIERS = (1, 3, 10, 30, 100)
PREFIX_BUDGET_MULTIPLIER = 30
CONTINUATION_BUDGET_MULTIPLIER = 100
BASE_PREFIX_SEED = 202608100
BASE_CONTINUATION_SEED = 202608200
BASE_CONTROL_SEED = 202608300
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class TaskSpec:
    family: str
    dimension: int
    instance: int
    lower: np.ndarray
    upper: np.ndarray
    optimum: float
    objective: Callable[[np.ndarray], float]

    @property
    def task_id(self) -> str:
        return f"{self.family}_d{self.dimension}_i{self.instance}"


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def orthogonal_matrix(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def make_task(family: str, dimension: int, instance: int) -> TaskSpec:
    family_index = FAMILIES.index(family) + 1
    rng = np.random.default_rng(
        991_000 + 10_000 * family_index + 100 * dimension + instance
    )
    lower = -5.0 * np.ones(dimension)
    upper = 5.0 * np.ones(dimension)
    rotation = orthogonal_matrix(rng, dimension)

    if family in {
        "rotated_rastrigin",
        "rotated_ackley",
        "rotated_griewank",
    }:
        shift = rng.uniform(-1.5, 1.5, size=dimension)

        def transformed(point: np.ndarray) -> np.ndarray:
            return rotation @ (np.asarray(point, dtype=float) - shift)

        if family == "rotated_rastrigin":
            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                value = 10.0 * dimension + np.sum(
                    z * z - 10.0 * np.cos(2.0 * np.pi * z)
                )
                return float(max(0.0, value))
        elif family == "rotated_ackley":
            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                term1 = -20.0 * np.exp(
                    -0.2 * np.sqrt(np.mean(z * z))
                )
                term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * z)))
                value = term1 + term2 + 20.0 + math.e
                return float(max(0.0, value))
        else:
            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                indices = np.arange(1, dimension + 1, dtype=float)
                value = (
                    np.sum(z * z) / 4000.0
                    - np.prod(np.cos(z / np.sqrt(indices)))
                    + 1.0
                )
                return float(max(0.0, value))

    elif family == "gallagher_mixture":
        basin_count = 14
        centers = rng.uniform(-3.2, 3.2, size=(basin_count, dimension))
        centers[0] = rng.uniform(-1.5, 1.5, size=dimension)
        offsets = np.linspace(0.0, 1.2, basin_count)
        rng.shuffle(offsets[1:])
        scales = np.exp(rng.uniform(np.log(0.25), np.log(1.6), size=(basin_count, dimension)))
        scales[0] = rng.uniform(0.20, 0.45, size=dimension)

        def objective(point: np.ndarray) -> float:
            z = np.asarray(point, dtype=float)
            normalized = (z[None, :] - centers) / scales
            basin_values = offsets + np.mean(normalized * normalized, axis=1)
            return float(max(0.0, np.min(basin_values)))

    elif family == "double_funnel":
        global_center = rng.uniform(-2.0, 2.0, size=dimension)
        direction = rng.normal(size=dimension)
        direction /= max(np.linalg.norm(direction), 1e-12)
        local_center = np.clip(global_center + 4.5 * direction, -3.8, 3.8)
        global_rotation = rotation
        local_rotation = orthogonal_matrix(rng, dimension)

        def objective(point: np.ndarray) -> float:
            z = np.asarray(point, dtype=float)
            zg = global_rotation @ (z - global_center)
            zl = local_rotation @ (z - local_center)
            global_value = (
                np.mean(zg * zg)
                + 0.035 * np.mean(1.0 - np.cos(2.0 * np.pi * zg))
            )
            local_value = (
                0.18
                + 0.55 * np.mean(zl * zl)
                + 0.020 * np.mean(1.0 - np.cos(2.0 * np.pi * zl))
            )
            return float(max(0.0, min(global_value, local_value)))

    elif family == "lunacek_birstrigin":
        shift = rng.uniform(-0.4, 0.4, size=dimension)
        mu_global = 0.70 * np.ones(dimension)
        mu_local = -0.70 * np.ones(dimension)

        def objective(point: np.ndarray) -> float:
            y = rotation @ (np.asarray(point, dtype=float) - shift)
            global_delta = y - mu_global
            local_delta = y - mu_local
            global_basin = np.mean(global_delta * global_delta)
            local_basin = 0.22 + 0.70 * np.mean(local_delta * local_delta)
            ripples = 4.0 * np.mean(
                1.0 - np.cos(2.0 * np.pi * global_delta)
            )
            return float(max(0.0, min(global_basin, local_basin) + ripples))
    else:
        raise KeyError(f"Unknown task family: {family}")

    return TaskSpec(
        family=family,
        dimension=dimension,
        instance=instance,
        lower=lower,
        upper=upper,
        optimum=0.0,
        objective=objective,
    )


def clone_archive(nodes: list[BasinNode], options: BasinGraphOptions) -> BasinArchive:
    archive = BasinArchive(
        merge_radius_factor=options.merge_radius_factor,
        max_nodes=options.archive_max_size,
    )
    archive.nodes = copy.deepcopy(nodes)
    maximum = max((node.node_id for node in archive.nodes), default=0)
    archive._id_counter = itertools.count(maximum + 1)
    return archive


def clone_graph(edges: list[TransitionEdge]) -> BasinTransitionGraph:
    graph = BasinTransitionGraph()
    copied = copy.deepcopy(edges)
    graph.edges = {
        (int(edge.source_id), int(edge.target_id)): edge
        for edge in copied
    }
    return graph


def shuffled_graph(
    edges: list[TransitionEdge],
    active_ids: list[int],
    seed: int,
) -> BasinTransitionGraph:
    if len(active_ids) < 2 or not edges:
        return BasinTransitionGraph()
    rng = np.random.default_rng(seed)
    permuted = np.asarray(active_ids, dtype=int).copy()
    # Avoid the identity permutation when possible.
    for _ in range(20):
        rng.shuffle(permuted)
        if not np.array_equal(permuted, np.asarray(active_ids, dtype=int)):
            break
    mapping = {
        int(source): int(target)
        for source, target in zip(active_ids, permuted.tolist())
    }
    output = BasinTransitionGraph()
    for edge in copy.deepcopy(edges):
        edge.source_id = mapping[int(edge.source_id)]
        edge.target_id = mapping[int(edge.target_id)]
        output.edges[(edge.source_id, edge.target_id)] = edge
    return output


def trace_only_archive(
    xbest: np.ndarray,
    fbest: float,
    lower: np.ndarray,
    upper: np.ndarray,
    options: BasinGraphOptions,
) -> BasinArchive:
    archive = BasinArchive(
        merge_radius_factor=options.merge_radius_factor,
        max_nodes=options.archive_max_size,
    )
    radius = 0.02 * float(np.linalg.norm(upper - lower) + 1e-300)
    archive.nodes = [
        BasinNode(
            node_id=1,
            center=np.asarray(xbest, dtype=float).copy(),
            f_center=float(fbest),
            radius=radius,
            curvature_proxy=float(1.0 / (1.0 + radius)),
            created_nfe=0,
            last_updated_nfe=0,
            novelty=1.0,
            source="matched_trace_incumbent",
        )
    ]
    archive._id_counter = itertools.count(2)
    return archive


def state_for_arm(
    arm: str,
    prefix_result,
    task: TaskSpec,
    options: BasinGraphOptions,
    control_seed: int,
) -> tuple[BasinArchive, BasinTransitionGraph]:
    if arm == "FullState":
        return (
            clone_archive(prefix_result.archive, options),
            clone_graph(prefix_result.graph_edges),
        )
    if arm == "ArchiveOnly":
        return clone_archive(prefix_result.archive, options), BasinTransitionGraph()
    if arm == "EdgeShuffled":
        archive = clone_archive(prefix_result.archive, options)
        active_ids = [int(node.node_id) for node in archive.nodes]
        return archive, shuffled_graph(
            prefix_result.graph_edges,
            active_ids,
            control_seed,
        )
    if arm == "TraceOnly":
        return (
            trace_only_archive(
                prefix_result.xbest,
                prefix_result.fbest,
                task.lower,
                task.upper,
                options,
            ),
            BasinTransitionGraph(),
        )
    if arm == "ColdRestart":
        return (
            BasinArchive(
                merge_radius_factor=options.merge_radius_factor,
                max_nodes=options.archive_max_size,
            ),
            BasinTransitionGraph(),
        )
    raise KeyError(arm)


def continuation_history_value(
    history: list[tuple[int, float]],
    checkpoint: int,
    initial_value: float,
) -> float:
    value = float(initial_value)
    for nfe, candidate in history:
        if int(nfe) > int(checkpoint):
            break
        value = min(value, float(candidate))
    return value


def target_evaluation(
    history: list[tuple[int, float]],
    threshold: float,
) -> int | None:
    for nfe, value in history:
        if float(value) <= float(threshold):
            return int(nfe)
    return None


def run_continuation(
    task: TaskSpec,
    prefix_result,
    arm: str,
    continuation_seed: int,
    control_seed: int,
) -> dict[str, Any]:
    options = BasinGraphOptions()
    budget = CONTINUATION_BUDGET_MULTIPLIER * task.dimension
    ledger = EvaluationLedger(
        task.objective,
        task.lower,
        task.upper,
        budget,
    )
    ledger.xbest = np.asarray(prefix_result.xbest, dtype=float).copy()
    ledger.fbest = float(prefix_result.fbest)
    archive, graph = state_for_arm(
        arm,
        prefix_result,
        task,
        options,
        control_seed,
    )
    rng = np.random.default_rng(continuation_seed)
    domain_radius = float(np.linalg.norm(task.upper - task.lower) + 1e-300)

    def add_node(
        point: np.ndarray,
        value: float,
        source: str,
        radius_fraction: float = 0.02,
    ) -> BasinNode:
        update = archive.add_or_merge(
            point,
            value,
            radius=radius_fraction * domain_radius,
            curvature_proxy=float(
                1.0 / (1.0 + radius_fraction * domain_radius)
            ),
            nfe=ledger.nfe,
            source=source,
            lb=task.lower,
            ub=task.upper,
        )
        if update.removed_node_ids:
            graph.remove_nodes(update.removed_node_ids)
        return update.node

    stall = 0
    with ledger.phase("matched_trace_continuation", None):
        while ledger.remaining > 0:
            previous_best = float(ledger.fbest)
            nodes = archive.sorted_nodes()

            if nodes and options.enable_graph_guidance and rng.random() < 0.80:
                scores = graph.guidance_scores(
                    nodes,
                    quality_weight=options.graph_quality_weight,
                    novelty_weight=options.graph_novelty_weight,
                    accessibility_weight=options.graph_accessibility_weight,
                )
                score_array = np.asarray(
                    [max(scores.get(node.node_id, 0.0), 0.0) for node in nodes],
                    dtype=float,
                )
                if score_array.sum() <= 0:
                    probabilities = np.full(len(nodes), 1.0 / len(nodes))
                else:
                    probabilities = score_array / score_array.sum()
                source_node = nodes[int(rng.choice(len(nodes), p=probabilities))]
                decay = max(
                    0.05,
                    (ledger.remaining / max(1, budget)) ** 0.5,
                )
                step = rng.standard_t(
                    df=2.0,
                    size=task.dimension,
                ) * (0.20 * decay * (task.upper - task.lower))
                point = ledger.project(source_node.center + step)
            elif nodes and rng.random() < 0.70:
                source_node = nodes[int(rng.integers(min(8, len(nodes))))]
                point = ledger.project(
                    source_node.center
                    + rng.standard_t(df=2.0, size=task.dimension)
                    * (0.10 * (task.upper - task.lower))
                )
            else:
                source_node = None
                point = ledger.project(
                    task.lower
                    + rng.random(task.dimension) * (task.upper - task.lower)
                )

            try:
                value = ledger.evaluate(point)
            except BudgetExhausted:
                break

            target = add_node(point, value, "matched_trace_continuation")
            if source_node is not None:
                graph.add_or_update(
                    source_node.node_id,
                    target.node_id,
                    evaluations=1,
                    improvement=source_node.f_center - target.f_center,
                    barrier_proxy=max(0.0, value - source_node.f_center),
                    nfe=ledger.nfe,
                    source_mode="matched_trace_continuation",
                )

            if ledger.fbest < previous_best - 1e-14:
                stall = 0
            else:
                stall += 1

            if (
                stall >= max(10, 2 * ledger.dimension)
                and ledger.remaining > max(10, 3 * ledger.dimension)
            ):
                x, f, used = _local_polish(
                    ledger,
                    ledger.xbest,
                    options=options,
                )
                target = add_node(x, f, "matched_trace_stall_polish")
                if source_node is not None:
                    graph.add_or_update(
                        source_node.node_id,
                        target.node_id,
                        evaluations=used,
                        improvement=source_node.f_center - target.f_center,
                        barrier_proxy=max(0.0, f - source_node.f_center),
                        nfe=ledger.nfe,
                        source_mode="matched_trace_stall_polish",
                    )
                stall = 0

    active_ids = {int(node.node_id) for node in archive.nodes}
    graph.remove_nodes(
        [
            node_id
            for edge in graph.to_edges()
            for node_id in (edge.source_id, edge.target_id)
            if int(node_id) not in active_ids
        ]
    )

    prefix_gap = max(
        float(prefix_result.fbest) - task.optimum,
        np.finfo(float).tiny,
    )
    final_gap = max(float(ledger.fbest) - task.optimum, np.finfo(float).tiny)
    target_thresholds = {
        str(ratio): task.optimum + ratio * prefix_gap
        for ratio in TARGET_RATIOS
    }
    target_evaluations = {
        ratio: target_evaluation(ledger.history, threshold)
        for ratio, threshold in target_thresholds.items()
    }
    final_hits = {
        ratio: evaluation is not None
        for ratio, evaluation in target_evaluations.items()
    }
    checkpoint_fractions: dict[str, float] = {}
    checkpoint_gaps: dict[str, float] = {}
    for multiplier in CHECKPOINT_MULTIPLIERS:
        checkpoint = min(multiplier * task.dimension, budget)
        value = continuation_history_value(
            ledger.history,
            checkpoint,
            prefix_result.fbest,
        )
        checkpoint_gaps[str(multiplier)] = max(
            float(value) - task.optimum,
            np.finfo(float).tiny,
        )
        checkpoint_fractions[str(multiplier)] = float(
            np.mean([
                evaluation is not None and evaluation <= checkpoint
                for evaluation in target_evaluations.values()
            ])
        )

    graph_valid = all(
        int(edge.source_id) in active_ids and int(edge.target_id) in active_ids
        for edge in graph.to_edges()
    )
    return {
        "arm": arm,
        "continuation_seed": int(continuation_seed),
        "control_seed": int(control_seed),
        "continuation_budget": int(budget),
        "continuation_nfe": int(ledger.nfe),
        "prefix_fbest": float(prefix_result.fbest),
        "final_fbest": float(ledger.fbest),
        "prefix_gap": float(prefix_gap),
        "final_gap": float(final_gap),
        "log10_gap_reduction": float(
            math.log10(prefix_gap) - math.log10(final_gap)
        ),
        "final_target_fraction": float(np.mean(list(final_hits.values()))),
        "target_evaluations_json": json.dumps(
            target_evaluations,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "checkpoint_target_fraction_json": json.dumps(
            checkpoint_fractions,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "checkpoint_gap_json": json.dumps(
            checkpoint_gaps,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "continuation_archive_nodes": int(len(archive.nodes)),
        "continuation_graph_edges": int(len(graph.to_edges())),
        "history_monotone": bool(all(
            later <= earlier + 1e-14
            for earlier, later in zip(
                [value for _, value in ledger.history[:-1]],
                [value for _, value in ledger.history[1:]],
            )
        )),
        "graph_referential_integrity": bool(graph_valid),
        "continuation_history_hash": stable_digest(ledger.history),
    }


def block_seed(base: int, family: str, dimension: int, instance: int) -> int:
    return int(
        base
        + 100_000 * (FAMILIES.index(family) + 1)
        + 1_000 * dimension
        + instance
    )


def run_block(family: str, dimension: int, instance: int) -> list[dict[str, Any]]:
    for key in THREAD_ENV:
        os.environ[key] = "1"
    task = make_task(family, dimension, instance)
    options = BasinGraphOptions()
    prefix_seed = block_seed(BASE_PREFIX_SEED, family, dimension, instance)
    continuation_seed = block_seed(
        BASE_CONTINUATION_SEED,
        family,
        dimension,
        instance,
    )
    control_seed = block_seed(BASE_CONTROL_SEED, family, dimension, instance)
    prefix_budget = PREFIX_BUDGET_MULTIPLIER * dimension
    prefix_started = time.perf_counter()
    prefix = minimize_basingraph_v2(
        task.objective,
        task.lower,
        task.upper,
        max_evals=prefix_budget,
        seed=prefix_seed,
        options=options,
    )
    prefix_time = time.perf_counter() - prefix_started
    if prefix.implementation_version != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected BasinGraph implementation version.")
    if prefix.options_hash != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected BasinGraph options hash.")
    if prefix.nfe != prefix_budget:
        raise RuntimeError("Prefix evaluation budget mismatch.")
    prefix_payload = prefix.to_jsonable()
    prefix_history_hash = stable_digest(prefix_payload["history"])
    prefix_state_hash = stable_digest({
        "archive": prefix_payload["archive"],
        "graph_edges": prefix_payload["graph_edges"],
        "xbest": prefix_payload["xbest"],
        "fbest": prefix_payload["fbest"],
    })
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        started = time.perf_counter()
        result = run_continuation(
            task,
            prefix,
            arm,
            continuation_seed,
            control_seed,
        )
        result.update({
            "task_id": task.task_id,
            "family": family,
            "dimension": int(dimension),
            "instance": int(instance),
            "known_optimum": float(task.optimum),
            "prefix_seed": int(prefix_seed),
            "prefix_budget": int(prefix_budget),
            "prefix_nfe": int(prefix.nfe),
            "prefix_archive_nodes": int(len(prefix.archive)),
            "prefix_graph_edges": int(len(prefix.graph_edges)),
            "prefix_history_hash": prefix_history_hash,
            "prefix_state_hash": prefix_state_hash,
            "prefix_wall_time_seconds": float(prefix_time),
            "continuation_wall_time_seconds": float(
                time.perf_counter() - started
            ),
            "implementation_version": prefix.implementation_version,
            "options_hash": prefix.options_hash,
        })
        rows.append(result)
    if len({row["prefix_history_hash"] for row in rows}) != 1:
        raise RuntimeError("Matched-prefix trace hash mismatch within block.")
    return rows


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[~np.isclose(differences, 0.0, atol=1e-14, rtol=1e-12)]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(np.sum(ranks[nonzero > 0]))
    negative = float(np.sum(ranks[nonzero < 0]))
    denominator = positive + negative
    return 0.0 if denominator <= 0 else (positive - negative) / denominator


def bootstrap_interval(
    differences: np.ndarray,
    seed: int = 20260810,
    resamples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(
        0,
        len(differences),
        size=(resamples, len(differences)),
    )
    means = np.mean(differences[sample_indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparison(frame: pd.DataFrame, baseline: str) -> dict[str, Any]:
    full = frame[frame.arm == "FullState"][[
        "task_id", "final_target_fraction", "log10_gap_reduction"
    ]].rename(columns={
        "final_target_fraction": "full_target_fraction",
        "log10_gap_reduction": "full_log_reduction",
    })
    other = frame[frame.arm == baseline][[
        "task_id", "final_target_fraction", "log10_gap_reduction"
    ]].rename(columns={
        "final_target_fraction": "baseline_target_fraction",
        "log10_gap_reduction": "baseline_log_reduction",
    })
    paired = full.merge(other, on="task_id", validate="one_to_one")
    differences = (
        paired.full_target_fraction.to_numpy(float)
        - paired.baseline_target_fraction.to_numpy(float)
    )
    nonzero = differences[
        ~np.isclose(differences, 0.0, atol=1e-14, rtol=1e-12)
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
    low, high = bootstrap_interval(differences)
    return {
        "baseline": baseline,
        "paired_blocks": int(len(paired)),
        "mean_target_fraction_difference": float(np.mean(differences)),
        "median_target_fraction_difference": float(np.median(differences)),
        "full_better_blocks": int(np.sum(differences > 1e-14)),
        "baseline_better_blocks": int(np.sum(differences < -1e-14)),
        "ties": int(np.sum(np.isclose(
            differences,
            0.0,
            atol=1e-14,
            rtol=1e-12,
        ))),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "rank_biserial": float(rank_biserial(differences)),
        "bootstrap_95_low": low,
        "bootstrap_95_high": high,
        "mean_log10_reduction_difference": float(np.mean(
            paired.full_log_reduction.to_numpy(float)
            - paired.baseline_log_reduction.to_numpy(float)
        )),
    }


def write_manifest(root: Path) -> None:
    path = root / "MANIFEST_SHA256.csv"
    rows = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file() or candidate == path:
            continue
        rows.append({
            "relative_path": candidate.relative_to(root).as_posix(),
            "sha256": sha256_file(candidate),
            "size_bytes": candidate.stat().st_size,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    for key in THREAD_ENV:
        os.environ[key] = "1"

    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    blocks = [
        (family, dimension, instance)
        for family in FAMILIES
        for dimension in DIMENSIONS
        for instance in INSTANCES
    ]
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(run_block, *block): block
            for block in blocks
        }
        for future in as_completed(futures):
            block = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(f"TRACK_E_BLOCK_OK {block}", flush=True)
            except Exception as exc:
                failures.append({
                    "family": block[0],
                    "dimension": str(block[1]),
                    "instance": str(block[2]),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"TRACK_E_BLOCK_FAILED {block}: {exc}", flush=True)

    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Track E block failures: {len(failures)}")

    frame = pd.DataFrame(all_rows).sort_values(
        ["family", "dimension", "instance", "arm"]
    )
    expected_rows = len(blocks) * len(ARMS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Unexpected row count: {len(frame)} != {expected_rows}")
    if frame.duplicated(["task_id", "arm"]).any():
        raise RuntimeError("Duplicate task-arm rows.")
    if not (frame.continuation_nfe == frame.continuation_budget).all():
        raise RuntimeError("Continuation budget accounting failed.")
    if not frame.history_monotone.all():
        raise RuntimeError("Non-monotone continuation history.")
    if not frame.graph_referential_integrity.all():
        raise RuntimeError("Graph referential integrity failed.")
    matched_counts = frame.groupby("task_id").prefix_history_hash.nunique()
    if not (matched_counts == 1).all():
        raise RuntimeError("Matched-prefix history hashes differ within blocks.")
    if set(frame.implementation_version) != {EXPECTED_IMPLEMENTATION}:
        raise RuntimeError("Implementation identity mismatch.")
    if set(frame.options_hash) != {EXPECTED_OPTIONS_HASH}:
        raise RuntimeError("Options identity mismatch.")

    frame.to_csv(output / "matched_trace_raw_results.csv", index=False)

    arm_summary = (
        frame.groupby("arm", as_index=False)
        .agg(
            mean_final_target_fraction=("final_target_fraction", "mean"),
            median_final_target_fraction=("final_target_fraction", "median"),
            mean_log10_gap_reduction=("log10_gap_reduction", "mean"),
            median_final_gap=("final_gap", "median"),
            runs=("task_id", "size"),
        )
        .sort_values("mean_final_target_fraction", ascending=False)
    )
    arm_summary.to_csv(output / "arm_summary.csv", index=False)

    family_summary = (
        frame.groupby(["family", "arm"], as_index=False)
        .agg(
            mean_final_target_fraction=("final_target_fraction", "mean"),
            mean_log10_gap_reduction=("log10_gap_reduction", "mean"),
            runs=("task_id", "size"),
        )
    )
    family_summary.to_csv(output / "family_arm_summary.csv", index=False)

    comparisons = [
        paired_comparison(frame, baseline)
        for baseline in ARMS
        if baseline != "FullState"
    ]
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(output / "paired_comparisons.csv", index=False)

    primary = next(
        item for item in comparisons if item["baseline"] == "TraceOnly"
    )
    family_pivot = family_summary.pivot(
        index="family",
        columns="arm",
        values="mean_final_target_fraction",
    )
    family_differences = (
        family_pivot["FullState"] - family_pivot["TraceOnly"]
    )
    positive_families = int(np.sum(family_differences > 0.0))
    success_conditions = {
        "mean_difference_at_least_0_03": bool(
            primary["mean_target_fraction_difference"] >= 0.03
        ),
        "wilcoxon_p_below_0_05": bool(primary["p_value"] < 0.05),
        "bootstrap_lower_bound_positive": bool(
            primary["bootstrap_95_low"] > 0.0
        ),
        "positive_in_at_least_four_families": bool(
            positive_families >= 4
        ),
        "integrity_checks_passed": True,
    }
    pilot_positive = bool(all(success_conditions.values()))

    decision = {
        "status": (
            "TRACK_E_PILOT_POSITIVE"
            if pilot_positive
            else "TRACK_E_PILOT_NOT_POSITIVE"
        ),
        "engineering_only": True,
        "confirmatory_evidence": False,
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "blocks": len(blocks),
        "continuation_runs": expected_rows,
        "families": list(FAMILIES),
        "dimensions": list(DIMENSIONS),
        "instances": list(INSTANCES),
        "prefix_budget_multiplier": PREFIX_BUDGET_MULTIPLIER,
        "continuation_budget_multiplier": CONTINUATION_BUDGET_MULTIPLIER,
        "primary_comparison": primary,
        "positive_family_count": positive_families,
        "family_differences": {
            str(key): float(value)
            for key, value in family_differences.items()
        },
        "success_conditions": success_conditions,
        "wall_time_seconds": float(time.perf_counter() - started),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "next_action": (
            "freeze_disjoint_confirmatory_matched_trace_protocol"
            if pilot_positive
            else "retain_auditability_only_claim"
        ),
    }
    (output / "pilot_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )

    # A compact checkpoint table for inspection.
    checkpoint_rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        values = json.loads(row.checkpoint_target_fraction_json)
        for multiplier, fraction in values.items():
            checkpoint_rows.append({
                "task_id": row.task_id,
                "family": row.family,
                "dimension": row.dimension,
                "instance": row.instance,
                "arm": row.arm,
                "checkpoint_evaluations_per_dimension": int(multiplier),
                "target_fraction": float(fraction),
            })
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    checkpoint_frame.to_csv(
        output / "checkpoint_target_fractions.csv",
        index=False,
    )
    checkpoint_summary = (
        checkpoint_frame.groupby(
            ["arm", "checkpoint_evaluations_per_dimension"],
            as_index=False,
        )
        .agg(
            mean_target_fraction=("target_fraction", "mean"),
            blocks=("task_id", "size"),
        )
    )
    checkpoint_summary.to_csv(
        output / "checkpoint_summary.csv",
        index=False,
    )

    metadata = {
        "status": "TRACK_E_PILOT_COMPLETE",
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "rows": len(frame),
        "blocks": len(blocks),
        "arms": list(ARMS),
        "target_ratios": list(TARGET_RATIOS),
        "checkpoint_multipliers": list(CHECKPOINT_MULTIPLIERS),
        "decision": decision["status"],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps(primary, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
