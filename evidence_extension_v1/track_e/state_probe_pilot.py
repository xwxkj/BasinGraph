"""Phase-boundary snapshot and state-probe continuation for Track E pilot 3."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
import json
import math
import os
import time
from typing import Any

import numpy as np

import basingraph_v2.optimizer as optimizer_module
from basingraph_v2.archive import BasinArchive as OriginalBasinArchive
from basingraph_v2.evaluation import (
    BudgetExhausted,
    EvaluationLedger as OriginalEvaluationLedger,
)
from basingraph_v2.graph import BasinTransitionGraph as OriginalBasinTransitionGraph
from basingraph_v2.optimizer import BasinGraphOptions
from basingraph_v2.types import BasinNode, TransitionEdge

from evidence_extension_v1.track_e import run_matched_trace_pilot as base


PLANNED_PARENT_BUDGET_MULTIPLIER = 250


@dataclass
class PhaseBoundarySnapshot:
    implementation_version: str
    options_hash: str
    xbest: np.ndarray
    fbest: float
    nfe: int
    history: list[tuple[int, float]]
    phase_evaluations: dict[str, int]
    archive: list[BasinNode]
    graph_edges: list[TransitionEdge]
    snapshot_phase: str


def capture_phase_boundary_snapshot(
    task,
    *,
    prefix_seed: int,
    options: BasinGraphOptions,
) -> PhaseBoundarySnapshot:
    planned_total = PLANNED_PARENT_BUDGET_MULTIPLIER * task.dimension
    holder: dict[str, Any] = {
        "coordinate_seen": False,
    }

    class CapturingArchive(OriginalBasinArchive):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["archive"] = self

    class CapturingGraph(OriginalBasinTransitionGraph):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["graph"] = self

    class CapturingLedger(OriginalEvaluationLedger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            holder["ledger"] = self

        def evaluate(self, point):
            phase = str(self.current_phase)
            if phase == "coordinate_sweep":
                holder["coordinate_seen"] = True
            if (
                holder.get("coordinate_seen", False)
                and phase != "coordinate_sweep"
                and "snapshot" not in holder
            ):
                archive = holder.get("archive")
                graph = holder.get("graph")
                if archive is None or graph is None:
                    raise RuntimeError("Track E snapshot objects were not registered.")
                holder["snapshot"] = PhaseBoundarySnapshot(
                    implementation_version=optimizer_module.IMPLEMENTATION_VERSION,
                    options_hash=options.stable_hash(),
                    xbest=np.asarray(self.xbest, dtype=float).copy(),
                    fbest=float(self.fbest),
                    nfe=int(self.nfe),
                    history=copy.deepcopy(self.history),
                    phase_evaluations=copy.deepcopy(self.phase_evaluations),
                    archive=copy.deepcopy(archive.sorted_nodes()),
                    graph_edges=copy.deepcopy(graph.to_edges()),
                    snapshot_phase=phase,
                )
                # Stop future objective calls after a fully updated phase boundary.
                self.max_evals = int(self.nfe)
            return super().evaluate(point)

    originals = (
        optimizer_module.EvaluationLedger,
        optimizer_module.BasinArchive,
        optimizer_module.BasinTransitionGraph,
    )
    optimizer_module.EvaluationLedger = CapturingLedger
    optimizer_module.BasinArchive = CapturingArchive
    optimizer_module.BasinTransitionGraph = CapturingGraph
    try:
        optimizer_module.minimize_basingraph_v2(
            task.objective,
            task.lower,
            task.upper,
            max_evals=planned_total,
            seed=prefix_seed,
            options=options,
        )
    finally:
        (
            optimizer_module.EvaluationLedger,
            optimizer_module.BasinArchive,
            optimizer_module.BasinTransitionGraph,
        ) = originals

    snapshot = holder.get("snapshot")
    if snapshot is None:
        raise RuntimeError("Track E phase-boundary snapshot was not captured.")
    if not snapshot.history or len(snapshot.history) != snapshot.nfe:
        raise RuntimeError("Phase-boundary history length mismatch.")
    if snapshot.snapshot_phase == "coordinate_sweep":
        raise RuntimeError("Snapshot did not occur after coordinate sweep.")
    return snapshot


def _clone_archive(nodes: list[BasinNode], options: BasinGraphOptions):
    archive = OriginalBasinArchive(
        merge_radius_factor=options.merge_radius_factor,
        max_nodes=options.archive_max_size,
    )
    archive.nodes = copy.deepcopy(nodes)
    maximum = max((node.node_id for node in archive.nodes), default=0)
    archive._id_counter = itertools.count(maximum + 1)
    return archive


def _clone_graph(edges: list[TransitionEdge]):
    graph = OriginalBasinTransitionGraph()
    copied = copy.deepcopy(edges)
    graph.edges = {
        (int(edge.source_id), int(edge.target_id)): edge
        for edge in copied
    }
    return graph


def _shuffled_graph(edges, active_ids, seed):
    if len(active_ids) < 2 or not edges:
        return OriginalBasinTransitionGraph()
    rng = np.random.default_rng(seed)
    original = np.asarray(active_ids, dtype=int)
    permuted = original.copy()
    for _ in range(50):
        rng.shuffle(permuted)
        if not np.array_equal(permuted, original):
            break
    mapping = {
        int(source): int(target)
        for source, target in zip(original.tolist(), permuted.tolist())
    }
    graph = OriginalBasinTransitionGraph()
    for edge in copy.deepcopy(edges):
        edge.source_id = mapping[int(edge.source_id)]
        edge.target_id = mapping[int(edge.target_id)]
        graph.edges[(edge.source_id, edge.target_id)] = edge
    return graph


def _trace_archive(snapshot, task, options):
    archive = OriginalBasinArchive(
        merge_radius_factor=options.merge_radius_factor,
        max_nodes=options.archive_max_size,
    )
    radius = 0.02 * float(np.linalg.norm(task.upper - task.lower) + 1e-300)
    archive.nodes = [
        BasinNode(
            node_id=1,
            center=np.asarray(snapshot.xbest, dtype=float).copy(),
            f_center=float(snapshot.fbest),
            radius=radius,
            curvature_proxy=float(1.0 / (1.0 + radius)),
            created_nfe=int(snapshot.nfe),
            last_updated_nfe=int(snapshot.nfe),
            novelty=1.0,
            source="matched_trace_incumbent",
        )
    ]
    archive._id_counter = itertools.count(2)
    return archive


def state_for_arm(arm, snapshot, task, options, control_seed):
    if arm == "FullState":
        return _clone_archive(snapshot.archive, options), _clone_graph(snapshot.graph_edges)
    if arm == "ArchiveOnly":
        return _clone_archive(snapshot.archive, options), OriginalBasinTransitionGraph()
    if arm == "EdgeShuffled":
        archive = _clone_archive(snapshot.archive, options)
        ids = [int(node.node_id) for node in archive.nodes]
        return archive, _shuffled_graph(snapshot.graph_edges, ids, control_seed)
    if arm == "TraceOnly":
        return _trace_archive(snapshot, task, options), OriginalBasinTransitionGraph()
    if arm == "ColdRestart":
        return (
            OriginalBasinArchive(
                merge_radius_factor=options.merge_radius_factor,
                max_nodes=options.archive_max_size,
            ),
            OriginalBasinTransitionGraph(),
        )
    raise KeyError(arm)


def _history_value(history, checkpoint, initial):
    value = float(initial)
    for nfe, candidate in history:
        if int(nfe) > int(checkpoint):
            break
        value = min(value, float(candidate))
    return value


def _target_eval(history, threshold):
    for nfe, value in history:
        if float(value) <= float(threshold):
            return int(nfe)
    return None


def run_state_probe_continuation(
    task,
    snapshot,
    arm,
    continuation_seed,
    control_seed,
):
    options = BasinGraphOptions()
    budget = base.CONTINUATION_BUDGET_MULTIPLIER * task.dimension
    evaluated_points: list[list[float]] = []

    def recorded_objective(point):
        z = np.asarray(point, dtype=float).copy()
        evaluated_points.append(z.tolist())
        return task.objective(z)

    ledger = OriginalEvaluationLedger(
        recorded_objective,
        task.lower,
        task.upper,
        budget,
    )
    ledger.xbest = np.asarray(snapshot.xbest, dtype=float).copy()
    ledger.fbest = float(snapshot.fbest)
    archive, graph = state_for_arm(
        arm,
        snapshot,
        task,
        options,
        control_seed,
    )
    rng = np.random.default_rng(continuation_seed)
    domain_radius = float(np.linalg.norm(task.upper - task.lower) + 1e-300)
    created_nodes = 0
    merged_nodes = 0

    def add_node(point, value):
        nonlocal created_nodes, merged_nodes
        update = archive.add_or_merge(
            point,
            value,
            radius=0.02 * domain_radius,
            curvature_proxy=float(1.0 / (1.0 + 0.02 * domain_radius)),
            nfe=ledger.nfe,
            source="matched_trace_state_probe",
            lb=task.lower,
            ub=task.upper,
        )
        if update.created:
            created_nodes += 1
        else:
            merged_nodes += 1
        if update.removed_node_ids:
            graph.remove_nodes(update.removed_node_ids)
        return update.node

    with ledger.phase("matched_trace_state_probe", None):
        while ledger.remaining > 0:
            nodes = archive.sorted_nodes()
            if nodes and rng.random() < 0.80:
                scores = graph.guidance_scores(
                    nodes,
                    quality_weight=options.graph_quality_weight,
                    novelty_weight=options.graph_novelty_weight,
                    accessibility_weight=options.graph_accessibility_weight,
                )
                weights = np.asarray(
                    [max(scores.get(node.node_id, 0.0), 0.0) for node in nodes],
                    dtype=float,
                )
                if weights.sum() <= 0:
                    probabilities = np.full(len(nodes), 1.0 / len(nodes))
                else:
                    probabilities = weights / weights.sum()
                source_node = nodes[int(rng.choice(len(nodes), p=probabilities))]
                decay = max(0.05, (ledger.remaining / max(1, budget)) ** 0.5)
                step = rng.standard_t(df=2.0, size=task.dimension) * (
                    0.20 * decay * (task.upper - task.lower)
                )
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
                    task.lower + rng.random(task.dimension) * (task.upper - task.lower)
                )

            try:
                value = ledger.evaluate(point)
            except BudgetExhausted:
                break
            target = add_node(point, value)
            if source_node is not None:
                graph.add_or_update(
                    source_node.node_id,
                    target.node_id,
                    evaluations=1,
                    improvement=source_node.f_center - target.f_center,
                    barrier_proxy=max(0.0, value - source_node.f_center),
                    nfe=ledger.nfe,
                    source_mode="matched_trace_state_probe",
                )

    active_ids = {int(node.node_id) for node in archive.nodes}
    graph.remove_nodes([
        node_id
        for edge in graph.to_edges()
        for node_id in (edge.source_id, edge.target_id)
        if int(node_id) not in active_ids
    ])
    graph_valid = all(
        int(edge.source_id) in active_ids and int(edge.target_id) in active_ids
        for edge in graph.to_edges()
    )

    prefix_gap = max(float(snapshot.fbest) - task.optimum, np.finfo(float).tiny)
    final_gap = max(float(ledger.fbest) - task.optimum, np.finfo(float).tiny)
    thresholds = {
        str(ratio): task.optimum + ratio * prefix_gap
        for ratio in base.TARGET_RATIOS
    }
    target_evaluations = {
        ratio: _target_eval(ledger.history, threshold)
        for ratio, threshold in thresholds.items()
    }
    checkpoint_fractions = {}
    checkpoint_gaps = {}
    for multiplier in base.CHECKPOINT_MULTIPLIERS:
        checkpoint = min(multiplier * task.dimension, budget)
        value = _history_value(ledger.history, checkpoint, snapshot.fbest)
        checkpoint_gaps[str(multiplier)] = max(
            float(value) - task.optimum,
            np.finfo(float).tiny,
        )
        checkpoint_fractions[str(multiplier)] = float(np.mean([
            evaluation is not None and evaluation <= checkpoint
            for evaluation in target_evaluations.values()
        ]))

    return {
        "arm": arm,
        "continuation_seed": int(continuation_seed),
        "control_seed": int(control_seed),
        "continuation_budget": int(budget),
        "continuation_nfe": int(ledger.nfe),
        "prefix_fbest": float(snapshot.fbest),
        "final_fbest": float(ledger.fbest),
        "prefix_gap": float(prefix_gap),
        "final_gap": float(final_gap),
        "log10_gap_reduction": float(math.log10(prefix_gap) - math.log10(final_gap)),
        "final_target_fraction": float(np.mean([
            evaluation is not None for evaluation in target_evaluations.values()
        ])),
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
        "created_nodes": int(created_nodes),
        "merged_nodes": int(merged_nodes),
        "history_monotone": bool(all(
            later <= earlier + 1e-14
            for earlier, later in zip(
                [value for _, value in ledger.history[:-1]],
                [value for _, value in ledger.history[1:]],
            )
        )),
        "graph_referential_integrity": bool(graph_valid),
        "continuation_history_hash": base.stable_digest(ledger.history),
        "evaluated_point_sequence_hash": base.stable_digest(evaluated_points),
    }


def run_block(family: str, dimension: int, instance: int):
    for key in base.THREAD_ENV:
        os.environ[key] = "1"
    task = base.make_task(family, dimension, instance)
    options = BasinGraphOptions()
    prefix_seed = base.block_seed(
        base.BASE_PREFIX_SEED + 20_000_000,
        family,
        dimension,
        instance,
    )
    continuation_seed = base.block_seed(
        base.BASE_CONTINUATION_SEED + 20_000_000,
        family,
        dimension,
        instance,
    )
    control_seed = base.block_seed(
        base.BASE_CONTROL_SEED + 20_000_000,
        family,
        dimension,
        instance,
    )

    started = time.perf_counter()
    snapshot = capture_phase_boundary_snapshot(
        task,
        prefix_seed=prefix_seed,
        options=options,
    )
    prefix_time = time.perf_counter() - started
    if snapshot.implementation_version != base.EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected BasinGraph implementation version.")
    if snapshot.options_hash != base.EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected BasinGraph options hash.")

    payload = {
        "history": snapshot.history,
        "archive": [node.to_jsonable() for node in snapshot.archive],
        "graph_edges": [edge.to_jsonable() for edge in snapshot.graph_edges],
        "xbest": np.asarray(snapshot.xbest, dtype=float).tolist(),
        "fbest": float(snapshot.fbest),
        "snapshot_phase": snapshot.snapshot_phase,
        "phase_evaluations": snapshot.phase_evaluations,
    }
    history_hash = base.stable_digest(payload["history"])
    state_hash = base.stable_digest({
        "archive": payload["archive"],
        "graph_edges": payload["graph_edges"],
        "xbest": payload["xbest"],
        "fbest": payload["fbest"],
    })

    rows = []
    for arm in base.ARMS:
        continuation_started = time.perf_counter()
        result = run_state_probe_continuation(
            task,
            snapshot,
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
            "prefix_budget": int(snapshot.nfe),
            "prefix_nfe": int(snapshot.nfe),
            "prefix_archive_nodes": int(len(snapshot.archive)),
            "prefix_graph_edges": int(len(snapshot.graph_edges)),
            "prefix_history_hash": history_hash,
            "prefix_state_hash": state_hash,
            "prefix_wall_time_seconds": float(prefix_time),
            "continuation_wall_time_seconds": float(
                time.perf_counter() - continuation_started
            ),
            "implementation_version": snapshot.implementation_version,
            "options_hash": snapshot.options_hash,
            "snapshot_phase": snapshot.snapshot_phase,
            "planned_parent_budget_multiplier": PLANNED_PARENT_BUDGET_MULTIPLIER,
        })
        rows.append(result)
    if len({row["prefix_history_hash"] for row in rows}) != 1:
        raise RuntimeError("Matched-prefix history hash mismatch within block.")
    return rows
