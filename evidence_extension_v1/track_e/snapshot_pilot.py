"""Exact mid-run snapshot support for Track E engineering pilot 2."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

import basingraph_v2.optimizer as optimizer_module
from basingraph_v2.archive import BasinArchive as OriginalBasinArchive
from basingraph_v2.evaluation import EvaluationLedger as OriginalEvaluationLedger
from basingraph_v2.graph import BasinTransitionGraph as OriginalBasinTransitionGraph
from basingraph_v2.optimizer import BasinGraphOptions
from basingraph_v2.types import BasinNode, TransitionEdge

from evidence_extension_v1.track_e import run_matched_trace_pilot as base


SNAPSHOT_CHECKPOINT_MULTIPLIER = 40
PLANNED_TOTAL_BUDGET_MULTIPLIER = 140


@dataclass
class PrefixSnapshot:
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


def capture_prefix_snapshot(
    task,
    *,
    prefix_seed: int,
    options: BasinGraphOptions,
) -> PrefixSnapshot:
    checkpoint = SNAPSHOT_CHECKPOINT_MULTIPLIER * task.dimension
    planned_total = PLANNED_TOTAL_BUDGET_MULTIPLIER * task.dimension
    holder: dict[str, Any] = {}

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
            if self.nfe >= checkpoint and "snapshot" not in holder:
                archive = holder.get("archive")
                graph = holder.get("graph")
                if archive is None or graph is None:
                    raise RuntimeError("Track E snapshot objects were not registered.")
                holder["snapshot"] = PrefixSnapshot(
                    implementation_version=optimizer_module.IMPLEMENTATION_VERSION,
                    options_hash=options.stable_hash(),
                    xbest=np.asarray(self.xbest, dtype=float).copy(),
                    fbest=float(self.fbest),
                    nfe=int(self.nfe),
                    history=copy.deepcopy(self.history),
                    phase_evaluations=copy.deepcopy(self.phase_evaluations),
                    archive=copy.deepcopy(archive.sorted_nodes()),
                    graph_edges=copy.deepcopy(graph.to_edges()),
                    snapshot_phase=str(self.current_phase),
                )
                # Stop all future objective calls while allowing the immutable
                # optimizer to unwind its normal budget-exhaustion paths.
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
        raise RuntimeError("Exact Track E snapshot was not captured.")
    if snapshot.nfe != checkpoint:
        raise RuntimeError(
            f"Snapshot evaluation mismatch: {snapshot.nfe} != {checkpoint}"
        )
    if len(snapshot.history) != checkpoint:
        raise RuntimeError("Snapshot best-so-far history length mismatch.")
    return snapshot


def run_block(family: str, dimension: int, instance: int):
    for key in base.THREAD_ENV:
        import os
        os.environ[key] = "1"

    task = base.make_task(family, dimension, instance)
    options = BasinGraphOptions()
    prefix_seed = base.block_seed(
        base.BASE_PREFIX_SEED + 10_000_000,
        family,
        dimension,
        instance,
    )
    continuation_seed = base.block_seed(
        base.BASE_CONTINUATION_SEED + 10_000_000,
        family,
        dimension,
        instance,
    )
    control_seed = base.block_seed(
        base.BASE_CONTROL_SEED + 10_000_000,
        family,
        dimension,
        instance,
    )

    started = time.perf_counter()
    snapshot = capture_prefix_snapshot(
        task,
        prefix_seed=prefix_seed,
        options=options,
    )
    prefix_time = time.perf_counter() - started

    if snapshot.implementation_version != base.EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected BasinGraph implementation version.")
    if snapshot.options_hash != base.EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected BasinGraph options hash.")

    prefix_payload = {
        "history": snapshot.history,
        "archive": [node.to_jsonable() for node in snapshot.archive],
        "graph_edges": [edge.to_jsonable() for edge in snapshot.graph_edges],
        "xbest": np.asarray(snapshot.xbest, dtype=float).tolist(),
        "fbest": float(snapshot.fbest),
        "snapshot_phase": snapshot.snapshot_phase,
        "phase_evaluations": snapshot.phase_evaluations,
    }
    prefix_history_hash = base.stable_digest(prefix_payload["history"])
    prefix_state_hash = base.stable_digest({
        "archive": prefix_payload["archive"],
        "graph_edges": prefix_payload["graph_edges"],
        "xbest": prefix_payload["xbest"],
        "fbest": prefix_payload["fbest"],
    })

    rows = []
    for arm in base.ARMS:
        continuation_started = time.perf_counter()
        result = base.run_continuation(
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
            "prefix_history_hash": prefix_history_hash,
            "prefix_state_hash": prefix_state_hash,
            "prefix_wall_time_seconds": float(prefix_time),
            "continuation_wall_time_seconds": float(
                time.perf_counter() - continuation_started
            ),
            "implementation_version": snapshot.implementation_version,
            "options_hash": snapshot.options_hash,
            "snapshot_phase": snapshot.snapshot_phase,
            "planned_total_budget_multiplier": (
                PLANNED_TOTAL_BUDGET_MULTIPLIER
            ),
        })
        rows.append(result)

    if len({row["prefix_history_hash"] for row in rows}) != 1:
        raise RuntimeError("Matched-prefix trace hash mismatch within block.")
    return rows
