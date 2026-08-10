"""Operational snapshot capture and feature construction for Track E2."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import itertools
import math
from typing import Any

import numpy as np

import basingraph_v2.optimizer as optimizer_module
from basingraph_v2.archive import BasinArchive as OriginalBasinArchive
from basingraph_v2.evaluation import EvaluationLedger as OriginalEvaluationLedger
from basingraph_v2.graph import BasinTransitionGraph as OriginalBasinTransitionGraph
from basingraph_v2.types import BasinNode, TransitionEdge


TRACE_FEATURES = (
    "dimension",
    "snapshot_budget_fraction",
    "log_fbest",
    "total_log_improvement",
    "improvement_count",
    "stall_fraction",
    "recent_improvement_10pct",
    "recent_improvement_25pct",
    "trace_log_area",
    "recent_trace_slope",
)

STATE_FEATURES = (
    "archive_size",
    "archive_loggap_mean",
    "archive_loggap_std",
    "archive_loggap_q50",
    "archive_loggap_q90",
    "archive_loggap_max",
    "archive_pairwise_dispersion",
    "archive_farthest_from_incumbent",
    "novelty_mean",
    "novelty_std",
    "visits_mean",
    "visits_max",
    "graph_edges",
    "graph_density",
    "accessibility_mean",
    "accessibility_std",
    "accessibility_max",
    "positive_improvement_edge_fraction",
    "positive_improvement_mean",
    "mode_coordinate_fraction",
    "mode_far_fraction",
    "mode_archive_fraction",
    "mode_final_fraction",
    "mode_completion_fraction",
    "mode_stall_fraction",
    "phase_initial_fraction",
    "phase_center_fraction",
    "phase_coordinate_fraction",
    "phase_far_fraction",
    "phase_archive_fraction",
    "phase_final_fraction",
    "phase_completion_fraction",
)


@dataclass
class Snapshot:
    target_nfe: int
    actual_nfe: int
    fbest: float
    xbest: np.ndarray
    history: list[tuple[int, float]]
    phase_evaluations: dict[str, int]
    archive: list[BasinNode]
    graph_edges: list[TransitionEdge]
    dropped_transient_edges: int
    graph_referential_integrity: bool


class CaptureContext:
    def __init__(self, thresholds: list[int]):
        self.thresholds = list(sorted(int(value) for value in thresholds))
        self.snapshots: list[Snapshot] = []
        self.ledger: OriginalEvaluationLedger | None = None
        self.archive: OriginalBasinArchive | None = None
        self.graph: OriginalBasinTransitionGraph | None = None

    def maybe_capture(self) -> None:
        if self.ledger is None or self.archive is None or self.graph is None:
            return
        while len(self.snapshots) < len(self.thresholds):
            target = self.thresholds[len(self.snapshots)]
            if self.ledger.nfe < target:
                return
            active_ids = {int(node.node_id) for node in self.archive.nodes}
            all_edges = copy.deepcopy(self.graph.to_edges())
            kept_edges = [
                edge
                for edge in all_edges
                if int(edge.source_id) in active_ids
                and int(edge.target_id) in active_ids
            ]
            dropped = len(all_edges) - len(kept_edges)
            integrity = all(
                int(edge.source_id) in active_ids
                and int(edge.target_id) in active_ids
                for edge in kept_edges
            )
            xbest = (
                self.ledger.xbest.copy()
                if self.ledger.xbest is not None
                else 0.5 * (self.ledger.lb + self.ledger.ub)
            )
            self.snapshots.append(
                Snapshot(
                    target_nfe=target,
                    actual_nfe=int(self.ledger.nfe),
                    fbest=float(self.ledger.fbest),
                    xbest=xbest,
                    history=copy.deepcopy(self.ledger.history),
                    phase_evaluations=dict(self.ledger.phase_evaluations),
                    archive=copy.deepcopy(self.archive.sorted_nodes()),
                    graph_edges=kept_edges,
                    dropped_transient_edges=dropped,
                    graph_referential_integrity=bool(integrity),
                )
            )


_ACTIVE_CONTEXT: CaptureContext | None = None


class CapturingLedger(OriginalEvaluationLedger):
    def __post_init__(self) -> None:
        super().__post_init__()
        if _ACTIVE_CONTEXT is not None:
            _ACTIVE_CONTEXT.ledger = self


class CapturingArchive(OriginalBasinArchive):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ACTIVE_CONTEXT is not None:
            _ACTIVE_CONTEXT.archive = self


class CapturingGraph(OriginalBasinTransitionGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ACTIVE_CONTEXT is not None:
            _ACTIVE_CONTEXT.graph = self

    def add_or_update(self, *args, **kwargs):
        edge = super().add_or_update(*args, **kwargs)
        if _ACTIVE_CONTEXT is not None:
            _ACTIVE_CONTEXT.maybe_capture()
        return edge


class capture_optimizer_state:
    """Temporarily instrument the frozen optimizer without changing decisions."""

    def __init__(self, thresholds: list[int]):
        self.context = CaptureContext(thresholds)
        self.originals: tuple[Any, Any, Any] | None = None

    def __enter__(self) -> CaptureContext:
        global _ACTIVE_CONTEXT
        if _ACTIVE_CONTEXT is not None:
            raise RuntimeError("Nested Track E2 capture context is not supported.")
        self.originals = (
            optimizer_module.EvaluationLedger,
            optimizer_module.BasinArchive,
            optimizer_module.BasinTransitionGraph,
        )
        _ACTIVE_CONTEXT = self.context
        optimizer_module.EvaluationLedger = CapturingLedger
        optimizer_module.BasinArchive = CapturingArchive
        optimizer_module.BasinTransitionGraph = CapturingGraph
        return self.context

    def __exit__(self, exc_type, exc, tb) -> None:
        global _ACTIVE_CONTEXT
        assert self.originals is not None
        (
            optimizer_module.EvaluationLedger,
            optimizer_module.BasinArchive,
            optimizer_module.BasinTransitionGraph,
        ) = self.originals
        _ACTIVE_CONTEXT = None


def _history_values(history: list[tuple[int, float]]) -> np.ndarray:
    values = np.asarray([float(value) for _, value in history], dtype=float)
    if values.size == 0:
        return np.asarray([np.inf], dtype=float)
    return values


def _improvement_count(values: np.ndarray) -> int:
    if len(values) < 2:
        return 0
    scale = np.maximum(1.0, np.abs(values[:-1]))
    return int(np.sum(values[1:] < values[:-1] - 1e-14 * scale))


def _stall_length(values: np.ndarray) -> int:
    if len(values) < 2:
        return int(len(values))
    best = float(values[-1])
    index = len(values) - 1
    while index > 0 and math.isclose(
        float(values[index - 1]),
        best,
        rel_tol=1e-13,
        abs_tol=1e-15,
    ):
        index -= 1
    return int(len(values) - 1 - index)


def _value_at_fraction(values: np.ndarray, fraction: float) -> float:
    index = max(0, min(len(values) - 1, int(math.floor((1.0 - fraction) * len(values)))))
    return float(values[index])


def trace_features(
    snapshot: Snapshot,
    *,
    dimension: int,
    parent_budget: int,
) -> dict[str, float]:
    values = _history_values(snapshot.history)
    log_values = np.log10(1.0 + np.maximum(values, 0.0))
    current = float(log_values[-1])
    first = float(log_values[0])
    recent_10 = math.log10(1.0 + max(_value_at_fraction(values, 0.10), 0.0)) - current
    recent_25 = math.log10(1.0 + max(_value_at_fraction(values, 0.25), 0.0)) - current
    tail_count = max(5, int(math.ceil(0.10 * len(log_values))))
    tail = log_values[-tail_count:]
    if len(tail) >= 2:
        x = np.arange(len(tail), dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
    else:
        slope = 0.0
    return {
        "dimension": float(dimension),
        "snapshot_budget_fraction": float(snapshot.actual_nfe / max(1, parent_budget)),
        "log_fbest": current,
        "total_log_improvement": first - current,
        "improvement_count": float(_improvement_count(values)),
        "stall_fraction": float(_stall_length(values) / max(1, len(values))),
        "recent_improvement_10pct": float(recent_10),
        "recent_improvement_25pct": float(recent_25),
        "trace_log_area": float(np.mean(log_values)),
        "recent_trace_slope": slope,
    }


def _safe_stats(values: np.ndarray) -> tuple[float, float, float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.quantile(values, 0.50)),
        float(np.quantile(values, 0.90)),
        float(np.max(values)),
    )


def state_features(
    snapshot: Snapshot,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    nodes = snapshot.archive
    edges = snapshot.graph_edges
    scale = float(np.linalg.norm(np.asarray(upper) - np.asarray(lower)) + 1e-300)
    fbest = max(float(snapshot.fbest), 0.0)
    gaps = np.asarray(
        [math.log10(1.0 + max(float(node.f_center) - fbest, 0.0)) for node in nodes],
        dtype=float,
    )
    gap_mean, gap_std, gap_q50, gap_q90, gap_max = _safe_stats(gaps)

    centers = np.asarray([node.center for node in nodes], dtype=float)
    if len(centers) >= 2:
        diff = centers[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(diff, axis=2)
        upper_triangle = distances[np.triu_indices(len(centers), k=1)]
        pairwise = float(np.mean(upper_triangle) / scale)
    else:
        pairwise = 0.0
    if len(centers):
        farthest = float(
            np.max(np.linalg.norm(centers - snapshot.xbest[None, :], axis=1))
            / scale
        )
    else:
        farthest = 0.0

    novelty = np.asarray([float(node.novelty) for node in nodes], dtype=float)
    visits = np.asarray([float(node.visits) for node in nodes], dtype=float)
    accessibility = np.asarray(
        [float(edge.accessibility) for edge in edges],
        dtype=float,
    )
    improvements = np.asarray(
        [float(edge.best_improvement) for edge in edges],
        dtype=float,
    )
    edge_count = len(edges)
    node_count = len(nodes)
    density = float(edge_count / max(1, node_count * max(1, node_count - 1)))

    modes = [str(edge.source_mode) for edge in edges]
    def mode_fraction(*prefixes: str) -> float:
        if not modes:
            return 0.0
        return float(
            sum(any(mode.startswith(prefix) for prefix in prefixes) for mode in modes)
            / len(modes)
        )

    nfe = max(1, snapshot.actual_nfe)
    phases = snapshot.phase_evaluations
    return {
        "archive_size": float(node_count),
        "archive_loggap_mean": gap_mean,
        "archive_loggap_std": gap_std,
        "archive_loggap_q50": gap_q50,
        "archive_loggap_q90": gap_q90,
        "archive_loggap_max": gap_max,
        "archive_pairwise_dispersion": pairwise,
        "archive_farthest_from_incumbent": farthest,
        "novelty_mean": float(np.mean(novelty)) if novelty.size else 0.0,
        "novelty_std": float(np.std(novelty)) if novelty.size else 0.0,
        "visits_mean": float(np.mean(visits)) if visits.size else 0.0,
        "visits_max": float(np.max(visits)) if visits.size else 0.0,
        "graph_edges": float(edge_count),
        "graph_density": density,
        "accessibility_mean": float(np.mean(accessibility)) if accessibility.size else 0.0,
        "accessibility_std": float(np.std(accessibility)) if accessibility.size else 0.0,
        "accessibility_max": float(np.max(accessibility)) if accessibility.size else 0.0,
        "positive_improvement_edge_fraction": float(np.mean(improvements > 0.0)) if improvements.size else 0.0,
        "positive_improvement_mean": float(np.mean(improvements[improvements > 0.0])) if np.any(improvements > 0.0) else 0.0,
        "mode_coordinate_fraction": mode_fraction("coordinate"),
        "mode_far_fraction": mode_fraction("far"),
        "mode_archive_fraction": mode_fraction("archive"),
        "mode_final_fraction": mode_fraction("final"),
        "mode_completion_fraction": mode_fraction("budget_completion"),
        "mode_stall_fraction": mode_fraction("stall"),
        "phase_initial_fraction": float(phases.get("initial_design", 0) / nfe),
        "phase_center_fraction": float(phases.get("center_local", 0) / nfe),
        "phase_coordinate_fraction": float(phases.get("coordinate_sweep", 0) / nfe),
        "phase_far_fraction": float(phases.get("far_basin", 0) / nfe),
        "phase_archive_fraction": float(phases.get("archive_fallback", 0) / nfe),
        "phase_final_fraction": float(phases.get("final_polish", 0) / nfe),
        "phase_completion_fraction": float(phases.get("budget_completion", 0) / nfe),
    }
