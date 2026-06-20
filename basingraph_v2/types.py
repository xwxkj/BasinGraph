"""
Public dataclasses for BasinGraph v2.0.0-rc1.

These objects define the algorithm-code-paper contract. Any manuscript
description of the final Route B algorithm must match these fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np


def _array_to_list(x) -> list[float]:
    return np.asarray(x, dtype=float).reshape(-1).tolist()


@dataclass
class BasinNode:
    node_id: int
    center: np.ndarray
    f_center: float
    radius: float
    curvature_proxy: float
    visits: int = 1
    created_nfe: int = 0
    last_updated_nfe: int = 0
    novelty: float = 1.0
    source: str = "unknown"

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["center"] = _array_to_list(self.center)
        return data


@dataclass
class TransitionEdge:
    source_id: int
    target_id: int
    evaluations: int
    best_improvement: float
    barrier_proxy: float
    accessibility: float
    created_nfe: int
    last_updated_nfe: int
    attempts: int = 1
    source_mode: str = "unknown"

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryDiagnostics:
    dimension: int
    mean_scale: float
    max_scale: float
    anisotropy: float
    boundary_signal: float
    ruggedness_score: float
    sign_change_rate: float
    finite_anchor_fraction: float
    valid_axis_triplets: int

    @property
    def local_mode_score(self) -> float:
        return float(
            1.0
            / (
                1.0
                + max(self.ruggedness_score, 0.0)
                + max(self.sign_change_rate, 0.0)
                + np.log1p(max(self.anisotropy - 1.0, 0.0))
            )
        )

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventRecord:
    nfe: int
    phase: str
    event: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BasinGraphResult:
    implementation_version: str
    options_hash: str
    options: dict[str, Any]
    xbest: np.ndarray
    fbest: float
    nfe: int
    phase_evaluations: dict[str, int]
    history: list[tuple[int, float]]
    archive: list[BasinNode]
    graph_edges: list[TransitionEdge]
    diagnostics: GeometryDiagnostics
    event_log: list[EventRecord]
    message: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "implementation_version": self.implementation_version,
            "options_hash": self.options_hash,
            "options": self.options,
            "xbest": _array_to_list(self.xbest),
            "fbest": float(self.fbest),
            "nfe": int(self.nfe),
            "phase_evaluations": {
                str(k): int(v) for k, v in self.phase_evaluations.items()
            },
            "history": [(int(a), float(b)) for a, b in self.history],
            "archive": [node.to_jsonable() for node in self.archive],
            "graph_edges": [edge.to_jsonable() for edge in self.graph_edges],
            "diagnostics": self.diagnostics.to_jsonable(),
            "event_log": [event.to_jsonable() for event in self.event_log],
            "message": self.message,
        }
