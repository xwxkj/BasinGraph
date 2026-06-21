"""
Public data model for BasinGraph v2.0.0-rc2.

The manuscript, implementation and serialized experiment records must use the
same semantics:

- ProbeRecord: an evaluated but not necessarily basin-certified point;
- BasinNode: a refined, certified basin representative;
- TransitionEdge: a directed transition between active certified nodes;
- GeometryDiagnostics: derivative-free landscape diagnostics;
- DirectionDiagnostics: provenance of principal search directions;
- BasinGraphResult: complete auditable optimizer output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np


def _array_to_list(value) -> list[float]:
    return np.asarray(value, dtype=float).reshape(-1).tolist()


@dataclass
class ProbeRecord:
    probe_id: int
    point: np.ndarray
    f_value: float
    source_phase: str
    created_nfe: int
    parent_node_id: int | None
    novelty: float
    quality_score: float = 0.0
    selected_for_refinement: bool = False
    refinement_mode: str | None = None
    committed_node_id: int | None = None

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["point"] = _array_to_list(self.point)
        return data


@dataclass
class BasinNode:
    node_id: int
    center: np.ndarray
    f_center: float
    radius: float
    curvature_proxy: float
    visits: int
    created_nfe: int
    last_updated_nfe: int
    novelty: float
    source: str
    certified: bool
    certification_mode: str
    parent_probe_id: int | None
    refinement_evaluations: int
    local_support_count: int
    local_support_radius: float
    certification_improvement: float

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
    attempts: int
    source_mode: str

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GeometryDiagnostics:
    dimension: int
    mean_scale: float
    max_scale: float
    domain_anisotropy: float
    curvature_anisotropy: float
    curvature_values: list[float]
    boundary_signal: float
    ruggedness_score: float
    sign_change_rate: float
    finite_anchor_fraction: float
    valid_axis_triplets: int

    @property
    def local_mode_score(self) -> float:
        curvature_term = min(
            1.0,
            np.log10(1.0 + max(self.curvature_anisotropy, 1.0)) / 6.0,
        )
        return float(
            1.0
            / (
                1.0
                + max(self.ruggedness_score, 0.0)
                + max(self.sign_change_rate, 0.0)
                + curvature_term
            )
        )

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectionDiagnostics:
    covariance_directions: int
    transition_directions: int
    coordinate_fallback_directions: int
    retained_directions: int
    direction_dimension: int

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
    probes: list[ProbeRecord]
    probe_count_total: int
    archive: list[BasinNode]
    graph_edges: list[TransitionEdge]
    diagnostics: GeometryDiagnostics
    direction_diagnostics: DirectionDiagnostics
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
                str(key): int(value)
                for key, value in self.phase_evaluations.items()
            },
            "history": [
                (int(nfe), float(best))
                for nfe, best in self.history
            ],
            "probes": [
                probe.to_jsonable()
                for probe in self.probes
            ],
            "probe_count_total": int(self.probe_count_total),
            "archive": [
                node.to_jsonable()
                for node in self.archive
            ],
            "graph_edges": [
                edge.to_jsonable()
                for edge in self.graph_edges
            ],
            "diagnostics": self.diagnostics.to_jsonable(),
            "direction_diagnostics": (
                self.direction_diagnostics.to_jsonable()
            ),
            "event_log": [
                event.to_jsonable()
                for event in self.event_log
            ],
            "message": self.message,
        }
