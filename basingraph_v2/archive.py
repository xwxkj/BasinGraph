"""Certified basin archive for BasinGraph v2.0.0-rc2."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Mapping
import numpy as np

from .types import BasinNode


@dataclass
class ArchiveUpdate:
    node: BasinNode
    created: bool
    active: bool
    removed_node_ids: list[int]
    merge_threshold: float


@dataclass
class BasinArchive:
    max_nodes: int = 80
    merge_radius_base: float = 0.025
    merge_radius_max: float = 0.080
    quality_weight: float = 0.50
    diversity_weight: float = 0.30
    accessibility_weight: float = 0.20
    nodes: list[BasinNode] = field(default_factory=list)
    _counter: itertools.count = field(
        default_factory=lambda: itertools.count(1)
    )

    def __len__(self) -> int:
        return len(self.nodes)

    def sorted_nodes(self) -> list[BasinNode]:
        return sorted(
            self.nodes,
            key=lambda node: node.f_center,
        )

    def node_by_id(self, node_id: int) -> BasinNode | None:
        for node in self.nodes:
            if node.node_id == int(node_id):
                return node
        return None

    def nearest(
        self,
        point: np.ndarray,
    ) -> tuple[BasinNode | None, float]:
        if not self.nodes:
            return None, np.inf
        point = np.asarray(point, dtype=float).reshape(-1)
        distances = np.asarray(
            [
                np.linalg.norm(point - node.center)
                for node in self.nodes
            ],
            dtype=float,
        )
        index = int(np.argmin(distances))
        return self.nodes[index], float(distances[index])

    @staticmethod
    def domain_radius(
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> float:
        return float(
            np.linalg.norm(
                np.asarray(ub) - np.asarray(lb)
            )
            + 1e-300
        )

    def occupancy(self) -> float:
        return float(
            len(self.nodes) / max(self.max_nodes, 1)
        )

    def merge_threshold(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> float:
        fraction = min(
            self.merge_radius_max,
            self.merge_radius_base
            * (1.0 + 2.0 * self.occupancy() ** 2),
        )
        return float(fraction * self.domain_radius(lb, ub))

    def _quality_scores(self) -> dict[int, float]:
        if not self.nodes:
            return {}
        ordered = sorted(
            self.nodes,
            key=lambda node: node.f_center,
        )
        if len(ordered) == 1:
            return {ordered[0].node_id: 1.0}
        return {
            node.node_id: float(
                1.0 - rank / (len(ordered) - 1)
            )
            for rank, node in enumerate(ordered)
        }

    def _diversity_scores(
        self,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> dict[int, float]:
        scale = self.domain_radius(lb, ub)
        if len(self.nodes) <= 1:
            return {
                node.node_id: 1.0
                for node in self.nodes
            }

        scores = {}
        for node in self.nodes:
            nearest_distance = min(
                np.linalg.norm(
                    node.center - other.center
                )
                for other in self.nodes
                if other.node_id != node.node_id
            )
            scores[node.node_id] = float(
                min(
                    1.0,
                    nearest_distance / max(scale, 1e-300),
                )
            )
        return scores

    def _retention_scores(
        self,
        *,
        lb: np.ndarray,
        ub: np.ndarray,
        accessibility: Mapping[int, float],
    ) -> dict[int, float]:
        quality = self._quality_scores()
        diversity = self._diversity_scores(lb, ub)
        return {
            node.node_id: float(
                self.quality_weight
                * quality.get(node.node_id, 0.0)
                + self.diversity_weight
                * diversity.get(node.node_id, 0.0)
                + self.accessibility_weight
                * float(accessibility.get(node.node_id, 0.5))
            )
            for node in self.nodes
        }

    def _enforce_capacity(
        self,
        *,
        lb: np.ndarray,
        ub: np.ndarray,
        accessibility: Mapping[int, float],
    ) -> list[int]:
        removed: list[int] = []

        while len(self.nodes) > self.max_nodes:
            best = min(
                self.nodes,
                key=lambda node: node.f_center,
            )
            scores = self._retention_scores(
                lb=lb,
                ub=ub,
                accessibility=accessibility,
            )
            candidates = [
                node
                for node in self.nodes
                if node.node_id != best.node_id
            ]
            victim = min(
                candidates,
                key=lambda node: (
                    scores.get(node.node_id, 0.0),
                    -node.f_center,
                    node.created_nfe,
                    node.node_id,
                ),
            )
            self.nodes.remove(victim)
            removed.append(victim.node_id)

        return removed

    def add_certified(
        self,
        point: np.ndarray,
        f_value: float,
        *,
        certification_mode: str,
        parent_probe_id: int | None,
        refinement_evaluations: int,
        local_support_count: int,
        local_support_radius: float,
        certification_improvement: float,
        curvature_proxy: float,
        nfe: int,
        source: str,
        lb: np.ndarray,
        ub: np.ndarray,
        accessibility: Mapping[int, float],
    ) -> ArchiveUpdate:
        point = np.asarray(point, dtype=float).reshape(-1)
        threshold = self.merge_threshold(lb, ub)
        nearest, distance = self.nearest(point)

        if nearest is not None and distance <= threshold:
            nearest.visits += 1
            nearest.last_updated_nfe = int(nfe)
            nearest.radius = max(
                nearest.radius,
                threshold,
                float(distance),
            )
            nearest.curvature_proxy = max(
                nearest.curvature_proxy,
                float(curvature_proxy),
            )
            nearest.novelty = min(
                1.0,
                distance
                / max(self.domain_radius(lb, ub), 1e-300),
            )
            nearest.local_support_count += int(local_support_count)
            nearest.local_support_radius = max(
                nearest.local_support_radius,
                float(local_support_radius),
            )
            nearest.refinement_evaluations += int(
                refinement_evaluations
            )
            nearest.certification_improvement = max(
                nearest.certification_improvement,
                float(certification_improvement),
            )

            if np.isfinite(f_value) and f_value < nearest.f_center:
                nearest.center = point.copy()
                nearest.f_center = float(f_value)
                nearest.source = str(source)
                nearest.certification_mode = str(
                    certification_mode
                )
                nearest.parent_probe_id = parent_probe_id

            return ArchiveUpdate(
                node=nearest,
                created=False,
                active=True,
                removed_node_ids=[],
                merge_threshold=threshold,
            )

        node = BasinNode(
            node_id=next(self._counter),
            center=point.copy(),
            f_center=float(f_value),
            radius=float(threshold),
            curvature_proxy=float(curvature_proxy),
            visits=1,
            created_nfe=int(nfe),
            last_updated_nfe=int(nfe),
            novelty=1.0,
            source=str(source),
            certified=True,
            certification_mode=str(certification_mode),
            parent_probe_id=parent_probe_id,
            refinement_evaluations=int(refinement_evaluations),
            local_support_count=int(local_support_count),
            local_support_radius=float(local_support_radius),
            certification_improvement=float(
                certification_improvement
            ),
        )
        self.nodes.append(node)

        removed = self._enforce_capacity(
            lb=lb,
            ub=ub,
            accessibility=accessibility,
        )
        active = self.node_by_id(node.node_id) is not None

        return ArchiveUpdate(
            node=node,
            created=active,
            active=active,
            removed_node_ids=removed,
            merge_threshold=threshold,
        )
