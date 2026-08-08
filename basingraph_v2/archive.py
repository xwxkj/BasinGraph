"""Merge-aware basin archive with capacity-safe node identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np

from .types import BasinNode


@dataclass
class ArchiveUpdate:
    node: BasinNode
    created: bool
    removed_node_ids: list[int]


@dataclass
class BasinArchive:
    merge_radius_factor: float = 0.035
    max_nodes: int = 80
    nodes: list[BasinNode] = field(default_factory=list)
    _id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))

    def __len__(self) -> int:
        return len(self.nodes)

    def sorted_nodes(self) -> list[BasinNode]:
        return sorted(self.nodes, key=lambda node: node.f_center)

    def node_by_id(self, node_id: int) -> BasinNode | None:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def nearest(self, x: np.ndarray) -> tuple[BasinNode | None, float]:
        if not self.nodes:
            return None, np.inf
        z = np.asarray(x, dtype=float).reshape(-1)
        distances = np.asarray(
            [np.linalg.norm(z - node.center) for node in self.nodes],
            dtype=float,
        )
        idx = int(np.argmin(distances))
        return self.nodes[idx], float(distances[idx])

    @staticmethod
    def problem_scale(lb: np.ndarray, ub: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(ub) - np.asarray(lb)) + 1e-300)

    def add_or_merge(
        self,
        x: np.ndarray,
        f: float,
        *,
        radius: float,
        curvature_proxy: float,
        nfe: int,
        source: str,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> ArchiveUpdate:
        x = np.asarray(x, dtype=float).reshape(-1)
        scale = self.problem_scale(lb, ub)
        merge_threshold = max(float(radius), self.merge_radius_factor * scale)

        nearest, distance = self.nearest(x)
        if nearest is not None and distance <= merge_threshold:
            nearest.visits += 1
            nearest.last_updated_nfe = int(nfe)
            nearest.radius = max(nearest.radius, float(radius), float(distance))
            nearest.curvature_proxy = max(
                nearest.curvature_proxy,
                float(curvature_proxy),
            )
            nearest.novelty = min(
                1.0,
                float(distance / max(merge_threshold, 1e-300)),
            )
            if np.isfinite(f) and f < nearest.f_center:
                nearest.center = x.copy()
                nearest.f_center = float(f)
                nearest.source = str(source)
            return ArchiveUpdate(nearest, False, [])

        node = BasinNode(
            node_id=next(self._id_counter),
            center=x.copy(),
            f_center=float(f),
            radius=float(radius),
            curvature_proxy=float(curvature_proxy),
            created_nfe=int(nfe),
            last_updated_nfe=int(nfe),
            novelty=1.0,
            source=str(source),
        )
        self.nodes.append(node)

        removed: list[int] = []
        if len(self.nodes) > self.max_nodes:
            # Preserve the best objective nodes; evict the current worst node.
            worst = max(self.nodes, key=lambda item: item.f_center)
            self.nodes.remove(worst)
            removed.append(worst.node_id)

        return ArchiveUpdate(node, node.node_id not in removed, removed)
