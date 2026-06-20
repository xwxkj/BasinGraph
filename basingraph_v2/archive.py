
from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np
from .types import BasinNode

@dataclass
class BasinArchive:
    merge_radius_factor: float = 0.05
    max_nodes: int = 80
    nodes: list[BasinNode] = field(default_factory=list)
    _id_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    def __len__(self): return len(self.nodes)
    def sorted_nodes(self) -> list[BasinNode]:
        return sorted(self.nodes, key=lambda n: n.f_center)
    def _scale(self, lb, ub) -> float:
        return float(np.linalg.norm(np.asarray(ub)-np.asarray(lb)) + 1e-300)
    def nearest(self, x) -> tuple[BasinNode | None, float]:
        if not self.nodes:
            return None, np.inf
        z = np.asarray(x, dtype=float).reshape(-1)
        ds = [float(np.linalg.norm(z-node.center)) for node in self.nodes]
        i = int(np.argmin(ds))
        return self.nodes[i], ds[i]
    def add_or_merge(self, x, f, *, radius, curvature_proxy, nfe, source, lb, ub):
        x = np.asarray(x, dtype=float).reshape(-1)
        threshold = max(float(radius), self.merge_radius_factor * self._scale(lb, ub))
        nearest, distance = self.nearest(x)
        if nearest is not None and distance <= threshold:
            nearest.visits += 1
            nearest.last_updated_nfe = int(nfe)
            nearest.radius = max(nearest.radius, float(radius), float(distance))
            nearest.curvature_proxy = max(nearest.curvature_proxy, float(curvature_proxy))
            if f < nearest.f_center:
                nearest.center = x.copy()
                nearest.f_center = float(f)
                nearest.source = source
            nearest.novelty = min(1.0, float(distance / max(threshold, 1e-300)))
            return nearest, False
        node = BasinNode(
            node_id=next(self._id_counter), center=x.copy(), f_center=float(f),
            radius=float(radius), curvature_proxy=float(curvature_proxy),
            created_nfe=int(nfe), last_updated_nfe=int(nfe), novelty=1.0, source=source
        )
        self.nodes.append(node)
        self.nodes.sort(key=lambda n: n.f_center)
        if len(self.nodes) > self.max_nodes:
            self.nodes = self.nodes[:self.max_nodes]
        return node, True
