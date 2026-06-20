
from __future__ import annotations
from dataclasses import dataclass, field
from .types import TransitionEdge

@dataclass
class BasinTransitionGraph:
    edges: dict[tuple[int, int], TransitionEdge] = field(default_factory=dict)
    def add_or_update(self, source_id, target_id, *, evaluations, improvement, barrier_proxy, nfe, source_mode):
        source_id, target_id = int(source_id), int(target_id)
        if source_id == target_id:
            return None
        accessibility = float(1.0 / (1.0 + max(float(barrier_proxy), 0.0) + max(int(evaluations), 0)))
        key = (source_id, target_id)
        if key not in self.edges:
            edge = TransitionEdge(
                source_id=source_id, target_id=target_id, evaluations=int(evaluations),
                best_improvement=float(improvement), barrier_proxy=float(barrier_proxy),
                accessibility=accessibility, created_nfe=int(nfe), last_updated_nfe=int(nfe),
                source_mode=source_mode
            )
            self.edges[key] = edge
            return edge
        edge = self.edges[key]
        edge.attempts += 1
        edge.evaluations += int(evaluations)
        edge.best_improvement = max(edge.best_improvement, float(improvement))
        edge.barrier_proxy = min(edge.barrier_proxy, float(barrier_proxy))
        edge.accessibility = max(edge.accessibility, accessibility)
        edge.last_updated_nfe = int(nfe)
        edge.source_mode = source_mode
        return edge
    def to_edges(self):
        return sorted(self.edges.values(), key=lambda e: (e.source_id, e.target_id))
