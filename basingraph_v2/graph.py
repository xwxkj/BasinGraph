"""Directed basin-transition graph and graph-guidance scores."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .types import BasinNode, TransitionEdge


@dataclass
class BasinTransitionGraph:
    edges: dict[tuple[int, int], TransitionEdge] = field(default_factory=dict)

    def remove_nodes(self, node_ids: list[int]) -> None:
        removed = set(int(i) for i in node_ids)
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if edge.source_id not in removed and edge.target_id not in removed
        }

    def add_or_update(
        self,
        source_id: int,
        target_id: int,
        *,
        evaluations: int,
        improvement: float,
        barrier_proxy: float,
        nfe: int,
        source_mode: str,
    ) -> TransitionEdge | None:
        if int(source_id) == int(target_id):
            return None

        barrier_proxy = float(max(barrier_proxy, 0.0))
        evaluations = int(max(evaluations, 0))
        accessibility = float(1.0 / (1.0 + barrier_proxy + evaluations))
        key = (int(source_id), int(target_id))

        if key not in self.edges:
            edge = TransitionEdge(
                source_id=int(source_id),
                target_id=int(target_id),
                evaluations=evaluations,
                best_improvement=float(improvement),
                barrier_proxy=barrier_proxy,
                accessibility=accessibility,
                created_nfe=int(nfe),
                last_updated_nfe=int(nfe),
                source_mode=str(source_mode),
            )
            self.edges[key] = edge
            return edge

        edge = self.edges[key]
        edge.attempts += 1
        edge.evaluations += evaluations
        edge.best_improvement = max(edge.best_improvement, float(improvement))
        edge.barrier_proxy = min(edge.barrier_proxy, barrier_proxy)
        edge.accessibility = max(edge.accessibility, accessibility)
        edge.last_updated_nfe = int(nfe)
        edge.source_mode = str(source_mode)
        return edge

    def incoming_accessibility(self, node_id: int) -> float:
        values = [
            edge.accessibility
            for edge in self.edges.values()
            if edge.target_id == int(node_id)
        ]
        return float(max(values)) if values else 0.5

    def guidance_scores(
        self,
        nodes: list[BasinNode],
        *,
        quality_weight: float,
        novelty_weight: float,
        accessibility_weight: float,
    ) -> dict[int, float]:
        if not nodes:
            return {}

        f = np.asarray([node.f_center for node in nodes], dtype=float)
        order = np.argsort(np.argsort(f, kind="stable"), kind="stable")
        if len(nodes) == 1:
            quality = np.ones(1)
        else:
            quality = 1.0 - order / (len(nodes) - 1)

        scores: dict[int, float] = {}
        for idx, node in enumerate(nodes):
            accessibility = self.incoming_accessibility(node.node_id)
            score = (
                quality_weight * float(quality[idx])
                + novelty_weight * float(node.novelty)
                + accessibility_weight * float(accessibility)
            )
            scores[node.node_id] = float(score)
        return scores

    def to_edges(self) -> list[TransitionEdge]:
        return sorted(
            self.edges.values(),
            key=lambda edge: (edge.source_id, edge.target_id),
        )
