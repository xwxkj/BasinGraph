"""Sparse directed transition graph for certified basin nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
import numpy as np

from .types import BasinNode, TransitionEdge


@dataclass
class BasinTransitionGraph:
    max_outgoing: int = 3
    max_incoming: int = 3
    improvement_weight: float = 0.45
    accessibility_weight: float = 0.35
    recency_weight: float = 0.20
    edges: dict[tuple[int, int], TransitionEdge] = field(
        default_factory=dict
    )

    def remove_nodes(self, node_ids: list[int]) -> None:
        removed = set(int(node_id) for node_id in node_ids)
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if (
                edge.source_id not in removed
                and edge.target_id not in removed
            )
        }

    def accessibility_map(self) -> dict[int, float]:
        node_ids = {
            node_id
            for edge in self.edges.values()
            for node_id in (
                edge.source_id,
                edge.target_id,
            )
        }
        return {
            node_id: self.incoming_accessibility(node_id)
            for node_id in node_ids
        }

    def incoming_accessibility(self, node_id: int) -> float:
        values = [
            edge.accessibility
            for edge in self.edges.values()
            if edge.target_id == int(node_id)
        ]
        return float(max(values)) if values else 0.5

    def _edge_score(
        self,
        edge: TransitionEdge,
        current_nfe: int,
    ) -> float:
        improvement = max(edge.best_improvement, 0.0)
        improvement_score = improvement / (1.0 + improvement)
        recency = 1.0 / (
            1.0
            + max(
                int(current_nfe) - edge.last_updated_nfe,
                0,
            )
        )
        return float(
            self.improvement_weight * improvement_score
            + self.accessibility_weight * edge.accessibility
            + self.recency_weight * recency
        )

    def prune(
        self,
        *,
        active_node_ids: set[int],
        current_nfe: int,
    ) -> None:
        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if (
                edge.source_id in active_node_ids
                and edge.target_id in active_node_ids
            )
        }

        outgoing: dict[int, list[TransitionEdge]] = {}
        for edge in self.edges.values():
            outgoing.setdefault(edge.source_id, []).append(edge)

        keep_keys: set[tuple[int, int]] = set()
        for source_id, candidates in outgoing.items():
            retained = sorted(
                candidates,
                key=lambda edge: (
                    self._edge_score(edge, current_nfe),
                    edge.best_improvement,
                    edge.accessibility,
                    -edge.target_id,
                ),
                reverse=True,
            )[: self.max_outgoing]
            keep_keys.update(
                (edge.source_id, edge.target_id)
                for edge in retained
            )

        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if key in keep_keys
        }

        incoming: dict[int, list[TransitionEdge]] = {}
        for edge in self.edges.values():
            incoming.setdefault(edge.target_id, []).append(edge)

        keep_keys = set()
        for target_id, candidates in incoming.items():
            retained = sorted(
                candidates,
                key=lambda edge: (
                    self._edge_score(edge, current_nfe),
                    edge.best_improvement,
                    edge.accessibility,
                    -edge.source_id,
                ),
                reverse=True,
            )[: self.max_incoming]
            keep_keys.update(
                (edge.source_id, edge.target_id)
                for edge in retained
            )

        self.edges = {
            key: edge
            for key, edge in self.edges.items()
            if key in keep_keys
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
        active_node_ids: set[int],
    ) -> TransitionEdge | None:
        source_id = int(source_id)
        target_id = int(target_id)

        if (
            source_id == target_id
            or source_id not in active_node_ids
            or target_id not in active_node_ids
        ):
            return None

        evaluations = int(max(evaluations, 0))
        barrier_proxy = float(max(barrier_proxy, 0.0))
        accessibility = float(
            1.0
            / (
                1.0
                + barrier_proxy
                + evaluations
            )
        )
        key = (source_id, target_id)

        if key not in self.edges:
            edge = TransitionEdge(
                source_id=source_id,
                target_id=target_id,
                evaluations=evaluations,
                best_improvement=float(improvement),
                barrier_proxy=barrier_proxy,
                accessibility=accessibility,
                created_nfe=int(nfe),
                last_updated_nfe=int(nfe),
                attempts=1,
                source_mode=str(source_mode),
            )
            self.edges[key] = edge
        else:
            edge = self.edges[key]
            edge.attempts += 1
            edge.evaluations += evaluations
            edge.best_improvement = max(
                edge.best_improvement,
                float(improvement),
            )
            edge.barrier_proxy = min(
                edge.barrier_proxy,
                barrier_proxy,
            )
            edge.accessibility = max(
                edge.accessibility,
                accessibility,
            )
            edge.last_updated_nfe = int(nfe)
            edge.source_mode = str(source_mode)

        self.prune(
            active_node_ids=active_node_ids,
            current_nfe=nfe,
        )
        return self.edges.get(key)

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

        values = np.asarray(
            [node.f_center for node in nodes],
            dtype=float,
        )
        order = np.argsort(
            np.argsort(values, kind="stable"),
            kind="stable",
        )
        if len(nodes) == 1:
            quality = np.ones(1)
        else:
            quality = 1.0 - order / (len(nodes) - 1)

        return {
            node.node_id: float(
                quality_weight * quality[index]
                + novelty_weight * node.novelty
                + accessibility_weight
                * self.incoming_accessibility(node.node_id)
            )
            for index, node in enumerate(nodes)
        }

    def successful_transition_directions(
        self,
        nodes_by_id: Mapping[int, BasinNode],
    ) -> list[tuple[float, np.ndarray]]:
        directions = []
        for edge in self.edges.values():
            if edge.best_improvement <= 0:
                continue
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None:
                continue
            direction = target.center - source.center
            norm = np.linalg.norm(direction)
            if norm <= 1e-14:
                continue
            score = float(
                edge.best_improvement
                * edge.accessibility
            )
            directions.append((score, direction / norm))
        return sorted(
            directions,
            key=lambda item: item[0],
            reverse=True,
        )

    def degree_statistics(self) -> dict[str, int]:
        outgoing: dict[int, int] = {}
        incoming: dict[int, int] = {}
        for edge in self.edges.values():
            outgoing[edge.source_id] = (
                outgoing.get(edge.source_id, 0) + 1
            )
            incoming[edge.target_id] = (
                incoming.get(edge.target_id, 0) + 1
            )
        return {
            "max_outgoing": max(outgoing.values(), default=0),
            "max_incoming": max(incoming.values(), default=0),
        }

    def to_edges(self) -> list[TransitionEdge]:
        return sorted(
            self.edges.values(),
            key=lambda edge: (
                edge.source_id,
                edge.target_id,
            ),
        )
