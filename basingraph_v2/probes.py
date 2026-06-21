"""Bounded probe pool used by BasinGraph v2.0.0-rc2."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np

from .types import BasinNode, ProbeRecord


@dataclass
class ProbePool:
    max_records: int = 256
    records: list[ProbeRecord] = field(default_factory=list)
    total_created: int = 0
    _counter: itertools.count = field(
        default_factory=lambda: itertools.count(1)
    )

    def _novelty(
        self,
        point: np.ndarray,
        nodes: list[BasinNode],
        domain_radius: float,
    ) -> float:
        if not nodes:
            return 1.0
        distances = [
            np.linalg.norm(point - node.center)
            for node in nodes
        ]
        return float(
            min(
                1.0,
                min(distances) / max(domain_radius, 1e-300),
            )
        )

    def add(
        self,
        point: np.ndarray,
        f_value: float,
        *,
        source_phase: str,
        created_nfe: int,
        parent_node_id: int | None,
        nodes: list[BasinNode],
        domain_radius: float,
    ) -> ProbeRecord:
        point = np.asarray(point, dtype=float).reshape(-1)
        record = ProbeRecord(
            probe_id=next(self._counter),
            point=point.copy(),
            f_value=float(f_value),
            source_phase=str(source_phase),
            created_nfe=int(created_nfe),
            parent_node_id=parent_node_id,
            novelty=self._novelty(
                point,
                nodes,
                domain_radius,
            ),
        )
        self.records.append(record)
        self.total_created += 1
        self._prune()
        return record

    def _prune(self) -> None:
        while len(self.records) > self.max_records:
            disposable = [
                record
                for record in self.records
                if (
                    not record.selected_for_refinement
                    and record.committed_node_id is None
                )
            ]
            candidates = disposable or list(self.records)
            victim = max(
                candidates,
                key=lambda record: (
                    record.f_value,
                    -record.novelty,
                    -record.created_nfe,
                ),
            )
            self.records.remove(victim)

    def select(
        self,
        candidates: list[ProbeRecord],
        count: int,
        *,
        quality_weight: float,
        novelty_weight: float,
    ) -> list[ProbeRecord]:
        if not candidates or count <= 0:
            return []

        finite_values = np.asarray(
            [record.f_value for record in candidates],
            dtype=float,
        )
        order = np.argsort(
            np.argsort(finite_values, kind="stable"),
            kind="stable",
        )
        if len(candidates) == 1:
            quality = np.ones(1)
        else:
            quality = 1.0 - order / (len(candidates) - 1)

        for index, record in enumerate(candidates):
            record.quality_score = float(
                quality_weight * quality[index]
                + novelty_weight * record.novelty
            )

        selected = sorted(
            candidates,
            key=lambda record: (
                record.quality_score,
                -record.f_value,
                record.novelty,
            ),
            reverse=True,
        )[:count]

        for record in selected:
            record.selected_for_refinement = True

        return selected

    def best_uncommitted(self, count: int) -> list[ProbeRecord]:
        candidates = [
            record
            for record in self.records
            if record.committed_node_id is None
        ]
        return sorted(
            candidates,
            key=lambda record: (
                record.f_value,
                -record.novelty,
                record.created_nfe,
            ),
        )[:count]
