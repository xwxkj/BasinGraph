"""Exact global, phase and local function-evaluation accounting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator
import numpy as np


class BudgetExhausted(RuntimeError):
    pass


class PhaseBudgetExhausted(RuntimeError):
    pass


class LocalBudgetExhausted(RuntimeError):
    pass


@dataclass
class EvaluationLedger:
    objective: Callable[[np.ndarray], float]
    lb: np.ndarray
    ub: np.ndarray
    max_evals: int
    nfe: int = 0
    xbest: np.ndarray | None = None
    fbest: float = np.inf
    history: list[tuple[int, float]] = field(default_factory=list)
    phase_evaluations: dict[str, int] = field(default_factory=dict)
    current_phase: str = "unassigned"
    current_phase_limit: int | None = None

    def __post_init__(self) -> None:
        self.lb = np.asarray(self.lb, dtype=float).reshape(-1)
        self.ub = np.asarray(self.ub, dtype=float).reshape(-1)
        self.max_evals = int(max(1, self.max_evals))

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.nfe)

    @property
    def dimension(self) -> int:
        return int(self.lb.size)

    def project(self, point: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(point, dtype=float).reshape(-1),
            self.lb,
            self.ub,
        )

    @contextmanager
    def phase(
        self,
        name: str,
        limit: int | None,
    ) -> Iterator[None]:
        previous_name = self.current_phase
        previous_limit = self.current_phase_limit
        self.current_phase = str(name)
        self.current_phase_limit = (
            None
            if limit is None
            else int(max(0, limit))
        )
        self.phase_evaluations.setdefault(
            self.current_phase,
            0,
        )
        try:
            yield
        finally:
            self.current_phase = previous_name
            self.current_phase_limit = previous_limit

    def evaluate(self, point: np.ndarray) -> float:
        if self.nfe >= self.max_evals:
            raise BudgetExhausted()

        used = self.phase_evaluations.get(
            self.current_phase,
            0,
        )
        if (
            self.current_phase_limit is not None
            and used >= self.current_phase_limit
        ):
            raise PhaseBudgetExhausted()

        projected = self.project(point)
        value = float(self.objective(projected))

        self.nfe += 1
        self.phase_evaluations[self.current_phase] = used + 1

        if np.isfinite(value) and value < self.fbest:
            self.fbest = value
            self.xbest = projected.copy()

        self.history.append((self.nfe, float(self.fbest)))
        return value
