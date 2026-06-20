
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np

class BudgetExhausted(RuntimeError):
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
    def __post_init__(self):
        self.lb = np.asarray(self.lb, dtype=float).reshape(-1)
        self.ub = np.asarray(self.ub, dtype=float).reshape(-1)
        self.max_evals = int(max(1, self.max_evals))
    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.nfe)
    @property
    def dimension(self) -> int:
        return int(self.lb.size)
    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float).reshape(-1), self.lb, self.ub)
    def evaluate(self, x: np.ndarray) -> float:
        if self.nfe >= self.max_evals:
            raise BudgetExhausted()
        z = self.project(x)
        val = float(self.objective(z))
        self.nfe += 1
        if np.isfinite(val) and val < self.fbest:
            self.fbest = val
            self.xbest = z.copy()
        self.history.append((self.nfe, float(self.fbest)))
        return val
