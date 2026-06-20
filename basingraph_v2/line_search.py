"""Coarse-grid, multi-bracket coordinate refinement."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

from .evaluation import (
    EvaluationLedger,
    BudgetExhausted,
    PhaseBudgetExhausted,
)


@dataclass
class LineCandidate:
    x: np.ndarray
    f: float
    coordinate: int
    bracket: tuple[float, float]
    evaluations: int
    barrier_proxy: float


def coordinate_multibracket_sweep(
    ledger: EvaluationLedger,
    start: np.ndarray,
    *,
    coarse_samples: int,
    refine_top_k: int,
    maxiter: int,
    enable_multibracket: bool,
) -> list[LineCandidate]:
    x0 = ledger.project(start)
    output: list[LineCandidate] = []

    for coordinate in range(ledger.dimension):
        if ledger.remaining <= 0:
            break

        lo = float(ledger.lb[coordinate])
        hi = float(ledger.ub[coordinate])
        grid = np.linspace(lo, hi, max(5, int(coarse_samples)))
        values: list[float] = []

        for t in grid:
            point = x0.copy()
            point[coordinate] = float(t)
            try:
                values.append(float(ledger.evaluate(point)))
            except (BudgetExhausted, PhaseBudgetExhausted):
                break

        if len(values) < 3:
            continue

        values_array = np.asarray(values, dtype=float)
        brackets: list[tuple[float, float, float]] = []
        for i in range(1, len(values_array) - 1):
            if (
                values_array[i] <= values_array[i - 1]
                and values_array[i] <= values_array[i + 1]
            ):
                brackets.append(
                    (
                        float(values_array[i]),
                        float(grid[i - 1]),
                        float(grid[i + 1]),
                    )
                )

        if not brackets:
            best_index = int(np.argmin(values_array))
            left = max(0, best_index - 1)
            right = min(len(values_array) - 1, best_index + 1)
            brackets = [
                (
                    float(values_array[best_index]),
                    float(grid[left]),
                    float(grid[right]),
                )
            ]

        top_k = int(refine_top_k) if enable_multibracket else 1
        brackets = sorted(brackets, key=lambda item: item[0])[: max(1, top_k)]

        for _, a, b in brackets:
            if not b > a:
                continue
            before = ledger.nfe

            def phi(t: float) -> float:
                point = x0.copy()
                point[coordinate] = float(t)
                try:
                    return ledger.evaluate(point)
                except (BudgetExhausted, PhaseBudgetExhausted):
                    return float(ledger.fbest + 1e100)

            try:
                result = minimize_scalar(
                    phi,
                    bounds=(a, b),
                    method="bounded",
                    options={
                        "maxiter": int(maxiter),
                        "xatol": max(1e-12, 1e-9 * (b - a)),
                    },
                )
            except Exception:
                continue

            point = x0.copy()
            point[coordinate] = float(result.x)
            output.append(
                LineCandidate(
                    x=ledger.project(point),
                    f=float(result.fun),
                    coordinate=coordinate,
                    bracket=(a, b),
                    evaluations=int(ledger.nfe - before),
                    barrier_proxy=float(
                        np.nanmax(values_array) - np.nanmin(values_array)
                    ),
                )
            )

    return output
