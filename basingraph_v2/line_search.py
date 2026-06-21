"""Multi-bracket bounded search along arbitrary feasible directions."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar

from .evaluation import (
    BudgetExhausted,
    EvaluationLedger,
    PhaseBudgetExhausted,
)


@dataclass
class DirectionCandidate:
    point: np.ndarray
    f_value: float
    direction: np.ndarray
    bracket: tuple[float, float]
    evaluations: int
    barrier_proxy: float
    local_support_count: int
    local_support_radius: float


def feasible_interval(
    start: np.ndarray,
    direction: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> tuple[float, float] | None:
    start = np.asarray(start, dtype=float)
    direction = np.asarray(direction, dtype=float)

    lower = -np.inf
    upper = np.inf

    for coordinate in range(len(start)):
        component = direction[coordinate]
        if abs(component) <= 1e-15:
            continue

        t1 = (
            lb[coordinate] - start[coordinate]
        ) / component
        t2 = (
            ub[coordinate] - start[coordinate]
        ) / component

        lower = max(lower, min(t1, t2))
        upper = min(upper, max(t1, t2))

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper - lower <= 1e-14
    ):
        return None

    return float(lower), float(upper)


def directional_multibracket_search(
    ledger: EvaluationLedger,
    start: np.ndarray,
    direction: np.ndarray,
    *,
    coarse_samples: int,
    refine_top_k: int,
    maxiter: int,
) -> list[DirectionCandidate]:
    start = ledger.project(start)
    direction = np.asarray(
        direction,
        dtype=float,
    ).reshape(-1)
    norm = np.linalg.norm(direction)
    if norm <= 1e-14:
        return []
    direction = direction / norm

    interval = feasible_interval(
        start,
        direction,
        ledger.lb,
        ledger.ub,
    )
    if interval is None:
        return []

    lower, upper = interval
    grid = np.linspace(
        lower,
        upper,
        max(5, int(coarse_samples)),
    )
    values: list[float] = []

    for scalar in grid:
        point = start + scalar * direction
        try:
            values.append(
                float(ledger.evaluate(point))
            )
        except (
            BudgetExhausted,
            PhaseBudgetExhausted,
        ):
            break

    if len(values) < 3:
        return []

    value_array = np.asarray(
        values,
        dtype=float,
    )
    grid = grid[: len(values)]

    brackets: list[
        tuple[float, float, float]
    ] = []
    for index in range(1, len(value_array) - 1):
        if (
            value_array[index]
            <= value_array[index - 1]
            and value_array[index]
            <= value_array[index + 1]
        ):
            brackets.append(
                (
                    float(value_array[index]),
                    float(grid[index - 1]),
                    float(grid[index + 1]),
                )
            )

    if not brackets:
        best_index = int(np.argmin(value_array))
        left = max(0, best_index - 1)
        right = min(
            len(value_array) - 1,
            best_index + 1,
        )
        brackets = [
            (
                float(value_array[best_index]),
                float(grid[left]),
                float(grid[right]),
            )
        ]

    brackets = sorted(
        brackets,
        key=lambda item: item[0],
    )[: max(1, int(refine_top_k))]

    output: list[DirectionCandidate] = []

    for _, a, b in brackets:
        if b - a <= 1e-14:
            continue

        before = ledger.nfe

        def objective(scalar: float) -> float:
            point = start + scalar * direction
            try:
                return ledger.evaluate(point)
            except (
                BudgetExhausted,
                PhaseBudgetExhausted,
            ):
                return float(
                    ledger.fbest + 1e100
                )

        try:
            result = minimize_scalar(
                objective,
                bounds=(a, b),
                method="bounded",
                options={
                    "maxiter": int(maxiter),
                    "xatol": max(
                        1e-12,
                        1e-9 * (b - a),
                    ),
                },
            )
        except Exception:
            continue

        point = ledger.project(
            start + float(result.x) * direction
        )
        output.append(
            DirectionCandidate(
                point=point,
                f_value=float(result.fun),
                direction=direction.copy(),
                bracket=(a, b),
                evaluations=int(
                    ledger.nfe - before
                ),
                barrier_proxy=float(
                    np.nanmax(value_array)
                    - np.nanmin(value_array)
                ),
                local_support_count=int(
                    len(value_array)
                    + ledger.nfe
                    - before
                ),
                local_support_radius=float(
                    abs(b - a)
                ),
            )
        )

    return output


def coordinate_multibracket_sweep(
    ledger: EvaluationLedger,
    start: np.ndarray,
    *,
    coarse_samples: int,
    refine_top_k: int,
    maxiter: int,
) -> list[DirectionCandidate]:
    output: list[DirectionCandidate] = []
    for coordinate in range(ledger.dimension):
        direction = np.zeros(
            ledger.dimension,
            dtype=float,
        )
        direction[coordinate] = 1.0
        output.extend(
            directional_multibracket_search(
                ledger,
                start,
                direction,
                coarse_samples=coarse_samples,
                refine_top_k=refine_top_k,
                maxiter=maxiter,
            )
        )
    return output
