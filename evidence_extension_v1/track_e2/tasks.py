"""Deterministic synthetic tasks for Track E2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


FAMILIES = (
    "rotated_rastrigin",
    "rotated_ackley",
    "rotated_griewank",
    "gallagher_mixture",
    "double_funnel",
    "lunacek_birstrigin",
)
DIMENSIONS = (10, 20)
DEVELOPMENT_INSTANCES = tuple(range(1, 9))
CONFIRMATORY_INSTANCES = tuple(range(101, 109))


@dataclass(frozen=True)
class TaskSpec:
    family: str
    dimension: int
    instance: int
    lower: np.ndarray
    upper: np.ndarray
    objective: Callable[[np.ndarray], float]

    @property
    def task_id(self) -> str:
        return f"{self.family}_d{self.dimension}_i{self.instance}"


def _orthogonal_matrix(
    rng: np.random.Generator,
    dimension: int,
) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


def make_task(family: str, dimension: int, instance: int) -> TaskSpec:
    if family not in FAMILIES:
        raise KeyError(f"Unknown Track E2 family: {family}")
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unexpected Track E2 dimension: {dimension}")

    family_index = FAMILIES.index(family) + 1
    rng = np.random.default_rng(
        2_608_100 + 100_000 * family_index + 100 * dimension + instance
    )
    lower = -5.0 * np.ones(dimension)
    upper = 5.0 * np.ones(dimension)
    rotation = _orthogonal_matrix(rng, dimension)

    if family in {
        "rotated_rastrigin",
        "rotated_ackley",
        "rotated_griewank",
    }:
        shift = rng.uniform(-1.5, 1.5, size=dimension)

        def transformed(point: np.ndarray) -> np.ndarray:
            return rotation @ (np.asarray(point, dtype=float) - shift)

        if family == "rotated_rastrigin":

            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                value = 10.0 * dimension + np.sum(
                    z * z - 10.0 * np.cos(2.0 * np.pi * z)
                )
                return float(max(0.0, value))

        elif family == "rotated_ackley":

            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                value = (
                    -20.0 * np.exp(-0.2 * np.sqrt(np.mean(z * z)))
                    - np.exp(np.mean(np.cos(2.0 * np.pi * z)))
                    + 20.0
                    + math.e
                )
                return float(max(0.0, value))

        else:

            def objective(point: np.ndarray) -> float:
                z = transformed(point)
                indices = np.arange(1, dimension + 1, dtype=float)
                value = (
                    np.sum(z * z) / 4000.0
                    - np.prod(np.cos(z / np.sqrt(indices)))
                    + 1.0
                )
                return float(max(0.0, value))

    elif family == "gallagher_mixture":
        basin_count = 18
        centers = rng.uniform(-3.2, 3.2, size=(basin_count, dimension))
        centers[0] = rng.uniform(-1.5, 1.5, size=dimension)
        offsets = np.linspace(0.0, 1.5, basin_count)
        shuffled = offsets[1:].copy()
        rng.shuffle(shuffled)
        offsets[1:] = shuffled
        scales = np.exp(
            rng.uniform(
                np.log(0.25),
                np.log(1.7),
                size=(basin_count, dimension),
            )
        )
        scales[0] = rng.uniform(0.20, 0.45, size=dimension)

        def objective(point: np.ndarray) -> float:
            z = np.asarray(point, dtype=float)
            normalized = (z[None, :] - centers) / scales
            values = offsets + np.mean(normalized * normalized, axis=1)
            return float(max(0.0, np.min(values)))

    elif family == "double_funnel":
        global_center = rng.uniform(-2.0, 2.0, size=dimension)
        direction = rng.normal(size=dimension)
        direction /= max(np.linalg.norm(direction), 1e-12)
        local_center = np.clip(global_center + 4.5 * direction, -3.8, 3.8)
        global_rotation = rotation
        local_rotation = _orthogonal_matrix(rng, dimension)

        def objective(point: np.ndarray) -> float:
            z = np.asarray(point, dtype=float)
            global_delta = global_rotation @ (z - global_center)
            local_delta = local_rotation @ (z - local_center)
            global_value = (
                np.mean(global_delta * global_delta)
                + 0.035
                * np.mean(1.0 - np.cos(2.0 * np.pi * global_delta))
            )
            local_value = (
                0.18
                + 0.55 * np.mean(local_delta * local_delta)
                + 0.020
                * np.mean(1.0 - np.cos(2.0 * np.pi * local_delta))
            )
            return float(max(0.0, min(global_value, local_value)))

    else:  # lunacek_birstrigin
        shift = rng.uniform(-0.4, 0.4, size=dimension)
        mu_global = 0.70 * np.ones(dimension)
        mu_local = -0.70 * np.ones(dimension)

        def objective(point: np.ndarray) -> float:
            y = rotation @ (np.asarray(point, dtype=float) - shift)
            global_delta = y - mu_global
            local_delta = y - mu_local
            global_basin = np.mean(global_delta * global_delta)
            local_basin = 0.22 + 0.70 * np.mean(local_delta * local_delta)
            ripples = 4.0 * np.mean(
                1.0 - np.cos(2.0 * np.pi * global_delta)
            )
            return float(max(0.0, min(global_basin, local_basin) + ripples))

    return TaskSpec(
        family=family,
        dimension=dimension,
        instance=instance,
        lower=lower,
        upper=upper,
        objective=objective,
    )
