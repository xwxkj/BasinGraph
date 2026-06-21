"""Derivative-free geometry diagnostics for BasinGraph v2.0.0-rc2."""

from __future__ import annotations

import numpy as np

from .types import GeometryDiagnostics


def lhs_sample(
    count: int,
    dimension: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    sample = np.empty((count, dimension), dtype=float)
    for coordinate in range(dimension):
        permutation = rng.permutation(count)
        sample[:, coordinate] = (
            permutation + rng.random(count)
        ) / count
    return lb + sample * (ub - lb)


def make_initial_anchors(
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    n_lhs: int,
) -> list[np.ndarray]:
    dimension = len(lb)
    center = 0.5 * (lb + ub)
    anchors = [
        center.copy(),
        lb.copy(),
        ub.copy(),
    ]

    alternating_a = lb.copy()
    alternating_b = ub.copy()
    alternating_a[1::2] = ub[1::2]
    alternating_b[1::2] = lb[1::2]
    anchors.extend(
        [
            alternating_a,
            alternating_b,
        ]
    )

    for coordinate in range(dimension):
        lower = center.copy()
        upper = center.copy()
        lower[coordinate] = lb[coordinate]
        upper[coordinate] = ub[coordinate]
        anchors.extend([lower, upper])

    if n_lhs > 0:
        anchors.extend(
            row.copy()
            for row in lhs_sample(
                n_lhs,
                dimension,
                lb,
                ub,
                rng,
            )
        )

    unique: list[np.ndarray] = []
    scale = np.linalg.norm(ub - lb) + 1e-300
    for candidate in anchors:
        if not any(
            np.linalg.norm(candidate - existing)
            / scale
            <= 1e-12
            for existing in unique
        ):
            unique.append(candidate)

    return unique


def _find_value(
    anchors: list[np.ndarray],
    values: list[float],
    target: np.ndarray,
    tolerance: float,
) -> float | None:
    for point, value in zip(anchors, values):
        if (
            np.linalg.norm(
                np.asarray(point) - target
            )
            <= tolerance
        ):
            return float(value)
    return None


def evaluate_diagnostics(
    *,
    lb: np.ndarray,
    ub: np.ndarray,
    anchors: list[np.ndarray],
    anchor_values: list[float],
) -> GeometryDiagnostics:
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    center = 0.5 * (lb + ub)
    widths = ub - lb
    dimension = len(lb)
    tolerance = 1e-11 * (
        np.linalg.norm(widths) + 1.0
    )

    finite_values = np.asarray(
        [
            value
            for value in anchor_values
            if np.isfinite(value)
        ],
        dtype=float,
    )
    finite_fraction = (
        len(finite_values)
        / max(len(anchor_values), 1)
    )

    mean_scale = float(np.mean(widths))
    max_scale = float(np.max(widths))
    domain_anisotropy = float(
        np.max(widths)
        / max(np.min(widths), 1e-300)
    )

    center_value = _find_value(
        anchors,
        anchor_values,
        center,
        tolerance,
    )

    sign_changes = 0
    valid_triplets = 0
    curvature_values: list[float] = []

    if (
        center_value is not None
        and np.isfinite(center_value)
    ):
        for coordinate in range(dimension):
            lower = center.copy()
            upper = center.copy()
            lower[coordinate] = lb[coordinate]
            upper[coordinate] = ub[coordinate]

            f_lower = _find_value(
                anchors,
                anchor_values,
                lower,
                tolerance,
            )
            f_upper = _find_value(
                anchors,
                anchor_values,
                upper,
                tolerance,
            )

            if (
                f_lower is None
                or f_upper is None
                or not np.isfinite(f_lower)
                or not np.isfinite(f_upper)
            ):
                continue

            h = max(
                0.5 * widths[coordinate],
                1e-300,
            )
            slope_left = (
                center_value - f_lower
            ) / h
            slope_right = (
                f_upper - center_value
            ) / h
            curvature = abs(
                f_lower
                - 2.0 * center_value
                + f_upper
            ) / (h * h)

            valid_triplets += 1
            curvature_values.append(float(curvature))

            if slope_left * slope_right < 0:
                sign_changes += 1

    sign_change_rate = (
        float(sign_changes / valid_triplets)
        if valid_triplets
        else 0.0
    )

    if curvature_values:
        curvature_array = np.asarray(
            curvature_values,
            dtype=float,
        )
        positive = curvature_array[
            curvature_array > 1e-18
        ]
        if len(positive) >= 2:
            q10, q90 = np.quantile(
                positive,
                [0.10, 0.90],
            )
            curvature_anisotropy = float(
                min(
                    1e6,
                    q90 / max(q10, 1e-18),
                )
            )
        else:
            curvature_anisotropy = 1.0
    else:
        curvature_anisotropy = 1.0

    boundary_values = []
    interior_values = []
    for point, value in zip(anchors, anchor_values):
        if not np.isfinite(value):
            continue
        normalized = np.abs(
            (np.asarray(point) - center)
            / np.maximum(widths, 1e-300)
        )
        if np.max(normalized) >= 0.49:
            boundary_values.append(value)
        else:
            interior_values.append(value)

    if (
        boundary_values
        and interior_values
        and len(finite_values)
    ):
        boundary_signal = float(
            (
                np.median(interior_values)
                - np.median(boundary_values)
            )
            / (
                1.0
                + abs(np.median(finite_values))
            )
        )
    else:
        boundary_signal = 0.0

    if len(finite_values) >= 4:
        median = float(np.median(finite_values))
        mad = float(
            np.median(
                np.abs(finite_values - median)
            )
        )
        ruggedness = float(
            mad / (1.0 + abs(median))
        )
    else:
        ruggedness = 0.0

    return GeometryDiagnostics(
        dimension=dimension,
        mean_scale=mean_scale,
        max_scale=max_scale,
        domain_anisotropy=domain_anisotropy,
        curvature_anisotropy=curvature_anisotropy,
        curvature_values=curvature_values,
        boundary_signal=boundary_signal,
        ruggedness_score=ruggedness,
        sign_change_rate=sign_change_rate,
        finite_anchor_fraction=float(finite_fraction),
        valid_axis_triplets=int(valid_triplets),
    )
