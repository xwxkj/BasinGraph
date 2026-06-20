"""Deterministic landscape diagnostics used by the v2 geometry controller."""

from __future__ import annotations

import numpy as np

from .types import GeometryDiagnostics


def lhs_sample(
    n: int,
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    x = np.empty((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        x[:, j] = (perm + rng.random(n)) / n
    return lb + x * (ub - lb)


def make_initial_anchors(
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
    n_lhs: int,
) -> list[np.ndarray]:
    dim = lb.size
    center = 0.5 * (lb + ub)
    anchors = [center, lb.copy(), ub.copy()]

    alternating_a = lb.copy()
    alternating_b = ub.copy()
    alternating_a[1::2] = ub[1::2]
    alternating_b[1::2] = lb[1::2]
    anchors.extend([alternating_a, alternating_b])

    # Exact coordinate triplets: lower endpoint, center, upper endpoint.
    for j in range(dim):
        xlo = center.copy()
        xhi = center.copy()
        xlo[j] = lb[j]
        xhi[j] = ub[j]
        anchors.extend([xlo, xhi])

    if n_lhs > 0:
        anchors.extend(
            [row.copy() for row in lhs_sample(n_lhs, dim, lb, ub, rng)]
        )

    out: list[np.ndarray] = []
    scale = np.linalg.norm(ub - lb) + 1e-300
    for candidate in anchors:
        if not any(
            np.linalg.norm(candidate - existing) / scale <= 1e-12
            for existing in out
        ):
            out.append(candidate)
    return out


def _find_value(
    anchors: list[np.ndarray],
    values: list[float],
    target: np.ndarray,
    tolerance: float,
) -> float | None:
    for x, value in zip(anchors, values):
        if np.linalg.norm(np.asarray(x) - target) <= tolerance:
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
    scale = ub - lb
    dim = len(lb)
    tolerance = 1e-11 * (np.linalg.norm(scale) + 1.0)

    finite_values = np.asarray(
        [value for value in anchor_values if np.isfinite(value)],
        dtype=float,
    )
    finite_fraction = len(finite_values) / max(1, len(anchor_values))

    mean_scale = float(np.mean(scale))
    max_scale = float(np.max(scale))
    anisotropy = float(np.max(scale) / max(np.min(scale), 1e-300))

    center_value = _find_value(anchors, anchor_values, center, tolerance)
    sign_changes = 0
    valid_triplets = 0

    if center_value is not None and np.isfinite(center_value):
        for j in range(dim):
            xlo = center.copy()
            xhi = center.copy()
            xlo[j] = lb[j]
            xhi[j] = ub[j]

            flo = _find_value(anchors, anchor_values, xlo, tolerance)
            fhi = _find_value(anchors, anchor_values, xhi, tolerance)

            if flo is None or fhi is None:
                continue
            if not (np.isfinite(flo) and np.isfinite(fhi)):
                continue

            left_den = max(center[j] - lb[j], 1e-300)
            right_den = max(ub[j] - center[j], 1e-300)
            slope_left = (center_value - flo) / left_den
            slope_right = (fhi - center_value) / right_den

            valid_triplets += 1
            if slope_left * slope_right < 0:
                sign_changes += 1

    sign_change_rate = (
        float(sign_changes / valid_triplets) if valid_triplets else 0.0
    )

    boundary_values = []
    interior_values = []
    for x, value in zip(anchors, anchor_values):
        if not np.isfinite(value):
            continue
        normalized = np.abs((np.asarray(x) - center) / np.maximum(scale, 1e-300))
        if np.max(normalized) >= 0.49:
            boundary_values.append(value)
        else:
            interior_values.append(value)

    if boundary_values and interior_values:
        boundary_signal = float(
            (np.median(interior_values) - np.median(boundary_values))
            / (1.0 + abs(np.median(finite_values)))
        )
    else:
        boundary_signal = 0.0

    if len(finite_values) >= 4:
        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)))
        ruggedness = mad / (1.0 + abs(median))
    else:
        ruggedness = 0.0

    return GeometryDiagnostics(
        dimension=dim,
        mean_scale=mean_scale,
        max_scale=max_scale,
        anisotropy=anisotropy,
        boundary_signal=boundary_signal,
        ruggedness_score=float(ruggedness),
        sign_change_rate=sign_change_rate,
        finite_anchor_fraction=float(finite_fraction),
        valid_axis_triplets=int(valid_triplets),
    )
