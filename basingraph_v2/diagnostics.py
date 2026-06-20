
from __future__ import annotations
import numpy as np
from .evaluation import EvaluationLedger
from .types import GeometryDiagnostics

def lhs_sample(n, dim, lb, ub, rng):
    X = np.empty((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return lb + X * (ub - lb)

def make_initial_anchors(lb, ub, rng, n_lhs):
    dim = lb.size
    center = 0.5 * (lb + ub)
    anchors = [center, lb.copy(), ub.copy()]
    alt1, alt2 = lb.copy(), ub.copy()
    alt1[1::2] = ub[1::2]
    alt2[1::2] = lb[1::2]
    anchors.extend([alt1, alt2])
    for j in range(min(dim, 8)):
        xlo, xhi = center.copy(), center.copy()
        xlo[j], xhi[j] = lb[j], ub[j]
        anchors.extend([xlo, xhi])
    if n_lhs > 0:
        anchors.extend([x.copy() for x in lhs_sample(n_lhs, dim, lb, ub, rng)])
    out, seen = [], []
    scale = np.linalg.norm(ub - lb) + 1e-300
    for x in anchors:
        if not any(np.linalg.norm(x-y)/scale <= 1e-12 for y in seen):
            out.append(x)
            seen.append(x)
    return out

def evaluate_diagnostics(ledger: EvaluationLedger, anchors, anchor_values) -> GeometryDiagnostics:
    lb, ub = ledger.lb, ledger.ub
    dim = ledger.dimension
    scale = ub - lb
    finite = np.asarray([v for v in anchor_values if np.isfinite(v)], dtype=float)
    finite_fraction = len(finite) / max(1, len(anchor_values))
    mean_scale = float(np.mean(scale))
    max_scale = float(np.max(scale))
    anisotropy = float(np.max(scale) / max(np.min(scale), 1e-300))
    center = 0.5 * (lb + ub)
    boundary, interior = [], []
    for x, val in zip(anchors, anchor_values):
        if not np.isfinite(val): continue
        normalized = np.abs((x-center)/np.maximum(scale, 1e-300))
        (boundary if np.max(normalized) > 0.45 else interior).append(val)
    if boundary and interior and len(finite):
        boundary_signal = float((np.median(interior) - np.median(boundary)) / (1.0 + abs(np.median(finite))))
    else:
        boundary_signal = 0.0
    if len(finite) >= 4:
        q25, q75 = np.quantile(finite, [0.25, 0.75])
        ruggedness = float((q75-q25) / (1.0 + abs(np.median(finite))))
    else:
        ruggedness = 0.0
    if len(finite) >= 3:
        diffs = np.diff(np.sort(finite))
        signs = np.sign(diffs)
        sign_changes = np.sum(signs[1:] * signs[:-1] < 0)
        sign_rate = float(sign_changes / max(1, len(signs)-1))
    else:
        sign_rate = 0.0
    return GeometryDiagnostics(dim, mean_scale, max_scale, anisotropy, boundary_signal, ruggedness, sign_rate, finite_fraction)
