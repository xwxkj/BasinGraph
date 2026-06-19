"""
BasinGraph optimizer: Python official-validation implementation.

This implementation is designed for official COCO/BBOB and PyCUTEst validation.
It follows the manuscript-level algorithmic structure:

    geometry controller
    + center-local contraction
    + coordinate-Brent basin sweep
    + far-basin sweep
    + archive fallback
    + final polishing

All function evaluations are counted explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize, minimize_scalar


class BudgetExhausted(RuntimeError):
    """Raised internally when the evaluation budget is exhausted."""
    pass


@dataclass
class BasinGraphOptions:
    # Archive and initialization
    n_initial_lhs: int = 24
    archive_max_size: int = 32

    # Geometry controller
    center_local_max_dim: int = 20
    large_box_scale: float = 100.0

    # Coordinate basin sweep
    coordinate_sweep_starts: int = 6
    coordinate_sweep_cycles: int = 2
    coordinate_line_maxiter: int = 18

    # Local contraction / polishing
    center_local_fraction: float = 0.25
    final_polish_fraction: float = 0.25
    final_polish_starts: int = 5

    # Far-basin exploration
    far_basin_fraction: float = 0.15
    far_basin_random: int = 24

    # Numerical safeguards
    local_ftol: float = 1e-12
    local_gtol: float = 1e-8
    eps_dedup: float = 1e-10


def _as_array(x, dtype=float) -> np.ndarray:
    return np.asarray(x, dtype=dtype).reshape(-1)


def _finite_bounds(lb: np.ndarray, ub: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    lb = _as_array(lb)
    ub = _as_array(ub)

    if lb.size != dim:
        lb = -5.0 * np.ones(dim)
    if ub.size != dim:
        ub = 5.0 * np.ones(dim)

    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)

    # Avoid zero-width boxes.
    bad = ub <= lb
    if np.any(bad):
        lb[bad] = -5.0
        ub[bad] = 5.0

    return lb.astype(float), ub.astype(float)


def _lhs(n: int, dim: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simple Latin hypercube sampling."""
    if n <= 0:
        return np.empty((0, dim))

    X = np.empty((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return lb + X * (ub - lb)


def _deduplicate(points: List[np.ndarray], lb: np.ndarray, ub: np.ndarray, eps: float) -> List[np.ndarray]:
    """Remove near-duplicate points under normalized Euclidean distance."""
    out: List[np.ndarray] = []
    scale = np.linalg.norm(ub - lb) + 1e-300
    for x in points:
        x = np.clip(_as_array(x), lb, ub)
        duplicate = False
        for y in out:
            if np.linalg.norm(x - y) / scale <= eps:
                duplicate = True
                break
        if not duplicate:
            out.append(x)
    return out


def _make_anchors(dim: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator,
                  n_lhs: int) -> List[np.ndarray]:
    """Construct deterministic and randomized anchors."""
    center = 0.5 * (lb + ub)

    anchors: List[np.ndarray] = [
        center,
        lb.copy(),
        ub.copy(),
    ]

    # Alternating corners. These are useful for far-basin landscapes.
    alt1 = lb.copy()
    alt1[1::2] = ub[1::2]
    alt2 = ub.copy()
    alt2[1::2] = lb[1::2]
    anchors.extend([alt1, alt2])

    # Axis-midpoint anchors.
    for j in range(min(dim, 10)):
        x1 = center.copy()
        x2 = center.copy()
        x1[j] = lb[j]
        x2[j] = ub[j]
        anchors.extend([x1, x2])

    X = _lhs(n_lhs, dim, lb, ub, rng)
    anchors.extend([X[i, :] for i in range(X.shape[0])])

    return _deduplicate(anchors, lb, ub, eps=1e-12)


def minimize_basingraph(
    objective: Callable[[np.ndarray], float],
    lb,
    ub,
    max_evals: int,
    seed: int = 0,
    options: BasinGraphOptions | None = None,
) -> Dict[str, Any]:
    """
    Minimize a bound-constrained black-box objective.

    Parameters
    ----------
    objective:
        Callable f(x). The caller is responsible for connecting this to COCO/CUTEst.
    lb, ub:
        Lower and upper bound vectors.
    max_evals:
        Maximum number of objective evaluations.
    seed:
        Random seed.
    options:
        BasinGraphOptions.

    Returns
    -------
    dict with xbest, fbest, nfe, history and message.
    """
    if options is None:
        options = BasinGraphOptions()

    rng = np.random.default_rng(seed)

    lb = _as_array(lb)
    ub = _as_array(ub)
    dim = lb.size
    lb, ub = _finite_bounds(lb, ub, dim)

    max_evals = int(max(1, max_evals))
    nfe = 0

    best_x = None
    best_f = np.inf
    archive: List[Tuple[float, np.ndarray]] = []
    history: List[Tuple[int, float]] = []

    def eval_x(x: np.ndarray) -> float:
        nonlocal nfe, best_x, best_f, archive, history

        if nfe >= max_evals:
            raise BudgetExhausted()

        z = np.clip(_as_array(x), lb, ub)
        val = float(objective(z))
        nfe += 1

        if np.isfinite(val):
            if val < best_f:
                best_f = val
                best_x = z.copy()
            archive.append((val, z.copy()))
            archive = sorted(archive, key=lambda t: t[0])[: max(4, options.archive_max_size)]
            history.append((nfe, best_f))

        return val

    def remaining() -> int:
        return max(0, max_evals - nfe)

    def local_polish(x0: np.ndarray, maxfun: int) -> None:
        """Budget-aware L-BFGS-B polishing."""
        if maxfun <= max(4, 2 * dim) or remaining() <= 0:
            return

        def fun(z):
            return eval_x(z)

        try:
            minimize(
                fun,
                np.clip(x0, lb, ub),
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={
                    "maxfun": int(maxfun),
                    "maxiter": int(maxfun),
                    "ftol": options.local_ftol,
                    "gtol": options.local_gtol,
                    "disp": False,
                },
            )
        except BudgetExhausted:
            return
        except Exception:
            # Solver failures should not terminate black-box validation.
            return

    def coordinate_brent_sweep(starts: List[np.ndarray], cycles: int) -> None:
        """Coordinate-wise bounded one-dimensional basin sweep."""
        if remaining() <= 0:
            return

        for x0 in starts:
            if remaining() <= 0:
                break

            x = np.clip(x0.copy(), lb, ub)

            for _ in range(cycles):
                improved = False

                for j in range(dim):
                    if remaining() <= 0:
                        break

                    lo = float(lb[j])
                    hi = float(ub[j])
                    if not np.isfinite(lo + hi) or hi <= lo:
                        continue

                    def phi(t):
                        z = x.copy()
                        z[j] = float(t)
                        try:
                            return eval_x(z)
                        except BudgetExhausted:
                            return best_f + 1e100

                    maxiter = int(min(options.coordinate_line_maxiter, max(3, remaining())))
                    try:
                        res = minimize_scalar(
                            phi,
                            bounds=(lo, hi),
                            method="bounded",
                            options={
                                "maxiter": maxiter,
                                "xatol": max(1e-12, 1e-9 * (hi - lo)),
                            },
                        )
                        if np.isfinite(res.fun) and res.fun <= best_f + 1e-14:
                            x[j] = float(res.x)
                            improved = True
                    except Exception:
                        continue

                if not improved:
                    break

    # ------------------------------------------------------------
    # 1. Initial anchor coverage
    # ------------------------------------------------------------
    anchors = _make_anchors(
        dim=dim,
        lb=lb,
        ub=ub,
        rng=rng,
        n_lhs=min(options.n_initial_lhs + dim, max(4, max_evals // 10)),
    )

    for x in anchors:
        if remaining() <= 0:
            break
        try:
            eval_x(x)
        except BudgetExhausted:
            break

    if best_x is None:
        # This should only happen for max_evals=0, but keep a safeguard.
        best_x = 0.5 * (lb + ub)
        best_f = np.inf

    # ------------------------------------------------------------
    # 2. Geometry controller
    # ------------------------------------------------------------
    box_scale = float(np.max(ub - lb))
    smooth_local_mode = dim <= options.center_local_max_dim
    far_basin_mode = box_scale >= options.large_box_scale

    # ------------------------------------------------------------
    # 3. Center-local contraction
    # ------------------------------------------------------------
    if smooth_local_mode and remaining() > 0:
        center_budget = int(options.center_local_fraction * max_evals)
        local_polish(0.5 * (lb + ub), min(center_budget, remaining()))

    # ------------------------------------------------------------
    # 4. Far-basin anchors and rugged escape probes
    # ------------------------------------------------------------
    if far_basin_mode and remaining() > 0:
        far_budget = int(options.far_basin_fraction * max_evals)

        far_points = [lb.copy(), ub.copy()]
        for _ in range(min(options.far_basin_random, far_budget)):
            if remaining() <= 0:
                break

            # Heavy-tailed random direction around the center.
            center = 0.5 * (lb + ub)
            direction = rng.standard_t(df=1.5, size=dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm

            radius = rng.random() ** 0.25
            x = center + radius * 0.5 * np.linalg.norm(ub - lb) * direction
            far_points.append(np.clip(x, lb, ub))

        for x in _deduplicate(far_points, lb, ub, eps=1e-10):
            if remaining() <= 0:
                break
            try:
                eval_x(x)
            except BudgetExhausted:
                break

    # ------------------------------------------------------------
    # 5. Coordinate-Brent basin sweep
    # ------------------------------------------------------------
    starts = [x for _, x in sorted(archive, key=lambda t: t[0])[: options.coordinate_sweep_starts]]
    if len(starts) > 0 and remaining() > 0:
        coordinate_brent_sweep(starts, cycles=options.coordinate_sweep_cycles)

    # ------------------------------------------------------------
    # 6. Archive fallback: refresh starts from best diverse archive points
    # ------------------------------------------------------------
    starts = [x for _, x in sorted(archive, key=lambda t: t[0])[: options.final_polish_starts]]

    # ------------------------------------------------------------
    # 7. Final elite polishing
    # ------------------------------------------------------------
    if remaining() > 0 and len(starts) > 0:
        per_start_budget = max(5, int(options.final_polish_fraction * max_evals / max(1, len(starts))))
        for x in starts:
            if remaining() <= 0:
                break
            local_polish(x, min(per_start_budget, remaining()))

    # ------------------------------------------------------------
    # 8. Budget-completion fallback
    # ------------------------------------------------------------
    # Official COCO/BBOB validation rewards algorithms that keep
    # improving toward target values under the prescribed FE budget.
    # Earlier development versions of BasinGraph could terminate after
    # the main basin-discovery phases. For final official validation,
    # we spend the remaining budget on archive-guided and global probes,
    # with occasional short local polishing calls.
    if remaining() > 0:
        stall_count = 0

        while remaining() > 0:
            previous_best = best_f

            try:
                # Prefer archive-guided probes, but retain global coverage.
                if len(archive) > 0 and rng.random() < 0.70:
                    elite_count = min(len(archive), 8)
                    idx = int(rng.integers(elite_count))
                    _, base = sorted(archive, key=lambda t: t[0])[idx]

                    # Heavy-tailed local/global perturbation around an elite basin.
                    decay = max(0.05, (remaining() / max(1, max_evals)) ** 0.5)
                    radius = 0.25 * decay * (ub - lb)
                    step = rng.standard_t(df=2.0, size=dim) * radius
                    x_probe = np.clip(base + step, lb, ub)
                else:
                    # Uniform global probe.
                    x_probe = lb + rng.random(dim) * (ub - lb)

                eval_x(x_probe)

            except BudgetExhausted:
                break
            except Exception:
                # Robustness safeguard for pathological objectives.
                continue

            if best_f < previous_best - 1e-14:
                stall_count = 0
            else:
                stall_count += 1

            # If probing stalls, perform a small local polish around current best.
            if remaining() > max(10, 3 * dim) and stall_count >= max(10, 2 * dim):
                try:
                    local_polish(best_x, min(max(10, 3 * dim), remaining()))
                except Exception:
                    pass
                stall_count = 0

    message = "budget_exhausted" if nfe >= max_evals else "completed"

    return {
        "xbest": best_x,
        "fbest": float(best_f),
        "nfe": int(nfe),
        "history": history,
        "message": message,
    }
