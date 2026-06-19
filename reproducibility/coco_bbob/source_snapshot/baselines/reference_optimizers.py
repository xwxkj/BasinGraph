"""
Reference optimizers for official COCO smoke tests.

These implementations are used to verify the benchmarking harness and provide
traceable baseline runs. For final manuscript baselines, implementation
provenance should be recorded in a baseline provenance table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Any, Tuple
import numpy as np
from scipy.optimize import minimize


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class BudgetedObjective:
    objective: Callable[[np.ndarray], float]
    lb: np.ndarray
    ub: np.ndarray
    max_evals: int

    def __post_init__(self):
        self.lb = np.asarray(self.lb, dtype=float).reshape(-1)
        self.ub = np.asarray(self.ub, dtype=float).reshape(-1)
        self.nfe = 0
        self.xbest = None
        self.fbest = np.inf
        self.history = []

    def __call__(self, x):
        if self.nfe >= self.max_evals:
            raise BudgetExhausted()

        z = np.clip(np.asarray(x, dtype=float).reshape(-1), self.lb, self.ub)
        y = float(self.objective(z))
        self.nfe += 1

        if np.isfinite(y) and y < self.fbest:
            self.fbest = y
            self.xbest = z.copy()

        self.history.append((self.nfe, self.fbest))
        return y

    @property
    def remaining(self):
        return max(0, int(self.max_evals) - int(self.nfe))


def _finish(bo: BudgetedObjective, message: str) -> Dict[str, Any]:
    if bo.xbest is None:
        bo.xbest = 0.5 * (bo.lb + bo.ub)
    return {
        "xbest": bo.xbest,
        "fbest": float(bo.fbest),
        "nfe": int(bo.nfe),
        "history": bo.history,
        "message": message,
    }


def _lhs(n: int, dim: int, lb: np.ndarray, ub: np.ndarray, rng: np.random.Generator):
    X = np.empty((n, dim), dtype=float)
    for j in range(dim):
        perm = rng.permutation(n)
        X[:, j] = (perm + rng.random(n)) / n
    return lb + X * (ub - lb)


def optimize_random_search(objective, lb, ub, max_evals, seed=0):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size
    bo = BudgetedObjective(objective, lb, ub, max_evals)

    try:
        bo(0.5 * (lb + ub))
        while bo.remaining > 0:
            x = lb + rng.random(dim) * (ub - lb)
            bo(x)
    except BudgetExhausted:
        pass

    return _finish(bo, "completed")


def optimize_lhs(objective, lb, ub, max_evals, seed=0):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size
    bo = BudgetedObjective(objective, lb, ub, max_evals)

    n = max(1, int(max_evals))
    X = _lhs(n, dim, lb, ub, rng)

    try:
        for x in X:
            bo(x)
    except BudgetExhausted:
        pass

    return _finish(bo, "completed")


def optimize_multistart_lbfgsb(objective, lb, ub, max_evals, seed=0):
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size
    bo = BudgetedObjective(objective, lb, ub, max_evals)

    n_starts = max(2, min(8, dim + 2))
    starts = [0.5 * (lb + ub)]
    starts.extend([lb + rng.random(dim) * (ub - lb) for _ in range(n_starts - 1)])

    for istart, x0 in enumerate(starts):
        if bo.remaining <= 0:
            break

        remaining_starts = max(1, len(starts) - istart)
        maxfun = max(5, bo.remaining // remaining_starts)

        try:
            minimize(
                bo,
                x0=np.clip(x0, lb, ub),
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={
                    "maxfun": int(maxfun),
                    "maxiter": int(maxfun),
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "disp": False,
                },
            )
        except BudgetExhausted:
            break
        except Exception:
            continue

    return _finish(bo, "completed")


def optimize_de(objective, lb, ub, max_evals, seed=0):
    """A compact differential evolution baseline with strict FE accounting."""
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size
    bo = BudgetedObjective(objective, lb, ub, max_evals)

    npop = max(8, min(10 * dim, max(8, max_evals // 4)))
    F = 0.5
    CR = 0.9

    X = lb + rng.random((npop, dim)) * (ub - lb)
    fit = np.full(npop, np.inf)

    try:
        for i in range(npop):
            if bo.remaining <= 0:
                break
            fit[i] = bo(X[i])

        while bo.remaining > 0:
            for i in range(npop):
                if bo.remaining <= 0:
                    break

                idx = np.arange(npop)
                idx = idx[idx != i]
                a, b, c = rng.choice(idx, size=3, replace=False)

                mutant = X[a] + F * (X[b] - X[c])
                mutant = np.clip(mutant, lb, ub)

                mask = rng.random(dim) < CR
                if not np.any(mask):
                    mask[rng.integers(dim)] = True

                trial = X[i].copy()
                trial[mask] = mutant[mask]

                ftrial = bo(trial)
                if ftrial <= fit[i]:
                    X[i] = trial
                    fit[i] = ftrial

    except BudgetExhausted:
        pass

    return _finish(bo, "completed")


def optimize_cmaes(objective, lb, ub, max_evals, seed=0, popsize_factor=1.0):
    """CMA-ES using pycma, with strict external FE accounting."""
    import cma

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)
    dim = lb.size

    bo = BudgetedObjective(objective, lb, ub, max_evals)

    x0 = 0.5 * (lb + ub)
    sigma0 = max(1e-12, 0.25 * float(np.mean(ub - lb)))

    opts = {
        "bounds": [lb.tolist(), ub.tolist()],
        "seed": int(seed),
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
    }

    if popsize_factor != 1.0:
        default_pop = int(4 + np.floor(3 * np.log(max(2, dim))))
        opts["popsize"] = max(4, int(np.ceil(popsize_factor * default_pop)))

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    try:
        while bo.remaining > 0 and not es.stop():
            nask = max(1, min(int(es.popsize), bo.remaining))
            xs = es.ask(number=nask)
            ys = []

            for x in xs:
                if bo.remaining <= 0:
                    break
                y = bo(np.asarray(x, dtype=float))
                ys.append(y)

            if len(ys) == 0:
                break

            es.tell(xs[:len(ys)], ys)

    except BudgetExhausted:
        pass
    except Exception:
        pass

    return _finish(bo, "completed")


def optimize_bipop_cmaes(objective, lb, ub, max_evals, seed=0):
    """
    BIPOP-CMA-ES using the official pycma restart interface.

    The BIPOP restart strategy is invoked through:
        cma.fmin2(..., restarts=9, bipop=True)

    Function evaluations are counted externally through BudgetedObjective.
    """
    import cma

    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    dim = lb.size

    bo = BudgetedObjective(objective, lb, ub, int(max_evals))

    # A callable initial point generates a new random start at every restart,
    # as recommended for the pycma BIPOP interface.
    def x0_factory():
        return lb + rng.random(dim) * (ub - lb)

    sigma0 = max(1e-12, 0.25 * float(np.mean(ub - lb)))

    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "seed": int(seed),
        "maxfevals": int(max_evals),
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
        "verb_time": 0,
    }

    try:
        cma.fmin2(
            bo,
            x0_factory,
            sigma0,
            options,
            restarts=9,
            bipop=True,
        )
    except BudgetExhausted:
        # The external objective wrapper prevents evaluations beyond budget.
        pass
    except Exception as exc:
        return _finish(
            bo,
            f"terminated_with_exception:{type(exc).__name__}"
        )

    message = "budget_exhausted" if bo.nfe >= int(max_evals) else "completed"
    return _finish(bo, message)

