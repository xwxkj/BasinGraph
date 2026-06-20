
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize_scalar
from .evaluation import EvaluationLedger, BudgetExhausted

@dataclass
class LineCandidate:
    x: np.ndarray
    f: float
    coordinate: int
    bracket: tuple[float, float]
    evaluations: int
    barrier_proxy: float

def coordinate_multibracket_sweep(ledger: EvaluationLedger, start, *, coarse_samples, refine_top_k, maxiter):
    x0 = ledger.project(start)
    dim = ledger.dimension
    out = []
    for j in range(dim):
        if ledger.remaining <= 0: break
        lo, hi = float(ledger.lb[j]), float(ledger.ub[j])
        grid = np.linspace(lo, hi, max(5, int(coarse_samples)))
        values, base = [], x0.copy()
        for t in grid:
            if ledger.remaining <= 0: break
            z = base.copy(); z[j] = t
            try:
                values.append(float(ledger.evaluate(z)))
            except BudgetExhausted:
                break
        if len(values) < 3: continue
        vals = np.asarray(values)
        brackets = []
        for i in range(1, len(vals)-1):
            if vals[i] <= vals[i-1] and vals[i] <= vals[i+1]:
                brackets.append((float(vals[i]), float(grid[i-1]), float(grid[i+1])))
        if not brackets:
            bi = int(np.argmin(vals))
            left, right = max(0, bi-1), min(len(vals)-1, bi+1)
            brackets.append((float(vals[bi]), float(grid[left]), float(grid[right])))
        brackets = sorted(brackets, key=lambda item: item[0])[:max(1, int(refine_top_k))]
        for _, a, b in brackets:
            if ledger.remaining <= 0 or not (b > a): continue
            before = ledger.nfe
            def phi(t):
                z = base.copy(); z[j] = float(t)
                try:
                    return ledger.evaluate(z)
                except BudgetExhausted:
                    return ledger.fbest + 1e100
            try:
                res = minimize_scalar(phi, bounds=(a, b), method="bounded",
                                      options={"maxiter": int(maxiter), "xatol": max(1e-12, 1e-9*(b-a))})
            except Exception:
                continue
            z = base.copy(); z[j] = float(res.x)
            out.append(LineCandidate(ledger.project(z), float(res.fun), j, (a,b), ledger.nfe-before, float(max(vals)-min(vals))))
    return out
