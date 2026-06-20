
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
import numpy as np
from scipy.optimize import minimize
from .archive import BasinArchive
from .diagnostics import make_initial_anchors, evaluate_diagnostics
from .evaluation import EvaluationLedger, BudgetExhausted
from .graph import BasinTransitionGraph
from .line_search import coordinate_multibracket_sweep
from .types import BasinGraphResult, EventRecord

@dataclass
class BasinGraphOptions:
    n_initial_lhs: int = 24
    archive_max_size: int = 80
    merge_radius_factor: float = 0.05
    center_local_fraction: float = 0.20
    far_basin_fraction: float = 0.15
    final_polish_fraction: float = 0.15
    center_local_max_dim: int = 20
    large_box_scale: float = 100.0
    line_coarse_samples: int = 11
    line_refine_top_k: int = 3
    line_refine_maxiter: int = 20
    coordinate_starts: int = 6
    far_basin_random: int = 32
    final_polish_starts: int = 5
    local_ftol: float = 1e-12
    local_gtol: float = 1e-8

def _curvature_proxy_from_radius(radius: float) -> float:
    return float(1.0 / (1.0 + max(radius, 0.0)))

def _local_polish(ledger, x0, *, maxfun, options):
    before = ledger.nfe
    if maxfun <= max(5, 2*ledger.dimension) or ledger.remaining <= 0:
        return ledger.project(x0), float("inf"), 0
    def fun(z):
        return ledger.evaluate(z)
    try:
        res = minimize(fun, ledger.project(x0), method="L-BFGS-B",
                       bounds=list(zip(ledger.lb, ledger.ub)),
                       options={"maxfun": int(min(maxfun, ledger.remaining)), "maxiter": int(min(maxfun, ledger.remaining)),
                                "ftol": options.local_ftol, "gtol": options.local_gtol, "disp": False})
        return ledger.project(res.x), float(res.fun), ledger.nfe-before
    except BudgetExhausted:
        x = ledger.xbest.copy() if ledger.xbest is not None else ledger.project(x0)
        return x, float(ledger.fbest), ledger.nfe-before
    except Exception:
        return ledger.project(x0), float("inf"), ledger.nfe-before

def minimize_basingraph_v2(objective: Callable[[np.ndarray], float], lb, ub, max_evals: int, seed: int = 0,
                           options: BasinGraphOptions | None = None) -> BasinGraphResult:
    if options is None: options = BasinGraphOptions()
    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)
    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)
    bad = ub <= lb
    lb[bad], ub[bad] = -5.0, 5.0

    ledger = EvaluationLedger(objective, lb, ub, max_evals)
    archive = BasinArchive(options.merge_radius_factor, options.archive_max_size)
    graph = BasinTransitionGraph()
    events = []

    def event(name, message, payload=None):
        events.append(EventRecord(ledger.nfe, name, message, payload or {}))

    def add_node(x, f, source, radius=None):
        if radius is None:
            radius = 0.02 * float(np.linalg.norm(ub-lb) + 1e-300)
        node, created = archive.add_or_merge(
            x, f, radius=radius, curvature_proxy=_curvature_proxy_from_radius(radius),
            nfe=ledger.nfe, source=source, lb=lb, ub=ub
        )
        event("basin_node_" + ("created" if created else "merged"), source, {"node_id": node.node_id, "f_center": node.f_center})
        return node, created

    anchors = make_initial_anchors(lb, ub, rng, n_lhs=min(options.n_initial_lhs + lb.size, max(4, max_evals//10)))
    anchor_values = []
    for x in anchors:
        if ledger.remaining <= 0: break
        try:
            f = ledger.evaluate(x)
            anchor_values.append(f)
            add_node(x, f, "anchor")
        except BudgetExhausted:
            break
    if ledger.xbest is None:
        ledger.xbest = 0.5*(lb+ub)
        ledger.fbest = float("inf")

    diagnostics = evaluate_diagnostics(ledger, anchors[:len(anchor_values)], anchor_values)
    event("diagnostics", "geometry diagnostics computed", diagnostics.to_jsonable())

    if diagnostics.dimension <= options.center_local_max_dim and ledger.remaining > 0:
        x, f, used = _local_polish(ledger, 0.5*(lb+ub), maxfun=int(options.center_local_fraction*max_evals), options=options)
        node, _ = add_node(x, f, "center_local")
        event("center_local", "center-local contraction completed", {"used": used, "node_id": node.node_id})

    starts = [n.center for n in archive.sorted_nodes()[:options.coordinate_starts]]
    for start in starts:
        if ledger.remaining <= 0: break
        source_node, _ = archive.nearest(start)
        candidates = coordinate_multibracket_sweep(ledger, start, coarse_samples=options.line_coarse_samples,
                                                   refine_top_k=options.line_refine_top_k, maxiter=options.line_refine_maxiter)
        for cand in candidates:
            node, _ = add_node(cand.x, cand.f, "coordinate_multibracket")
            if source_node is not None:
                graph.add_or_update(source_node.node_id, node.node_id, evaluations=cand.evaluations,
                                    improvement=source_node.f_center-node.f_center, barrier_proxy=cand.barrier_proxy,
                                    nfe=ledger.nfe, source_mode="coordinate_multibracket")

    far_count = min(options.far_basin_random, int(options.far_basin_fraction*max_evals))
    center = 0.5*(lb+ub)
    for _ in range(far_count):
        if ledger.remaining <= 0: break
        direction = rng.standard_t(df=1.5, size=ledger.dimension)
        norm = np.linalg.norm(direction)
        if norm <= 0: continue
        direction /= norm
        x = ledger.project(center + (rng.random()**0.25)*0.5*np.linalg.norm(ub-lb)*direction)
        source_node, _ = archive.nearest(ledger.xbest)
        try:
            f = ledger.evaluate(x)
        except BudgetExhausted:
            break
        node, _ = add_node(x, f, "far_basin", radius=0.05*np.linalg.norm(ub-lb))
        if source_node is not None:
            graph.add_or_update(source_node.node_id, node.node_id, evaluations=1,
                                improvement=source_node.f_center-node.f_center, barrier_proxy=max(0.0, f-source_node.f_center),
                                nfe=ledger.nfe, source_mode="far_basin")

    starts = [n.center for n in archive.sorted_nodes()[:options.final_polish_starts]]
    per_start = max(5, int(options.final_polish_fraction*max_evals/max(1, len(starts))))
    for start in starts:
        if ledger.remaining <= 0: break
        source_node, _ = archive.nearest(start)
        x, f, used = _local_polish(ledger, start, maxfun=min(per_start, ledger.remaining), options=options)
        node, _ = add_node(x, f, "archive_fallback_polish")
        if source_node is not None:
            graph.add_or_update(source_node.node_id, node.node_id, evaluations=used,
                                improvement=source_node.f_center-node.f_center, barrier_proxy=max(0.0, f-source_node.f_center),
                                nfe=ledger.nfe, source_mode="archive_fallback_polish")

    stall = 0
    while ledger.remaining > 0:
        previous = ledger.fbest
        elite = archive.sorted_nodes()[:max(1, min(8, len(archive.nodes)))]
        if elite and rng.random() < 0.75:
            source_node = elite[int(rng.integers(len(elite)))]
            decay = max(0.05, (ledger.remaining/max(1,max_evals))**0.5)
            x = ledger.project(source_node.center + rng.standard_t(df=2.0, size=ledger.dimension) * (0.20*decay*(ub-lb)))
        else:
            x = ledger.project(lb + rng.random(ledger.dimension)*(ub-lb))
            source_node, _ = archive.nearest(ledger.xbest)
        try:
            f = ledger.evaluate(x)
        except BudgetExhausted:
            break
        node, _ = add_node(x, f, "budget_completion")
        if source_node is not None:
            graph.add_or_update(source_node.node_id, node.node_id, evaluations=1,
                                improvement=source_node.f_center-node.f_center, barrier_proxy=max(0.0, f-source_node.f_center),
                                nfe=ledger.nfe, source_mode="budget_completion")
        stall = 0 if ledger.fbest < previous - 1e-14 else stall + 1
        if stall >= max(10, 2*ledger.dimension) and ledger.remaining > max(10, 3*ledger.dimension):
            x, f, used = _local_polish(ledger, ledger.xbest, maxfun=min(max(10,3*ledger.dimension), ledger.remaining), options=options)
            add_node(x, f, "stall_polish")
            stall = 0

    return BasinGraphResult(
        xbest=ledger.xbest.copy(), fbest=float(ledger.fbest), nfe=int(ledger.nfe),
        history=ledger.history, archive=archive.sorted_nodes(), graph_edges=graph.to_edges(),
        diagnostics=diagnostics, event_log=events,
        message="budget_exhausted" if ledger.remaining == 0 else "completed"
    )
