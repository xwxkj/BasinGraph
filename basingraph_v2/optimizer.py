"""Full BasinGraph v2.0.0-rc1 with enforced phase budgets and ablation flags."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Callable, Any

import numpy as np
from scipy.optimize import minimize

from .archive import BasinArchive
from .diagnostics import make_initial_anchors, evaluate_diagnostics
from .evaluation import (
    EvaluationLedger,
    BudgetExhausted,
    PhaseBudgetExhausted,
)
from .graph import BasinTransitionGraph
from .line_search import coordinate_multibracket_sweep
from .types import BasinGraphResult, EventRecord


IMPLEMENTATION_VERSION = "2.0.0-rc1"


@dataclass(frozen=True)
class BasinGraphOptions:
    # Phase allocations. Unused allocations flow to budget completion.
    initial_design_fraction: float = 0.10
    center_local_fraction: float = 0.15
    coordinate_sweep_fraction: float = 0.30
    far_basin_fraction: float = 0.15
    archive_fallback_fraction: float = 0.10
    final_polish_fraction: float = 0.05

    n_initial_lhs: int = 24
    archive_max_size: int = 80
    merge_radius_factor: float = 0.035

    center_local_max_dim: int = 20
    local_mode_min_score: float = 0.12
    large_box_scale: float = 100.0
    far_basin_ruggedness_threshold: float = 0.10
    far_basin_boundary_threshold: float = 0.05

    line_coarse_samples: int = 11
    line_refine_top_k: int = 3
    line_refine_maxiter: int = 20
    coordinate_starts: int = 6

    far_basin_random: int = 32
    archive_fallback_starts: int = 4
    final_polish_starts: int = 2

    graph_quality_weight: float = 0.55
    graph_novelty_weight: float = 0.25
    graph_accessibility_weight: float = 0.20

    local_ftol: float = 1e-12
    local_gtol: float = 1e-8

    # Ablation switches.
    enable_geometry_controller: bool = True
    enable_graph_guidance: bool = True
    enable_multibracket: bool = True
    enable_far_basin: bool = True
    enable_archive_fallback: bool = True
    enable_final_polish: bool = True

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.to_jsonable(),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _phase_limit(max_evals: int, fraction: float) -> int:
    return max(0, int(np.floor(max_evals * float(fraction))))


def _curvature_proxy(radius: float) -> float:
    return float(1.0 / (1.0 + max(radius, 0.0)))


def _local_polish(
    ledger: EvaluationLedger,
    x0: np.ndarray,
    *,
    options: BasinGraphOptions,
) -> tuple[np.ndarray, float, int]:
    before = ledger.nfe

    def objective(point):
        return ledger.evaluate(point)

    try:
        result = minimize(
            objective,
            ledger.project(x0),
            method="L-BFGS-B",
            bounds=list(zip(ledger.lb, ledger.ub)),
            options={
                "maxfun": int(max(1, ledger.remaining)),
                "maxiter": int(max(1, ledger.remaining)),
                "ftol": options.local_ftol,
                "gtol": options.local_gtol,
                "disp": False,
            },
        )
        x = ledger.project(result.x)
        f = float(result.fun)
    except (BudgetExhausted, PhaseBudgetExhausted):
        x = (
            ledger.xbest.copy()
            if ledger.xbest is not None
            else ledger.project(x0)
        )
        f = float(ledger.fbest)
    except Exception:
        x = ledger.project(x0)
        f = float("inf")

    return x, f, int(ledger.nfe - before)


def minimize_basingraph_v2(
    objective: Callable[[np.ndarray], float],
    lb,
    ub,
    max_evals: int,
    seed: int = 0,
    options: BasinGraphOptions | None = None,
) -> BasinGraphResult:
    if options is None:
        options = BasinGraphOptions()

    rng = np.random.default_rng(seed)
    lb = np.asarray(lb, dtype=float).reshape(-1)
    ub = np.asarray(ub, dtype=float).reshape(-1)

    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)
    invalid = ub <= lb
    lb[invalid], ub[invalid] = -5.0, 5.0

    ledger = EvaluationLedger(objective, lb, ub, max_evals)
    archive = BasinArchive(
        merge_radius_factor=options.merge_radius_factor,
        max_nodes=options.archive_max_size,
    )
    graph = BasinTransitionGraph()
    events: list[EventRecord] = []
    domain_radius = float(np.linalg.norm(ub - lb) + 1e-300)

    def event(name: str, message: str, payload=None):
        events.append(
            EventRecord(
                nfe=int(ledger.nfe),
                phase=str(ledger.current_phase),
                event=str(name),
                message=str(message),
                payload=payload or {},
            )
        )

    def add_node(
        x,
        f,
        source: str,
        *,
        radius_fraction: float = 0.02,
    ):
        update = archive.add_or_merge(
            x,
            f,
            radius=radius_fraction * domain_radius,
            curvature_proxy=_curvature_proxy(radius_fraction * domain_radius),
            nfe=ledger.nfe,
            source=source,
            lb=lb,
            ub=ub,
        )
        if update.removed_node_ids:
            graph.remove_nodes(update.removed_node_ids)
        event(
            "node_created" if update.created else "node_merged",
            source,
            {
                "node_id": update.node.node_id,
                "f_center": update.node.f_center,
                "removed_node_ids": update.removed_node_ids,
            },
        )
        return update.node

    # --------------------------------------------------------------
    # Phase 1: initial design
    # --------------------------------------------------------------
    initial_limit = _phase_limit(max_evals, options.initial_design_fraction)
    anchors = make_initial_anchors(
        lb,
        ub,
        rng,
        n_lhs=options.n_initial_lhs,
    )
    anchor_values: list[float] = []

    with ledger.phase("initial_design", initial_limit):
        for point in anchors:
            try:
                value = ledger.evaluate(point)
            except (BudgetExhausted, PhaseBudgetExhausted):
                break
            anchor_values.append(value)
            add_node(point, value, "initial_design")

    if ledger.xbest is None:
        ledger.xbest = 0.5 * (lb + ub)
        ledger.fbest = float("inf")

    diagnostics = evaluate_diagnostics(
        lb=lb,
        ub=ub,
        anchors=anchors[: len(anchor_values)],
        anchor_values=anchor_values,
    )
    event("diagnostics", "geometry diagnostics computed", diagnostics.to_jsonable())

    if options.enable_geometry_controller:
        use_center_local = (
            diagnostics.dimension <= options.center_local_max_dim
            and diagnostics.local_mode_score >= options.local_mode_min_score
        )
        use_far_basin = (
            options.enable_far_basin
            and (
                diagnostics.ruggedness_score
                >= options.far_basin_ruggedness_threshold
                or diagnostics.boundary_signal
                >= options.far_basin_boundary_threshold
                or diagnostics.max_scale >= options.large_box_scale
            )
        )
    else:
        use_center_local = True
        use_far_basin = options.enable_far_basin

    event(
        "controller_decision",
        "phase enablement fixed",
        {
            "use_center_local": use_center_local,
            "use_far_basin": use_far_basin,
            "enable_graph_guidance": options.enable_graph_guidance,
            "enable_multibracket": options.enable_multibracket,
        },
    )

    # --------------------------------------------------------------
    # Phase 2: center-local contraction
    # --------------------------------------------------------------
    if use_center_local and ledger.remaining > 0:
        with ledger.phase(
            "center_local",
            _phase_limit(max_evals, options.center_local_fraction),
        ):
            x, f, used = _local_polish(
                ledger,
                0.5 * (lb + ub),
                options=options,
            )
            node = add_node(x, f, "center_local")
            event(
                "center_local_complete",
                "center-local contraction finished",
                {"node_id": node.node_id, "evaluations": used},
            )

    # --------------------------------------------------------------
    # Phase 3: coordinate multi-bracket sweep
    # --------------------------------------------------------------
    with ledger.phase(
        "coordinate_sweep",
        _phase_limit(max_evals, options.coordinate_sweep_fraction),
    ):
        starts = [
            node.center.copy()
            for node in archive.sorted_nodes()[: options.coordinate_starts]
        ]
        for start in starts:
            source_node, _ = archive.nearest(start)
            candidates = coordinate_multibracket_sweep(
                ledger,
                start,
                coarse_samples=options.line_coarse_samples,
                refine_top_k=options.line_refine_top_k,
                maxiter=options.line_refine_maxiter,
                enable_multibracket=options.enable_multibracket,
            )
            for candidate in candidates:
                target = add_node(
                    candidate.x,
                    candidate.f,
                    "coordinate_sweep",
                )
                if source_node is not None:
                    graph.add_or_update(
                        source_node.node_id,
                        target.node_id,
                        evaluations=candidate.evaluations,
                        improvement=source_node.f_center - target.f_center,
                        barrier_proxy=candidate.barrier_proxy,
                        nfe=ledger.nfe,
                        source_mode="coordinate_sweep",
                    )

    # --------------------------------------------------------------
    # Phase 4: far-basin exploration
    # --------------------------------------------------------------
    if use_far_basin and ledger.remaining > 0:
        with ledger.phase(
            "far_basin",
            _phase_limit(max_evals, options.far_basin_fraction),
        ):
            center = 0.5 * (lb + ub)
            for _ in range(options.far_basin_random):
                direction = rng.standard_t(df=1.5, size=ledger.dimension)
                norm = np.linalg.norm(direction)
                if norm <= 0:
                    continue
                direction /= norm
                radius = (rng.random() ** 0.25) * 0.5 * domain_radius
                point = ledger.project(center + radius * direction)
                source_node, _ = archive.nearest(ledger.xbest)
                try:
                    value = ledger.evaluate(point)
                except (BudgetExhausted, PhaseBudgetExhausted):
                    break
                target = add_node(
                    point,
                    value,
                    "far_basin",
                    radius_fraction=0.05,
                )
                if source_node is not None:
                    graph.add_or_update(
                        source_node.node_id,
                        target.node_id,
                        evaluations=1,
                        improvement=source_node.f_center - target.f_center,
                        barrier_proxy=max(0.0, value - source_node.f_center),
                        nfe=ledger.nfe,
                        source_mode="far_basin",
                    )

    # --------------------------------------------------------------
    # Phase 5: archive fallback
    # --------------------------------------------------------------
    if options.enable_archive_fallback and ledger.remaining > 0:
        with ledger.phase(
            "archive_fallback",
            _phase_limit(max_evals, options.archive_fallback_fraction),
        ):
            candidates = archive.sorted_nodes()
            if options.enable_graph_guidance:
                scores = graph.guidance_scores(
                    candidates,
                    quality_weight=options.graph_quality_weight,
                    novelty_weight=options.graph_novelty_weight,
                    accessibility_weight=options.graph_accessibility_weight,
                )
                candidates = sorted(
                    candidates,
                    key=lambda node: scores.get(node.node_id, 0.0),
                    reverse=True,
                )

            for source_node in candidates[: options.archive_fallback_starts]:
                x, f, used = _local_polish(
                    ledger,
                    source_node.center,
                    options=options,
                )
                target = add_node(x, f, "archive_fallback")
                graph.add_or_update(
                    source_node.node_id,
                    target.node_id,
                    evaluations=used,
                    improvement=source_node.f_center - target.f_center,
                    barrier_proxy=max(0.0, f - source_node.f_center),
                    nfe=ledger.nfe,
                    source_mode="archive_fallback",
                )

    # --------------------------------------------------------------
    # Phase 6: final polishing
    # --------------------------------------------------------------
    if options.enable_final_polish and ledger.remaining > 0:
        with ledger.phase(
            "final_polish",
            _phase_limit(max_evals, options.final_polish_fraction),
        ):
            for source_node in archive.sorted_nodes()[: options.final_polish_starts]:
                x, f, used = _local_polish(
                    ledger,
                    source_node.center,
                    options=options,
                )
                target = add_node(x, f, "final_polish")
                graph.add_or_update(
                    source_node.node_id,
                    target.node_id,
                    evaluations=used,
                    improvement=source_node.f_center - target.f_center,
                    barrier_proxy=max(0.0, f - source_node.f_center),
                    nfe=ledger.nfe,
                    source_mode="final_polish",
                )

    # --------------------------------------------------------------
    # Phase 7: graph-aware budget completion
    # --------------------------------------------------------------
    with ledger.phase("budget_completion", None):
        stall = 0
        while ledger.remaining > 0:
            previous_best = ledger.fbest
            nodes = archive.sorted_nodes()

            if nodes and options.enable_graph_guidance and rng.random() < 0.80:
                scores = graph.guidance_scores(
                    nodes,
                    quality_weight=options.graph_quality_weight,
                    novelty_weight=options.graph_novelty_weight,
                    accessibility_weight=options.graph_accessibility_weight,
                )
                score_array = np.asarray(
                    [max(scores.get(node.node_id, 0.0), 0.0) for node in nodes],
                    dtype=float,
                )
                if score_array.sum() <= 0:
                    probabilities = np.full(len(nodes), 1.0 / len(nodes))
                else:
                    probabilities = score_array / score_array.sum()
                source_node = nodes[int(rng.choice(len(nodes), p=probabilities))]
                decay = max(
                    0.05,
                    (ledger.remaining / max(1, max_evals)) ** 0.5,
                )
                step = rng.standard_t(
                    df=2.0,
                    size=ledger.dimension,
                ) * (0.20 * decay * (ub - lb))
                point = ledger.project(source_node.center + step)
            elif nodes and rng.random() < 0.70:
                source_node = nodes[int(rng.integers(min(8, len(nodes))))]
                point = ledger.project(
                    source_node.center
                    + rng.standard_t(df=2.0, size=ledger.dimension)
                    * (0.10 * (ub - lb))
                )
            else:
                source_node, _ = archive.nearest(ledger.xbest)
                point = ledger.project(lb + rng.random(ledger.dimension) * (ub - lb))

            try:
                value = ledger.evaluate(point)
            except BudgetExhausted:
                break

            target = add_node(point, value, "budget_completion")
            if source_node is not None:
                graph.add_or_update(
                    source_node.node_id,
                    target.node_id,
                    evaluations=1,
                    improvement=source_node.f_center - target.f_center,
                    barrier_proxy=max(0.0, value - source_node.f_center),
                    nfe=ledger.nfe,
                    source_mode="budget_completion",
                )

            if ledger.fbest < previous_best - 1e-14:
                stall = 0
            else:
                stall += 1

            if (
                stall >= max(10, 2 * ledger.dimension)
                and ledger.remaining > max(10, 3 * ledger.dimension)
            ):
                x, f, used = _local_polish(
                    ledger,
                    ledger.xbest,
                    options=options,
                )
                target = add_node(x, f, "stall_polish")
                if source_node is not None:
                    graph.add_or_update(
                        source_node.node_id,
                        target.node_id,
                        evaluations=used,
                        improvement=source_node.f_center - target.f_center,
                        barrier_proxy=max(0.0, f - source_node.f_center),
                        nfe=ledger.nfe,
                        source_mode="stall_polish",
                    )
                stall = 0

    # Guarantee graph/archive referential integrity.
    active_node_ids = {node.node_id for node in archive.nodes}
    graph.remove_nodes(
        [
            node_id
            for edge in graph.to_edges()
            for node_id in (edge.source_id, edge.target_id)
            if node_id not in active_node_ids
        ]
    )

    message = "budget_exhausted" if ledger.remaining == 0 else "completed"
    return BasinGraphResult(
        implementation_version=IMPLEMENTATION_VERSION,
        options_hash=options.stable_hash(),
        options=options.to_jsonable(),
        xbest=ledger.xbest.copy(),
        fbest=float(ledger.fbest),
        nfe=int(ledger.nfe),
        phase_evaluations=dict(ledger.phase_evaluations),
        history=ledger.history,
        archive=archive.sorted_nodes(),
        graph_edges=graph.to_edges(),
        diagnostics=diagnostics,
        event_log=events,
        message=message,
    )
