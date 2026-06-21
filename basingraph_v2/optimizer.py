"""
BasinGraph v2.0.0-rc2.

Structural revision frozen before implementation:
- probes and certified basin nodes are separate;
- adaptive merge radius;
- quality-diversity-accessibility archive retention;
- landscape curvature anisotropy;
- principal-direction multi-bracket refinement;
- probe-refine-commit exploration;
- sparse transition graph;
- stricter center-local controller;
- exact phase-level FE accounting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable
import numpy as np
from scipy.optimize import minimize

from .archive import BasinArchive
from .diagnostics import (
    evaluate_diagnostics,
    make_initial_anchors,
)
from .directions import build_principal_directions
from .evaluation import (
    BudgetExhausted,
    EvaluationLedger,
    LocalBudgetExhausted,
    PhaseBudgetExhausted,
)
from .graph import BasinTransitionGraph
from .line_search import (
    coordinate_multibracket_sweep,
    directional_multibracket_search,
)
from .probes import ProbePool
from .types import (
    BasinGraphResult,
    DirectionDiagnostics,
    EventRecord,
    ProbeRecord,
)


IMPLEMENTATION_VERSION = "2.0.0-rc2"


@dataclass(frozen=True)
class BasinGraphOptions:
    initial_design_fraction: float = 0.10
    center_local_fraction: float = 0.08
    coordinate_sweep_fraction: float = 0.25
    principal_direction_fraction: float = 0.15
    far_basin_fraction: float = 0.12
    archive_fallback_fraction: float = 0.10
    final_polish_fraction: float = 0.08

    n_initial_lhs: int = 24
    probe_pool_max_size: int = 256
    archive_max_size: int = 80

    merge_radius_base: float = 0.025
    merge_radius_max: float = 0.080
    archive_quality_weight: float = 0.50
    archive_diversity_weight: float = 0.30
    archive_accessibility_weight: float = 0.20

    center_local_max_dim: int = 20
    center_local_min_score: float = 0.20
    center_local_max_curvature_anisotropy: float = 100.0
    center_local_max_ruggedness: float = 0.10

    large_box_scale: float = 100.0
    far_basin_ruggedness_threshold: float = 0.10
    far_basin_boundary_threshold: float = 0.05

    line_coarse_samples: int = 11
    line_refine_top_k: int = 3
    line_refine_maxiter: int = 20
    coordinate_start_count: int = 6

    principal_elite_count: int = 10
    principal_covariance_directions: int = 4
    principal_transition_directions: int = 3
    principal_max_directions: int = 8
    principal_cosine_threshold: float = 0.98
    principal_start_count: int = 4

    far_probe_batch_size: int = 8
    far_refine_per_batch: int = 2
    completion_probe_batch_size: int = 4
    completion_refine_per_batch: int = 1
    completion_local_budget_factor: int = 4

    archive_fallback_starts: int = 4
    final_polish_starts: int = 2

    graph_quality_weight: float = 0.55
    graph_novelty_weight: float = 0.25
    graph_accessibility_weight: float = 0.20
    graph_max_outgoing: int = 3
    graph_max_incoming: int = 3
    graph_edge_improvement_weight: float = 0.45
    graph_edge_accessibility_weight: float = 0.35
    graph_edge_recency_weight: float = 0.20

    local_ftol: float = 1e-12
    local_gtol: float = 1e-8
    event_log_max_size: int = 4000

    enable_geometry_controller: bool = True
    enable_graph_guidance: bool = True
    enable_multibracket: bool = True
    enable_principal_directions: bool = True
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
        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()


def _phase_limit(
    max_evals: int,
    fraction: float,
) -> int:
    return int(
        max(
            0,
            np.floor(
                max_evals * float(fraction)
            ),
        )
    )


def _curvature_proxy(radius: float) -> float:
    return float(
        1.0 / (1.0 + max(radius, 0.0))
    )


def _local_polish(
    ledger: EvaluationLedger,
    start: np.ndarray,
    *,
    options: BasinGraphOptions,
    max_local_evals: int | None = None,
) -> tuple[np.ndarray, float, int]:
    before = ledger.nfe
    local_calls = 0
    local_limit = (
        ledger.remaining
        if max_local_evals is None
        else int(
            max(
                1,
                min(
                    max_local_evals,
                    ledger.remaining,
                ),
            )
        )
    )

    def objective(point):
        nonlocal local_calls
        if local_calls >= local_limit:
            raise LocalBudgetExhausted()
        local_calls += 1
        return ledger.evaluate(point)

    try:
        result = minimize(
            objective,
            ledger.project(start),
            method="L-BFGS-B",
            bounds=list(
                zip(
                    ledger.lb,
                    ledger.ub,
                )
            ),
            options={
                "maxfun": int(local_limit),
                "maxiter": int(local_limit),
                "ftol": options.local_ftol,
                "gtol": options.local_gtol,
                "disp": False,
            },
        )
        point = ledger.project(result.x)
        value = float(result.fun)
    except (
        BudgetExhausted,
        PhaseBudgetExhausted,
        LocalBudgetExhausted,
    ):
        point = (
            ledger.xbest.copy()
            if ledger.xbest is not None
            else ledger.project(start)
        )
        value = float(ledger.fbest)
    except Exception:
        point = ledger.project(start)
        value = float("inf")

    return (
        point,
        value,
        int(ledger.nfe - before),
    )


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
    lb[invalid] = -5.0
    ub[invalid] = 5.0

    ledger = EvaluationLedger(
        objective,
        lb,
        ub,
        max_evals,
    )
    archive = BasinArchive(
        max_nodes=options.archive_max_size,
        merge_radius_base=options.merge_radius_base,
        merge_radius_max=options.merge_radius_max,
        quality_weight=options.archive_quality_weight,
        diversity_weight=options.archive_diversity_weight,
        accessibility_weight=(
            options.archive_accessibility_weight
        ),
    )
    graph = BasinTransitionGraph(
        max_outgoing=options.graph_max_outgoing,
        max_incoming=options.graph_max_incoming,
        improvement_weight=(
            options.graph_edge_improvement_weight
        ),
        accessibility_weight=(
            options.graph_edge_accessibility_weight
        ),
        recency_weight=(
            options.graph_edge_recency_weight
        ),
    )
    probes = ProbePool(
        max_records=options.probe_pool_max_size
    )
    events: list[EventRecord] = []
    domain_radius = float(
        np.linalg.norm(ub - lb) + 1e-300
    )

    def event(
        event_name: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        record = EventRecord(
            nfe=int(ledger.nfe),
            phase=str(ledger.current_phase),
            event=str(event_name),
            message=str(message),
            payload=payload or {},
        )
        events.append(record)
        if len(events) > options.event_log_max_size:
            # Preserve the first half and most recent half.
            half = options.event_log_max_size // 2
            del events[half:-half]

    def active_node_ids() -> set[int]:
        return {
            node.node_id
            for node in archive.nodes
        }

    def record_probe(
        point: np.ndarray,
        value: float,
        *,
        phase: str,
        parent_node_id: int | None,
    ) -> ProbeRecord:
        return probes.add(
            point,
            value,
            source_phase=phase,
            created_nfe=ledger.nfe,
            parent_node_id=parent_node_id,
            nodes=archive.nodes,
            domain_radius=domain_radius,
        )

    def commit_certified(
        point: np.ndarray,
        value: float,
        *,
        certification_mode: str,
        parent_probe: ProbeRecord | None,
        refinement_evaluations: int,
        local_support_count: int,
        local_support_radius: float,
        start_value: float,
        source_node_id: int | None,
        barrier_proxy: float,
    ):
        improvement = float(
            start_value - value
        )
        update = archive.add_certified(
            point,
            value,
            certification_mode=certification_mode,
            parent_probe_id=(
                None
                if parent_probe is None
                else parent_probe.probe_id
            ),
            refinement_evaluations=(
                refinement_evaluations
            ),
            local_support_count=local_support_count,
            local_support_radius=local_support_radius,
            certification_improvement=improvement,
            curvature_proxy=_curvature_proxy(
                local_support_radius
            ),
            nfe=ledger.nfe,
            source=certification_mode,
            lb=lb,
            ub=ub,
            accessibility=graph.accessibility_map(),
        )

        if update.removed_node_ids:
            graph.remove_nodes(
                update.removed_node_ids
            )

        if (
            parent_probe is not None
            and update.active
        ):
            parent_probe.refinement_mode = (
                certification_mode
            )
            parent_probe.committed_node_id = (
                update.node.node_id
            )

        if (
            update.active
            and source_node_id is not None
        ):
            graph.add_or_update(
                source_node_id,
                update.node.node_id,
                evaluations=refinement_evaluations,
                improvement=improvement,
                barrier_proxy=barrier_proxy,
                nfe=ledger.nfe,
                source_mode=certification_mode,
                active_node_ids=active_node_ids(),
            )

        graph.prune(
            active_node_ids=active_node_ids(),
            current_nfe=ledger.nfe,
        )

        event(
            (
                "certified_node_created"
                if update.created
                else "certified_node_merged"
            ),
            certification_mode,
            {
                "node_id": update.node.node_id,
                "active": update.active,
                "parent_probe_id": (
                    None
                    if parent_probe is None
                    else parent_probe.probe_id
                ),
                "removed_node_ids": (
                    update.removed_node_ids
                ),
                "merge_threshold": (
                    update.merge_threshold
                ),
            },
        )
        return (
            update.node
            if update.active
            else None
        )

    # ----------------------------------------------------------
    # 1. Initial design: probes only.
    # ----------------------------------------------------------
    anchors = make_initial_anchors(
        lb,
        ub,
        rng,
        options.n_initial_lhs,
    )
    anchor_values: list[float] = []
    anchor_probes: list[ProbeRecord] = []

    with ledger.phase(
        "initial_design",
        _phase_limit(
            max_evals,
            options.initial_design_fraction,
        ),
    ):
        for point in anchors:
            try:
                value = ledger.evaluate(point)
            except (
                BudgetExhausted,
                PhaseBudgetExhausted,
            ):
                break
            anchor_values.append(value)
            anchor_probes.append(
                record_probe(
                    point,
                    value,
                    phase="initial_design",
                    parent_node_id=None,
                )
            )

    if ledger.xbest is None:
        ledger.xbest = 0.5 * (lb + ub)
        ledger.fbest = float("inf")

    diagnostics = evaluate_diagnostics(
        lb=lb,
        ub=ub,
        anchors=anchors[: len(anchor_values)],
        anchor_values=anchor_values,
    )

    event(
        "geometry_diagnostics",
        "derivative-free diagnostics computed",
        diagnostics.to_jsonable(),
    )

    if options.enable_geometry_controller:
        use_center_local = (
            diagnostics.dimension
            <= options.center_local_max_dim
            and diagnostics.local_mode_score
            >= options.center_local_min_score
            and diagnostics.curvature_anisotropy
            <= options.center_local_max_curvature_anisotropy
            and diagnostics.ruggedness_score
            <= options.center_local_max_ruggedness
        )
        use_far_basin = (
            options.enable_far_basin
            and (
                diagnostics.ruggedness_score
                >= options.far_basin_ruggedness_threshold
                or diagnostics.boundary_signal
                >= options.far_basin_boundary_threshold
                or diagnostics.max_scale
                >= options.large_box_scale
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
            "local_mode_score": (
                diagnostics.local_mode_score
            ),
            "curvature_anisotropy": (
                diagnostics.curvature_anisotropy
            ),
        },
    )

    # ----------------------------------------------------------
    # 2. Center-local certification.
    # ----------------------------------------------------------
    if use_center_local and ledger.remaining > 0:
        with ledger.phase(
            "center_local",
            _phase_limit(
                max_evals,
                options.center_local_fraction,
            ),
        ):
            center = 0.5 * (lb + ub)
            parent = min(
                anchor_probes,
                key=lambda probe: np.linalg.norm(
                    probe.point - center
                ),
            ) if anchor_probes else None
            start_value = (
                parent.f_value
                if parent is not None
                else float("inf")
            )
            point, value, used = _local_polish(
                ledger,
                center,
                options=options,
            )
            commit_certified(
                point,
                value,
                certification_mode="center_local",
                parent_probe=parent,
                refinement_evaluations=used,
                local_support_count=max(used, 1),
                local_support_radius=float(
                    np.linalg.norm(point - center)
                ),
                start_value=start_value,
                source_node_id=None,
                barrier_proxy=0.0,
            )

    # ----------------------------------------------------------
    # 3. Coordinate multi-bracket certification.
    # ----------------------------------------------------------
    with ledger.phase(
        "coordinate_sweep",
        _phase_limit(
            max_evals,
            options.coordinate_sweep_fraction,
        ),
    ):
        starts: list[
            tuple[
                np.ndarray,
                float,
                ProbeRecord | None,
                int | None,
            ]
        ] = []

        for node in archive.sorted_nodes()[
            : options.coordinate_start_count
        ]:
            starts.append(
                (
                    node.center.copy(),
                    node.f_center,
                    None,
                    node.node_id,
                )
            )

        for probe in probes.best_uncommitted(
            options.coordinate_start_count
        ):
            starts.append(
                (
                    probe.point.copy(),
                    probe.f_value,
                    probe,
                    probe.parent_node_id,
                )
            )

        unique_starts = []
        for item in starts:
            if not any(
                np.linalg.norm(
                    item[0] - existing[0]
                )
                / domain_radius
                <= 1e-12
                for existing in unique_starts
            ):
                unique_starts.append(item)

        for (
            start,
            start_value,
            parent_probe,
            source_node_id,
        ) in unique_starts[
            : options.coordinate_start_count
        ]:
            candidates = coordinate_multibracket_sweep(
                ledger,
                start,
                coarse_samples=options.line_coarse_samples,
                refine_top_k=(
                    options.line_refine_top_k
                    if options.enable_multibracket
                    else 1
                ),
                maxiter=options.line_refine_maxiter,
            )
            for candidate in candidates:
                commit_certified(
                    candidate.point,
                    candidate.f_value,
                    certification_mode=(
                        "coordinate_multibracket"
                    ),
                    parent_probe=parent_probe,
                    refinement_evaluations=(
                        candidate.evaluations
                    ),
                    local_support_count=(
                        candidate.local_support_count
                    ),
                    local_support_radius=(
                        candidate.local_support_radius
                    ),
                    start_value=start_value,
                    source_node_id=source_node_id,
                    barrier_proxy=(
                        candidate.barrier_proxy
                    ),
                )

    # Emergency certified node if coordinate phase did not create one.
    if not archive.nodes and ledger.remaining > 0:
        with ledger.phase(
            "principal_direction",
            _phase_limit(
                max_evals,
                options.principal_direction_fraction,
            ),
        ):
            parent = min(
                probes.records,
                key=lambda probe: probe.f_value,
            ) if probes.records else None
            start = (
                parent.point
                if parent is not None
                else ledger.xbest
            )
            start_value = (
                parent.f_value
                if parent is not None
                else ledger.fbest
            )
            point, value, used = _local_polish(
                ledger,
                start,
                options=options,
            )
            commit_certified(
                point,
                value,
                certification_mode="emergency_certification",
                parent_probe=parent,
                refinement_evaluations=used,
                local_support_count=max(used, 1),
                local_support_radius=float(
                    np.linalg.norm(point - start)
                ),
                start_value=start_value,
                source_node_id=None,
                barrier_proxy=0.0,
            )

    # ----------------------------------------------------------
    # 4. Principal-direction refinement.
    # ----------------------------------------------------------
    direction_diagnostics = DirectionDiagnostics(
        covariance_directions=0,
        transition_directions=0,
        coordinate_fallback_directions=0,
        retained_directions=0,
        direction_dimension=ledger.dimension,
    )

    if (
        options.enable_principal_directions
        and ledger.remaining > 0
    ):
        with ledger.phase(
            "principal_direction",
            _phase_limit(
                max_evals,
                options.principal_direction_fraction,
            ),
        ):
            directions, direction_diagnostics = (
                build_principal_directions(
                    archive,
                    graph,
                    dimension=ledger.dimension,
                    elite_count=(
                        options.principal_elite_count
                    ),
                    covariance_direction_count=(
                        options.principal_covariance_directions
                    ),
                    transition_direction_count=(
                        options.principal_transition_directions
                    ),
                    max_directions=(
                        options.principal_max_directions
                    ),
                    cosine_threshold=(
                        options.principal_cosine_threshold
                    ),
                )
            )
            event(
                "principal_directions_built",
                "principal direction set fixed",
                direction_diagnostics.to_jsonable(),
            )

            starts = archive.sorted_nodes()[
                : options.principal_start_count
            ]
            for source_node in starts:
                for direction in directions:
                    candidates = (
                        directional_multibracket_search(
                            ledger,
                            source_node.center,
                            direction,
                            coarse_samples=(
                                options.line_coarse_samples
                            ),
                            refine_top_k=(
                                options.line_refine_top_k
                                if options.enable_multibracket
                                else 1
                            ),
                            maxiter=(
                                options.line_refine_maxiter
                            ),
                        )
                    )
                    for candidate in candidates:
                        commit_certified(
                            candidate.point,
                            candidate.f_value,
                            certification_mode=(
                                "principal_direction"
                            ),
                            parent_probe=None,
                            refinement_evaluations=(
                                candidate.evaluations
                            ),
                            local_support_count=(
                                candidate.local_support_count
                            ),
                            local_support_radius=(
                                candidate.local_support_radius
                            ),
                            start_value=(
                                source_node.f_center
                            ),
                            source_node_id=(
                                source_node.node_id
                            ),
                            barrier_proxy=(
                                candidate.barrier_proxy
                            ),
                        )

    # ----------------------------------------------------------
    # 5. Far-basin probe-refine-commit.
    # ----------------------------------------------------------
    if use_far_basin and ledger.remaining > 0:
        with ledger.phase(
            "far_basin",
            _phase_limit(
                max_evals,
                options.far_basin_fraction,
            ),
        ):
            center = 0.5 * (lb + ub)

            while ledger.remaining > 0:
                batch: list[ProbeRecord] = []

                for _ in range(
                    options.far_probe_batch_size
                ):
                    direction = rng.standard_t(
                        df=1.5,
                        size=ledger.dimension,
                    )
                    norm = np.linalg.norm(direction)
                    if norm <= 1e-14:
                        continue
                    direction /= norm
                    radius = (
                        rng.random() ** 0.25
                    ) * 0.5 * domain_radius
                    point = ledger.project(
                        center + radius * direction
                    )
                    source_node, _ = archive.nearest(
                        ledger.xbest
                    )
                    try:
                        value = ledger.evaluate(point)
                    except (
                        BudgetExhausted,
                        PhaseBudgetExhausted,
                    ):
                        break
                    batch.append(
                        record_probe(
                            point,
                            value,
                            phase="far_basin",
                            parent_node_id=(
                                None
                                if source_node is None
                                else source_node.node_id
                            ),
                        )
                    )

                if not batch:
                    break

                selected = probes.select(
                    batch,
                    options.far_refine_per_batch,
                    quality_weight=0.60,
                    novelty_weight=0.40,
                )

                for probe in selected:
                    source_node = (
                        archive.node_by_id(
                            probe.parent_node_id
                        )
                        if probe.parent_node_id
                        is not None
                        else None
                    )
                    point, value, used = _local_polish(
                        ledger,
                        probe.point,
                        options=options,
                        max_local_evals=max(
                            8,
                            3 * ledger.dimension,
                        ),
                    )
                    commit_certified(
                        point,
                        value,
                        certification_mode=(
                            "far_basin_refinement"
                        ),
                        parent_probe=probe,
                        refinement_evaluations=used,
                        local_support_count=max(used, 1),
                        local_support_radius=float(
                            np.linalg.norm(
                                point - probe.point
                            )
                        ),
                        start_value=probe.f_value,
                        source_node_id=(
                            None
                            if source_node is None
                            else source_node.node_id
                        ),
                        barrier_proxy=max(
                            0.0,
                            probe.f_value
                            - (
                                source_node.f_center
                                if source_node is not None
                                else probe.f_value
                            ),
                        ),
                    )

    # ----------------------------------------------------------
    # 6. Graph-guided archive fallback.
    # ----------------------------------------------------------
    if (
        options.enable_archive_fallback
        and archive.nodes
        and ledger.remaining > 0
    ):
        with ledger.phase(
            "archive_fallback",
            _phase_limit(
                max_evals,
                options.archive_fallback_fraction,
            ),
        ):
            candidates = archive.sorted_nodes()

            if options.enable_graph_guidance:
                scores = graph.guidance_scores(
                    candidates,
                    quality_weight=(
                        options.graph_quality_weight
                    ),
                    novelty_weight=(
                        options.graph_novelty_weight
                    ),
                    accessibility_weight=(
                        options.graph_accessibility_weight
                    ),
                )
                candidates = sorted(
                    candidates,
                    key=lambda node: scores.get(
                        node.node_id,
                        0.0,
                    ),
                    reverse=True,
                )

            for source_node in candidates[
                : options.archive_fallback_starts
            ]:
                point, value, used = _local_polish(
                    ledger,
                    source_node.center,
                    options=options,
                )
                commit_certified(
                    point,
                    value,
                    certification_mode="archive_fallback",
                    parent_probe=None,
                    refinement_evaluations=used,
                    local_support_count=max(used, 1),
                    local_support_radius=float(
                        np.linalg.norm(
                            point - source_node.center
                        )
                    ),
                    start_value=source_node.f_center,
                    source_node_id=source_node.node_id,
                    barrier_proxy=max(
                        0.0,
                        value - source_node.f_center,
                    ),
                )

    # ----------------------------------------------------------
    # 7. Final polishing.
    # ----------------------------------------------------------
    if (
        options.enable_final_polish
        and archive.nodes
        and ledger.remaining > 0
    ):
        with ledger.phase(
            "final_polish",
            _phase_limit(
                max_evals,
                options.final_polish_fraction,
            ),
        ):
            for source_node in archive.sorted_nodes()[
                : options.final_polish_starts
            ]:
                point, value, used = _local_polish(
                    ledger,
                    source_node.center,
                    options=options,
                )
                commit_certified(
                    point,
                    value,
                    certification_mode="final_polish",
                    parent_probe=None,
                    refinement_evaluations=used,
                    local_support_count=max(used, 1),
                    local_support_radius=float(
                        np.linalg.norm(
                            point - source_node.center
                        )
                    ),
                    start_value=source_node.f_center,
                    source_node_id=source_node.node_id,
                    barrier_proxy=max(
                        0.0,
                        value - source_node.f_center,
                    ),
                )

    # ----------------------------------------------------------
    # 8. Graph-aware probe-refine-commit completion.
    # ----------------------------------------------------------
    with ledger.phase(
        "budget_completion",
        None,
    ):
        while ledger.remaining > 0:
            batch: list[ProbeRecord] = []
            nodes = archive.sorted_nodes()

            for _ in range(
                options.completion_probe_batch_size
            ):
                if ledger.remaining <= 0:
                    break

                source_node = None
                if (
                    nodes
                    and options.enable_graph_guidance
                    and rng.random() < 0.80
                ):
                    scores = graph.guidance_scores(
                        nodes,
                        quality_weight=(
                            options.graph_quality_weight
                        ),
                        novelty_weight=(
                            options.graph_novelty_weight
                        ),
                        accessibility_weight=(
                            options.graph_accessibility_weight
                        ),
                    )
                    weights = np.asarray(
                        [
                            max(
                                scores.get(
                                    node.node_id,
                                    0.0,
                                ),
                                0.0,
                            )
                            for node in nodes
                        ],
                        dtype=float,
                    )
                    if weights.sum() <= 0:
                        weights = np.ones(len(nodes))
                    weights /= weights.sum()
                    source_node = nodes[
                        int(
                            rng.choice(
                                len(nodes),
                                p=weights,
                            )
                        )
                    ]
                    decay = max(
                        0.04,
                        (
                            ledger.remaining
                            / max(max_evals, 1)
                        )
                        ** 0.5,
                    )
                    point = ledger.project(
                        source_node.center
                        + rng.standard_t(
                            df=2.0,
                            size=ledger.dimension,
                        )
                        * (
                            0.18
                            * decay
                            * (ub - lb)
                        )
                    )
                else:
                    point = ledger.project(
                        lb
                        + rng.random(
                            ledger.dimension
                        )
                        * (ub - lb)
                    )
                    source_node, _ = archive.nearest(
                        ledger.xbest
                    )

                try:
                    value = ledger.evaluate(point)
                except BudgetExhausted:
                    break

                batch.append(
                    record_probe(
                        point,
                        value,
                        phase="budget_completion",
                        parent_node_id=(
                            None
                            if source_node is None
                            else source_node.node_id
                        ),
                    )
                )

            if not batch:
                break

            selected = probes.select(
                batch,
                options.completion_refine_per_batch,
                quality_weight=0.65,
                novelty_weight=0.35,
            )

            for probe in selected:
                source_node = (
                    archive.node_by_id(
                        probe.parent_node_id
                    )
                    if probe.parent_node_id
                    is not None
                    else None
                )
                local_budget = min(
                    ledger.remaining,
                    max(
                        8,
                        options.completion_local_budget_factor
                        * ledger.dimension,
                    ),
                )
                if local_budget <= 0:
                    continue

                point, value, used = _local_polish(
                    ledger,
                    probe.point,
                    options=options,
                    max_local_evals=local_budget,
                )
                commit_certified(
                    point,
                    value,
                    certification_mode=(
                        "budget_completion_refinement"
                    ),
                    parent_probe=probe,
                    refinement_evaluations=used,
                    local_support_count=max(used, 1),
                    local_support_radius=float(
                        np.linalg.norm(
                            point - probe.point
                        )
                    ),
                    start_value=probe.f_value,
                    source_node_id=(
                        None
                        if source_node is None
                        else source_node.node_id
                    ),
                    barrier_proxy=max(
                        0.0,
                        probe.f_value
                        - (
                            source_node.f_center
                            if source_node is not None
                            else probe.f_value
                        ),
                    ),
                )

    graph.prune(
        active_node_ids=active_node_ids(),
        current_nfe=ledger.nfe,
    )

    message = (
        "budget_exhausted"
        if ledger.remaining == 0
        else "completed"
    )

    return BasinGraphResult(
        implementation_version=(
            IMPLEMENTATION_VERSION
        ),
        options_hash=options.stable_hash(),
        options=options.to_jsonable(),
        xbest=ledger.xbest.copy(),
        fbest=float(ledger.fbest),
        nfe=int(ledger.nfe),
        phase_evaluations=dict(
            ledger.phase_evaluations
        ),
        history=ledger.history,
        probes=list(probes.records),
        probe_count_total=probes.total_created,
        archive=archive.sorted_nodes(),
        graph_edges=graph.to_edges(),
        diagnostics=diagnostics,
        direction_diagnostics=(
            direction_diagnostics
        ),
        event_log=events,
        message=message,
    )
