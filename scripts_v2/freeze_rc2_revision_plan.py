#!/usr/bin/env python3
"""
Route B Step B6D1: freeze the single rc2 structural revision and its
development-only acceptance gate.

This script performs no optimizer runs and does not access holdout data.
It reads only the completed rc1 COCO development partition (instances 1-3),
its serialized BasinGraph diagnostics, and the reviewed cocopp output.

The resulting files prevent post-hoc movement of the rc2 design or its
acceptance criteria.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_v2" / "formal_development" / "coco_rc1"
OUT = ROOT / "protocols" / "route_b" / "rc2_revision"

RC1_IMPLEMENTATION = "2.0.0-rc1"
RC1_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
RC1_ALGORITHM_FREEZE_TAG = "route-b-v2.0.0-rc1-ablationfreeze"
RC1_PROTOCOL_FREEZE_TAG = "route-b-v2.0.0-rc1-protocolfreeze"

# Values read from the official development cocopp aggregate ECDF SVGs.
# Fractions are proportions of function-target pairs reached by the indicated
# evaluation budget per dimension.
RC1_COCOPP_AGGREGATE = {
    "2D": {
        "10d": 0.1441,
        "100d": 0.5365,
        "1000d": 0.7436,
    },
    "5D": {
        "10d": 0.0591,
        "100d": 0.3400,
        "1000d": 0.5605,
    },
    "10D": {
        "10d": 0.1268,
        "100d": 0.2876,
        "1000d": 0.4630,
    },
}

RC1_HIGH_CONDITIONING_1000D = {
    "2D": 0.8680,
    "5D": 0.7150,
    "10D": 0.6118,
}

RC1_HARDEST_TARGET_SUCCESSES = {
    "2D": 40,
    "5D": 22,
    "10D": 15,
    "total": 77,
    "total_problems": 216,
}


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    branch = git_output("branch", "--show-current")
    if branch != "route-b/full-basingraph-v2":
        raise RuntimeError(f"Wrong branch: {branch}")

    run_id = (RESULT_ROOT / "LAST_RUN_ID.txt").read_text().strip()
    run_root = RESULT_ROOT / run_id
    raw_path = run_root / "coco_v2_development_raw_results.csv"
    validation_path = run_root / "validation_report.json"
    diag_root = run_root / "development_diagnostics"
    diag_report_path = diag_root / "development_diagnostic_report.json"
    group_path = diag_root / "algorithm_function_group_summary.csv"
    internal_path = diag_root / "basingraph_internal_by_group_dimension.csv"
    correlations_path = diag_root / "basingraph_rank_correlations.csv"
    overall_path = diag_root / "algorithm_overall_summary.csv"
    cocopp_path_file = run_root / "cocopp_output_path.txt"

    for path in [
        raw_path,
        validation_path,
        diag_report_path,
        group_path,
        internal_path,
        correlations_path,
        overall_path,
        cocopp_path_file,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    raw = pd.read_csv(raw_path)
    validation = json.loads(validation_path.read_text())
    diagnostic = json.loads(diag_report_path.read_text())
    groups = pd.read_csv(group_path)
    internal = pd.read_csv(internal_path)
    correlations = pd.read_csv(correlations_path)
    overall = pd.read_csv(overall_path)

    if set(raw["instance_index"].unique()) != {1, 2, 3}:
        raise RuntimeError("Development data contains non-development instances.")
    if validation["holdout_leakage"] is not False:
        raise RuntimeError("Validation report indicates holdout leakage.")
    if validation["status"] != "V2_COCO_DEVELOPMENT_VALIDATION_OK":
        raise RuntimeError("Development validation did not pass.")

    bg_overall = overall[overall["algorithm"] == "BasinGraph_v2"].iloc[0]
    bg_groups = groups[groups["algorithm"] == "BasinGraph_v2"].copy()
    center_row = correlations[
        correlations["feature"] == "phase_center_local_fraction"
    ].iloc[0]

    baseline_snapshot = {
        "status": "RC1_DEVELOPMENT_EVIDENCE_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "development_run_id": run_id,
        "development_instances": [1, 2, 3],
        "holdout_accessed": False,
        "rc1_implementation": RC1_IMPLEMENTATION,
        "rc1_options_hash": RC1_OPTIONS_HASH,
        "rc1_algorithm_freeze_tag": RC1_ALGORITHM_FREEZE_TAG,
        "rc1_algorithm_freeze_commit": git_output(
            "rev-list", "-n", "1", RC1_ALGORITHM_FREEZE_TAG
        ),
        "rc1_protocol_freeze_tag": RC1_PROTOCOL_FREEZE_TAG,
        "rc1_protocol_freeze_commit": git_output(
            "rev-list", "-n", "1", RC1_PROTOCOL_FREEZE_TAG
        ),
        "plan_freeze_commit_parent": git_output("rev-parse", "HEAD"),
        "rows": int(len(raw)),
        "problems": int(raw["problem_id"].nunique()),
        "basingraph_final_value_mean_rank": float(bg_overall["mean_rank"]),
        "basingraph_final_value_wins": int(bg_overall["wins"]),
        "basingraph_group_mean_ranks": {
            str(row.function_group): float(row.mean_rank)
            for row in bg_groups.itertuples(index=False)
        },
        "archive_saturation_fraction": float(
            diagnostic["internal_summary"]["archive_saturation_fraction"]
        ),
        "mean_archive_nodes": float(
            diagnostic["internal_summary"]["mean_archive_nodes"]
        ),
        "mean_graph_edges": float(
            diagnostic["internal_summary"]["mean_graph_edges"]
        ),
        "mean_graph_edges_per_node": float(
            diagnostic["internal_summary"]["mean_graph_edges_per_node"]
        ),
        "center_local_activation_fraction": float(
            diagnostic["internal_summary"]["center_local_activation_fraction"]
        ),
        "far_basin_activation_fraction": float(
            diagnostic["internal_summary"]["far_basin_activation_fraction"]
        ),
        "center_local_rank_correlation": {
            "spearman_rho": float(center_row["spearman_rho_with_rank"]),
            "p_value": float(center_row["p_value"]),
        },
        "cocopp_aggregate_target_fractions": RC1_COCOPP_AGGREGATE,
        "cocopp_high_conditioning_1000d": RC1_HIGH_CONDITIONING_1000D,
        "hardest_target_successes": RC1_HARDEST_TARGET_SUCCESSES,
        "source_files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in [
                raw_path,
                validation_path,
                diag_report_path,
                group_path,
                internal_path,
                correlations_path,
                overall_path,
            ]
        },
        "cocopp_output_path": cocopp_path_file.read_text().strip(),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    snapshot_path = OUT / "RC1_DEVELOPMENT_EVIDENCE_SNAPSHOT.json"
    snapshot_path.write_text(json.dumps(baseline_snapshot, indent=2))

    rationale = """# Decision: one structural rc2 revision is required

## Scope

This decision uses only the frozen COCO development partition, instances 1-3.
No prospective COCO or CUTEst holdout result was accessed.

## Evidence

The official cocopp development output shows a consistent pattern:

- BasinGraph is competitive at early and intermediate budgets;
- its aggregate ECDF plateaus relative to CMA-ES and BIPOP-CMA-ES near the
  1000d budget;
- the late-budget deficit is strongest on high-conditioning/unimodal
  functions and at finer targets;
- hardest-target successes are 40/72, 22/72 and 15/72 in 2D, 5D and 10D.

The serialized mechanism diagnostics show:

- all 216 development runs saturate the 80-node archive;
- low-dimensional graphs are much denser than higher-dimensional graphs;
- center-local contraction is activated in almost every run;
- center-local budget fraction has a weak positive association with worse
  final-value rank.

## Semantic defect in rc1

In rc1, raw initial anchors, far-basin probes and budget-completion probes can
be committed directly as `BasinNode` objects. Consequently, archive saturation
does not necessarily mean that 80 distinct attraction basins were discovered.
This is inconsistent with the intended manuscript semantics of nodes as
refined basin representatives.

The rc1 anisotropy statistic is based on box-width ratios. BBOB uses isotropic
box widths, so this statistic cannot diagnose objective-landscape
conditioning.

## Decision

Implement exactly one structural release candidate, `v2.0.0-rc2`, before
opening any prospective holdout. The revision is mechanism-driven rather than
a post-hoc parameter search. Its design and acceptance gate are frozen in this
directory before code is changed.
"""
    (OUT / "RC1_DEVELOPMENT_REVIEW_DECISION.md").write_text(rationale)

    rc2_spec = """# BasinGraph v2.0.0-rc2 frozen revision specification

## 1. Certified basin semantics

`ProbeRecord` and `BasinNode` become distinct objects.

Raw points from initial design, far-basin probing and budget completion are
stored only as probes. A point may enter the basin archive only after a
certification operation:

- local polishing;
- coordinate multi-bracket refinement;
- principal-direction multi-bracket refinement;
- archive-fallback polishing;
- stall-triggered polishing.

Each active node records `certified=True`, `certification_mode`, parent probe
identifier, refinement evaluations and local-support metadata.

## 2. Adaptive basin identity

The merge threshold is

`r_merge = min(r_max, r_base * (1 + 2 * occupancy^2)) * ||ub-lb||_2`.

Frozen constants:

- `r_base = 0.025`;
- `r_max = 0.080`;
- `occupancy = active_nodes / archive_capacity`.

## 3. Quality-diversity-accessibility retention

The global-best node is protected. When capacity is exceeded, all other nodes
are ranked by

`0.50 * quality + 0.30 * diversity + 0.20 * accessibility`.

Quality is reverse objective rank, diversity is normalized nearest-neighbour
distance, and accessibility is the maximum incoming edge accessibility.
The lowest-scoring unprotected node is evicted, with all incident edges.

## 4. Landscape curvature anisotropy

Replace box-width anisotropy by a derivative-free curvature proxy. For each
valid lower-centre-upper coordinate triplet,

`kappa_j = |f(x_j^-) - 2 f(c) + f(x_j^+)| / h_j^2`.

Curvature anisotropy is the robust ratio `q90(kappa)/max(q10(kappa), eps)`,
capped at `1e6`. The box-width ratio remains available as
`domain_anisotropy` but must not be described as landscape anisotropy.

## 5. Principal-direction refinement

Add directional coarse sampling plus top-K bounded refinement along:

- leading eigenvectors of elite-node covariance;
- normalized successful transition directions;
- coordinate directions as deterministic fallback.

The directional line-search routine must share the same bracket semantics as
the coordinate multi-bracket routine.

## 6. Probe-refine-commit exploration

Far-basin and budget-completion phases use batches:

1. generate probes;
2. score probes by objective quality and distance from certified nodes;
3. refine only the best/most novel probes;
4. commit only certified refined results;
5. create graph edges only between certified nodes.

No raw probe may become a graph node.

## 7. Graph sparsification

For each active node retain at most:

- three outgoing edges;
- three incoming edges.

Edge retention score combines best improvement, accessibility and recency.
Graph pruning occurs after each node or edge update.

## 8. Controller and phase allocation

Frozen rc2 phase fractions:

- initial design: 0.10;
- center-local: 0.08;
- coordinate multi-bracket: 0.25;
- principal-direction refinement: 0.15;
- far-basin probe/refine: 0.12;
- archive fallback: 0.10;
- final polishing: 0.08;
- remaining/unused budget: graph-aware completion.

Center-local contraction is enabled only when:

- dimension <= 20;
- local-mode score >= 0.20;
- curvature anisotropy <= 100;
- ruggedness score <= 0.10.

## 9. Required result contract

Every rc2 result must serialize:

- implementation version and options hash;
- probes and certified archive nodes separately;
- certification metadata for every node;
- sparse transition graph;
- curvature and domain anisotropy separately;
- principal-direction diagnostics;
- exact phase evaluation counts;
- event log;
- graph/archive referential integrity.

## 10. Change budget

No other algorithmic mechanism or parameter search is permitted before the
first rc2 development evaluation. Any correction necessary for a software bug
must be documented separately and must not use holdout evidence.
"""
    (OUT / "RC2_FROZEN_REVISION_SPEC.md").write_text(rc2_spec)

    gate = {
        "status": "RC2_DEVELOPMENT_ACCEPTANCE_GATE_FROZEN",
        "development_only": True,
        "prospective_holdout_must_remain_unopened": True,
        "comparison": "paired rc2 versus rc1 on identical COCO instances 1-3",
        "integrity_gates_all_required": {
            "implementation_version": "2.0.0-rc2",
            "all_runs_complete": True,
            "exact_function_evaluation_accounting": True,
            "phase_evaluation_sum_equals_nfe": True,
            "all_archive_nodes_certified": True,
            "raw_probe_nodes_in_archive": 0,
            "graph_edges_reference_certified_active_nodes_only": True,
            "landscape_anisotropy_not_constant_on_bbob": True,
            "principal_direction_phase_exercised": True,
        },
        "mechanism_gates_all_required": {
            "archive_saturation_fraction_max": 0.75,
            "maximum_group_dimension_mean_edges_per_node": 3.0,
            "center_local_activation_fraction_max": 0.80,
        },
        "noninferiority_gates_all_required": {
            "aggregate_ecdf_100d_max_absolute_drop_per_dimension": 0.02,
            "fixed_budget_mean_rank_max": 3.49,
            "weak_structure_multimodal_mean_rank_max": 3.21,
        },
        "improvement_gates_require_at_least": 2,
        "improvement_gates": {
            "mean_aggregate_ecdf_1000d_min_gain": 0.02,
            "mean_high_conditioning_ecdf_1000d_5d_10d_min_gain": 0.03,
            "hardest_target_successes_min": 82,
            "fixed_budget_mean_rank_target": 3.34,
        },
        "rc1_reference": {
            "aggregate_ecdf": RC1_COCOPP_AGGREGATE,
            "high_conditioning_1000d": RC1_HIGH_CONDITIONING_1000D,
            "hardest_target_successes": RC1_HARDEST_TARGET_SUCCESSES,
            "fixed_budget_mean_rank": float(bg_overall["mean_rank"]),
            "weak_structure_multimodal_mean_rank": float(
                bg_groups[
                    bg_groups["function_group"]
                    == "Multimodal with weak structure (f20-f24)"
                ]["mean_rank"].iloc[0]
            ),
        },
        "decision": {
            "accept_rc2": (
                "all integrity, mechanism and noninferiority gates pass, "
                "and at least two improvement gates pass"
            ),
            "reject_rc2": (
                "otherwise retain rc1 or explicitly terminate optimization "
                "development; do not inspect holdout"
            ),
        },
    }
    gate_path = OUT / "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json"
    gate_path.write_text(json.dumps(gate, indent=2))

    implementation_plan = """# rc2 implementation order

1. Add `ProbeRecord`, node certification metadata and result serialization.
2. Refactor archive so only certified nodes can be inserted.
3. Implement adaptive merge threshold and quality-diversity-accessibility
   eviction.
4. Implement curvature anisotropy and retain domain anisotropy separately.
5. Generalize the line search to arbitrary directions.
6. Implement principal-direction generation and refinement.
7. Convert far-basin and budget completion to probe-refine-commit batches.
8. Add graph degree pruning.
9. Update controller and frozen phase allocations.
10. Add unit tests for every semantic invariant.
11. Run smoke tests only.
12. Run paired rc1-versus-rc2 development evaluation.
13. Apply the frozen acceptance gate.
14. Only after acceptance, create the final v2.0.0 code tag and open holdout.
"""
    (OUT / "RC2_IMPLEMENTATION_ORDER.md").write_text(implementation_plan)

    manifest_path = OUT / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(OUT.iterdir()):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    print("RC2_REVISION_PLAN_FROZEN")
    print("run id:", run_id)
    print("rc1 mean rank:", float(bg_overall["mean_rank"]))
    print(
        "archive saturation:",
        baseline_snapshot["archive_saturation_fraction"],
    )
    print(
        "center-local correlation:",
        baseline_snapshot["center_local_rank_correlation"],
    )
    print("output:", OUT)


if __name__ == "__main__":
    main()
