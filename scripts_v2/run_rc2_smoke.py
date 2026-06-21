#!/usr/bin/env python3
"""Run deterministic BasinGraph v2.0.0-rc2 smoke problems."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


def rastrigin(x):
    x = np.asarray(x)
    return float(
        10 * x.size
        + np.sum(
            x * x
            - 10 * np.cos(2 * np.pi * x)
        )
    )


def ellipsoid(x):
    x = np.asarray(x)
    weights = np.logspace(0, 6, x.size)
    return float(
        np.sum(weights * (x - 0.25) ** 2)
        / weights.sum()
    )


out = ROOT / "results_v2" / "rc2_smoke"
out.mkdir(parents=True, exist_ok=True)

problems = [
    (
        "rastrigin_d5",
        rastrigin,
        -5.12 * np.ones(5),
        5.12 * np.ones(5),
        1200,
        20260619,
    ),
    (
        "ellipsoid_d10",
        ellipsoid,
        -5.0 * np.ones(10),
        5.0 * np.ones(10),
        2000,
        20260620,
    ),
]

summaries = []

for (
    name,
    objective,
    lb,
    ub,
    budget,
    seed,
) in problems:
    result = minimize_basingraph_v2(
        objective,
        lb,
        ub,
        budget,
        seed,
    )
    payload = result.to_jsonable()
    path = out / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2)
    )

    active = {
        node.node_id
        for node in result.archive
    }
    raw_modes = {
        "initial_design",
        "far_basin",
        "budget_completion",
    }

    assert result.nfe == budget
    assert sum(
        result.phase_evaluations.values()
    ) == budget
    assert all(
        node.certified
        and node.certification_mode
        not in raw_modes
        for node in result.archive
    )
    assert all(
        edge.source_id in active
        and edge.target_id in active
        for edge in result.graph_edges
    )

    summaries.append(
        {
            "problem": name,
            "implementation_version": (
                result.implementation_version
            ),
            "options_hash": result.options_hash,
            "budget": budget,
            "nfe": result.nfe,
            "fbest": result.fbest,
            "probe_count_total": (
                result.probe_count_total
            ),
            "retained_probes": len(result.probes),
            "certified_nodes": len(result.archive),
            "graph_edges": len(result.graph_edges),
            "archive_saturated": (
                len(result.archive)
                >= BasinGraphOptions().archive_max_size
            ),
            "curvature_anisotropy": (
                result.diagnostics.curvature_anisotropy
            ),
            "domain_anisotropy": (
                result.diagnostics.domain_anisotropy
            ),
            "principal_directions": (
                result.direction_diagnostics.retained_directions
            ),
            "phase_evaluations": (
                result.phase_evaluations
            ),
        }
    )

summary_path = out / "rc2_smoke_summary.json"
summary_path.write_text(
    json.dumps(summaries, indent=2)
)

print("RC2_SMOKE_OK")
print("implementation:", IMPLEMENTATION_VERSION)
print(
    "options hash:",
    BasinGraphOptions().stable_hash(),
)
for summary in summaries:
    print(json.dumps(summary, indent=2))
