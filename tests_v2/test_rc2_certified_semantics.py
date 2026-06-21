import numpy as np

from basingraph_v2.optimizer import (
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


def objective(x):
    x = np.asarray(x)
    return float(
        np.sum((x - 0.7) ** 2)
        + 0.15 * np.sum(np.cos(6 * x))
    )


def test_rc2_probe_and_certified_node_semantics():
    result = minimize_basingraph_v2(
        objective,
        -4 * np.ones(4),
        4 * np.ones(4),
        max_evals=800,
        seed=12,
    )

    assert IMPLEMENTATION_VERSION == "2.0.0-rc2"
    assert result.nfe == 800
    assert result.probe_count_total > 0
    assert len(result.probes) > 0
    assert len(result.archive) > 0
    assert all(node.certified for node in result.archive)
    assert all(
        node.certification_mode
        not in {
            "initial_design",
            "far_basin",
            "budget_completion",
        }
        for node in result.archive
    )
    assert sum(result.phase_evaluations.values()) == result.nfe
