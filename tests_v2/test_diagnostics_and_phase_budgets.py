import numpy as np

from basingraph_v2.diagnostics import (
    evaluate_diagnostics,
    make_initial_anchors,
)
from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


def test_axis_triplet_sign_change_and_curvature_are_computable():
    lb = -np.ones(2)
    ub = np.ones(2)
    rng = np.random.default_rng(1)
    anchors = make_initial_anchors(lb, ub, rng, n_lhs=0)

    def saddle(x):
        return float(x[0] ** 2 - x[1] ** 2)

    values = [saddle(x) for x in anchors]
    diagnostics = evaluate_diagnostics(
        lb=lb,
        ub=ub,
        anchors=anchors,
        anchor_values=values,
    )

    assert diagnostics.valid_axis_triplets == 2
    assert 0.0 <= diagnostics.sign_change_rate <= 1.0
    assert diagnostics.domain_anisotropy == 1.0
    assert len(diagnostics.curvature_values) == 2


def test_phase_counts_sum_to_total_budget_and_rc2_identity():
    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))

    options = BasinGraphOptions()
    result = minimize_basingraph_v2(
        sphere,
        -5 * np.ones(4),
        5 * np.ones(4),
        max_evals=500,
        seed=2,
        options=options,
    )

    assert result.nfe == 500
    assert sum(result.phase_evaluations.values()) == 500
    assert result.options_hash == options.stable_hash()
    assert IMPLEMENTATION_VERSION == "2.0.0-rc2"
    assert result.implementation_version == "2.0.0-rc2"
    assert result.probe_count_total >= len(result.probes)
    assert all(node.certified for node in result.archive)
