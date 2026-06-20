import numpy as np

from basingraph_v2.diagnostics import make_initial_anchors, evaluate_diagnostics
from basingraph_v2.optimizer import minimize_basingraph_v2


def test_axis_triplet_sign_change_is_computable():
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


def test_phase_counts_sum_to_total_budget():
    def sphere(x):
        return float(np.sum(np.asarray(x) ** 2))

    result = minimize_basingraph_v2(
        sphere,
        -5 * np.ones(4),
        5 * np.ones(4),
        max_evals=500,
        seed=2,
    )

    assert result.nfe == 500
    assert sum(result.phase_evaluations.values()) == 500
    assert result.options_hash
    assert result.implementation_version == "2.0.0-rc1"
