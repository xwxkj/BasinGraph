import numpy as np

from basingraph_v2.diagnostics import (
    evaluate_diagnostics,
    make_initial_anchors,
)
from basingraph_v2.optimizer import (
    minimize_basingraph_v2,
)


def test_curvature_anisotropy_detects_ellipsoid():
    lb = -np.ones(4)
    ub = np.ones(4)
    rng = np.random.default_rng(1)
    anchors = make_initial_anchors(
        lb,
        ub,
        rng,
        n_lhs=0,
    )
    weights = np.asarray([1.0, 10.0, 100.0, 1000.0])
    values = [
        float(np.sum(weights * point * point))
        for point in anchors
    ]
    diagnostics = evaluate_diagnostics(
        lb=lb,
        ub=ub,
        anchors=anchors,
        anchor_values=values,
    )

    assert diagnostics.domain_anisotropy == 1.0
    assert diagnostics.curvature_anisotropy > 100.0
    assert len(diagnostics.curvature_values) == 4


def test_principal_direction_phase_is_exercised():
    def function(x):
        x = np.asarray(x)
        return float(
            np.sum(
                np.logspace(0, 4, x.size)
                * (x - 0.3) ** 2
            )
        )

    result = minimize_basingraph_v2(
        function,
        -3 * np.ones(6),
        3 * np.ones(6),
        max_evals=1200,
        seed=14,
    )

    assert (
        result.phase_evaluations.get(
            "principal_direction",
            0,
        )
        > 0
    )
    assert (
        result.direction_diagnostics.retained_directions
        > 0
    )
