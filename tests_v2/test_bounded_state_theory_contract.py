"""Machine checks for the bounded operational-state propositions."""

from __future__ import annotations

import numpy as np

from basingraph_v2.optimizer import BasinGraphOptions, minimize_basingraph_v2


def test_best_so_far_trace_is_not_history_identifier() -> None:
    # Two different evaluated points and identical objective sequences produce
    # the same best-so-far trace. The theorem is combinatorial; this test keeps
    # its constructive witness executable.
    first_history = [
        (np.array([0.0]), 0.0),
        (np.array([1.0]), 1.0),
    ]
    second_history = [
        (np.array([0.0]), 0.0),
        (np.array([-1.0]), 1.0),
    ]
    first_trace = np.minimum.accumulate([value for _, value in first_history])
    second_trace = np.minimum.accumulate([value for _, value in second_history])
    assert np.array_equal(first_trace, second_trace)
    assert not np.array_equal(first_history[1][0], second_history[1][0])


def test_returned_state_capacity_integrity_and_accounting() -> None:
    options = BasinGraphOptions()
    dimension = 5

    def objective(point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float)
        return float(np.sum((point - 0.37) ** 2))

    result = minimize_basingraph_v2(
        objective,
        -5.0 * np.ones(dimension),
        5.0 * np.ones(dimension),
        max_evals=500,
        seed=20260811,
        options=options,
    )
    active = {int(node.node_id) for node in result.archive}
    assert len(result.archive) <= options.archive_max_size == 80
    assert all(
        int(edge.source_id) in active and int(edge.target_id) in active
        for edge in result.graph_edges
    )
    assert all(int(edge.source_id) != int(edge.target_id) for edge in result.graph_edges)
    assert len(result.graph_edges) <= len(active) * max(0, len(active) - 1)
    assert len(result.history) == result.nfe
    assert sum(result.phase_evaluations.values()) == result.nfe
    values = [float(value) for _, value in result.history]
    assert all(later <= earlier for earlier, later in zip(values[:-1], values[1:]))
