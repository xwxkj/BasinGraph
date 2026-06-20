import numpy as np

from basingraph_v2.optimizer import BasinGraphOptions, minimize_basingraph_v2


def objective(x):
    x = np.asarray(x)
    return float(np.sum((x - 0.5) ** 2) + 0.1 * np.sum(np.cos(7 * x)))


def test_graph_references_only_active_nodes():
    result = minimize_basingraph_v2(
        objective,
        -3 * np.ones(3),
        3 * np.ones(3),
        max_evals=600,
        seed=3,
        options=BasinGraphOptions(archive_max_size=20),
    )
    active = {node.node_id for node in result.archive}
    assert all(edge.source_id in active for edge in result.graph_edges)
    assert all(edge.target_id in active for edge in result.graph_edges)


def test_no_graph_guidance_flag_is_serialized():
    options = BasinGraphOptions(enable_graph_guidance=False)
    result = minimize_basingraph_v2(
        objective,
        -2 * np.ones(2),
        2 * np.ones(2),
        max_evals=300,
        seed=4,
        options=options,
    )
    assert result.options["enable_graph_guidance"] is False
    assert result.nfe == 300
