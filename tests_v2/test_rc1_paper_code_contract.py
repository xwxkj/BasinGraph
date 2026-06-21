import numpy as np

from basingraph_v2.archive import BasinArchive
from basingraph_v2.graph import BasinTransitionGraph
from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)

EXPECTED_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)


def test_selected_rc1_identity():
    assert IMPLEMENTATION_VERSION == "2.0.0-rc1"
    assert BasinGraphOptions().stable_hash() == EXPECTED_HASH


def test_operational_archive_merge_and_edge_formula():
    archive = BasinArchive(merge_radius_factor=0.035, max_nodes=4)
    graph = BasinTransitionGraph()
    lb = -np.ones(2)
    ub = np.ones(2)

    update_a = archive.add_or_merge(
        np.asarray([0.0, 0.0]),
        1.0,
        radius=0.01,
        curvature_proxy=1.0,
        nfe=1,
        source="anchor",
        lb=lb,
        ub=ub,
    )
    update_b = archive.add_or_merge(
        np.asarray([0.8, 0.8]),
        0.5,
        radius=0.01,
        curvature_proxy=1.0,
        nfe=2,
        source="far_basin",
        lb=lb,
        ub=ub,
    )
    assert update_a.node.node_id != update_b.node.node_id

    edge = graph.add_or_update(
        update_a.node.node_id,
        update_b.node.node_id,
        evaluations=2,
        improvement=0.5,
        barrier_proxy=3.0,
        nfe=2,
        source_mode="test",
    )
    assert edge is not None
    assert np.isclose(edge.accessibility, 1.0 / 6.0)


def test_selected_rc1_result_contract():
    def sphere(x):
        x = np.asarray(x, dtype=float)
        return float(np.sum(x * x))

    result = minimize_basingraph_v2(
        sphere,
        -5.0 * np.ones(3),
        5.0 * np.ones(3),
        max_evals=300,
        seed=9,
    )
    payload = result.to_jsonable()

    assert result.nfe == 300
    assert sum(result.phase_evaluations.values()) == 300
    assert result.options_hash == EXPECTED_HASH
    assert len(payload["archive"]) > 0
    assert "graph_edges" in payload
    assert "diagnostics" in payload
    assert payload["diagnostics"]["anisotropy"] == 1.0
