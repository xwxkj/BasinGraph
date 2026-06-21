import numpy as np

from basingraph_v2.archive import BasinArchive
from basingraph_v2.graph import BasinTransitionGraph
from basingraph_v2.optimizer import (
    BasinGraphOptions,
    minimize_basingraph_v2,
)


def test_adaptive_merge_radius_increases_with_occupancy():
    archive = BasinArchive(
        max_nodes=10,
        merge_radius_base=0.025,
        merge_radius_max=0.080,
    )
    lb = -np.ones(2)
    ub = np.ones(2)

    low = archive.merge_threshold(lb, ub)

    for i in range(8):
        archive.add_certified(
            np.asarray([-0.9 + 0.2 * i, 0.8]),
            float(i),
            certification_mode="test",
            parent_probe_id=None,
            refinement_evaluations=1,
            local_support_count=1,
            local_support_radius=0.01,
            certification_improvement=1.0,
            curvature_proxy=1.0,
            nfe=i + 1,
            source="test",
            lb=lb,
            ub=ub,
            accessibility={},
        )

    high = archive.merge_threshold(lb, ub)
    assert high > low


def test_sparse_graph_and_certified_references():
    def function(x):
        x = np.asarray(x)
        return float(
            np.sum(x * x)
            + 0.2 * np.sum(np.sin(5 * x))
        )

    result = minimize_basingraph_v2(
        function,
        -5 * np.ones(3),
        5 * np.ones(3),
        max_evals=1000,
        seed=13,
        options=BasinGraphOptions(
            archive_max_size=30,
        ),
    )

    active = {
        node.node_id
        for node in result.archive
    }
    assert all(
        edge.source_id in active
        and edge.target_id in active
        for edge in result.graph_edges
    )

    outgoing = {}
    incoming = {}
    for edge in result.graph_edges:
        outgoing[edge.source_id] = (
            outgoing.get(edge.source_id, 0) + 1
        )
        incoming[edge.target_id] = (
            incoming.get(edge.target_id, 0) + 1
        )

    assert max(outgoing.values(), default=0) <= 3
    assert max(incoming.values(), default=0) <= 3
