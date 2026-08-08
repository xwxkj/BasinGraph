
import numpy as np
from basingraph_v2.optimizer import minimize_basingraph_v2

def test_archive_and_graph_edges_are_explicit():
    def multimodal(x): return float(np.sum((x-1.0)**2) + 0.2*np.sum(np.sin(5*x)))
    result = minimize_basingraph_v2(multimodal, -3*np.ones(4), 3*np.ones(4), 300, seed=7)
    assert len(result.archive) >= 1
    assert all(hasattr(node, "node_id") for node in result.archive)
    assert all(hasattr(edge, "accessibility") for edge in result.graph_edges)
    data = result.to_jsonable()
    assert "archive" in data and "graph_edges" in data
