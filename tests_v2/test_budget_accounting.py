
import numpy as np
from basingraph_v2.optimizer import minimize_basingraph_v2

def test_budget_is_exhausted_and_outputs_exist():
    def sphere(x): return float(np.sum(x*x))
    result = minimize_basingraph_v2(sphere, -5*np.ones(3), 5*np.ones(3), 200, seed=123)
    assert result.nfe == 200
    assert result.message == "budget_exhausted"
    assert len(result.archive) > 0
    assert result.diagnostics.dimension == 3
    assert isinstance(result.to_jsonable(), dict)
