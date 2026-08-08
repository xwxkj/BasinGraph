#!/usr/bin/env python3
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
from basingraph_v2.optimizer import minimize_basingraph_v2

def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return float(10*x.size + np.sum(x*x - 10*np.cos(2*np.pi*x)))

out = Path("results_v2/smoke")
out.mkdir(parents=True, exist_ok=True)
result = minimize_basingraph_v2(rastrigin, -5.12*np.ones(5), 5.12*np.ones(5), 1000, seed=20260619)
(out/"v2_smoke_result.json").write_text(json.dumps(result.to_jsonable(), indent=2))
print("V2_SMOKE_OK")
print("nfe:", result.nfe)
print("fbest:", result.fbest)
print("archive nodes:", len(result.archive))
print("graph edges:", len(result.graph_edges))
print("result:", out/"v2_smoke_result.json")
