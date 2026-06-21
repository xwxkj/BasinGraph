#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

import cocoex


ROOT = Path(__file__).resolve().parents[1]

suite = cocoex.Suite(
    "bbob",
    "instances: 4-15",
    "dimensions: 2,3,5,10,20 function_indices: 1-24",
)

ids = list(suite.ids())
pattern = re.compile(r"bbob_f(\d+)_i(\d+)_d(\d+)")

parsed = []
for problem_id in ids:
    match = pattern.fullmatch(problem_id)
    if not match:
        raise RuntimeError(f"Unexpected problem ID: {problem_id}")
    parsed.append(tuple(int(value) for value in match.groups()))

functions = sorted({function for function, _, _ in parsed})
instances = sorted({instance for _, instance, _ in parsed})
dimensions = sorted({dimension for _, _, dimension in parsed})

report = {
    "status": "COCO_HOLDOUT_SUITE_PREFLIGHT_OK",
    "suite": "bbob",
    "suite_instance": "instances: 4-15",
    "suite_options": "dimensions: 2,3,5,10,20 function_indices: 1-24",
    "problem_count": len(ids),
    "functions": functions,
    "instances": instances,
    "dimensions": dimensions,
    "first_problem_id": ids[0],
    "last_problem_id": ids[-1],
    "objective_evaluations_performed": 0,
}

assert len(ids) == 24 * 5 * 12
assert functions == list(range(1, 25))
assert instances == list(range(4, 16))
assert dimensions == [2, 3, 5, 10, 20]

output = (
    ROOT
    / "protocols"
    / "route_b"
    / "final_rc1"
    / "COCO_HOLDOUT_SUITE_PREFLIGHT.json"
)
output.write_text(json.dumps(report, indent=2) + "\n")

print("COCO_HOLDOUT_SUITE_PREFLIGHT_OK")
print(json.dumps(report, indent=2))
