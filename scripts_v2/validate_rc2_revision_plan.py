#!/usr/bin/env python3
"""Validate the frozen rc2 revision plan without running an optimizer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "protocols" / "route_b" / "rc2_revision"

required = [
    "RC1_DEVELOPMENT_EVIDENCE_SNAPSHOT.json",
    "RC1_DEVELOPMENT_REVIEW_DECISION.md",
    "RC2_FROZEN_REVISION_SPEC.md",
    "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json",
    "RC2_IMPLEMENTATION_ORDER.md",
    "MANIFEST_SHA256.csv",
]
for name in required:
    assert (OUT / name).exists(), name

snapshot = json.loads(
    (OUT / "RC1_DEVELOPMENT_EVIDENCE_SNAPSHOT.json").read_text()
)
gate = json.loads(
    (OUT / "RC2_DEVELOPMENT_ACCEPTANCE_GATE.json").read_text()
)
manifest = pd.read_csv(OUT / "MANIFEST_SHA256.csv")

assert snapshot["status"] == "RC1_DEVELOPMENT_EVIDENCE_FROZEN"
assert snapshot["development_instances"] == [1, 2, 3]
assert snapshot["holdout_accessed"] is False
assert snapshot["rows"] == 1512
assert snapshot["problems"] == 216
assert snapshot["archive_saturation_fraction"] == 1.0
assert snapshot["hardest_target_successes"]["total"] == 77

assert gate["status"] == "RC2_DEVELOPMENT_ACCEPTANCE_GATE_FROZEN"
assert gate["prospective_holdout_must_remain_unopened"] is True
assert gate["improvement_gates_require_at_least"] == 2
assert gate["integrity_gates_all_required"]["raw_probe_nodes_in_archive"] == 0

for row in manifest.itertuples(index=False):
    path = OUT / row.filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row.sha256, row.filename

print("RC2_REVISION_PLAN_VALIDATION_OK")
print("manifest entries:", len(manifest))
print("rc1 mean rank:", snapshot["basingraph_final_value_mean_rank"])
print("rc1 hardest-target successes:", snapshot["hardest_target_successes"]["total"])
print("holdout accessed:", snapshot["holdout_accessed"])
