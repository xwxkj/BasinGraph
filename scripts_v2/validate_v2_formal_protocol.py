#!/usr/bin/env python3
"""Validate the frozen Route B formal experiment protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "protocols" / "route_b" / "formal_v2_protocol"

required = [
    "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv",
    "CUTEST_V2_HOLDOUT_SELECTION_AUDIT.csv",
    "CUTEST_V2_ABLATION_DEVELOPMENT_15.csv",
    "COCO_V2_PARTITIONS.csv",
    "V2_FINAL_ABLATION_DESIGN.json",
    "V2_BASELINE_PROVENANCE_PLAN.csv",
    "V2_CLAIM_EVIDENCE_MAP.csv",
    "CUTEST_V2_HOLDOUT_SUMMARY.json",
    "V2_FORMAL_EXPERIMENT_PROTOCOL.md",
    "V2_FORMAL_PROTOCOL_LOCK.json",
    "V2_FORMAL_PROTOCOL_MANIFEST_SHA256.csv",
]

for name in required:
    assert (OUT / name).exists(), name

holdout = pd.read_csv(OUT / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv")
summary = json.loads((OUT / "CUTEST_V2_HOLDOUT_SUMMARY.json").read_text())
lock = json.loads((OUT / "V2_FORMAL_PROTOCOL_LOCK.json").read_text())
manifest = pd.read_csv(OUT / "V2_FORMAL_PROTOCOL_MANIFEST_SHA256.csv")

assert len(holdout) == 24
assert holdout["instance_id"].nunique() == 24
assert holdout["dimension_group"].value_counts().to_dict() == {
    "small_2_20": 11,
    "medium_21_100": 7,
    "large_101_500": 6,
}
assert summary["optimizer_runs_performed_by_this_script"] == 0
assert summary["prospective_holdout_instances"] == 24
assert lock["status"] == "V2_FORMAL_PROTOCOL_FROZEN"
assert lock["implementation_version"] == "2.0.0-rc1"
assert lock["options_hash"] == (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
assert lock["coco"]["development_instances"] == [1, 2, 3]
assert lock["coco"]["holdout_instances"] == list(range(4, 16))

for row in manifest.itertuples(index=False):
    path = OUT / row.filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row.sha256, row.filename

print("V2_FORMAL_PROTOCOL_VALIDATION_OK")
print("CUTEst holdout rows:", len(holdout))
print(
    "CUTEst holdout groups:",
    holdout["dimension_group"].value_counts().sort_index().to_dict(),
)
print("CUTEst unique base problems:", holdout["problem_name"].nunique())
print("Protocol manifest entries:", len(manifest))
