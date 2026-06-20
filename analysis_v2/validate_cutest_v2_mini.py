#!/usr/bin/env python3
"""Validate Route B Step B4 CUTEst mini outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results_v2" / "cutest_mini"

raw = OUT / "cutest_v2_mini_raw_results.csv"
summary = OUT / "cutest_v2_mini_summary.json"

rows = list(csv.DictReader(raw.open()))
summary_data = json.loads(summary.read_text())

assert len(rows) == 24, len(rows)
assert summary_data["status"] == "V2_CUTEST_MINI_OK"

assert Counter(r["algorithm"] for r in rows) == {
    "BasinGraph_v2": 6,
    "CMA_ES": 6,
    "BIPOP_CMA_ES": 6,
    "Multi_start_LBFGSB": 6,
}

assert Counter(r["dimension_group"] for r in rows) == {
    "small_2_20": 8,
    "medium_21_100": 8,
    "large_101_500": 8,
}

failed = [r for r in rows if r["runner_status"] != "completed"]
assert not failed, failed

bg_rows = [r for r in rows if r["algorithm"] == "BasinGraph_v2"]
assert all(int(r["nfe_internal"]) == int(r["budget"]) for r in bg_rows)
assert all(int(r["archive_nodes"]) > 0 for r in bg_rows)
assert sum(int(r["graph_edges"]) for r in bg_rows) > 0

for row in bg_rows:
    path = PROJECT_ROOT / row["detail_json"]
    data = json.loads(path.read_text())
    assert data["nfe"] == int(row["budget"])
    assert "archive" in data
    assert "graph_edges" in data
    assert "diagnostics" in data
    assert "event_log" in data
    assert len(data["archive"]) == int(row["archive_nodes"])
    assert len(data["graph_edges"]) == int(row["graph_edges"])

print("V2_CUTEST_MINI_VALIDATION_OK")
print("rows:", len(rows))
print("algorithms:", dict(Counter(r["algorithm"] for r in rows)))
print("dimension groups:", dict(Counter(r["dimension_group"] for r in rows)))
print("BasinGraph_v2 total archive nodes:", sum(int(r["archive_nodes"]) for r in bg_rows))
print("BasinGraph_v2 total graph edges:", sum(int(r["graph_edges"]) for r in bg_rows))
