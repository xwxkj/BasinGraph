#!/usr/bin/env python3
"""Validate Route B Step B3 COCO mini outputs."""

from __future__ import annotations

from pathlib import Path
import csv
import json
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results_v2" / "coco_mini"
EXDATA_ROOT = PROJECT_ROOT / "exdata" / "routeb_v2_coco_mini"

raw = OUT / "coco_v2_mini_raw_results.csv"
summary = OUT / "coco_v2_mini_summary.json"

rows = list(csv.DictReader(raw.open()))
summary_data = json.loads(summary.read_text())

assert len(rows) == 27, len(rows)

alg_counts = Counter(row["algorithm"] for row in rows)
assert alg_counts == {
    "BasinGraph_v2": 9,
    "CMA_ES": 9,
    "BIPOP_CMA_ES": 9,
}, alg_counts

problem_counts = Counter(row["function_instance"] for row in rows)
assert len(problem_counts) == 9, problem_counts
assert all(v == 3 for v in problem_counts.values()), problem_counts

bg_rows = [row for row in rows if row["algorithm"] == "BasinGraph_v2"]
assert all(int(row["nfe_internal"]) == int(row["budget"]) for row in bg_rows)
assert all(int(row["archive_nodes"]) > 0 for row in bg_rows)
assert sum(int(row["graph_edges"]) for row in bg_rows) > 0

for row in bg_rows:
    path = PROJECT_ROOT / row["detail_json"]
    data = json.loads(path.read_text())

    assert "archive" in data
    assert "graph_edges" in data
    assert "diagnostics" in data
    assert "event_log" in data
    assert data["nfe"] == int(row["budget"])
    assert len(data["archive"]) == int(row["archive_nodes"])
    assert len(data["graph_edges"]) == int(row["graph_edges"])

info_files = list(EXDATA_ROOT.rglob("*.info"))
dat_files = list(EXDATA_ROOT.rglob("*.dat"))

assert len(info_files) > 0
assert len(dat_files) > 0
assert summary_data["status"] == "V2_COCO_MINI_OK"

print("V2_COCO_MINI_VALIDATION_OK")
print("rows:", len(rows))
print("algorithms:", dict(alg_counts))
print("problems:", len(problem_counts))
print("COCO info files:", len(info_files))
print("COCO dat files:", len(dat_files))
print("BasinGraph_v2 total archive nodes:", sum(int(r["archive_nodes"]) for r in bg_rows))
print("BasinGraph_v2 total graph edges:", sum(int(r["graph_edges"]) for r in bg_rows))
