#!/usr/bin/env python3
# Validate the selected rc1 paper-code semantic contract.

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)

EXPECTED_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
OUT = ROOT / "protocols" / "route_b" / "final_rc1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract = json.loads(
        (OUT / "RC1_PAPER_CODE_CONTRACT.json").read_text()
    )
    assert contract["status"] == "RC1_PAPER_CODE_CONTRACT_FROZEN"
    assert contract["implementation_version"] == "2.0.0-rc1"
    assert contract["options_hash"] == EXPECTED_HASH
    assert contract["holdout_accessed"] is False
    assert IMPLEMENTATION_VERSION == "2.0.0-rc1"
    assert BasinGraphOptions().stable_hash() == EXPECTED_HASH

    codefreeze = subprocess.check_output(
        ["git", "rev-list", "-n", "1", contract["code_freeze_tag"]],
        cwd=ROOT,
        text=True,
    ).strip()
    assert codefreeze == contract["code_freeze_commit"]

    for relative, expected in contract["source_hashes"].items():
        assert sha256_file(ROOT / relative) == expected, relative

    def objective(x):
        x = np.asarray(x, dtype=float)
        return float(
            np.sum((x - 0.4) ** 2)
            + 0.1 * np.sum(np.sin(5.0 * x))
        )

    result = minimize_basingraph_v2(
        objective,
        -3.0 * np.ones(3),
        3.0 * np.ones(3),
        max_evals=400,
        seed=20260620,
    )
    payload = result.to_jsonable()

    assert result.nfe == 400
    assert sum(result.phase_evaluations.values()) == 400
    assert result.implementation_version == "2.0.0-rc1"
    assert result.options_hash == EXPECTED_HASH
    assert len(payload["archive"]) > 0
    assert "graph_edges" in payload
    assert "diagnostics" in payload
    assert "event_log" in payload
    assert payload["diagnostics"]["anisotropy"] == 1.0

    active = {node["node_id"] for node in payload["archive"]}
    assert all(
        edge["source_id"] in active and edge["target_id"] in active
        for edge in payload["graph_edges"]
    )

    language_rows = list(
        csv.DictReader((OUT / "RC1_MANUSCRIPT_LANGUAGE_RULES.csv").open())
    )
    prohibited = {
        row["language"]
        for row in language_rows
        if row["category"] == "prohibited"
    }
    assert "certified basin node" in prohibited
    assert "objective curvature anisotropy" in prohibited

    manifest_rows = list(
        csv.DictReader((OUT / "MANIFEST_SHA256.csv").open())
    )
    for row in manifest_rows:
        assert sha256_file(OUT / row["filename"]) == row["sha256"]

    print("RC1_PAPER_CODE_CONTRACT_VALIDATION_OK")
    print("implementation:", IMPLEMENTATION_VERSION)
    print("options hash:", EXPECTED_HASH)
    print("semantic smoke nfe:", result.nfe)
    print("semantic smoke archive nodes:", len(result.archive))
    print("semantic smoke graph edges:", len(result.graph_edges))
    print("manifest entries:", len(manifest_rows))
    print("holdout accessed:", False)


if __name__ == "__main__":
    main()
