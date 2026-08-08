#!/usr/bin/env python3
"""Audit BasinGraph repository identity before any B21 experiment.

The audit is intentionally offline. It checks the frozen source-tree Git blob
identities, public import routing, options hash, manuscript-level defaults,
metadata/DOI files, exact evaluation accounting and graph/archive integrity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import tomllib
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "protocols"
    / "evidence_extension_v1"
    / "STEP0_IDENTITY_CONTRACT.json"
)


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results_b21/step0/identity_audit_report.json",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--require-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    expected = contract["result_bearing"]
    expected_software_doi = contract["public_records"]["software_doi"]
    expected_dataset_doi = contract["public_records"]["dataset_doi"]

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Any = None) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "details": details,
            }
        )

    # --------------------------------------------------------------
    # Frozen result-bearing source tree
    # --------------------------------------------------------------
    for relative, expected_blob in expected["source_git_blob_sha1"].items():
        path = ROOT / relative
        observed = git_blob_sha1(path) if path.is_file() else None
        record(
            f"frozen_source_blob::{relative}",
            observed == expected_blob,
            {"expected": expected_blob, "observed": observed},
        )

    # --------------------------------------------------------------
    # Public import routing and frozen defaults
    # --------------------------------------------------------------
    from basingraph import (  # noqa: WPS433
        BasinGraphOptions as PublicOptions,
        BasinGraphResult as PublicResult,
        IMPLEMENTATION_VERSION as public_version,
        minimize_basingraph,
    )
    from basingraph_v2.optimizer import (  # noqa: WPS433
        BasinGraphOptions as FrozenOptions,
        IMPLEMENTATION_VERSION as frozen_version,
        minimize_basingraph_v2,
    )
    from basingraph_v2.types import BasinGraphResult as FrozenResult  # noqa: WPS433

    record(
        "public_function_is_frozen_function",
        minimize_basingraph is minimize_basingraph_v2,
        {
            "public_module": minimize_basingraph.__module__,
            "public_name": minimize_basingraph.__name__,
        },
    )
    record("public_options_is_frozen_options", PublicOptions is FrozenOptions)
    record("public_result_is_frozen_result", PublicResult is FrozenResult)
    record(
        "implementation_version",
        public_version == frozen_version == expected["internal_implementation_version"],
        {
            "expected": expected["internal_implementation_version"],
            "public": public_version,
            "frozen": frozen_version,
        },
    )

    options = FrozenOptions()
    observed_hash = options.stable_hash()
    record(
        "frozen_options_hash",
        observed_hash == expected["options_hash"],
        {"expected": expected["options_hash"], "observed": observed_hash},
    )
    record(
        "archive_capacity",
        options.archive_max_size == expected["archive_capacity"],
        {
            "expected": expected["archive_capacity"],
            "observed": options.archive_max_size,
        },
    )
    record(
        "line_coarse_samples",
        options.line_coarse_samples == expected["line_coarse_samples"],
    )
    record(
        "line_refine_top_k",
        options.line_refine_top_k == expected["line_refine_top_k"],
    )

    phase_fields = {
        "initial_design": "initial_design_fraction",
        "center_local": "center_local_fraction",
        "coordinate_sweep": "coordinate_sweep_fraction",
        "far_basin": "far_basin_fraction",
        "archive_fallback": "archive_fallback_fraction",
        "final_polish": "final_polish_fraction",
    }
    for phase, field in phase_fields.items():
        expected_value = float(expected["phase_fractions"][phase])
        observed_value = float(getattr(options, field))
        record(
            f"phase_fraction::{phase}",
            np.isclose(observed_value, expected_value, rtol=0.0, atol=1e-15),
            {"expected": expected_value, "observed": observed_value},
        )

    # --------------------------------------------------------------
    # Packaging and metadata
    # --------------------------------------------------------------
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_includes = set(
        pyproject.get("tool", {})
        .get("setuptools", {})
        .get("packages", {})
        .get("find", {})
        .get("include", [])
    )
    record(
        "package_includes_result_bearing_source",
        {"basingraph", "basingraph_v2"}.issubset(package_includes),
        sorted(package_includes),
    )
    record(
        "package_version_marks_step0_development",
        pyproject["project"]["version"] == "3.0.1.dev0",
        pyproject["project"]["version"],
    )

    authoritative_files = {
        "README.md": (expected_software_doi, expected_dataset_doi),
        "CITATION.cff": (expected_software_doi,),
        "CODE_AVAILABILITY_TEMPLATE.md": (expected_software_doi,),
        "DATA_AVAILABILITY_TEMPLATE.md": (expected_dataset_doi,),
        ".zenodo.json": (expected_software_doi, expected_dataset_doi),
    }
    superseded_or_unverified = {
        "10.5281/zenodo.20791905",
        "10.5281/zenodo.20800092",
        "10.5281/zenodo.20800093",
        "10.5281/zenodo.20791231",
    }
    for relative, required_values in authoritative_files.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        record(
            f"authoritative_doi_present::{relative}",
            all(value in text for value in required_values),
            {"required": list(required_values)},
        )
        found_forbidden = sorted(value for value in superseded_or_unverified if value in text)
        record(
            f"no_unverified_doi::{relative}",
            not found_forbidden,
            {"found": found_forbidden},
        )

    compatibility_source = (ROOT / "basingraph" / "optimizer.py").read_text(
        encoding="utf-8"
    )
    record(
        "legacy_default_implementation_removed",
        "class BudgetExhausted" not in compatibility_source
        and "from basingraph_v2.optimizer import" in compatibility_source,
    )

    # --------------------------------------------------------------
    # Deterministic local smoke test
    # --------------------------------------------------------------
    smoke_report: dict[str, Any] | None = None
    if not args.skip_smoke:
        shift = np.asarray([1.25, -0.75, 0.5], dtype=float)

        def objective(x: np.ndarray) -> float:
            z = np.asarray(x, dtype=float) - shift
            return float(np.dot(z, z))

        budget = 240
        result = minimize_basingraph(
            objective,
            -5.0 * np.ones(3),
            5.0 * np.ones(3),
            max_evals=budget,
            seed=20260807,
            options=PublicOptions(),
        )
        active_ids = {node.node_id for node in result.archive}
        graph_valid = all(
            edge.source_id in active_ids and edge.target_id in active_ids
            for edge in result.graph_edges
        )
        history_values = [float(value) for _, value in result.history]
        monotone = all(
            later <= earlier + 1e-14
            for earlier, later in zip(history_values[:-1], history_values[1:])
        )
        phase_sum = int(sum(result.phase_evaluations.values()))
        smoke_report = {
            "budget": budget,
            "nfe": result.nfe,
            "fbest": result.fbest,
            "phase_sum": phase_sum,
            "archive_nodes": len(result.archive),
            "graph_edges": len(result.graph_edges),
            "graph_referential_integrity": graph_valid,
            "best_history_monotone": monotone,
            "message": result.message,
        }
        record("smoke_budget_exhausted", result.nfe == budget, smoke_report)
        record("smoke_phase_accounting", phase_sum == result.nfe, smoke_report)
        record("smoke_archive_capacity", 0 < len(result.archive) <= 80, smoke_report)
        record("smoke_graph_integrity", graph_valid, smoke_report)
        record("smoke_best_history_monotone", monotone, smoke_report)

    # --------------------------------------------------------------
    # Optional root-manifest verification
    # --------------------------------------------------------------
    manifest_path = ROOT / "MANIFEST_SHA256.csv"
    if manifest_path.is_file():
        mismatches = []
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                path = ROOT / row["relative_path"]
                if not path.is_file():
                    mismatches.append({"path": row["relative_path"], "reason": "missing"})
                    continue
                observed_size = path.stat().st_size
                observed_sha = sha256(path)
                if observed_size != int(row["size_bytes"]) or observed_sha != row["sha256"]:
                    mismatches.append(
                        {
                            "path": row["relative_path"],
                            "reason": "mismatch",
                            "expected_size": int(row["size_bytes"]),
                            "observed_size": observed_size,
                            "expected_sha256": row["sha256"],
                            "observed_sha256": observed_sha,
                        }
                    )
        record("root_manifest_verifies", not mismatches, mismatches[:25])
    else:
        record(
            "root_manifest_present",
            not args.require_manifest,
            "Run scripts_v2/generate_repository_manifest.py before release.",
        )

    passed = all(item["passed"] for item in checks)
    report = {
        "status": "B21_STEP0_IDENTITY_AUDIT_OK" if passed else "B21_STEP0_IDENTITY_AUDIT_FAILED",
        "contract": str(CONTRACT_PATH.relative_to(ROOT)),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "checks_total": len(checks),
        "checks_passed": sum(item["passed"] for item in checks),
        "checks_failed": sum(not item["passed"] for item in checks),
        "smoke": smoke_report,
        "checks": checks,
    }

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report["status"])
    print(json.dumps({k: report[k] for k in ("checks_total", "checks_passed", "checks_failed")}, indent=2))
    print(f"Report: {output_path}")

    if not passed:
        for item in checks:
            if not item["passed"]:
                print(f"FAILED: {item['name']} :: {item['details']}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
