#!/usr/bin/env python3
"""Zero-objective-evaluation preflight for B21 Track C."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    C1_INSTANCES,
    NIST_DATASETS,
    PAIRED_SEEDS,
    SMOKE_TASKS,
    sha256_file,
    verify_source_identity,
)

C1_DIMENSIONS = {
    "elliptic_pde_inverse": [6, 8, 10],
    "lorenz63_calibration": [6, 6, 6],
    "phase_retrieval": [16, 24, 32],
    "noisy_phase_retrieval": [16, 24, 32],
    "matrix_factorization": [36, 60, 96],
    "large_matrix_factorization": [120, 200, 320],
    "burgers_control": [8, 12, 16],
    "allen_cahn_energy": [32, 64, 96],
    "sparse_nonlinear_inverse": [20, 40, 80],
}
C1_BUDGET_MULTIPLIERS = {
    "elliptic_pde_inverse": 300,
    "lorenz63_calibration": 300,
    "phase_retrieval": 200,
    "noisy_phase_retrieval": 200,
    "matrix_factorization": 150,
    "large_matrix_factorization": 75,
    "burgers_control": 200,
    "allen_cahn_energy": 100,
    "sparse_nonlinear_inverse": 150,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory preflight requires authorization.")
    identity = verify_source_identity(require_clean=True)
    checks: dict[str, bool] = {
        "algorithm_count": len(ALGORITHMS) == 7,
        "c1_family_count": len(C1_FAMILIES) == 9,
        "nist_dataset_count": len(NIST_DATASETS) == 6,
        "c1_instances": C1_INSTANCES == [11, 12, 13],
        "paired_seeds": PAIRED_SEEDS == list(range(10)),
        "dimension_table": set(C1_DIMENSIONS) == set(C1_FAMILIES),
        "budget_table": set(C1_BUDGET_MULTIPLIERS) == set(C1_FAMILIES),
        "positive_dimensions": all(d > 0 for values in C1_DIMENSIONS.values() for d in values),
        "positive_budgets": all(v > 0 for v in C1_BUDGET_MULTIPLIERS.values()),
    }
    provenance_path = ROOT / "protocols/evidence_extension_v1/track_c/TRACK_C_NIST_PROVENANCE.csv"
    rows = list(csv.DictReader(provenance_path.open(encoding="utf-8")))
    checks["nist_provenance_rows"] = len(rows) == 6
    checks["nist_provenance_names"] = {row["dataset"] for row in rows} == set(NIST_DATASETS)
    for row in rows:
        path = ROOT / row["repository_path"]
        checks[f"nist_file::{row['dataset']}"] = path.is_file()
        if path.is_file():
            checks[f"nist_sha::{row['dataset']}"] = sha256_file(path) == row["sha256"]

    if args.mode == "smoke":
        expected_rows = len(SMOKE_TASKS) * len(ALGORITHMS)
        task_units = len(SMOKE_TASKS)
    else:
        c1_rows = len(C1_FAMILIES) * len(C1_INSTANCES) * len(PAIRED_SEEDS) * len(ALGORITHMS)
        c2_rows = len(NIST_DATASETS) * len(PAIRED_SEEDS) * len(ALGORITHMS)
        expected_rows = c1_rows + c2_rows
        task_units = len(C1_FAMILIES) * len(C1_INSTANCES) + len(NIST_DATASETS)
        checks["confirmatory_rows"] = expected_rows == 2310
        checks["partition_disjoint"] = not set(C1_INSTANCES).intersection({1, 2})

    report = {
        "status": "TRACK_C_PREFLIGHT_OK" if all(checks.values()) else "TRACK_C_PREFLIGHT_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "identity": identity,
        "task_units": task_units,
        "algorithms": len(ALGORITHMS),
        "expected_rows": expected_rows,
        "algorithm_objective_evaluations": 0,
        "checks_total": len(checks),
        "checks_passed": sum(checks.values()),
        "checks_failed": sum(not v for v in checks.values()),
        "checks": checks,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError("Track C preflight failed.")
    print("TRACK_C_PREFLIGHT_OK")
    print(json.dumps({"expected_rows": expected_rows, "algorithm_objective_evaluations": 0}, indent=2))


if __name__ == "__main__":
    main()
