#!/usr/bin/env python3
"""Run registered descriptive task-specific references for Track C."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_c.common import (  # noqa: E402
    C1_INSTANCES,
    NIST_DATASETS,
    verify_source_identity,
)
from evidence_extension_v1.track_c.nist import make_nist_task  # noqa: E402
from evidence_extension_v1.track_c.tasks import make_c1_task  # noqa: E402


SMOKE_REFERENCES = [
    ("c1", "phase_retrieval", 1),
    ("c1", "matrix_factorization", 1),
    ("c2", "BoxBOD", 0),
]

CONFIRMATORY_REFERENCES = (
    [
        ("c1", family, instance)
        for family in [
            "phase_retrieval",
            "noisy_phase_retrieval",
            "matrix_factorization",
            "large_matrix_factorization",
        ]
        for instance in C1_INSTANCES
    ]
    + [("c2", name, 0) for name in NIST_DATASETS]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError(
            "Task-specific confirmatory references require authorization."
        )

    identity = verify_source_identity(require_clean=True)
    specifications = (
        SMOKE_REFERENCES
        if args.mode == "smoke"
        else CONFIRMATORY_REFERENCES
    )

    rows: list[dict[str, object]] = []
    for domain, name, instance in specifications:
        task = (
            make_c1_task(name, instance)
            if domain == "c1"
            else make_nist_task(name)
        )
        if task.specialized_reference is None:
            raise RuntimeError(
                f"No registered task-specific reference for {task.task_id}"
            )

        started = time.perf_counter()
        result = task.specialized_reference()
        elapsed = time.perf_counter() - started
        xbest = np.asarray(result["xbest"], dtype=float)
        fbest = float(result["fbest"])

        if xbest.shape != (task.dimension,):
            raise RuntimeError(
                f"Reference shape mismatch for {task.task_id}: {xbest.shape}"
            )
        if not np.all(np.isfinite(xbest)) or not np.isfinite(fbest):
            raise RuntimeError(
                f"Non-finite reference result for {task.task_id}"
            )
        if np.any(xbest < task.lb - 1e-12) or np.any(xbest > task.ub + 1e-12):
            raise RuntimeError(
                f"Reference result outside bounds for {task.task_id}"
            )

        metrics = task.metrics(xbest)
        rows.append(
            {
                "mode": args.mode,
                "domain": domain,
                "task_name": name,
                "task_id": task.task_id,
                "instance": instance,
                "dimension": task.dimension,
                "method": str(result["method"]),
                "fbest": fbest,
                "normalized_gap": float(
                    max(
                        0.0,
                        (fbest - task.f_ref)
                        / max(task.f_base - task.f_ref, 1e-15),
                    )
                ),
                "iterations": int(result.get("iterations", 0)),
                "objective_evaluations": int(
                    result.get("objective_evaluations", 0)
                ),
                "wall_time_seconds": elapsed,
                "secondary_metrics_json": json.dumps(
                    metrics,
                    sort_keys=True,
                    allow_nan=False,
                ),
                "method_metadata_json": json.dumps(
                    result.get("metadata", {}),
                    sort_keys=True,
                    allow_nan=False,
                ),
                "xbest_json": json.dumps(
                    xbest.tolist(),
                    separators=(",", ":"),
                ),
            }
        )

    expected = 3 if args.mode == "smoke" else 18
    if len(rows) != expected:
        raise RuntimeError(
            f"Reference count mismatch: {len(rows)} != {expected}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "status": "TRACK_C_TASK_SPECIFIC_REFERENCES_OK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "rows": len(rows),
        "source_identity_version": identity.get("identity_version"),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("TRACK_C_TASK_SPECIFIC_REFERENCES_OK")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
