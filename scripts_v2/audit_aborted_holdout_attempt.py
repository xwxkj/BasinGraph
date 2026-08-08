#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path.cwd()
    run_id = args.run_id
    locations = [
        root / "logs_v2" / run_id,
        root / "results_v2" / "formal_holdout" / "coco_rc1" / run_id,
        root / "exdata" / "routeb_v2_holdout" / run_id,
    ]

    inventory = []
    problem_ids = set()

    for base in locations:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            inventory.append(
                {
                    "relative_path": relative,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
            problem_ids.update(
                re.findall(
                    r"bbob_f\d{3}_i\d{2,3}_d\d{2,3}",
                    relative,
                )
            )

    result_root = (
        root
        / "results_v2"
        / "formal_holdout"
        / "coco_rc1"
        / run_id
    )
    detail_files = sorted(
        str(path.relative_to(root))
        for path in result_root.rglob("*.json.gz")
    )

    audit_dir = (
        root
        / "protocols"
        / "route_b"
        / "final_rc1"
        / "aborted_holdout_attempts"
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "status": "ABORTED_HOLDOUT_RUNNER_SELECTION_ERROR",
        "run_id": run_id,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "root_cause": (
            "The runner used suite_options instance_indices: 4-15. "
            "COCO interprets these as ordinal positions in the suite instance "
            "list, not as actual BBOB instance identifiers. Actual identifiers "
            "4-15 must be supplied with suite_instance='instances: 4-15'."
        ),
        "observed_failure_problem_id": "bbob_f001_i71_d02",
        "partial_basingraph_detail_files": detail_files,
        "problem_ids_visible_in_paths": sorted(problem_ids),
        "performance_summary_generated": False,
        "performance_values_inspected": False,
        "operator_review_scope": (
            "Only exception logs, file paths and file counts were inspected."
        ),
        "algorithm_source_or_options_modified": False,
        "disposition": (
            "Retain as an audit record, exclude all partial files from final "
            "analysis, apply a runner-only fix, and rerun all algorithms under "
            "a new run ID."
        ),
        "file_count": len(inventory),
        "files": inventory,
    }

    json_path = audit_dir / f"{run_id}.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n")

    md_path = audit_dir / f"{run_id}.md"
    md_path.write_text(
        f"# Aborted COCO holdout attempt\n\n"
        f"Run ID: `{run_id}`\n\n"
        "The runner used positional `instance_indices: 4-15` and stopped "
        "upon reaching the unexpected identifier `bbob_f001_i71_d02`.\n\n"
        "No performance summary was generated, no objective values or rankings "
        "were inspected, and no algorithm source or option was changed. All "
        "partial files are excluded from final analysis. The complete holdout "
        "will be rerun under a new run ID after a runner-only correction.\n"
    )

    print("ABORTED_HOLDOUT_ATTEMPT_AUDIT_OK")
    print("run id:", run_id)
    print("inventory files:", len(inventory))
    print("detail files:", len(detail_files))
    print("json:", json_path)
    print("markdown:", md_path)


if __name__ == "__main__":
    main()
