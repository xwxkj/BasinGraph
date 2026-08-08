#!/usr/bin/env python3
"""Zero-objective-evaluation preflight for B21 Track D."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_d.common import (  # noqa: E402
    ALGORITHMS,
    MODE_CONFIG,
    SUITE_NAME,
    parse_problem_id,
    suite_options,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "overhead", "confirmatory"), required=True
    )
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    if config["confirmatory"] and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory preflight requires explicit authorization.")
    identity = verify_source_identity(require_clean=True)

    import cocoex

    problem_ids: list[str] = []
    objective_evaluations = 0
    dimensions_seen: set[int] = set()
    functions_seen: set[int] = set()
    instances_seen: set[int] = set()

    for dimension in config["dimensions"]:
        suite_instance, suite_options_text = suite_options(args.mode, dimension)
        suite = cocoex.Suite(SUITE_NAME, suite_instance, suite_options_text)
        ids = list(suite.ids())
        for problem_id in ids:
            function_index, instance_index, observed_dimension = parse_problem_id(
                problem_id
            )
            functions_seen.add(function_index)
            instances_seen.add(instance_index)
            dimensions_seen.add(observed_dimension)
        problem_ids.extend(ids)
        objective_evaluations += sum(
            int(getattr(problem, "evaluations", 0)) for problem in suite
        )

    expected_problems = (
        len(config["functions"])
        * len(config["dimensions"])
        * len(config["instances"])
    )
    checks = {
        "problem_count": len(problem_ids) == expected_problems,
        "unique_problem_ids": len(set(problem_ids)) == expected_problems,
        "functions": sorted(functions_seen) == config["functions"],
        "dimensions": sorted(dimensions_seen) == config["dimensions"],
        "instances": sorted(instances_seen) == config["instances"],
        "expected_rows": expected_problems * len(ALGORITHMS)
        == config["expected_rows"],
        "objective_evaluations_zero": objective_evaluations == 0,
    }
    if args.mode == "confirmatory":
        checks.update(
            {
                "confirmatory_instances_disjoint_from_development": set(
                    config["instances"]
                ).isdisjoint({1, 2, 3}),
                "confirmatory_instances_are_registered": config["instances"]
                == [16, 17, 18],
            }
        )
    passed = all(checks.values())
    report = {
        "status": "TRACK_D_PREFLIGHT_OK" if passed else "TRACK_D_PREFLIGHT_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "suite": SUITE_NAME,
        "identity": identity,
        "functions": config["functions"],
        "dimensions": config["dimensions"],
        "instances": config["instances"],
        "budget_multiplier": config["budget_multiplier"],
        "problems": len(problem_ids),
        "algorithms": len(ALGORITHMS),
        "expected_rows": config["expected_rows"],
        "objective_evaluations": objective_evaluations,
        "checks": checks,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError("Track D preflight failed: " + json.dumps(checks))
    print("TRACK_D_PREFLIGHT_OK")
    print(json.dumps({
        "mode": args.mode,
        "problems": len(problem_ids),
        "algorithms": len(ALGORITHMS),
        "expected_rows": config["expected_rows"],
        "objective_evaluations": objective_evaluations,
    }, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
