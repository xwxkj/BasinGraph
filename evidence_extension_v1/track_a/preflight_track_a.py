#!/usr/bin/env python3
"""Zero-evaluation preflight for B21 Track A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_a.common import (  # noqa: E402
    MODE_CONFIG,
    VARIANT_ORDER,
    expected_variant_hashes,
    parse_problem_id,
    suite_options,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("smoke", "confirmatory"),
        default="smoke",
    )
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument(
        "--output",
        default="",
        help="JSON output path; a mode-specific default is used when omitted.",
    )
    parser.add_argument(
        "--authorize-confirmatory",
        action="store_true",
        help="Required for confirmatory instances 16-20.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "confirmatory" and not args.authorize_confirmatory:
        raise RuntimeError(
            "Confirmatory preflight requires --authorize-confirmatory."
        )

    identity = verify_source_identity(require_clean=True)
    config = MODE_CONFIG[args.mode]

    import cocoex

    observed_ids = []
    objective_evaluations = 0
    per_dimension = {}

    for dimension in config["dimensions"]:
        suite_instance, suite_options_text = suite_options(
            args.mode,
            dimension,
        )
        suite = cocoex.Suite(
            "bbob",
            suite_instance,
            suite_options_text,
        )
        ids = list(suite.ids())
        parsed = [parse_problem_id(problem_id) for problem_id in ids]
        per_dimension[str(dimension)] = {
            "suite_instance": suite_instance,
            "suite_options": suite_options_text,
            "problem_count": len(ids),
            "functions": sorted({item[0] for item in parsed}),
            "instances": sorted({item[1] for item in parsed}),
            "dimensions": sorted({item[2] for item in parsed}),
        }
        observed_ids.extend(ids)
        del suite

    parsed_all = [parse_problem_id(problem_id) for problem_id in observed_ids]
    expected_problem_count = (
        len(config["functions"])
        * len(config["dimensions"])
        * len(config["instances"])
    )
    expected_rows = expected_problem_count * len(VARIANT_ORDER)

    checks = {
        "problem_count": len(observed_ids) == expected_problem_count,
        "unique_problem_ids": len(observed_ids) == len(set(observed_ids)),
        "functions": sorted({item[0] for item in parsed_all})
        == config["functions"],
        "dimensions": sorted({item[2] for item in parsed_all})
        == config["dimensions"],
        "instances": sorted({item[1] for item in parsed_all})
        == config["instances"],
        "objective_evaluations_zero": objective_evaluations == 0,
        "expected_rows": expected_rows == config["expected_rows"],
        "variant_hashes_unique": len(set(expected_variant_hashes().values()))
        == len(VARIANT_ORDER),
    }
    if args.mode == "confirmatory":
        checks["disjoint_from_development_instances"] = not (
            set(config["instances"]) & {1, 2, 3}
        )
        checks["disjoint_from_original_holdout"] = not (
            set(config["instances"]) & set(range(4, 16))
        )

    report = {
        "status": (
            "TRACK_A_PREFLIGHT_OK"
            if all(checks.values())
            else "TRACK_A_PREFLIGHT_FAILED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "confirmatory": bool(config["confirmatory"]),
        "identity": identity,
        "suite": "bbob",
        "functions": config["functions"],
        "dimensions": config["dimensions"],
        "instances": config["instances"],
        "budget_multiplier": config["budget_multiplier"],
        "problems": expected_problem_count,
        "variants": VARIANT_ORDER,
        "expected_rows": expected_rows,
        "base_seed": args.base_seed,
        "seed_formula": (
            "base_seed + 100000*function + 1000*dimension + instance"
        ),
        "per_dimension": per_dimension,
        "checks": checks,
    }

    output = (
        ROOT / args.output
        if args.output
        else ROOT
        / "results_b21"
        / "track_a"
        / f"preflight_{args.mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(report["status"])
    print(json.dumps(
        {
            "problems": expected_problem_count,
            "variants": len(VARIANT_ORDER),
            "expected_rows": expected_rows,
            "objective_evaluations": objective_evaluations,
        },
        indent=2,
    ))
    print(f"Report: {output}")

    if report["status"].endswith("FAILED"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
