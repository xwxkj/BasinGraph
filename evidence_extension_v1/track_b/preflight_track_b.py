#!/usr/bin/env python3
"""Zero-objective-evaluation preflight for B21 Track B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_b.common import (  # noqa: E402
    ALGORITHMS,
    MODE_CONFIG,
    parse_problem_id,
    suite_options,
    verify_source_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "confirmatory"), required=True)
    parser.add_argument("--base-seed", type=int, default=20260808)
    parser.add_argument("--authorize-confirmatory", action="store_true")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MODE_CONFIG[args.mode]
    if config["confirmatory"] and not args.authorize_confirmatory:
        raise RuntimeError("Confirmatory preflight requires --authorize-confirmatory.")
    identity = verify_source_identity(require_clean=True)

    import cma
    import cocoex
    import cocopp
    import numpy as np
    import pandas as pd
    import scipy

    all_ids = []
    objective_evaluations = 0
    by_dimension = {}
    for dimension in config["dimensions"]:
        suite_instance, suite_options_text = suite_options(args.mode, dimension)
        suite = cocoex.Suite("bbob", suite_instance, suite_options_text)
        ids = list(suite.ids())
        all_ids.extend(ids)
        by_dimension[str(dimension)] = len(ids)
        for problem_id in ids:
            function_index, instance_index, parsed_dimension = parse_problem_id(problem_id)
            if function_index not in config["functions"]:
                raise RuntimeError(f"Unexpected function in suite: {problem_id}")
            if instance_index not in config["instances"]:
                raise RuntimeError(f"Unexpected instance in suite: {problem_id}")
            if parsed_dimension != dimension:
                raise RuntimeError(f"Unexpected dimension in suite: {problem_id}")
        del suite

    expected_problems = (
        len(config["functions"])
        * len(config["dimensions"])
        * len(config["instances"])
    )
    expected_rows = expected_problems * len(ALGORITHMS)
    if len(all_ids) != expected_problems:
        raise RuntimeError(
            f"Problem count mismatch: {len(all_ids)} != {expected_problems}"
        )
    if expected_rows != config["expected_rows"]:
        raise RuntimeError(
            f"Expected-row contract mismatch: {expected_rows} != {config['expected_rows']}"
        )

    development = {1, 2, 3}
    original_holdout = set(range(4, 16))
    track_a = set(range(16, 21))
    current = set(config["instances"])
    if args.mode == "confirmatory":
        if current & development or current & original_holdout or current & track_a:
            raise RuntimeError("Track B confirmatory partition overlaps prior partitions.")
        if current != set(range(21, 31)):
            raise RuntimeError("Track B confirmatory instances are not 21–30.")

    report = {
        "status": "TRACK_B_PREFLIGHT_OK",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "partition": config["partition"],
        "suite": "bbob",
        "functions": config["functions"],
        "dimensions": config["dimensions"],
        "instances": config["instances"],
        "budget_multiplier": config["budget_multiplier"],
        "algorithms": ALGORITHMS,
        "problems": expected_problems,
        "problems_by_dimension": by_dimension,
        "expected_rows": expected_rows,
        "objective_evaluations": objective_evaluations,
        "partition_disjoint_from": {
            "development_instances": sorted(development),
            "original_holdout_instances": sorted(original_holdout),
            "track_a_instances": sorted(track_a),
        },
        "base_seed": args.base_seed,
        "seed_formula": "base_seed + 100000*function + 1000*dimension + instance",
        "identity": identity,
        "environment": {
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "cma": getattr(cma, "__version__", "unknown"),
            "cocoex": getattr(cocoex, "__version__", "unknown"),
            "cocopp": getattr(cocopp, "__version__", "unknown"),
        },
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT / "results_b21" / "track_b" / f"preflight_{args.mode}.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("TRACK_B_PREFLIGHT_OK")
    print(json.dumps({
        "problems": expected_problems,
        "algorithms": len(ALGORITHMS),
        "expected_rows": expected_rows,
        "objective_evaluations": objective_evaluations,
    }, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
