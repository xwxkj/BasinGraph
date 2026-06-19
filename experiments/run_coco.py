"""
Official COCO/BBOB runner for BasinGraph and core baselines.

Examples
--------
Smoke test:

python experiments/run_coco.py \
    --algorithm all-core \
    --dimensions 2 \
    --functions 1 \
    --instances 1 \
    --budget-multiplier 100 \
    --result-folder step4_core_smoke \
    --summary-csv processed_results/step4_core_smoke.csv
"""

from __future__ import annotations

# Make project root importable when running:
#     python experiments/run_coco.py
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
from pathlib import Path
import numpy as np
import cocoex

from basingraph.optimizer import minimize_basingraph
from baselines.reference_optimizers import (
    optimize_cmaes,
    optimize_bipop_cmaes,
    optimize_de,
    optimize_multistart_lbfgsb,
    optimize_random_search,
    optimize_lhs,
)


ALGORITHMS = {
    "basingraph": ("BasinGraph", minimize_basingraph),
    "cmaes": ("CMA-ES", optimize_cmaes),
    "bipop-cmaes": ("BIPOP-CMA-ES", optimize_bipop_cmaes),
    "de": ("Differential Evolution", optimize_de),
    "ms-lbfgsb": ("Multi-start L-BFGS-B", optimize_multistart_lbfgsb),
    "random": ("Random Search", optimize_random_search),
    "lhs": ("Latin Hypercube Sampling", optimize_lhs),
}

ALL_CORE = [
    "basingraph",
    "cmaes",
    "bipop-cmaes",
    "de",
    "ms-lbfgsb",
    "random",
    "lhs",
]


# COCO requires short algorithm_name labels without spaces.
COCO_SHORT_NAMES = {
    "basingraph": "BasinGraph",
    "cmaes": "CMA_ES",
    "bipop-cmaes": "BIPOP_CMA_ES",
    "de": "DE",
    "ms-lbfgsb": "MS_LBFGSB",
    "random": "Random",
    "lhs": "LHS",
}

# Detailed provenance-oriented descriptions stored in official .info logs.
COCO_ALGORITHM_INFO = {
    "basingraph":
        "BasinGraph geometry-controlled basin-graph optimizer implemented in this work",
    "cmaes":
        "CMA-ES implemented with the pycma package under a strict COCO evaluation budget",
    "bipop-cmaes":
        "BIPOP-CMA-ES implemented with pycma fmin2 restarts=9 and bipop=True",
    "de":
        "Self-contained differential evolution reference implementation with fixed F and CR",
    "ms-lbfgsb":
        "Multi-start L-BFGS-B using scipy.optimize.minimize with fixed budget allocation",
    "random":
        "Uniform random search under the same function-evaluation budget",
    "lhs":
        "Latin hypercube sampling under the same function-evaluation budget",
}


def safe_name(name: str) -> str:
    out = name.replace(" ", "_").replace("-", "_").replace("/", "_")
    out = out.replace("(", "").replace(")", "")
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="basingraph",
                        help="Algorithm id, comma-separated list, or all-core.")
    parser.add_argument("--dimensions", type=str, default="2")
    parser.add_argument("--functions", type=str, default="1")
    parser.add_argument("--instances", type=str, default="1")
    parser.add_argument("--budget-multiplier", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--result-folder", type=str, default="BasinGraph_on_bbob")
    parser.add_argument("--summary-csv", type=str, default="processed_results/coco_run_summary.csv")
    return parser.parse_args()


def parse_algorithm_list(s: str):
    s = s.strip().lower()
    if s == "all-core":
        return ALL_CORE

    algs = [a.strip().lower() for a in s.split(",") if a.strip()]
    for a in algs:
        if a not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm id: {a}. Available: {sorted(ALGORITHMS)}")
    return algs


def get_bounds(problem):
    dim = int(problem.dimension)

    try:
        lb = np.asarray(problem.lower_bounds, dtype=float)
        ub = np.asarray(problem.upper_bounds, dtype=float)
    except Exception:
        lb = -5.0 * np.ones(dim)
        ub = 5.0 * np.ones(dim)

    if lb.size != dim:
        lb = -5.0 * np.ones(dim)
    if ub.size != dim:
        ub = 5.0 * np.ones(dim)

    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)

    bad = ub <= lb
    if np.any(bad):
        lb[bad] = -5.0
        ub[bad] = 5.0

    return lb, ub


def run_one_problem(problem, algorithm_id, max_evals, seed):
    display_name, optimizer = ALGORITHMS[algorithm_id]
    lb, ub = get_bounds(problem)

    if algorithm_id == "basingraph":
        result = optimizer(
            objective=problem,
            lb=lb,
            ub=ub,
            max_evals=max_evals,
            seed=seed,
        )
    else:
        result = optimizer(
            objective=problem,
            lb=lb,
            ub=ub,
            max_evals=max_evals,
            seed=seed,
        )

    return display_name, result


def run_one_algorithm(args, algorithm_id):
    display_name, _ = ALGORITHMS[algorithm_id]

    suite_options = (
        f"dimensions: {args.dimensions} "
        f"function_indices: {args.functions} "
        f"instance_indices: {args.instances}"
    )

    result_folder = args.result_folder
    if len(parse_algorithm_list(args.algorithm)) > 1:
        result_folder = f"{args.result_folder}_{safe_name(display_name)}"

    suite = cocoex.Suite("bbob", "", suite_options)

    coco_short_name = COCO_SHORT_NAMES[algorithm_id]
    algorithm_info = COCO_ALGORITHM_INFO[algorithm_id]

    observer_options = (
        f"result_folder: {result_folder} "
        f"algorithm_name: {coco_short_name} "
        f'algorithm_info: "{algorithm_info}"'
    )

    observer = cocoex.Observer("bbob", observer_options)

    rows = []

    for iproblem, problem in enumerate(suite):
        problem.observe_with(observer)

        dim = int(problem.dimension)
        max_evals = int(args.budget_multiplier * dim)
        seed = int(args.seed + 100000 * list(ALGORITHMS).index(algorithm_id) + iproblem)

        alg_display, result = run_one_problem(
            problem=problem,
            algorithm_id=algorithm_id,
            max_evals=max_evals,
            seed=seed,
        )

        rows.append({
            "algorithm": alg_display,
            "algorithm_id": algorithm_id,
            "problem_id": problem.id,
            "dimension": dim,
            "budget": max_evals,
            "seed": seed,
            "fbest": result["fbest"],
            "nfe": result["nfe"],
            "message": result["message"],
            "result_folder": result_folder,
        })

    # Force COCO observer cleanup so files are flushed.
    del observer
    del suite

    return rows


def main():
    args = parse_args()
    algs = parse_algorithm_list(args.algorithm)

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for alg in algs:
        rows = run_one_algorithm(args, alg)
        all_rows.extend(rows)

    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "algorithm", "algorithm_id", "problem_id", "dimension", "budget",
            "seed", "fbest", "nfe", "message", "result_folder"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    print("COCO run finished.")
    print("Algorithms:", ", ".join([ALGORITHMS[a][0] for a in algs]))
    print("Dimensions:", args.dimensions)
    print("Functions:", args.functions)
    print("Instances:", args.instances)
    print("Budget multiplier:", args.budget_multiplier)
    print("Summary CSV:", summary_path)
    print("Number of rows:", len(all_rows))
    print("OFFICIAL_COCO_RUN_FINISHED")


if __name__ == "__main__":
    main()
