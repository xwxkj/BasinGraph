#!/usr/bin/env python3
"""
Route B Step B3: official COCO/BBOB mini benchmark for BasinGraph v2.

This is a mini validation, not the final manuscript experiment.

Protocol:
- suite: official COCO/BBOB noiseless
- dimensions: 2
- functions: 1-3
- instances: 1-3
- budget: 1000 * dimension
- algorithms:
  1. BasinGraph_v2
  2. CMA-ES
  3. BIPOP-CMA-ES

The purpose is to verify algorithm-runner-result consistency before launching
the full official COCO/BBOB rerun.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import re
import shutil
import sys
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import minimize_basingraph_v2


OUT = PROJECT_ROOT / "results_v2" / "coco_mini"
OUT.mkdir(parents=True, exist_ok=True)

EXDATA_ROOT = PROJECT_ROOT / "exdata" / "routeb_v2_coco_mini"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def get_bounds(problem, dim: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lb = np.asarray(problem.lower_bounds, dtype=float)
        ub = np.asarray(problem.upper_bounds, dtype=float)
    except Exception:
        lb = -5.0 * np.ones(dim)
        ub = 5.0 * np.ones(dim)

    lb = np.where(np.isfinite(lb), lb, -5.0)
    ub = np.where(np.isfinite(ub), ub, 5.0)

    bad = ub <= lb
    lb[bad] = -5.0
    ub[bad] = 5.0

    return lb, ub


def run_basingraph_v2(problem, lb, ub, budget: int, seed: int, problem_tag: str) -> dict:
    result = minimize_basingraph_v2(
        objective=problem,
        lb=lb,
        ub=ub,
        max_evals=budget,
        seed=seed,
    )

    payload = result.to_jsonable()
    detail_path = OUT / f"BasinGraph_v2_{problem_tag}.json"
    detail_path.write_text(json.dumps(payload, indent=2))

    return {
        "algorithm": "BasinGraph_v2",
        "budget": budget,
        "nfe_internal": result.nfe,
        "fbest": result.fbest,
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "diagnostics_keys": ";".join(sorted(payload["diagnostics"].keys())),
        "detail_json": str(detail_path.relative_to(PROJECT_ROOT)),
        "message": result.message,
    }


def run_cma_variant(problem, lb, ub, budget: int, seed: int, *, bipop: bool) -> dict:
    import cma

    dim = len(lb)
    x0 = 0.5 * (lb + ub)
    sigma0 = 0.25 * float(np.max(ub - lb))

    calls = 0
    fbest = float("inf")

    def objective(x):
        nonlocal calls, fbest
        calls += 1
        value = float(problem(np.asarray(x, dtype=float)))
        if np.isfinite(value) and value < fbest:
            fbest = value
        return value

    options = {
        "bounds": [lb.tolist(), ub.tolist()],
        "maxfevals": int(budget),
        "seed": int(seed),
        "verbose": -9,
        "verb_disp": 0,
        "verb_log": 0,
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if bipop:
            xbest, es = cma.fmin2(
                objective,
                x0,
                sigma0,
                options=options,
                restarts=9,
                bipop=True,
            )
            name = "BIPOP_CMA_ES"
        else:
            xbest, es = cma.fmin2(
                objective,
                x0,
                sigma0,
                options=options,
                restarts=0,
            )
            name = "CMA_ES"

    try:
        fbest = float(es.result.fbest)
    except Exception:
        fbest = float(fbest)

    return {
        "algorithm": name,
        "budget": budget,
        "nfe_internal": int(calls),
        "fbest": fbest,
        "archive_nodes": "",
        "graph_edges": "",
        "diagnostics_keys": "",
        "detail_json": "",
        "message": "completed",
    }


def run_algorithm_on_suite(algorithm_name: str, algo_index: int) -> list[dict]:
    import cocoex

    suite = cocoex.Suite(
        "bbob",
        "",
        "dimensions: 2 function_indices: 1-3 instance_indices: 1-3",
    )

    result_folder = f"routeb_v2_coco_mini/{algorithm_name}"

    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {result_folder} "
        f"algorithm_name: {algorithm_name} "
        f'algorithm_info: "Route B BasinGraph v2 COCO mini benchmark"',
    )

    rows = []

    for problem_index, problem in enumerate(suite):
        problem.observe_with(observer)

        dim = int(problem.dimension)
        budget = int(1000 * dim)
        lb, ub = get_bounds(problem, dim)

        problem_tag = safe_name(problem.id)
        seed = 20260619 + 100000 * algo_index + problem_index

        if algorithm_name == "BasinGraph_v2":
            row = run_basingraph_v2(problem, lb, ub, budget, seed, problem_tag)
        elif algorithm_name == "CMA_ES":
            row = run_cma_variant(problem, lb, ub, budget, seed, bipop=False)
        elif algorithm_name == "BIPOP_CMA_ES":
            row = run_cma_variant(problem, lb, ub, budget, seed, bipop=True)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        row.update({
            "suite": "bbob",
            "dimension": dim,
            "function_instance": problem.id,
            "seed": seed,
        })

        rows.append(row)

    del observer
    del suite

    return rows


def main() -> None:
    if EXDATA_ROOT.exists():
        shutil.rmtree(EXDATA_ROOT)

    all_rows = []
    algorithms = ["BasinGraph_v2", "CMA_ES", "BIPOP_CMA_ES"]

    for algo_index, algorithm_name in enumerate(algorithms):
        rows = run_algorithm_on_suite(algorithm_name, algo_index)
        all_rows.extend(rows)

    csv_path = OUT / "coco_v2_mini_raw_results.csv"
    fieldnames = [
        "suite",
        "dimension",
        "function_instance",
        "algorithm",
        "seed",
        "budget",
        "nfe_internal",
        "fbest",
        "archive_nodes",
        "graph_edges",
        "diagnostics_keys",
        "detail_json",
        "message",
    ]

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    info_files = list(EXDATA_ROOT.rglob("*.info"))
    dat_files = list(EXDATA_ROOT.rglob("*.dat"))

    summary = {
        "status": "V2_COCO_MINI_OK",
        "rows": len(all_rows),
        "algorithms": algorithms,
        "expected_rows": 27,
        "info_files": len(info_files),
        "dat_files": len(dat_files),
        "raw_results": str(csv_path.relative_to(PROJECT_ROOT)),
        "exdata_root": str(EXDATA_ROOT.relative_to(PROJECT_ROOT)),
    }

    (OUT / "coco_v2_mini_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    assert len(all_rows) == 27, summary
    assert len(info_files) > 0, summary
    assert len(dat_files) > 0, summary

    bg_rows = [r for r in all_rows if r["algorithm"] == "BasinGraph_v2"]
    assert len(bg_rows) == 9
    assert all(int(r["nfe_internal"]) == int(r["budget"]) for r in bg_rows)
    assert all(int(r["archive_nodes"]) > 0 for r in bg_rows)
    assert sum(int(r["graph_edges"]) for r in bg_rows) > 0

    print("V2_COCO_MINI_OK")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
