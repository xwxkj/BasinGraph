#!/usr/bin/env python3
"""
Route B Step B4: CUTEst mini benchmark for BasinGraph v2.

This is not a final manuscript experiment.

Protocol:
- CUTEst pre-registered list v2
- selected representatives: global_protocol_order 1, 21, 41
- dimension groups: small, medium, large
- algorithms:
  1. BasinGraph_v2
  2. CMA_ES
  3. BIPOP_CMA_ES
  4. Multi_start_LBFGSB
- seeds: 2
"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import math
import re
import sys
import warnings

import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import minimize_basingraph_v2


OUT = PROJECT_ROOT / "results_v2" / "cutest_mini"
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL_LIST = PROJECT_ROOT / "protocols" / "cutest_pre_registered_problem_list_v2.csv"
TARGET_ORDERS = {1, 21, 41}
SEEDS = [20260619, 20260620]


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def load_target_rows() -> list[dict]:
    rows = list(csv.DictReader(PROTOCOL_LIST.open()))
    selected = [r for r in rows if int(r["global_protocol_order"]) in TARGET_ORDERS]
    selected = sorted(selected, key=lambda r: int(r["global_protocol_order"]))
    assert len(selected) == 3, selected
    return selected


def sif_params_from_row(row: dict) -> dict | None:
    if row.get("source_type") != "scalable":
        return None
    val = row.get("sif_N", "")
    if val == "" or val.lower() == "nan":
        return None
    return {"N": int(float(val))}


def import_problem(row: dict):
    import pycutest

    name = row["problem_name"]
    params = sif_params_from_row(row)

    if params is None:
        return pycutest.import_problem(
            name,
            quiet=True,
            drop_fixed_variables=True,
        )

    return pycutest.import_problem(
        name,
        sifParams=params,
        quiet=True,
        drop_fixed_variables=True,
    )


def bounds_from_problem(problem, row: dict) -> tuple[np.ndarray, np.ndarray, str]:
    n = int(problem.n)

    try:
        lb = np.asarray(problem.bl, dtype=float).reshape(-1)
        ub = np.asarray(problem.bu, dtype=float).reshape(-1)
    except Exception:
        lb = np.full(n, float(row["minimum_bound"]))
        ub = np.full(n, float(row["maximum_bound"]))

    fallback_lb = float(row["minimum_bound"])
    fallback_ub = float(row["maximum_bound"])

    if not np.isfinite(fallback_lb) or abs(fallback_lb) > 1e20:
        fallback_lb = -5.0
    if not np.isfinite(fallback_ub) or abs(fallback_ub) > 1e20:
        fallback_ub = 5.0

    lb = np.where(np.isfinite(lb) & (np.abs(lb) < 1e20), lb, fallback_lb)
    ub = np.where(np.isfinite(ub) & (np.abs(ub) < 1e20), ub, fallback_ub)

    bad = ub <= lb
    lb[bad] = -5.0
    ub[bad] = 5.0

    return lb, ub, "cutest_bounds_with_protocol_fallback"


def budget_for_dim(dim: int) -> int:
    return int(min(3000, max(300, 50 * dim)))


def make_objective(problem):
    def objective(x):
        return float(problem.obj(np.asarray(x, dtype=float)))
    return objective


def run_basingraph(problem, row, lb, ub, budget, seed):
    result = minimize_basingraph_v2(
        objective=make_objective(problem),
        lb=lb,
        ub=ub,
        max_evals=budget,
        seed=seed,
    )

    tag = f"{safe_name(row['instance_id'])}_seed{seed}"
    detail_path = OUT / f"BasinGraph_v2_{tag}.json"
    detail_path.write_text(json.dumps(result.to_jsonable(), indent=2))

    return {
        "algorithm": "BasinGraph_v2",
        "nfe_internal": result.nfe,
        "fbest": result.fbest,
        "archive_nodes": len(result.archive),
        "graph_edges": len(result.graph_edges),
        "detail_json": str(detail_path.relative_to(PROJECT_ROOT)),
        "message": result.message,
    }


def run_cma(problem, row, lb, ub, budget, seed, *, bipop: bool):
    import cma

    dim = len(lb)
    x0 = np.clip(np.asarray(problem.x0, dtype=float).reshape(-1), lb, ub)
    scale = float(np.median(ub - lb))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    sigma0 = 0.25 * scale

    objective0 = make_objective(problem)
    calls = 0
    fbest = float("inf")

    def objective(x):
        nonlocal calls, fbest
        if calls >= budget:
            return fbest + 1e100
        calls += 1
        val = objective0(np.clip(np.asarray(x, dtype=float), lb, ub))
        if np.isfinite(val) and val < fbest:
            fbest = float(val)
        return float(val)

    opts = {
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
            _, es = cma.fmin2(objective, x0, sigma0, options=opts, restarts=9, bipop=True)
            name = "BIPOP_CMA_ES"
        else:
            _, es = cma.fmin2(objective, x0, sigma0, options=opts, restarts=0)
            name = "CMA_ES"

    try:
        fbest = min(float(fbest), float(es.result.fbest))
    except Exception:
        pass

    return {
        "algorithm": name,
        "nfe_internal": int(calls),
        "fbest": float(fbest),
        "archive_nodes": "",
        "graph_edges": "",
        "detail_json": "",
        "message": "completed",
    }


def run_multistart_lbfgsb(problem, row, lb, ub, budget, seed):
    rng = np.random.default_rng(seed)
    objective0 = make_objective(problem)

    calls = 0
    fbest = float("inf")
    dim = len(lb)

    def objective(x):
        nonlocal calls, fbest
        if calls >= budget:
            return fbest + 1e100
        calls += 1
        val = objective0(np.clip(np.asarray(x, dtype=float), lb, ub))
        if np.isfinite(val) and val < fbest:
            fbest = float(val)
        return float(val)

    starts = [np.clip(np.asarray(problem.x0, dtype=float).reshape(-1), lb, ub)]
    for _ in range(4):
        starts.append(lb + rng.random(dim) * (ub - lb))

    per_start = max(10, budget // len(starts))

    for x0 in starts:
        if calls >= budget:
            break
        try:
            minimize(
                objective,
                x0,
                method="L-BFGS-B",
                bounds=list(zip(lb, ub)),
                options={
                    "maxfun": int(min(per_start, budget - calls)),
                    "maxiter": int(min(per_start, budget - calls)),
                    "ftol": 1e-12,
                    "gtol": 1e-8,
                    "disp": False,
                },
            )
        except Exception:
            continue

    return {
        "algorithm": "Multi_start_LBFGSB",
        "nfe_internal": int(calls),
        "fbest": float(fbest),
        "archive_nodes": "",
        "graph_edges": "",
        "detail_json": "",
        "message": "completed",
    }


def run_one(row: dict, algorithm: str, seed: int) -> dict:
    problem = import_problem(row)
    try:
        lb, ub, box_policy = bounds_from_problem(problem, row)
        dim = int(problem.n)
        budget = budget_for_dim(dim)

        if algorithm == "BasinGraph_v2":
            out = run_basingraph(problem, row, lb, ub, budget, seed)
        elif algorithm == "CMA_ES":
            out = run_cma(problem, row, lb, ub, budget, seed, bipop=False)
        elif algorithm == "BIPOP_CMA_ES":
            out = run_cma(problem, row, lb, ub, budget, seed, bipop=True)
        elif algorithm == "Multi_start_LBFGSB":
            out = run_multistart_lbfgsb(problem, row, lb, ub, budget, seed)
        else:
            raise ValueError(algorithm)

        out.update({
            "global_protocol_order": row["global_protocol_order"],
            "problem_name": row["problem_name"],
            "instance_id": row["instance_id"],
            "source_type": row["source_type"],
            "dimension_group": row["dimension_group"],
            "dimension": dim,
            "objective_type": row["objective_type"],
            "constraints": row["constraints"],
            "budget": budget,
            "seed": seed,
            "box_policy": box_policy,
            "runner_status": "completed",
            "error": "",
        })
        return out

    except Exception as exc:
        return {
            "global_protocol_order": row.get("global_protocol_order", ""),
            "problem_name": row.get("problem_name", ""),
            "instance_id": row.get("instance_id", ""),
            "source_type": row.get("source_type", ""),
            "dimension_group": row.get("dimension_group", ""),
            "dimension": "",
            "objective_type": row.get("objective_type", ""),
            "constraints": row.get("constraints", ""),
            "algorithm": algorithm,
            "budget": "",
            "seed": seed,
            "nfe_internal": "",
            "fbest": "",
            "archive_nodes": "",
            "graph_edges": "",
            "detail_json": "",
            "box_policy": "",
            "message": "",
            "runner_status": "failed",
            "error": repr(exc),
        }
    finally:
        try:
            if hasattr(problem, "terminate"):
                problem.terminate()
        except Exception:
            pass


def main():
    rows = load_target_rows()
    algorithms = ["BasinGraph_v2", "CMA_ES", "BIPOP_CMA_ES", "Multi_start_LBFGSB"]

    all_rows = []
    for row in rows:
        for algorithm in algorithms:
            for seed in SEEDS:
                print("RUN", row["instance_id"], algorithm, seed, flush=True)
                all_rows.append(run_one(row, algorithm, seed))

    raw_path = OUT / "cutest_v2_mini_raw_results.csv"

    fieldnames = [
        "global_protocol_order",
        "problem_name",
        "instance_id",
        "source_type",
        "dimension_group",
        "dimension",
        "objective_type",
        "constraints",
        "algorithm",
        "budget",
        "seed",
        "nfe_internal",
        "fbest",
        "archive_nodes",
        "graph_edges",
        "detail_json",
        "box_policy",
        "message",
        "runner_status",
        "error",
    ]

    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "status": "V2_CUTEST_MINI_OK",
        "rows": len(all_rows),
        "expected_rows": 24,
        "problems": [r["instance_id"] for r in rows],
        "algorithms": algorithms,
        "seeds": SEEDS,
        "raw_results": str(raw_path.relative_to(PROJECT_ROOT)),
    }

    (OUT / "cutest_v2_mini_summary.json").write_text(json.dumps(summary, indent=2))

    failed = [r for r in all_rows if r["runner_status"] != "completed"]
    assert len(all_rows) == 24, summary
    assert not failed, failed

    bg_rows = [r for r in all_rows if r["algorithm"] == "BasinGraph_v2"]
    assert len(bg_rows) == 6
    assert all(int(r["nfe_internal"]) == int(r["budget"]) for r in bg_rows)
    assert all(int(r["archive_nodes"]) > 0 for r in bg_rows)
    assert sum(int(r["graph_edges"]) for r in bg_rows) > 0

    print("V2_CUTEST_MINI_OK")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
