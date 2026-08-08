#!/usr/bin/env python3
"""
Route B Step B2 smoke runner.

This script verifies that BasinGraph v2 can connect to:

1. official COCO/BBOB through cocoex Observer;
2. CUTEst / PyCUTEst through a small bound-constrained problem;
3. JSON serialization of archive, graph, diagnostics and event log.

This is NOT a final experiment.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basingraph_v2.optimizer import minimize_basingraph_v2


OUT = PROJECT_ROOT / "results_v2" / "smoke"
OUT.mkdir(parents=True, exist_ok=True)


def run_coco_smoke() -> None:
    import cocoex

    folder = "routeb_v2_coco_smoke"
    exdata_folder = PROJECT_ROOT / "exdata" / folder

    if exdata_folder.exists():
        shutil.rmtree(exdata_folder)

    suite = cocoex.Suite(
        "bbob",
        "",
        "dimensions: 2 function_indices: 1 instance_indices: 1",
    )

    observer = cocoex.Observer(
        "bbob",
        f"result_folder: {folder} algorithm_name: BasinGraph_v2 "
        f'algorithm_info: "Full BasinGraph v2.0.0 Route B smoke test"',
    )

    rows = []

    for problem in suite:
        problem.observe_with(observer)

        dim = int(problem.dimension)

        try:
            lb = np.asarray(problem.lower_bounds, dtype=float)
            ub = np.asarray(problem.upper_bounds, dtype=float)
        except Exception:
            lb = -5.0 * np.ones(dim)
            ub = 5.0 * np.ones(dim)

        result = minimize_basingraph_v2(
            objective=problem,
            lb=lb,
            ub=ub,
            max_evals=100 * dim,
            seed=20260619,
        )

        payload = result.to_jsonable()

        rows.append({
            "problem_id": problem.id,
            "dimension": dim,
            "budget": 100 * dim,
            "nfe": result.nfe,
            "fbest": result.fbest,
            "archive_nodes": len(result.archive),
            "graph_edges": len(result.graph_edges),
            "message": result.message,
        })

        (OUT / "routeb_v2_coco_smoke_result.json").write_text(
            json.dumps(payload, indent=2)
        )

    del observer
    del suite

    (OUT / "routeb_v2_coco_smoke_summary.json").write_text(
        json.dumps(rows, indent=2)
    )

    info_files = list(exdata_folder.rglob("*.info"))
    dat_files = list(exdata_folder.rglob("*.dat"))

    assert rows, "No COCO rows were produced."
    assert rows[0]["nfe"] == rows[0]["budget"], rows[0]
    assert rows[0]["archive_nodes"] > 0, rows[0]
    assert len(info_files) > 0, "No COCO .info files produced."
    assert len(dat_files) > 0, "No COCO .dat files produced."

    print("V2_COCO_SMOKE_OK")
    print("COCO rows:", rows)
    print("COCO .info files:", len(info_files))
    print("COCO .dat files:", len(dat_files))


def run_cutest_smoke() -> None:
    import pycutest

    problem = pycutest.import_problem(
        "ROSENBR",
        quiet=True,
        drop_fixed_variables=True,
    )

    try:
        lb = np.asarray(problem.bl, dtype=float)
        ub = np.asarray(problem.bu, dtype=float)

        # ROSENBR can have infinite bounds under some CUTEst builds.
        # For smoke testing only, impose a standard finite box.
        lb = np.where(np.isfinite(lb) & (np.abs(lb) < 1e19), lb, -5.0)
        ub = np.where(np.isfinite(ub) & (np.abs(ub) < 1e19), ub, 5.0)

        def objective(x):
            return float(problem.obj(np.asarray(x, dtype=float)))

        result = minimize_basingraph_v2(
            objective=objective,
            lb=lb,
            ub=ub,
            max_evals=300,
            seed=20260620,
        )

        payload = result.to_jsonable()

        (OUT / "routeb_v2_cutest_rosenbr_smoke_result.json").write_text(
            json.dumps(payload, indent=2)
        )

        summary = {
            "problem": problem.name,
            "dimension": int(problem.n),
            "budget": 300,
            "nfe": result.nfe,
            "fbest": result.fbest,
            "archive_nodes": len(result.archive),
            "graph_edges": len(result.graph_edges),
            "message": result.message,
        }

        (OUT / "routeb_v2_cutest_rosenbr_smoke_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

        assert result.nfe == 300, summary
        assert len(result.archive) > 0, summary

        print("V2_CUTEST_SMOKE_OK")
        print("CUTEst summary:", summary)

    finally:
        if hasattr(problem, "terminate"):
            problem.terminate()


def main() -> None:
    run_coco_smoke()
    run_cutest_smoke()
    print("ROUTE_B_STEP_B2_OK")
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
