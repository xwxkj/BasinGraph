#!/usr/bin/env python3
"""
Run a seed batch of the frozen prospective CUTEst holdout.

One atomic job is one (CUTEst instance, paired seed) combination. Each job
imports the problem once and runs all seven algorithms sequentially with the
same deterministic seed and budget. Results are written atomically as one
gzip-compressed JSON record per problem-seed job.

No performance summary or ranking is printed by this runner.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.reference_optimizers import (
    optimize_bipop_cmaes,
    optimize_cmaes,
    optimize_de,
    optimize_lhs,
    optimize_multistart_lbfgsb,
    optimize_random_search,
)
from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)


EXPECTED_BRANCH = "route-b/finalize-rc1-consistent"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
CODE_FREEZE_TAG = "route-b-v2.0.0-rc1-codefreeze"
FINAL_CANDIDATE_TAG = "route-b-v2.0.0-rc1-selected-final-candidate"
RUNNER_FREEZE_TAG = "route-b-v2.0.0-rc1-cutest-holdout-runner-freeze"

HOLDOUT_LIST = (
    ROOT
    / "protocols"
    / "route_b"
    / "formal_v2_protocol"
    / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
)
PREFLIGHT_PATH = (
    ROOT
    / "protocols"
    / "route_b"
    / "final_rc1"
    / "CUTEST_HOLDOUT_PREFLIGHT.json"
)

ALGORITHMS = [
    "BasinGraph_v2",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]

DISPLAY_NAMES = {
    "BasinGraph_v2": "BasinGraph 2.0.0-rc1",
    "CMA_ES": "CMA-ES",
    "BIPOP_CMA_ES": "BIPOP-CMA-ES",
    "DE": "Differential Evolution",
    "MS_LBFGSB": "Multi-start L-BFGS-B",
    "LHS": "Latin Hypercube Sampling",
    "Random": "Random Search",
}

BASELINE_FUNCTIONS = {
    "CMA_ES": optimize_cmaes,
    "BIPOP_CMA_ES": optimize_bipop_cmaes,
    "DE": optimize_de,
    "MS_LBFGSB": optimize_multistart_lbfgsb,
    "LHS": optimize_lhs,
    "Random": optimize_random_search,
}


class ObjectiveBudgetExceeded(RuntimeError):
    pass


@dataclass
class ObjectiveTracker:
    problem: Any
    budget: int
    calls: int = 0
    best: float = math.inf
    nonfinite_calls: int = 0

    def __post_init__(self) -> None:
        self.improvements: list[list[float | int]] = []

    def __call__(self, x: np.ndarray) -> float:
        if self.calls >= self.budget:
            raise ObjectiveBudgetExceeded(
                f"Objective budget exceeded: {self.calls}/{self.budget}"
            )

        raw = float(self.problem.obj(np.asarray(x, dtype=float)))
        self.calls += 1

        if np.isfinite(raw):
            value = raw
        else:
            self.nonfinite_calls += 1
            value = 1.0e300

        if value < self.best:
            self.best = float(value)
            self.improvements.append([int(self.calls), float(self.best)])

        return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260621)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--result-root",
        default="results_v2/formal_holdout/cutest_rc1",
    )
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_contract() -> dict[str, Any]:
    branch = git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Wrong branch: {branch}")

    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(IMPLEMENTATION_VERSION)

    options = BasinGraphOptions()
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Selected candidate options hash mismatch.")

    if subprocess.run(
        ["git", "diff", "--quiet", CODE_FREEZE_TAG, "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("basingraph_v2 differs from selected code freeze.")

    runner_commit = git_output("rev-list", "-n", "1", RUNNER_FREEZE_TAG)
    current_commit = git_output("rev-parse", "HEAD")
    if current_commit != runner_commit:
        raise RuntimeError(
            "HEAD does not equal the frozen CUTEst holdout runner tag:\n"
            f"HEAD={current_commit}\nrunner={runner_commit}"
        )

    preflight = json.loads(PREFLIGHT_PATH.read_text())
    assert preflight["status"] == "CUTEST_HOLDOUT_PREFLIGHT_OK"
    assert preflight["objective_evaluations_performed"] == 0
    assert preflight["optimizer_runs_performed"] == 0
    assert preflight["holdout_list_sha256"] == sha256_file(HOLDOUT_LIST)

    return {
        "branch": branch,
        "current_commit": current_commit,
        "runner_freeze_tag": RUNNER_FREEZE_TAG,
        "runner_freeze_commit": runner_commit,
        "code_freeze_tag": CODE_FREEZE_TAG,
        "code_freeze_commit": git_output("rev-list", "-n", "1", CODE_FREEZE_TAG),
        "final_candidate_tag": FINAL_CANDIDATE_TAG,
        "final_candidate_commit": git_output(
            "rev-list", "-n", "1", FINAL_CANDIDATE_TAG
        ),
        "implementation_version": IMPLEMENTATION_VERSION,
        "options_hash": options.stable_hash(),
        "holdout_list_sha256": sha256_file(HOLDOUT_LIST),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "baseline_source_sha256": sha256_file(
            ROOT / "baselines" / "reference_optimizers.py"
        ),
    }


def safe_token(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"{clean[:60]}_{digest}"


def safe_destination(instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()[:16]
    return f"BGHOLD_{digest}"


def sif_params(row: dict[str, Any]) -> dict[str, int] | None:
    if str(row["source_type"]) != "scalable":
        return None
    value = row.get("sif_N")
    if value is None or str(value).strip() in {"", "nan", "None"}:
        raise RuntimeError(f"Missing sif_N for {row['instance_id']}")
    return {"N": int(float(value))}


def import_problem(row: dict[str, Any]):
    import pycutest

    kwargs: dict[str, Any] = {
        "quiet": True,
        "drop_fixed_variables": True,
        "destination": safe_destination(str(row["instance_id"])),
    }
    params = sif_params(row)
    if params is not None:
        kwargs["sifParams"] = params
    return pycutest.import_problem(str(row["problem_name"]), **kwargs)


def budget_for_dimension(dimension: int) -> int:
    return int(min(20_000, max(1_000, 50 * int(dimension))))


def paired_seed(
    base_seed: int,
    global_holdout_order: int,
    protocol_seed_index: int,
) -> int:
    return int(
        base_seed
        + 100_000 * int(global_holdout_order)
        + int(protocol_seed_index)
    )


def bounds_from_problem(problem, expected_dimension: int) -> tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(problem.bl, dtype=float).reshape(-1)
    ub = np.asarray(problem.bu, dtype=float).reshape(-1)

    if len(lb) != expected_dimension or len(ub) != expected_dimension:
        raise RuntimeError(
            f"Bound dimension mismatch: {len(lb)}, {len(ub)}, {expected_dimension}"
        )
    if not np.isfinite(lb).all() or not np.isfinite(ub).all():
        raise RuntimeError("Nonfinite CUTEst bounds in frozen holdout.")
    if not (ub > lb).all():
        raise RuntimeError("Nonpositive CUTEst box width.")
    return lb, ub


def run_algorithm(
    algorithm: str,
    problem,
    lb: np.ndarray,
    ub: np.ndarray,
    budget: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    tracker = ObjectiveTracker(problem=problem, budget=budget)
    started = time.perf_counter()

    if algorithm == "BasinGraph_v2":
        result = minimize_basingraph_v2(
            objective=tracker,
            lb=lb,
            ub=ub,
            max_evals=budget,
            seed=seed,
        )
        returned_nfe = int(result.nfe)
        fbest = float(result.fbest)
        message = str(result.message)
        bg_detail = result.to_jsonable()
        implementation_version = result.implementation_version
        options_hash = result.options_hash
        archive_nodes = len(result.archive)
        graph_edges = len(result.graph_edges)
    else:
        result = BASELINE_FUNCTIONS[algorithm](
            objective=tracker,
            lb=lb,
            ub=ub,
            max_evals=budget,
            seed=seed,
        )
        returned_nfe = int(result["nfe"])
        fbest = float(result["fbest"])
        message = str(result["message"])
        bg_detail = None
        implementation_version = ""
        options_hash = ""
        archive_nodes = ""
        graph_edges = ""

    elapsed = time.perf_counter() - started

    if returned_nfe != tracker.calls:
        raise RuntimeError(
            f"{algorithm} returned nfe={returned_nfe}, "
            f"tracker recorded {tracker.calls}"
        )
    if tracker.calls > budget:
        raise RuntimeError(
            f"{algorithm} exceeded budget: {tracker.calls}/{budget}"
        )
    if not np.isclose(
        fbest,
        tracker.best,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=False,
    ):
        raise RuntimeError(
            f"{algorithm} fbest mismatch: result={fbest}, tracker={tracker.best}"
        )

    row = {
        "algorithm": algorithm,
        "algorithm_display": DISPLAY_NAMES[algorithm],
        "budget": budget,
        "nfe": tracker.calls,
        "budget_ratio": tracker.calls / budget,
        "fbest": fbest,
        "algorithm_message": message,
        "implementation_version": implementation_version,
        "options_hash": options_hash,
        "archive_nodes": archive_nodes,
        "graph_edges": graph_edges,
        "nonfinite_evaluations": tracker.nonfinite_calls,
        "history_points": len(tracker.improvements),
        "wall_time_seconds": elapsed,
        "runner_status": "completed",
        "error": "",
    }
    history = {
        "algorithm": algorithm,
        "improvements": tracker.improvements,
        "total_evaluations": tracker.calls,
        "nonfinite_evaluations": tracker.nonfinite_calls,
    }
    return row, history, bg_detail


def atomic_write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def validate_existing_job(
    path: Path,
    *,
    instance_id: str,
    protocol_seed_index: int,
    seed: int,
    runner_commit: str,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        meta = payload["job_metadata"]
        rows = payload["rows"]
        if (
            meta["instance_id"] == instance_id
            and int(meta["protocol_seed_index"]) == protocol_seed_index
            and int(meta["seed"]) == seed
            and meta["runner_freeze_commit"] == runner_commit
            and len(rows) == 7
            and all(row["runner_status"] == "completed" for row in rows)
        ):
            return payload
    except Exception:
        return None
    return None


def run_job(payload: dict[str, Any]) -> dict[str, Any]:
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = "1"

    row = payload["problem"]
    protocol_seed_index = int(payload["protocol_seed_index"])
    seed = int(payload["seed"])
    run_root = Path(payload["run_root"])
    contract = payload["contract"]

    instance_id = str(row["instance_id"])
    job_path = (
        run_root
        / "jobs"
        / safe_token(instance_id)
        / f"seed_{protocol_seed_index:02d}.json.gz"
    )

    existing = validate_existing_job(
        job_path,
        instance_id=instance_id,
        protocol_seed_index=protocol_seed_index,
        seed=seed,
        runner_commit=contract["runner_freeze_commit"],
    )
    if existing is not None:
        return {
            "job_status": "skipped_existing",
            "job_path": str(job_path),
            "rows": existing["rows"],
        }

    problem = None
    try:
        problem = import_problem(row)
        dimension = int(problem.n)
        if int(problem.m) != 0:
            raise RuntimeError(f"General constraints present: m={problem.m}")
        if dimension != int(float(row["dimension"])):
            raise RuntimeError(
                f"Dimension mismatch for {instance_id}: {dimension} "
                f"vs {row['dimension']}"
            )

        lb, ub = bounds_from_problem(problem, dimension)
        budget = budget_for_dimension(dimension)
        f_at_x0 = float(row["f_at_x0"])

        rows: list[dict[str, Any]] = []
        histories: dict[str, Any] = {}
        basingraph_result: dict[str, Any] | None = None

        for algorithm in ALGORITHMS:
            algorithm_row, history, detail = run_algorithm(
                algorithm,
                problem,
                lb,
                ub,
                budget,
                seed,
            )
            algorithm_row.update(
                {
                    "global_holdout_order": int(row["global_holdout_order"]),
                    "problem_name": str(row["problem_name"]),
                    "instance_id": instance_id,
                    "source_type": str(row["source_type"]),
                    "sif_N": (
                        ""
                        if row.get("sif_N") is None
                        or str(row.get("sif_N")) in {"", "nan", "None"}
                        else int(float(row["sif_N"]))
                    ),
                    "dimension": dimension,
                    "dimension_group": str(row["dimension_group"]),
                    "objective_type": str(row["objective_type"]),
                    "protocol_seed_index": protocol_seed_index,
                    "seed": seed,
                    "f_at_cutest_x0": f_at_x0,
                    "job_file": str(job_path.relative_to(ROOT)),
                }
            )
            rows.append(algorithm_row)
            histories[algorithm] = history
            if detail is not None:
                basingraph_result = detail

        job_metadata = {
            **contract,
            "run_id": payload["run_id"],
            "global_holdout_order": int(row["global_holdout_order"]),
            "problem_name": str(row["problem_name"]),
            "instance_id": instance_id,
            "source_type": str(row["source_type"]),
            "sif_N": (
                None
                if row.get("sif_N") is None
                or str(row.get("sif_N")) in {"", "nan", "None"}
                else int(float(row["sif_N"]))
            ),
            "dimension": dimension,
            "dimension_group": str(row["dimension_group"]),
            "protocol_seed_index": protocol_seed_index,
            "seed": seed,
            "budget": budget,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        output_payload = {
            "job_metadata": job_metadata,
            "rows": rows,
            "improvement_histories": histories,
            "basingraph_result": basingraph_result,
        }
        atomic_write_gzip_json(job_path, output_payload)

        return {
            "job_status": "completed",
            "job_path": str(job_path),
            "rows": rows,
        }

    except Exception as exc:
        failure_root = run_root / "failures"
        failure_root.mkdir(parents=True, exist_ok=True)
        failure_path = (
            failure_root
            / f"{safe_token(instance_id)}_seed_{protocol_seed_index:02d}.json"
        )
        failure = {
            "status": "job_failed",
            "problem": row,
            "protocol_seed_index": protocol_seed_index,
            "seed": seed,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=50),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        failure_path.write_text(json.dumps(failure, indent=2, default=str))
        return {
            "job_status": "job_failed",
            "job_path": str(job_path),
            "failure_path": str(failure_path),
            "rows": [],
        }
    finally:
        if problem is not None and hasattr(problem, "terminate"):
            try:
                problem.terminate()
            except Exception:
                pass


def load_all_completed_rows(run_root: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    jobs = 0
    for path in sorted((run_root / "jobs").rglob("*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if (
            len(payload.get("rows", [])) == 7
            and all(
                row.get("runner_status") == "completed"
                for row in payload["rows"]
            )
        ):
            rows.extend(payload["rows"])
            jobs += 1
    return rows, jobs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if args.seed_start < 0 or args.seed_count <= 0:
        raise ValueError("Invalid seed range.")
    if args.seed_start + args.seed_count > 30:
        raise ValueError("Seed range exceeds frozen 30-seed protocol.")

    contract = verify_contract()
    holdout = pd.read_csv(HOLDOUT_LIST)
    if len(holdout) != 24:
        raise RuntimeError(f"Expected 24 holdout rows, found {len(holdout)}")

    result_root = ROOT / args.result_root
    run_root = result_root / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    metadata_path = run_root / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["runner_freeze_commit"] != contract["runner_freeze_commit"]:
            raise RuntimeError("Existing run uses a different runner freeze.")
    else:
        metadata = {
            **contract,
            "status": "CUTEST_HOLDOUT_RUN_STARTED",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "partition": "prospective_holdout",
            "problem_count": 24,
            "algorithms": ALGORITHMS,
            "paired_seeds": 30,
            "base_seed": args.base_seed,
            "seed_formula": (
                "base_seed + 100000*global_holdout_order "
                "+ protocol_seed_index"
            ),
            "budget_formula": "min(20000, max(1000, 50*n))",
            "performance_analysis_performed": False,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2))

    jobs = []
    for problem_row in holdout.sort_values("global_holdout_order").to_dict("records"):
        order = int(problem_row["global_holdout_order"])
        for seed_index in range(
            args.seed_start,
            args.seed_start + args.seed_count,
        ):
            jobs.append(
                {
                    "problem": problem_row,
                    "protocol_seed_index": seed_index,
                    "seed": paired_seed(args.base_seed, order, seed_index),
                    "run_id": args.run_id,
                    "run_root": str(run_root),
                    "contract": contract,
                }
            )

    results = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        mp_context=context,
    ) as executor:
        futures = [executor.submit(run_job, job) for job in jobs]
        for completed_index, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            results.append(future.result())
            if completed_index % 24 == 0 or completed_index == len(futures):
                print(
                    f"[{completed_index}/{len(futures)}] jobs finished",
                    flush=True,
                )

    status_counts: dict[str, int] = {}
    batch_rows: list[dict[str, Any]] = []
    for result in results:
        status = result["job_status"]
        status_counts[status] = status_counts.get(status, 0) + 1
        batch_rows.extend(result["rows"])

    batch_id = (
        f"seed{args.seed_start + 1:02d}_to_"
        f"{args.seed_start + args.seed_count:02d}"
    )
    batch_root = run_root / "batch_metadata"
    batch_root.mkdir(parents=True, exist_ok=True)

    if batch_rows:
        write_csv(
            batch_root / f"{batch_id}_rows.csv",
            sorted(
                batch_rows,
                key=lambda row: (
                    int(row["global_holdout_order"]),
                    int(row["protocol_seed_index"]),
                    ALGORITHMS.index(row["algorithm"]),
                ),
            ),
        )

    all_rows, completed_jobs = load_all_completed_rows(run_root)
    if all_rows:
        write_csv(
            run_root / "cutest_holdout_raw_results_all_available.csv",
            sorted(
                all_rows,
                key=lambda row: (
                    int(row["global_holdout_order"]),
                    int(row["protocol_seed_index"]),
                    ALGORITHMS.index(row["algorithm"]),
                ),
            ),
        )

    batch_report = {
        "status": (
            "CUTEST_HOLDOUT_BATCH_OK"
            if status_counts.get("job_failed", 0) == 0
            else "CUTEST_HOLDOUT_BATCH_FAILED"
        ),
        "run_id": args.run_id,
        "batch_id": batch_id,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "problem_seed_jobs": len(jobs),
        "expected_batch_rows": len(jobs) * 7,
        "observed_batch_rows": len(batch_rows),
        "all_available_jobs": completed_jobs,
        "all_available_rows": len(all_rows),
        "status_counts": status_counts,
        "performance_analysis_performed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = batch_root / f"{batch_id}_validation.json"
    report_path.write_text(json.dumps(batch_report, indent=2))

    if status_counts.get("job_failed", 0):
        raise RuntimeError(
            f"CUTEst holdout batch failures: {status_counts}; "
            f"see {run_root / 'failures'}"
        )

    print("CUTEST_HOLDOUT_BATCH_OK")
    print("RUN_ID=" + args.run_id)
    print("BATCH_ID=" + batch_id)
    print("Problem-seed jobs:", len(jobs))
    print("Expected batch rows:", len(jobs) * 7)
    print("Observed batch rows:", len(batch_rows))
    print("All available jobs:", completed_jobs)
    print("All available rows:", len(all_rows))
    print("Status counts:", status_counts)
    print("Performance analysis performed:", False)


if __name__ == "__main__":
    main()
