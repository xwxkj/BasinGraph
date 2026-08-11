#!/usr/bin/env python3
"""Execute locked PDE non-identifiability flagship v2.1 confirmation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION  # noqa: E402
from evidence_extension_v1.pde_flagship.reanalyze_v2_scale_normalized import (  # noqa: E402
    analyze_task,
)
from evidence_extension_v1.pde_flagship.run_development_v2 import (  # noqa: E402
    make_task,
    run_one,
)


LOCK_PATH = (
    ROOT
    / "protocols"
    / "evidence_extension_v1"
    / "pde_flagship"
    / "PDE_V2_1_CONFIRMATORY_LOCK.json"
)
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
EXPECTED_OPTIMIZER_BLOB = "4433d2b92075dcd858529f7f13342631847def4c"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="results_b21/pde_flagship/confirmatory_v2_1",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root: Path) -> None:
    target = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target:
            rows.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    for key in THREAD_ENV:
        os.environ[key] = "1"
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "PDE_FLAGSHIP_V2_1_LOCKED_BEFORE_CONFIRMATORY_ACCESS":
        raise RuntimeError("Invalid PDE v2.1 lock status.")
    if lock["confirmatory_objective_evaluations_before_lock"] != 0:
        raise RuntimeError("PDE v2.1 lock was not created before access.")
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected implementation version.")
    if BasinGraphOptions().stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected options hash.")
    if git_output("rev-parse", "HEAD:basingraph_v2/optimizer.py") != EXPECTED_OPTIMIZER_BLOB:
        raise RuntimeError("Frozen optimizer source changed.")

    instances = tuple(int(value) for value in lock["confirmatory_instances"])
    seeds_per_instance = int(lock["seeds_per_instance"])
    jobs = [
        (instance, replicate)
        for instance in instances
        for replicate in range(1, seeds_per_instance + 1)
    ]
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    records = []
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(run_one, instance, replicate): (instance, replicate)
            for instance, replicate in jobs
        }
        for future in as_completed(futures):
            instance, replicate = futures[future]
            try:
                records.append(future.result())
                print("PDE_V2_1_CONFIRMATORY_RUN_OK", instance, replicate, flush=True)
            except Exception as exc:
                failures.append(
                    {
                        "instance": instance,
                        "replicate": replicate,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=40),
                    }
                )
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"PDE v2.1 confirmatory failures: {len(failures)}")

    run_rows = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "candidates"}
        row["candidate_json"] = json.dumps(record["candidates"], separators=(",", ":"))
        run_rows.append(row)
    run_frame = pd.DataFrame(run_rows).sort_values(["instance", "replicate"])
    if len(run_frame) != len(jobs):
        raise RuntimeError("Unexpected PDE v2.1 confirmatory run count.")
    integrity_ok = bool(
        run_frame["graph_referential_integrity"].all()
        and (run_frame["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
        and (run_frame["options_hash"] == EXPECTED_OPTIONS_HASH).all()
        and (run_frame["nfe"] == run_frame["budget"]).all()
        and (run_frame["phase_sum"] == run_frame["budget"]).all()
    )
    if not integrity_ok:
        raise RuntimeError("PDE v2.1 confirmatory integrity failed.")
    run_frame.to_csv(output / "pde_v2_1_confirmatory_runs.csv", index=False)

    summaries = []
    design_rows = []
    for instance in instances:
        task = make_task(instance)
        selected_records = [record for record in records if record["instance"] == instance]
        summary, rows = analyze_task(task, selected_records)
        summaries.append(summary)
        design_rows.extend(rows)
    task_frame = pd.DataFrame(summaries).sort_values("instance")
    design_frame = pd.DataFrame(design_rows).sort_values(["instance", "design_id"])
    task_frame.to_csv(output / "pde_v2_1_confirmatory_task_summary.csv", index=False)
    design_frame.to_csv(output / "pde_v2_1_confirmatory_design_results.csv", index=False)

    rules = lock["success_rule"]
    conditions = {
        "six_tasks_with_four_explanations": int(
            (task_frame["selected_explanations"] >= 4).sum()
        ) >= int(rules["minimum_tasks_with_four_explanations"]),
        "six_tasks_with_25pct_separation_gain": int(
            (task_frame["separation_gain_over_median"] >= 0.25).sum()
        ) >= int(rules["minimum_tasks_with_25pct_score_gain"]),
        "six_tasks_with_20pct_effective_count_reduction": int(
            (task_frame["effective_count_reduction_vs_baseline"] >= 0.20).sum()
        ) >= int(rules["minimum_tasks_with_20pct_effective_count_reduction"]),
        "six_tasks_with_20pct_dispersion_reduction": int(
            (task_frame["dispersion_reduction_vs_baseline"] >= 0.20).sum()
        ) >= int(rules["minimum_tasks_with_20pct_dispersion_reduction"]),
        "selected_no_worse_than_median_on_six_tasks": int(
            task_frame["chosen_ambiguity_no_worse_than_median_design"].sum()
        ) >= int(rules["minimum_tasks_no_worse_than_median_design"]),
        "truth_not_used_for_design_selection": bool(
            rules["truth_not_used_for_selection"]
        ),
        "integrity_checks_pass": integrity_ok,
    }
    success = bool(all(conditions.values()))
    decision = {
        "status": (
            "PDE_FLAGSHIP_V2_1_CONFIRMATORY_SUCCESS"
            if success
            else "PDE_FLAGSHIP_V2_1_CONFIRMATORY_NOT_SUPPORTED"
        ),
        "success": success,
        "conditions": conditions,
        "source_commit": git_output("rev-parse", "HEAD"),
        "lock_sha256": sha256_file(LOCK_PATH),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "optimizer_blob": EXPECTED_OPTIMIZER_BLOB,
        "confirmatory_instances": list(instances),
        "runs": len(run_frame),
        "tasks": len(task_frame),
        "counts": {
            "four_explanations": int(
                (task_frame["selected_explanations"] >= 4).sum()
            ),
            "separation_gain": int(
                (task_frame["separation_gain_over_median"] >= 0.25).sum()
            ),
            "effective_count_reduction": int(
                (task_frame["effective_count_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "dispersion_reduction": int(
                (task_frame["dispersion_reduction_vs_baseline"] >= 0.20).sum()
            ),
            "no_worse_than_median": int(
                task_frame["chosen_ambiguity_no_worse_than_median_design"].sum()
            ),
        },
        "median_selected_explanations": float(
            task_frame["selected_explanations"].median()
        ),
        "median_score_gain": float(
            task_frame["separation_gain_over_median"].median()
        ),
        "median_effective_count_reduction": float(
            task_frame["effective_count_reduction_vs_baseline"].median()
        ),
        "median_dispersion_reduction": float(
            task_frame["dispersion_reduction_vs_baseline"].median()
        ),
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    (output / "confirmatory_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps(conditions, indent=2))


if __name__ == "__main__":
    main()
