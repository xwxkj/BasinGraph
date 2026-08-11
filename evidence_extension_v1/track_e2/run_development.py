#!/usr/bin/env python3
"""Execute the frozen Track E2 development-stage predictive-information test."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import (  # noqa: E402
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)
from evidence_extension_v1.track_e2.capture_features import (  # noqa: E402
    STATE_FEATURES,
    TRACE_FEATURES,
    capture_optimizer_state,
    state_features,
    trace_features,
)
from evidence_extension_v1.track_e2.model import (  # noqa: E402
    EVENT_THRESHOLD,
    binary_log_loss,
    instance_bootstrap_interval,
    make_cyclic_shuffled_state,
    metrics,
    nested_leave_one_instance_out,
    nested_logistic_leave_one_instance_out,
    paired_mse_difference,
    per_family_metrics,
    select_full_development_alpha,
)
from evidence_extension_v1.track_e2.tasks import (  # noqa: E402
    DEVELOPMENT_INSTANCES,
    DIMENSIONS,
    FAMILIES,
    make_task,
)


EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
EXPECTED_OPTIMIZER_BLOB = "4433d2b92075dcd858529f7f13342631847def4c"
PARENT_BUDGET_MULTIPLIER = 250
SNAPSHOT_MULTIPLIERS = (80, 140, 200)
BASE_PARENT_SEED = 202608120
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
        default="results_b21/track_e2/development",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def set_single_thread_environment() -> None:
    for key in THREAD_ENV:
        os.environ[key] = "1"


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def parent_seed(family: str, dimension: int, instance: int) -> int:
    return int(
        BASE_PARENT_SEED
        + 100_000 * (FAMILIES.index(family) + 1)
        + 1_000 * dimension
        + instance
    )


def history_monotone(history: list[tuple[int, float]]) -> bool:
    values = [float(value) for _, value in history]
    return all(
        later <= earlier + 1e-13 * max(1.0, abs(earlier))
        for earlier, later in zip(values[:-1], values[1:])
    )


def run_parent(family: str, dimension: int, instance: int) -> list[dict[str, Any]]:
    set_single_thread_environment()
    task = make_task(family, dimension, instance)
    options = BasinGraphOptions()
    budget = int(PARENT_BUDGET_MULTIPLIER * dimension)
    thresholds = [int(multiplier * dimension) for multiplier in SNAPSHOT_MULTIPLIERS]
    seed = parent_seed(family, dimension, instance)
    started = time.perf_counter()

    with capture_optimizer_state(thresholds) as context:
        result = minimize_basingraph_v2(
            task.objective,
            task.lower,
            task.upper,
            max_evals=budget,
            seed=seed,
            options=options,
        )

    if len(context.snapshots) != len(thresholds):
        raise RuntimeError(
            f"{task.task_id}: captured {len(context.snapshots)} snapshots, "
            f"expected {len(thresholds)}"
        )
    if result.nfe != budget:
        raise RuntimeError(f"{task.task_id}: nfe {result.nfe} != budget {budget}")
    if result.options_hash != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(f"{task.task_id}: options hash changed")
    if not history_monotone(result.history):
        raise RuntimeError(f"{task.task_id}: non-monotone parent history")

    final_log_gap = float(np.log10(1.0 + max(float(result.fbest), 0.0)))
    rows: list[dict[str, Any]] = []
    for snapshot_index, snapshot in enumerate(context.snapshots, start=1):
        trace = trace_features(
            snapshot,
            dimension=dimension,
            parent_budget=budget,
        )
        state = state_features(
            snapshot,
            lower=task.lower,
            upper=task.upper,
        )
        snapshot_log_gap = float(
            np.log10(1.0 + max(float(snapshot.fbest), 0.0))
        )
        future_improvement = max(0.0, snapshot_log_gap - final_log_gap)
        row: dict[str, Any] = {
            "task_id": task.task_id,
            "family": family,
            "dimension": int(dimension),
            "instance": int(instance),
            "parent_seed": int(seed),
            "parent_budget": int(budget),
            "snapshot_index": int(snapshot_index),
            "snapshot_target_multiplier": int(
                SNAPSHOT_MULTIPLIERS[snapshot_index - 1]
            ),
            "snapshot_target_nfe": int(snapshot.target_nfe),
            "snapshot_actual_nfe": int(snapshot.actual_nfe),
            "snapshot_fbest": float(snapshot.fbest),
            "parent_final_fbest": float(result.fbest),
            "future_log_improvement": float(future_improvement),
            "event_label": int(future_improvement >= EVENT_THRESHOLD),
            "archive_nodes": int(len(snapshot.archive)),
            "graph_edges_retained": int(len(snapshot.graph_edges)),
            "dropped_transient_edges": int(snapshot.dropped_transient_edges),
            "graph_referential_integrity": bool(
                snapshot.graph_referential_integrity
            ),
            "parent_history_monotone": True,
            "parent_phase_sum": int(sum(result.phase_evaluations.values())),
            "implementation_version": result.implementation_version,
            "options_hash": result.options_hash,
            "parent_wall_time_seconds": float(time.perf_counter() - started),
        }
        row.update(trace)
        row.update(state)
        row["trace_feature_hash"] = stable_digest(
            {name: row[name] for name in TRACE_FEATURES}
        )
        row["state_feature_hash"] = stable_digest(
            {name: row[name] for name in STATE_FEATURES}
        )
        rows.append(row)
    return rows


def write_manifest(root: Path) -> None:
    manifest_path = root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def combine_family_metrics(
    trace_predictions: pd.DataFrame,
    state_predictions: pd.DataFrame,
) -> pd.DataFrame:
    trace = per_family_metrics(trace_predictions).rename(
        columns={
            "mse": "trace_mse",
            "mae": "trace_mae",
            "r_squared": "trace_r_squared",
        }
    )
    state = per_family_metrics(state_predictions).rename(
        columns={
            "mse": "state_mse",
            "mae": "state_mae",
            "r_squared": "state_r_squared",
            "records": "state_records",
        }
    )
    merged = trace.merge(state, on="family", validate="one_to_one")
    merged["mse_difference_trace_minus_state"] = (
        merged["trace_mse"] - merged["state_mse"]
    )
    merged["relative_mse_improvement"] = (
        merged["mse_difference_trace_minus_state"]
        / merged["trace_mse"].clip(lower=1e-300)
    )
    return merged


def main() -> None:
    args = parse_args()
    set_single_thread_environment()
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected BasinGraph implementation version.")
    if BasinGraphOptions().stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected BasinGraph options hash.")
    if git_output("rev-parse", "HEAD:basingraph_v2/optimizer.py") != EXPECTED_OPTIMIZER_BLOB:
        raise RuntimeError("Frozen optimizer source blob changed.")

    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    tasks = [
        (family, dimension, instance)
        for family in FAMILIES
        for dimension in DIMENSIONS
        for instance in DEVELOPMENT_INSTANCES
    ]
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(run_parent, family, dimension, instance): (
                family,
                dimension,
                instance,
            )
            for family, dimension, instance in tasks
        }
        for future in as_completed(futures):
            family, dimension, instance = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                print(
                    "TRACK_E2_PARENT_OK",
                    family,
                    dimension,
                    instance,
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "family": family,
                        "dimension": str(dimension),
                        "instance": str(instance),
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=40),
                    }
                )
                print(
                    "TRACK_E2_PARENT_FAILED",
                    family,
                    dimension,
                    instance,
                    exc,
                    flush=True,
                )
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2),
            encoding="utf-8",
        )
        raise RuntimeError(f"Track E2 parent failures: {len(failures)}")

    frame = pd.DataFrame(all_rows).sort_values(
        ["family", "dimension", "instance", "snapshot_index"]
    ).reset_index(drop=True)
    expected_records = len(tasks) * len(SNAPSHOT_MULTIPLIERS)
    if len(frame) != expected_records:
        raise RuntimeError(f"Unexpected snapshot rows: {len(frame)} != {expected_records}")
    if not frame["graph_referential_integrity"].all():
        raise RuntimeError("Snapshot graph referential integrity failed.")
    if not (frame["parent_phase_sum"] == frame["parent_budget"]).all():
        raise RuntimeError("Parent phase accounting failed.")
    frame.to_csv(output / "track_e2_development_snapshots.csv", index=False)

    trace_names = list(TRACE_FEATURES)
    state_names = list(TRACE_FEATURES + STATE_FEATURES)
    shuffled_frame = make_cyclic_shuffled_state(frame, STATE_FEATURES)

    trace_predictions, trace_alpha_by_fold = nested_leave_one_instance_out(
        frame,
        trace_names,
        model_name="TraceOnly",
    )
    state_predictions, state_alpha_by_fold = nested_leave_one_instance_out(
        frame,
        state_names,
        model_name="TracePlusState",
    )
    shuffled_predictions, shuffled_alpha_by_fold = nested_leave_one_instance_out(
        shuffled_frame,
        state_names,
        model_name="TracePlusShuffledState",
    )

    predictions = pd.concat(
        [trace_predictions, state_predictions, shuffled_predictions],
        ignore_index=True,
    )
    predictions.to_csv(output / "nested_oof_predictions.csv", index=False)

    trace_metric = metrics(trace_predictions)
    state_metric = metrics(state_predictions)
    shuffled_metric = metrics(shuffled_predictions)
    paired = paired_mse_difference(trace_predictions, state_predictions)
    paired.to_csv(output / "paired_squared_errors.csv", index=False)
    bootstrap = instance_bootstrap_interval(
        paired,
        frame,
        replicates=int(args.bootstrap_replicates),
    )
    family_metrics = combine_family_metrics(trace_predictions, state_predictions)
    family_metrics.to_csv(output / "per_family_metrics.csv", index=False)

    logistic_status = "not_run_single_class"
    logistic_metrics: dict[str, Any] = {}
    if frame["event_label"].nunique() == 2:
        trace_logistic, trace_logistic_alpha = nested_logistic_leave_one_instance_out(
            frame,
            trace_names,
            model_name="TraceOnly",
        )
        state_logistic, state_logistic_alpha = nested_logistic_leave_one_instance_out(
            frame,
            state_names,
            model_name="TracePlusState",
        )
        logistic_predictions = pd.concat(
            [trace_logistic, state_logistic],
            ignore_index=True,
        )
        logistic_predictions.to_csv(
            output / "nested_oof_event_predictions.csv",
            index=False,
        )
        logistic_status = "completed"
        logistic_metrics = {
            "event_threshold": EVENT_THRESHOLD,
            "positive_records": int(frame["event_label"].sum()),
            "negative_records": int((1 - frame["event_label"]).sum()),
            "trace_log_loss": binary_log_loss(
                trace_logistic["event_label"].to_numpy(float),
                trace_logistic["event_probability"].to_numpy(float),
            ),
            "state_log_loss": binary_log_loss(
                state_logistic["event_label"].to_numpy(float),
                state_logistic["event_probability"].to_numpy(float),
            ),
            "trace_alpha_by_outer_fold": trace_logistic_alpha,
            "state_alpha_by_outer_fold": state_logistic_alpha,
        }

    absolute_improvement = float(trace_metric["mse"] - state_metric["mse"])
    relative_improvement = float(
        absolute_improvement / max(trace_metric["mse"], 1e-300)
    )
    shuffled_absolute_improvement = float(
        trace_metric["mse"] - shuffled_metric["mse"]
    )
    nonnegative_families = int(
        (family_metrics["mse_difference_trace_minus_state"] >= -1e-15).sum()
    )
    integrity_ok = bool(
        frame["graph_referential_integrity"].all()
        and (frame["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
        and (frame["options_hash"] == EXPECTED_OPTIONS_HASH).all()
        and (frame["parent_phase_sum"] == frame["parent_budget"]).all()
    )

    gate_conditions = {
        "relative_mse_improvement_at_least_5pct": relative_improvement >= 0.05,
        "bootstrap_lower_bound_above_zero": bootstrap["lower_95"] > 0.0,
        "real_state_exceeds_shuffled_state": absolute_improvement > shuffled_absolute_improvement,
        "at_least_five_nonnegative_families": nonnegative_families >= 5,
        "integrity_checks_pass": integrity_ok,
    }
    gate_passed = bool(all(gate_conditions.values()))

    selected_penalties = {
        "trace_ridge_alpha": select_full_development_alpha(frame, trace_names),
        "state_ridge_alpha": select_full_development_alpha(frame, state_names),
        "shuffled_state_ridge_alpha": select_full_development_alpha(
            shuffled_frame,
            state_names,
        ),
        "trace_feature_order": trace_names,
        "state_feature_order": state_names,
    }
    if frame["event_label"].nunique() == 2:
        selected_penalties.update(
            {
                "trace_logistic_alpha": select_full_development_alpha(
                    frame,
                    trace_names,
                    logistic=True,
                ),
                "state_logistic_alpha": select_full_development_alpha(
                    frame,
                    state_names,
                    logistic=True,
                ),
            }
        )

    summary = {
        "status": (
            "TRACK_E2_DEVELOPMENT_GATE_PASSED"
            if gate_passed
            else "TRACK_E2_DEVELOPMENT_GATE_NOT_PASSED"
        ),
        "source_commit": git_output("rev-parse", "HEAD"),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "optimizer_blob": EXPECTED_OPTIMIZER_BLOB,
        "parent_runs": len(tasks),
        "snapshot_records": len(frame),
        "families": list(FAMILIES),
        "dimensions": list(DIMENSIONS),
        "development_instances": list(DEVELOPMENT_INSTANCES),
        "parent_budget_multiplier": PARENT_BUDGET_MULTIPLIER,
        "snapshot_multipliers": list(SNAPSHOT_MULTIPLIERS),
        "trace_metrics": trace_metric,
        "state_metrics": state_metric,
        "shuffled_state_metrics": shuffled_metric,
        "absolute_mse_improvement": absolute_improvement,
        "relative_mse_improvement": relative_improvement,
        "shuffled_absolute_mse_improvement": shuffled_absolute_improvement,
        "bootstrap": bootstrap,
        "nonnegative_families": nonnegative_families,
        "gate_conditions": gate_conditions,
        "gate_passed": gate_passed,
        "selected_penalties_candidate": selected_penalties,
        "trace_alpha_by_outer_fold": trace_alpha_by_fold,
        "state_alpha_by_outer_fold": state_alpha_by_fold,
        "shuffled_alpha_by_outer_fold": shuffled_alpha_by_fold,
        "logistic_status": logistic_status,
        "logistic_metrics": logistic_metrics,
        "dropped_transient_edges_total": int(
            frame.drop_duplicates(["task_id", "snapshot_index"])[
                "dropped_transient_edges"
            ].sum()
        ),
        "wall_time_seconds": float(time.perf_counter() - started),
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
    }
    (output / "development_decision.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (output / "confirmatory_lock_candidate.json").write_text(
        json.dumps(
            {
                "status": "candidate_only_not_authorized",
                "development_gate_passed": gate_passed,
                "confirmatory_instances": list(range(101, 109)),
                **selected_penalties,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_manifest(output)
    print(summary["status"])
    print(json.dumps({
        "relative_mse_improvement": relative_improvement,
        "bootstrap_95": [bootstrap["lower_95"], bootstrap["upper_95"]],
        "nonnegative_families": nonnegative_families,
        "gate_conditions": gate_conditions,
    }, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
