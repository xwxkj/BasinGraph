#!/usr/bin/env python3
"""Execute the locked Track E2-B event-prediction confirmatory study."""

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
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION  # noqa: E402
from evidence_extension_v1.track_e2.capture_features import STATE_FEATURES  # noqa: E402
from evidence_extension_v1.track_e2.model import (  # noqa: E402
    binary_log_loss,
    make_cyclic_shuffled_state,
)
from evidence_extension_v1.track_e2.run_development import run_parent  # noqa: E402
from evidence_extension_v1.track_e2.tasks import DIMENSIONS, FAMILIES  # noqa: E402


LOCK_PATH = (
    ROOT
    / "protocols"
    / "evidence_extension_v1"
    / "track_e2"
    / "TRACK_E2B_CONFIRMATORY_LOCK.json"
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
        default="results_b21/track_e2/event_confirmatory",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    return parser.parse_args()


def set_single_thread_environment() -> None:
    for key in THREAD_ENV:
        os.environ[key] = "1"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
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


def predict_locked(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    feature_order = list(model["feature_order"])
    matrix = frame[feature_order].to_numpy(float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    if matrix.shape[1] != len(mean) or len(mean) != len(coefficients):
        raise RuntimeError("Locked model dimension does not match feature matrix.")
    constant = model.get("constant_probability")
    if constant is not None:
        return np.full(len(frame), float(constant), dtype=float)
    standardized = (matrix - mean) / scale
    score = float(model["intercept"]) + standardized @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def per_record_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return -(
        labels * np.log(probabilities)
        + (1.0 - labels) * np.log(1.0 - probabilities)
    )


def bootstrap_difference(
    frame: pd.DataFrame,
    *,
    trace_loss: np.ndarray,
    state_loss: np.ndarray,
    replicates: int,
    seed: int = 20260813,
) -> dict[str, float]:
    working = frame[["instance"]].copy()
    working["difference"] = trace_loss - state_loss
    instances = np.asarray(sorted(working["instance"].unique()), dtype=int)
    groups = {
        int(instance): working[working["instance"] == instance][
            "difference"
        ].to_numpy(float)
        for instance in instances
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(instances, size=len(instances), replace=True)
        values = np.concatenate([groups[int(instance)] for instance in sampled])
        draws[index] = float(np.mean(values))
    return {
        "estimate": float(np.mean(trace_loss - state_loss)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def family_log_losses(
    frame: pd.DataFrame,
    labels: np.ndarray,
    trace_probability: np.ndarray,
    state_probability: np.ndarray,
    shuffled_probability: np.ndarray,
) -> pd.DataFrame:
    working = frame[["family"]].copy()
    working["label"] = labels
    working["trace_probability"] = trace_probability
    working["state_probability"] = state_probability
    working["shuffled_probability"] = shuffled_probability
    rows = []
    for family, group in working.groupby("family", sort=True):
        y = group["label"].to_numpy(float)
        trace = binary_log_loss(y, group["trace_probability"].to_numpy(float))
        state = binary_log_loss(y, group["state_probability"].to_numpy(float))
        shuffled = binary_log_loss(
            y,
            group["shuffled_probability"].to_numpy(float),
        )
        rows.append(
            {
                "family": family,
                "records": len(group),
                "positive_records": int(np.sum(y)),
                "trace_log_loss": trace,
                "state_log_loss": state,
                "shuffled_state_log_loss": shuffled,
                "trace_minus_state": trace - state,
                "relative_state_improvement": (
                    (trace - state) / max(trace, 1e-300)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_single_thread_environment()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "TRACK_E2B_CONFIRMATORY_LOCKED_BEFORE_ACCESS":
        raise RuntimeError("Track E2-B lock status is invalid.")
    if lock["confirmatory_objective_evaluations_before_lock"] != 0:
        raise RuntimeError("Track E2-B was not locked before confirmatory access.")
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected BasinGraph implementation version.")
    if BasinGraphOptions().stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected BasinGraph options hash.")
    if lock["options_hash"] != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Lock options hash mismatch.")
    if lock["optimizer_blob"] != EXPECTED_OPTIMIZER_BLOB:
        raise RuntimeError("Lock optimizer blob mismatch.")
    if git_output("rev-parse", "HEAD:basingraph_v2/optimizer.py") != EXPECTED_OPTIMIZER_BLOB:
        raise RuntimeError("Frozen optimizer source changed.")

    instances = tuple(int(value) for value in lock["confirmatory_instances"])
    tasks = [
        (family, dimension, instance)
        for family in FAMILIES
        for dimension in DIMENSIONS
        for instance in instances
    ]
    output = ROOT / args.output
    if output.exists():
        raise RuntimeError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
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
                rows.extend(future.result())
                print(
                    "TRACK_E2B_PARENT_OK",
                    family,
                    dimension,
                    instance,
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "family": family,
                        "dimension": dimension,
                        "instance": instance,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=40),
                    }
                )
                print(
                    "TRACK_E2B_PARENT_FAILED",
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
        raise RuntimeError(f"Track E2-B parent failures: {len(failures)}")

    frame = pd.DataFrame(rows).sort_values(
        ["family", "dimension", "instance", "snapshot_index"]
    ).reset_index(drop=True)
    expected_records = len(tasks) * len(lock["snapshot_multipliers"])
    if len(frame) != expected_records:
        raise RuntimeError(
            f"Unexpected confirmatory snapshot rows: {len(frame)} != {expected_records}"
        )
    integrity_ok = bool(
        frame["graph_referential_integrity"].all()
        and (frame["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
        and (frame["options_hash"] == EXPECTED_OPTIONS_HASH).all()
        and (frame["parent_phase_sum"] == frame["parent_budget"]).all()
    )
    if not integrity_ok:
        raise RuntimeError("Track E2-B confirmatory integrity checks failed.")

    threshold = float(lock["event_threshold"])
    labels = (frame["future_log_improvement"] >= threshold).to_numpy(float)
    shuffled_frame = make_cyclic_shuffled_state(frame, STATE_FEATURES)
    trace_probability = predict_locked(frame, lock["models"]["trace_only"])
    state_probability = predict_locked(
        frame,
        lock["models"]["trace_plus_state"],
    )
    shuffled_probability = predict_locked(
        shuffled_frame,
        lock["models"]["trace_plus_shuffled_state"],
    )

    trace_loss = per_record_log_loss(labels, trace_probability)
    state_loss = per_record_log_loss(labels, state_probability)
    shuffled_loss = per_record_log_loss(labels, shuffled_probability)
    trace_log_loss = float(np.mean(trace_loss))
    state_log_loss = float(np.mean(state_loss))
    shuffled_log_loss = float(np.mean(shuffled_loss))
    absolute_improvement = trace_log_loss - state_log_loss
    relative_improvement = absolute_improvement / max(trace_log_loss, 1e-300)
    shuffled_improvement = trace_log_loss - shuffled_log_loss
    bootstrap = bootstrap_difference(
        frame,
        trace_loss=trace_loss,
        state_loss=state_loss,
        replicates=int(args.bootstrap_replicates),
    )

    predictions = frame[
        [
            "task_id",
            "family",
            "dimension",
            "instance",
            "snapshot_index",
            "snapshot_actual_nfe",
            "future_log_improvement",
        ]
    ].copy()
    predictions["event_label"] = labels.astype(int)
    predictions["trace_probability"] = trace_probability
    predictions["state_probability"] = state_probability
    predictions["shuffled_state_probability"] = shuffled_probability
    predictions["trace_log_loss_record"] = trace_loss
    predictions["state_log_loss_record"] = state_loss
    predictions["shuffled_state_log_loss_record"] = shuffled_loss
    predictions.to_csv(output / "confirmatory_event_predictions.csv", index=False)
    frame.to_csv(output / "confirmatory_snapshots.csv", index=False)
    family = family_log_losses(
        frame,
        labels,
        trace_probability,
        state_probability,
        shuffled_probability,
    )
    family.to_csv(output / "confirmatory_family_log_loss.csv", index=False)

    conditions = {
        "relative_log_loss_improvement_at_least_5pct": relative_improvement >= 0.05,
        "bootstrap_lower_bound_above_zero": bootstrap["lower_95"] > 0.0,
        "real_state_exceeds_shuffled_state": absolute_improvement > shuffled_improvement,
        "both_event_classes_present": len(np.unique(labels)) == 2,
        "integrity_checks_pass": integrity_ok,
    }
    success = bool(all(conditions.values()))
    decision = {
        "status": (
            "TRACK_E2B_CONFIRMATORY_SUCCESS"
            if success
            else "TRACK_E2B_CONFIRMATORY_NOT_SUPPORTED"
        ),
        "success": success,
        "conditions": conditions,
        "source_commit": git_output("rev-parse", "HEAD"),
        "lock_sha256": sha256_file(LOCK_PATH),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "optimizer_blob": EXPECTED_OPTIMIZER_BLOB,
        "parent_runs": len(tasks),
        "snapshot_records": len(frame),
        "confirmatory_instances": list(instances),
        "positive_records": int(np.sum(labels)),
        "negative_records": int(len(labels) - np.sum(labels)),
        "trace_log_loss": trace_log_loss,
        "state_log_loss": state_log_loss,
        "shuffled_state_log_loss": shuffled_log_loss,
        "absolute_log_loss_improvement": absolute_improvement,
        "relative_log_loss_improvement": relative_improvement,
        "shuffled_absolute_log_loss_improvement": shuffled_improvement,
        "bootstrap": bootstrap,
        "families_with_nonnegative_improvement": int(
            (family["trace_minus_state"] >= -1e-15).sum()
        ),
        "dropped_transient_edges_total": int(
            frame["dropped_transient_edges"].sum()
        ),
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    (output / "confirmatory_decision.json").write_text(
        json.dumps(decision, indent=2),
        encoding="utf-8",
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps({
        "trace_log_loss": trace_log_loss,
        "state_log_loss": state_log_loss,
        "shuffled_state_log_loss": shuffled_log_loss,
        "relative_log_loss_improvement": relative_improvement,
        "bootstrap_95": [bootstrap["lower_95"], bootstrap["upper_95"]],
        "conditions": conditions,
    }, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
