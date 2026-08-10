#!/usr/bin/env python3
"""Execute Track E2-C alignment-specificity confirmatory study."""

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
from evidence_extension_v1.track_e2.run_development import run_parent  # noqa: E402
from evidence_extension_v1.track_e2.tasks import DIMENSIONS, FAMILIES  # noqa: E402


LOCK_PATH = (
    ROOT
    / "protocols"
    / "evidence_extension_v1"
    / "track_e2"
    / "TRACK_E2C_ALIGNMENT_LOCK.json"
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
        default="results_b21/track_e2/alignment_confirmatory",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
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


def predict_locked(frame: pd.DataFrame, model: dict[str, Any]) -> np.ndarray:
    order = list(model["feature_order"])
    matrix = frame[order].to_numpy(float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    constant = model.get("constant_probability")
    if constant is not None:
        return np.full(len(frame), float(constant), dtype=float)
    score = float(model["intercept"]) + ((matrix - mean) / scale) @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def cyclic_shift_state(frame: pd.DataFrame, shift: int) -> pd.DataFrame:
    output = frame.copy()
    strata = ["family", "dimension", "snapshot_index"]
    features = list(STATE_FEATURES)
    for _, indices in output.groupby(strata, sort=True).groups.items():
        ordered = sorted(indices, key=lambda idx: int(output.loc[idx, "instance"]))
        effective = int(shift) % len(ordered)
        source = ordered[-effective:] + ordered[:-effective] if effective else ordered
        output.loc[ordered, features] = output.loc[source, features].to_numpy()
    return output


def per_record_loss(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    labels = np.asarray(labels, dtype=float)
    return -(
        labels * np.log(probability)
        + (1.0 - labels) * np.log(1.0 - probability)
    )


def bootstrap(
    frame: pd.DataFrame,
    difference: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    working = frame[["instance"]].copy()
    working["difference"] = difference
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
        draws[index] = float(
            np.mean(np.concatenate([groups[int(value)] for value in sampled]))
        )
    return {
        "estimate": float(np.mean(difference)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def main() -> None:
    args = parse_args()
    for key in THREAD_ENV:
        os.environ[key] = "1"
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock["status"] != "TRACK_E2C_ALIGNMENT_LOCKED_BEFORE_ACCESS":
        raise RuntimeError("Invalid Track E2-C lock status.")
    if lock["confirmatory_objective_evaluations_before_lock"] != 0:
        raise RuntimeError("Track E2-C lock was not created before access.")
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("Unexpected implementation version.")
    if BasinGraphOptions().stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Unexpected options hash.")
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
    rows = []
    failures = []
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
                print("TRACK_E2C_PARENT_OK", family, dimension, instance, flush=True)
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
    if failures:
        (output / "failures.json").write_text(
            json.dumps(failures, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Track E2-C parent failures: {len(failures)}")

    frame = pd.DataFrame(rows).sort_values(
        ["family", "dimension", "instance", "snapshot_index"]
    ).reset_index(drop=True)
    expected_records = len(tasks) * len(lock["snapshot_multipliers"])
    if len(frame) != expected_records:
        raise RuntimeError("Unexpected Track E2-C snapshot count.")
    integrity_ok = bool(
        frame["graph_referential_integrity"].all()
        and (frame["implementation_version"] == EXPECTED_IMPLEMENTATION).all()
        and (frame["options_hash"] == EXPECTED_OPTIONS_HASH).all()
        and (frame["parent_phase_sum"] == frame["parent_budget"]).all()
    )
    if not integrity_ok:
        raise RuntimeError("Track E2-C integrity checks failed.")

    threshold = float(lock["event_threshold"])
    labels = (frame["future_log_improvement"] >= threshold).to_numpy(float)
    trace_probability = predict_locked(frame, lock["trace_only_model"])
    real_probability = predict_locked(frame, lock["real_state_model"])
    control_probabilities = []
    for model in lock["permutation_control_models"]:
        shifted = cyclic_shift_state(frame, int(model["shift"]))
        control_probabilities.append(predict_locked(shifted, model))
    control_matrix = np.asarray(control_probabilities, dtype=float)
    ensemble_probability = np.mean(control_matrix, axis=0)

    trace_loss = per_record_loss(labels, trace_probability)
    real_loss = per_record_loss(labels, real_probability)
    ensemble_loss = per_record_loss(labels, ensemble_probability)
    trace_mean = float(np.mean(trace_loss))
    real_mean = float(np.mean(real_loss))
    ensemble_mean = float(np.mean(ensemble_loss))
    trace_difference = trace_loss - real_loss
    specificity_difference = ensemble_loss - real_loss
    trace_bootstrap = bootstrap(
        frame,
        trace_difference,
        replicates=int(args.bootstrap_replicates),
        seed=20260815,
    )
    specificity_bootstrap = bootstrap(
        frame,
        specificity_difference,
        replicates=int(args.bootstrap_replicates),
        seed=20260816,
    )
    relative_trace_improvement = (trace_mean - real_mean) / max(trace_mean, 1e-300)
    relative_specificity_improvement = (
        ensemble_mean - real_mean
    ) / max(ensemble_mean, 1e-300)

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
    predictions["real_state_probability"] = real_probability
    predictions["permutation_ensemble_probability"] = ensemble_probability
    for index, probability in enumerate(control_probabilities, start=1):
        predictions[f"shift_{index}_probability"] = probability
    predictions["trace_log_loss_record"] = trace_loss
    predictions["real_state_log_loss_record"] = real_loss
    predictions["permutation_ensemble_log_loss_record"] = ensemble_loss
    predictions.to_csv(output / "alignment_confirmatory_predictions.csv", index=False)
    frame.to_csv(output / "alignment_confirmatory_snapshots.csv", index=False)

    conditions = {
        "real_vs_permutation_relative_improvement_at_least_2pct": (
            relative_specificity_improvement >= 0.02
        ),
        "specificity_bootstrap_lower_above_zero": (
            specificity_bootstrap["lower_95"] > 0.0
        ),
        "real_vs_trace_relative_improvement_at_least_5pct": (
            relative_trace_improvement >= 0.05
        ),
        "trace_bootstrap_lower_above_zero": trace_bootstrap["lower_95"] > 0.0,
        "both_classes_present": len(np.unique(labels)) == 2,
        "integrity_checks_pass": integrity_ok,
    }
    success = bool(all(conditions.values()))
    decision = {
        "status": (
            "TRACK_E2C_ALIGNMENT_SPECIFICITY_SUPPORTED"
            if success
            else "TRACK_E2C_ALIGNMENT_SPECIFICITY_NOT_SUPPORTED"
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
        "positive_records": int(np.sum(labels)),
        "negative_records": int(len(labels) - np.sum(labels)),
        "trace_log_loss": trace_mean,
        "real_state_log_loss": real_mean,
        "permutation_ensemble_log_loss": ensemble_mean,
        "relative_trace_improvement": relative_trace_improvement,
        "relative_specificity_improvement": relative_specificity_improvement,
        "trace_bootstrap": trace_bootstrap,
        "specificity_bootstrap": specificity_bootstrap,
        "dropped_transient_edges_total": int(
            frame["dropped_transient_edges"].sum()
        ),
        "wall_time_seconds": float(time.perf_counter() - started),
    }
    (output / "alignment_confirmatory_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_manifest(output)
    print(decision["status"])
    print(json.dumps({
        "trace_log_loss": trace_mean,
        "real_state_log_loss": real_mean,
        "permutation_ensemble_log_loss": ensemble_mean,
        "relative_trace_improvement": relative_trace_improvement,
        "relative_specificity_improvement": relative_specificity_improvement,
        "trace_bootstrap_95": [trace_bootstrap["lower_95"], trace_bootstrap["upper_95"]],
        "specificity_bootstrap_95": [
            specificity_bootstrap["lower_95"],
            specificity_bootstrap["upper_95"],
        ],
        "conditions": conditions,
    }, indent=2))


if __name__ == "__main__":
    main()
