#!/usr/bin/env python3
"""Freeze Track E2-C permutation-control models before confirmatory access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from evidence_extension_v1.track_e2.capture_features import STATE_FEATURES
from evidence_extension_v1.track_e2.model import (
    EVENT_THRESHOLD,
    fit_logistic_ridge,
    select_full_development_alpha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dir", required=True)
    parser.add_argument("--e2b-lock", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def cyclic_shift_state(
    frame: pd.DataFrame,
    state_features: list[str],
    shift: int,
) -> pd.DataFrame:
    output = frame.copy()
    strata = ["family", "dimension", "snapshot_index"]
    for _, indices in output.groupby(strata, sort=True).groups.items():
        ordered = sorted(indices, key=lambda idx: int(output.loc[idx, "instance"]))
        if len(ordered) <= 1:
            continue
        effective = int(shift) % len(ordered)
        source = ordered[-effective:] + ordered[:-effective] if effective else ordered
        output.loc[ordered, state_features] = output.loc[source, state_features].to_numpy()
    return output


def serialize_model(model, feature_order: list[str], shift: int) -> dict:
    return {
        "shift": int(shift),
        "feature_order": feature_order,
        "alpha": float(model.alpha),
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "intercept": float(model.intercept),
        "coefficients": model.coefficients.tolist(),
        "constant_probability": model.constant_probability,
    }


def main() -> None:
    args = parse_args()
    development = Path(args.development_dir)
    frame = pd.read_csv(development / "track_e2_development_snapshots.csv")
    e2b_lock = json.loads(Path(args.e2b_lock).read_text(encoding="utf-8"))
    if e2b_lock["status"] != "TRACK_E2B_CONFIRMATORY_LOCKED_BEFORE_ACCESS":
        raise RuntimeError("Invalid Track E2-B lock.")
    state_order = list(
        e2b_lock["models"]["trace_plus_state"]["feature_order"]
    )
    labels = (frame["future_log_improvement"] >= EVENT_THRESHOLD).to_numpy(float)
    controls = []
    for shift in range(1, 8):
        shifted = cyclic_shift_state(frame, list(STATE_FEATURES), shift)
        alpha = select_full_development_alpha(
            shifted,
            state_order,
            logistic=True,
        )
        model = fit_logistic_ridge(
            shifted[state_order].to_numpy(float),
            labels,
            alpha,
        )
        controls.append(serialize_model(model, state_order, shift))

    lock = {
        "status": "TRACK_E2C_ALIGNMENT_LOCKED_BEFORE_ACCESS",
        "confirmatory_objective_evaluations_before_lock": 0,
        "development_source_commit": e2b_lock["development_source_commit"],
        "implementation_version": e2b_lock["implementation_version"],
        "options_hash": e2b_lock["options_hash"],
        "optimizer_blob": e2b_lock["optimizer_blob"],
        "event_threshold": EVENT_THRESHOLD,
        "confirmatory_instances": list(range(201, 225)),
        "parent_budget_multiplier": 250,
        "snapshot_multipliers": [80, 140, 200],
        "trace_only_model": e2b_lock["models"]["trace_only"],
        "real_state_model": e2b_lock["models"]["trace_plus_state"],
        "permutation_control_models": controls,
        "permutation_ensemble_rule": "arithmetic_mean_probability",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print("TRACK_E2C_ALIGNMENT_LOCK_CREATED")
    print(output)


if __name__ == "__main__":
    main()
