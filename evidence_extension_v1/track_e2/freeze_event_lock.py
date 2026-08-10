#!/usr/bin/env python3
"""Freeze Track E2-B logistic models from the audited development artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from evidence_extension_v1.track_e2.capture_features import STATE_FEATURES
from evidence_extension_v1.track_e2.model import (
    EVENT_THRESHOLD,
    fit_logistic_ridge,
    make_cyclic_shuffled_state,
    select_full_development_alpha,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def serialize_model(model, feature_order: list[str]) -> dict:
    return {
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
    decision = json.loads((development / "development_decision.json").read_text())
    if decision["gate_passed"]:
        raise RuntimeError("Expected the original continuous Track E2 gate to remain closed.")

    trace_features = list(
        decision["selected_penalties_candidate"]["trace_feature_order"]
    )
    state_features = list(
        decision["selected_penalties_candidate"]["state_feature_order"]
    )
    shuffled = make_cyclic_shuffled_state(frame, STATE_FEATURES)
    labels = (frame["future_log_improvement"] >= EVENT_THRESHOLD).to_numpy(float)
    shuffled_labels = (
        shuffled["future_log_improvement"] >= EVENT_THRESHOLD
    ).to_numpy(float)

    trace_alpha = select_full_development_alpha(
        frame,
        trace_features,
        logistic=True,
    )
    state_alpha = select_full_development_alpha(
        frame,
        state_features,
        logistic=True,
    )
    shuffled_alpha = select_full_development_alpha(
        shuffled,
        state_features,
        logistic=True,
    )

    trace_model = fit_logistic_ridge(
        frame[trace_features].to_numpy(float),
        labels,
        trace_alpha,
    )
    state_model = fit_logistic_ridge(
        frame[state_features].to_numpy(float),
        labels,
        state_alpha,
    )
    shuffled_model = fit_logistic_ridge(
        shuffled[state_features].to_numpy(float),
        shuffled_labels,
        shuffled_alpha,
    )

    lock = {
        "status": "TRACK_E2B_CONFIRMATORY_LOCKED_BEFORE_ACCESS",
        "development_source_commit": decision["source_commit"],
        "confirmatory_objective_evaluations_before_lock": 0,
        "implementation_version": decision["implementation_version"],
        "options_hash": decision["options_hash"],
        "optimizer_blob": decision["optimizer_blob"],
        "event_threshold": EVENT_THRESHOLD,
        "confirmatory_instances": list(range(101, 125)),
        "parent_budget_multiplier": 250,
        "snapshot_multipliers": [80, 140, 200],
        "development_continuous_endpoint": {
            "relative_mse_improvement": decision["relative_mse_improvement"],
            "gate_passed": False,
        },
        "development_event_endpoint": decision["logistic_metrics"],
        "models": {
            "trace_only": serialize_model(trace_model, trace_features),
            "trace_plus_state": serialize_model(state_model, state_features),
            "trace_plus_shuffled_state": serialize_model(
                shuffled_model,
                state_features,
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    print("TRACK_E2B_CONFIRMATORY_LOCK_CREATED")
    print(output)


if __name__ == "__main__":
    main()
