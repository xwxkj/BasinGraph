"""Transparent grouped ridge comparison for Track E2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    y_mean: float
    coefficients: np.ndarray
    alpha: float

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        standardized = (matrix - self.mean) / self.scale
        return self.y_mean + standardized @ self.coefficients


def fit_ridge(matrix: np.ndarray, response: np.ndarray, alpha: float) -> RidgeFit:
    matrix = np.asarray(matrix, dtype=float)
    response = np.asarray(response, dtype=float)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (matrix - mean) / scale
    y_mean = float(np.mean(response))
    centered = response - y_mean
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        standardized.T @ centered,
    )
    return RidgeFit(mean, scale, y_mean, coefficients, float(alpha))


def _choose_alpha_inner(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    instance_column: str,
) -> float:
    instances = sorted(int(value) for value in frame[instance_column].unique())
    if len(instances) < 3:
        return 1.0
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHA_GRID}
    for held_out in instances:
        train = frame[frame[instance_column] != held_out]
        valid = frame[frame[instance_column] == held_out]
        x_train = train[feature_names].to_numpy(float)
        y_train = train["future_log_improvement"].to_numpy(float)
        x_valid = valid[feature_names].to_numpy(float)
        y_valid = valid["future_log_improvement"].to_numpy(float)
        for alpha in ALPHA_GRID:
            fit = fit_ridge(x_train, y_train, alpha)
            prediction = fit.predict(x_valid)
            scores[alpha].extend(((prediction - y_valid) ** 2).tolist())
    return min(
        ALPHA_GRID,
        key=lambda alpha: (float(np.mean(scores[alpha])), float(alpha)),
    )


def nested_leave_one_instance_out(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    model_name: str,
    instance_column: str = "instance",
) -> tuple[pd.DataFrame, dict[int, float]]:
    predictions: list[pd.DataFrame] = []
    selected: dict[int, float] = {}
    for held_out in sorted(int(value) for value in frame[instance_column].unique()):
        train = frame[frame[instance_column] != held_out].copy()
        valid = frame[frame[instance_column] == held_out].copy()
        alpha = _choose_alpha_inner(
            train,
            feature_names,
            instance_column=instance_column,
        )
        selected[held_out] = float(alpha)
        fit = fit_ridge(
            train[feature_names].to_numpy(float),
            train["future_log_improvement"].to_numpy(float),
            alpha,
        )
        valid["prediction"] = fit.predict(valid[feature_names].to_numpy(float))
        valid["model"] = model_name
        valid["outer_instance"] = held_out
        valid["selected_alpha"] = alpha
        predictions.append(valid)
    return pd.concat(predictions, ignore_index=True), selected


def metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["future_log_improvement"].to_numpy(float)
    p = predictions["prediction"].to_numpy(float)
    residual = p - y
    mse = float(np.mean(residual * residual))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - float(np.sum(residual * residual)) / max(denominator, 1e-300)
    return {"mse": mse, "mae": mae, "r_squared": r_squared}


def per_family_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, group in predictions.groupby("family", sort=True):
        values = metrics(group)
        rows.append({"family": family, **values, "records": len(group)})
    return pd.DataFrame(rows)


def paired_mse_difference(
    trace_predictions: pd.DataFrame,
    state_predictions: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["task_id", "snapshot_index"]
    left = trace_predictions[keys + ["future_log_improvement", "prediction"]].rename(
        columns={"prediction": "trace_prediction"}
    )
    right = state_predictions[keys + ["prediction"]].rename(
        columns={"prediction": "state_prediction"}
    )
    merged = left.merge(right, on=keys, validate="one_to_one")
    y = merged["future_log_improvement"].to_numpy(float)
    merged["trace_squared_error"] = (merged["trace_prediction"] - y) ** 2
    merged["state_squared_error"] = (merged["state_prediction"] - y) ** 2
    merged["mse_advantage_trace_minus_state"] = (
        merged["trace_squared_error"] - merged["state_squared_error"]
    )
    return merged


def instance_bootstrap_interval(
    paired: pd.DataFrame,
    instance_lookup: pd.DataFrame,
    *,
    replicates: int = 10_000,
    seed: int = 20260811,
) -> dict[str, float]:
    merged = paired.merge(
        instance_lookup[["task_id", "instance"]].drop_duplicates(),
        on="task_id",
        validate="many_to_one",
    )
    instances = np.asarray(sorted(merged["instance"].unique()), dtype=int)
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    groups = {
        int(instance): merged[merged["instance"] == instance][
            "mse_advantage_trace_minus_state"
        ].to_numpy(float)
        for instance in instances
    }
    for index in range(replicates):
        sampled = rng.choice(instances, size=len(instances), replace=True)
        values = np.concatenate([groups[int(instance)] for instance in sampled])
        draws[index] = float(np.mean(values))
    return {
        "estimate": float(merged["mse_advantage_trace_minus_state"].mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def make_cyclic_shuffled_state(
    frame: pd.DataFrame,
    state_features: Iterable[str],
) -> pd.DataFrame:
    output = frame.copy()
    state_features = list(state_features)
    strata = ["family", "dimension", "snapshot_index"]
    for _, indices in output.groupby(strata, sort=True).groups.items():
        ordered = sorted(indices, key=lambda idx: int(output.loc[idx, "instance"]))
        if len(ordered) <= 1:
            continue
        source = ordered[-1:] + ordered[:-1]
        output.loc[ordered, state_features] = output.loc[source, state_features].to_numpy()
    return output
