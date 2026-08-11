"""Transparent grouped ridge comparisons for Track E2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


ALPHA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
EVENT_THRESHOLD = 0.10


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


@dataclass
class LogisticFit:
    mean: np.ndarray
    scale: np.ndarray
    intercept: float
    coefficients: np.ndarray
    alpha: float
    constant_probability: float | None = None

    def predict_probability(self, matrix: np.ndarray) -> np.ndarray:
        if self.constant_probability is not None:
            return np.full(len(matrix), self.constant_probability, dtype=float)
        standardized = (matrix - self.mean) / self.scale
        score = self.intercept + standardized @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))


def _standardization(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def fit_ridge(matrix: np.ndarray, response: np.ndarray, alpha: float) -> RidgeFit:
    standardized, mean, scale = _standardization(matrix)
    response = np.asarray(response, dtype=float)
    y_mean = float(np.mean(response))
    centered = response - y_mean
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        standardized.T @ centered,
    )
    return RidgeFit(mean, scale, y_mean, coefficients, float(alpha))


def fit_logistic_ridge(
    matrix: np.ndarray,
    labels: np.ndarray,
    alpha: float,
) -> LogisticFit:
    standardized, mean, scale = _standardization(matrix)
    labels = np.asarray(labels, dtype=float)
    prevalence = float(np.mean(labels))
    if prevalence <= 0.0 or prevalence >= 1.0:
        probability = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
        return LogisticFit(
            mean,
            scale,
            float(np.log(probability / (1.0 - probability))),
            np.zeros(standardized.shape[1]),
            float(alpha),
            probability,
        )

    initial_intercept = float(np.log(prevalence / (1.0 - prevalence)))
    initial = np.zeros(standardized.shape[1] + 1, dtype=float)
    initial[0] = initial_intercept

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        score = intercept + standardized @ coefficients
        probability = 1.0 / (1.0 + np.exp(-np.clip(score, -40.0, 40.0)))
        loss = float(
            np.mean(np.logaddexp(0.0, score) - labels * score)
            + 0.5 * float(alpha) * np.dot(coefficients, coefficients)
        )
        residual = probability - labels
        gradient = np.empty_like(parameters)
        gradient[0] = float(np.mean(residual))
        gradient[1:] = (
            standardized.T @ residual / len(labels)
            + float(alpha) * coefficients
        )
        return loss, gradient

    result = minimize(
        lambda parameters: objective(parameters)[0],
        initial,
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    parameters = np.asarray(result.x, dtype=float)
    return LogisticFit(
        mean,
        scale,
        float(parameters[0]),
        parameters[1:].copy(),
        float(alpha),
        None,
    )


def binary_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    return float(
        -np.mean(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )
    )


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


def _choose_logistic_alpha_inner(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    instance_column: str,
) -> float:
    instances = sorted(int(value) for value in frame[instance_column].unique())
    labels_all = (frame["future_log_improvement"] >= EVENT_THRESHOLD).astype(float)
    if labels_all.nunique() < 2 or len(instances) < 3:
        return 1.0
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHA_GRID}
    for held_out in instances:
        train = frame[frame[instance_column] != held_out]
        valid = frame[frame[instance_column] == held_out]
        x_train = train[feature_names].to_numpy(float)
        y_train = (train["future_log_improvement"] >= EVENT_THRESHOLD).to_numpy(float)
        x_valid = valid[feature_names].to_numpy(float)
        y_valid = (valid["future_log_improvement"] >= EVENT_THRESHOLD).to_numpy(float)
        for alpha in ALPHA_GRID:
            fit = fit_logistic_ridge(x_train, y_train, alpha)
            probability = fit.predict_probability(x_valid)
            scores[alpha].append(binary_log_loss(y_valid, probability))
    return min(
        ALPHA_GRID,
        key=lambda alpha: (float(np.mean(scores[alpha])), float(alpha)),
    )


def select_full_development_alpha(
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    logistic: bool = False,
    instance_column: str = "instance",
) -> float:
    if logistic:
        return _choose_logistic_alpha_inner(
            frame,
            feature_names,
            instance_column=instance_column,
        )
    return _choose_alpha_inner(
        frame,
        feature_names,
        instance_column=instance_column,
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


def nested_logistic_leave_one_instance_out(
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
        alpha = _choose_logistic_alpha_inner(
            train,
            feature_names,
            instance_column=instance_column,
        )
        selected[held_out] = float(alpha)
        fit = fit_logistic_ridge(
            train[feature_names].to_numpy(float),
            (train["future_log_improvement"] >= EVENT_THRESHOLD).to_numpy(float),
            alpha,
        )
        valid["event_label"] = (
            valid["future_log_improvement"] >= EVENT_THRESHOLD
        ).astype(int)
        valid["event_probability"] = fit.predict_probability(
            valid[feature_names].to_numpy(float)
        )
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
