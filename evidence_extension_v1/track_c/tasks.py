"""Deterministic scientific-model tasks for B21 Track C."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


Array = np.ndarray


@dataclass
class ScientificTask:
    domain: str
    family: str
    instance: int
    dimension: int
    budget_multiplier: int
    lb: Array
    ub: Array
    objective: Callable[[Array], float]
    metrics: Callable[[Array], dict[str, float]]
    f_ref: float
    f_base: float
    reference_x: Array
    metadata: dict[str, Any]
    specialized_reference: Callable[[], dict[str, Any]] | None = None

    @property
    def task_id(self) -> str:
        return f"{self.family}_i{self.instance}_d{self.dimension}"

    @property
    def budget(self) -> int:
        return int(self.budget_multiplier * self.dimension)


def _safe_reference(objective: Callable[[Array], float], truth: Array, base: Array) -> tuple[float, float]:
    f_ref = float(objective(np.asarray(truth, dtype=float)))
    f_base = float(objective(np.asarray(base, dtype=float)))
    if not np.isfinite(f_ref) or not np.isfinite(f_base):
        raise RuntimeError("Non-finite reference objective.")
    if f_base <= f_ref + 1e-13:
        # A fixed deterministic perturbation provides a non-degenerate baseline
        # without using any optimizer output.
        candidate = np.asarray(base, dtype=float) + 0.25
        f_base = float(objective(candidate))
    if f_base <= f_ref + 1e-13:
        f_base = f_ref + max(1.0, abs(f_ref))
    return f_ref, f_base


def _thomas(lower: Array, diagonal: Array, upper: Array, rhs: Array) -> Array:
    n = len(diagonal)
    c = np.array(upper, dtype=float, copy=True)
    d = np.array(rhs, dtype=float, copy=True)
    b = np.array(diagonal, dtype=float, copy=True)
    for i in range(1, n):
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.empty(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def make_elliptic_pde(instance: int) -> ScientificTask:
    dimensions = {11: 6, 12: 8, 13: 10, 1: 6, 2: 8}
    d = dimensions[instance]
    rng = np.random.default_rng(110_000 + instance)
    n = 64 + 4 * d
    x = np.linspace(0.0, 1.0, n + 2)
    xi = x[1:-1]
    basis = np.stack(
        [np.sin(math.pi * (j + 1) * x) / (j + 1) for j in range(d)], axis=1
    )
    truth = np.zeros(d)
    truth[: min(4, d)] = rng.normal(0.0, [0.55, 0.35, 0.25, 0.18][: min(4, d)])
    source = 1.0 + 0.35 * np.sin(2.0 * math.pi * xi)
    h = 1.0 / (n + 1)

    def state(theta: Array) -> Array:
        log_k = np.clip(basis @ np.asarray(theta, dtype=float), -3.0, 3.0)
        k_nodes = np.exp(log_k)
        k_half = 0.5 * (k_nodes[:-1] + k_nodes[1:])
        # k_half contains the n+1 conductivity values at interfaces.
        left = k_half[:-1]
        right = k_half[1:]
        diagonal = (left + right) / (h * h)
        lower = -left[1:] / (h * h)
        upper = -right[:-1] / (h * h)
        return _thomas(lower, diagonal, upper, source)

    truth_state = state(truth)
    obs_indices = np.linspace(2, n - 3, 18, dtype=int)
    noise_scale = 0.0025 * max(np.std(truth_state[obs_indices]), 1e-6)
    observations = truth_state[obs_indices] + rng.normal(0.0, noise_scale, len(obs_indices))
    scale = max(float(np.var(observations)), 1e-10)

    def objective(theta: Array) -> float:
        pred = state(theta)[obs_indices]
        return float(np.mean((pred - observations) ** 2) / scale + 1e-5 * np.mean(np.asarray(theta) ** 2))

    def metrics(theta: Array) -> dict[str, float]:
        pred_state = state(theta)
        return {
            "parameter_relative_error": float(np.linalg.norm(theta - truth) / max(np.linalg.norm(truth), 1e-12)),
            "state_relative_error": float(np.linalg.norm(pred_state - truth_state) / max(np.linalg.norm(truth_state), 1e-12)),
            "observation_rmse": float(np.sqrt(np.mean((pred_state[obs_indices] - observations) ** 2))),
        }

    base = np.zeros(d)
    f_ref, f_base = _safe_reference(objective, truth, base)
    return ScientificTask(
        "c1", "elliptic_pde_inverse", instance, d, 300,
        -2.0 * np.ones(d), 2.0 * np.ones(d), objective, metrics,
        f_ref, f_base, truth, {"grid_points": n, "observations": len(obs_indices)},
    )


def _lorenz_trajectory(params: Array, steps: int = 60, dt: float = 0.02) -> Array:
    sigma, rho, beta, x, y, z = map(float, params)
    out = np.empty((steps + 1, 3), dtype=float)
    out[0] = [x, y, z]

    def rhs(a: float, b: float, c: float) -> tuple[float, float, float]:
        return sigma * (b - a), a * (rho - c) - b, a * b - beta * c

    for k in range(steps):
        a, b, c = out[k]
        k1 = rhs(a, b, c)
        k2 = rhs(a + 0.5 * dt * k1[0], b + 0.5 * dt * k1[1], c + 0.5 * dt * k1[2])
        k3 = rhs(a + 0.5 * dt * k2[0], b + 0.5 * dt * k2[1], c + 0.5 * dt * k2[2])
        k4 = rhs(a + dt * k3[0], b + dt * k3[1], c + dt * k3[2])
        out[k + 1] = [
            a + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6,
            b + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6,
            c + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6,
        ]
        if not np.all(np.isfinite(out[k + 1])) or np.max(np.abs(out[k + 1])) > 1e4:
            out[k + 1 :] = 1e4
            break
    return out


def make_lorenz63(instance: int) -> ScientificTask:
    rng = np.random.default_rng(120_000 + instance)
    truth = np.array([
        10.0 + 0.15 * (instance - 12),
        28.0 + 0.25 * (instance - 12),
        8.0 / 3.0,
        -7.0 + 0.4 * (instance - 12),
        8.0,
        25.0,
    ])
    trajectory = _lorenz_trajectory(truth)
    idx = np.arange(0, len(trajectory), 3)
    obs = trajectory[idx]
    scale_vec = np.maximum(np.std(obs, axis=0), 1.0)
    noisy = obs + rng.normal(0.0, 0.015 * scale_vec, obs.shape)

    def objective(params: Array) -> float:
        tr = _lorenz_trajectory(params)[idx]
        if not np.all(np.isfinite(tr)):
            return 1e12
        return float(np.mean(((tr - noisy) / scale_vec) ** 2))

    def metrics(params: Array) -> dict[str, float]:
        tr = _lorenz_trajectory(params)[idx]
        return {
            "parameter_relative_error": float(np.linalg.norm((params - truth) / np.array([10, 28, 3, 10, 10, 25]))),
            "trajectory_normalized_rmse": float(np.sqrt(np.mean(((tr - trajectory[idx]) / scale_vec) ** 2))),
        }

    lb = np.array([5.0, 20.0, 1.5, -15.0, -10.0, 10.0])
    ub = np.array([18.0, 40.0, 4.0, 10.0, 20.0, 40.0])
    base = 0.5 * (lb + ub)
    f_ref, f_base = _safe_reference(objective, truth, base)
    return ScientificTask(
        "c1", "lorenz63_calibration", instance, 6, 300,
        lb, ub, objective, metrics, f_ref, f_base, truth,
        {"steps": 60, "dt": 0.02, "observations": int(obs.size)},
    )


def _make_phase(instance: int, noisy: bool) -> ScientificTask:
    dimensions = {11: 16, 12: 24, 13: 32, 1: 16, 2: 24}
    d = dimensions[instance]
    rng = np.random.default_rng((140_000 if noisy else 130_000) + instance)
    truth = rng.normal(0.0, 1.0, d)
    truth /= max(np.linalg.norm(truth), 1e-12)
    m = 2 * d
    matrix = rng.normal(0.0, 1.0 / math.sqrt(d), (m, d))
    clean = (matrix @ truth) ** 2
    if noisy:
        sigma = 0.03 * max(np.std(clean), 1e-8)
        measurements = np.maximum(0.0, clean + rng.normal(0.0, sigma, m))
    else:
        sigma = 0.0
        measurements = clean
    scale = max(float(np.mean(measurements ** 2)), 1e-12)

    def objective(vector: Array) -> float:
        predicted = (matrix @ np.asarray(vector, dtype=float)) ** 2
        return float(np.mean((predicted - measurements) ** 2) / scale)

    def metrics(vector: Array) -> dict[str, float]:
        v = np.asarray(vector, dtype=float)
        error = min(np.linalg.norm(v - truth), np.linalg.norm(v + truth)) / max(np.linalg.norm(truth), 1e-12)
        predicted = (matrix @ v) ** 2
        return {
            "phase_invariant_relative_error": float(error),
            "measurement_relative_error": float(np.linalg.norm(predicted - clean) / max(np.linalg.norm(clean), 1e-12)),
        }

    base = np.zeros(d)
    f_ref, f_base = _safe_reference(objective, truth, base)
    family = "noisy_phase_retrieval" if noisy else "phase_retrieval"

    def specialized_reference() -> dict[str, Any]:
        covariance = matrix.T @ (measurements[:, None] * matrix) / m
        _, eigenvectors = np.linalg.eigh(covariance)
        norm_estimate = math.sqrt(
            max(float(np.mean(measurements)) * d, 1e-12)
        )
        vector = np.clip(
            norm_estimate * eigenvectors[:, -1],
            -2.0,
            2.0,
        )
        current = objective(vector)
        best = vector.copy()
        best_value = current
        evaluations = 1
        step = 0.25
        iterations = 0
        for iterations in range(1, 501):
            projected = matrix @ vector
            gradient = (4.0 / (m * scale)) * matrix.T @ (
                (projected * projected - measurements) * projected
            )
            norm_sq = float(np.dot(gradient, gradient))
            if not np.isfinite(norm_sq) or norm_sq <= 1e-24:
                break
            local_step = step
            accepted = False
            candidate = vector
            candidate_value = current
            for _ in range(24):
                candidate = np.clip(
                    vector - local_step * gradient,
                    -2.0,
                    2.0,
                )
                candidate_value = objective(candidate)
                evaluations += 1
                if (
                    candidate_value
                    <= current - 1e-4 * local_step * norm_sq
                    or local_step <= 1e-12
                ):
                    accepted = True
                    break
                local_step *= 0.5
            if not accepted:
                break
            if candidate_value < best_value:
                best_value = candidate_value
                best = candidate.copy()
            change = float(np.linalg.norm(candidate - vector))
            vector = candidate
            current = candidate_value
            step = min(1.0, local_step * 1.25)
            if change <= 1e-10 * max(
                1.0,
                float(np.linalg.norm(vector)),
            ):
                break
        return {
            "method": "spectral_wirtinger_flow",
            "xbest": best,
            "fbest": best_value,
            "iterations": iterations,
            "objective_evaluations": evaluations,
            "metadata": {
                "spectral_initialization": True,
                "backtracking": True,
            },
        }

    return ScientificTask(
        "c1", family, instance, d, 200,
        -2.0 * np.ones(d), 2.0 * np.ones(d), objective, metrics,
        f_ref, f_base, truth, {"measurements": m, "noise_sigma": sigma},
        specialized_reference,
    )


def make_phase_retrieval(instance: int) -> ScientificTask:
    return _make_phase(instance, False)


def make_noisy_phase_retrieval(instance: int) -> ScientificTask:
    return _make_phase(instance, True)


def _factor_config(instance: int, large: bool) -> tuple[int, int, int]:
    if large:
        return {11: (15, 15, 4), 12: (20, 20, 5), 13: (32, 32, 5), 1: (15, 15, 4), 2: (20, 20, 5)}[instance]
    return {11: (6, 6, 3), 12: (8, 7, 4), 13: (12, 12, 4), 1: (6, 6, 3), 2: (8, 7, 4)}[instance]


def _make_matrix_factorization(instance: int, large: bool) -> ScientificTask:
    m, n, rank = _factor_config(instance, large)
    d = (m + n) * rank
    rng = np.random.default_rng((160_000 if large else 150_000) + instance)
    u_true = rng.normal(0.0, 0.7, (m, rank))
    v_true = rng.normal(0.0, 0.7, (n, rank))
    truth = np.concatenate([u_true.ravel(), v_true.ravel()])
    full = u_true @ v_true.T
    sample_count = min(m * n, max(64, 2 * d if not large else d))
    flat_indices = rng.choice(m * n, sample_count, replace=False)
    rows, cols = np.unravel_index(flat_indices, (m, n))
    observations = full[rows, cols]
    scale = max(float(np.mean(observations ** 2)), 1e-12)

    def unpack(vector: Array) -> tuple[Array, Array]:
        vector = np.asarray(vector, dtype=float)
        split = m * rank
        return vector[:split].reshape(m, rank), vector[split:].reshape(n, rank)

    def objective(vector: Array) -> float:
        u, v = unpack(vector)
        predicted = np.sum(u[rows] * v[cols], axis=1)
        balance = (np.sum(u * u) - np.sum(v * v)) ** 2 / max(d * d, 1)
        return float(np.mean((predicted - observations) ** 2) / scale + 1e-7 * balance)

    def metrics(vector: Array) -> dict[str, float]:
        u, v = unpack(vector)
        reconstruction = u @ v.T
        return {
            "matrix_relative_error": float(np.linalg.norm(reconstruction - full) / max(np.linalg.norm(full), 1e-12)),
            "sample_rmse": float(np.sqrt(np.mean((reconstruction[rows, cols] - observations) ** 2))),
        }

    base = np.zeros(d)
    f_ref, f_base = _safe_reference(objective, truth, base)
    family = "large_matrix_factorization" if large else "matrix_factorization"
    budget_multiplier = 75 if large else 150

    def specialized_reference() -> dict[str, Any]:
        reference_rng = np.random.default_rng(
            260_000 + instance + (10_000 if large else 0)
        )
        best_vector = base.copy()
        best_value = objective(best_vector)
        evaluations = 1
        total_iterations = 0
        ridge = 1e-8
        row_observations = [
            np.flatnonzero(rows == row) for row in range(m)
        ]
        col_observations = [
            np.flatnonzero(cols == col) for col in range(n)
        ]
        for _restart in range(3):
            u = reference_rng.normal(0.0, 0.25, (m, rank))
            v = reference_rng.normal(0.0, 0.25, (n, rank))
            for _ in range(80):
                total_iterations += 1
                for row_index, locations in enumerate(row_observations):
                    if len(locations):
                        design = v[cols[locations]]
                        u[row_index] = np.linalg.solve(
                            design.T @ design + ridge * np.eye(rank),
                            design.T @ observations[locations],
                        )
                for col_index, locations in enumerate(col_observations):
                    if len(locations):
                        design = u[rows[locations]]
                        v[col_index] = np.linalg.solve(
                            design.T @ design + ridge * np.eye(rank),
                            design.T @ observations[locations],
                        )
                u_norm = max(float(np.linalg.norm(u)), 1e-12)
                v_norm = max(float(np.linalg.norm(v)), 1e-12)
                balance_scale = math.sqrt(v_norm / u_norm)
                u *= balance_scale
                v /= balance_scale
                vector = np.clip(
                    np.concatenate([u.ravel(), v.ravel()]),
                    -3.0,
                    3.0,
                )
                value = objective(vector)
                evaluations += 1
                if value < best_value:
                    best_value = value
                    best_vector = vector.copy()
            if best_value <= 1e-14:
                break
        return {
            "method": "alternating_ridge_least_squares",
            "xbest": best_vector,
            "fbest": best_value,
            "iterations": total_iterations,
            "objective_evaluations": evaluations,
            "metadata": {"restarts": 3, "ridge": ridge},
        }

    return ScientificTask(
        "c1", family, instance, d, budget_multiplier,
        -3.0 * np.ones(d), 3.0 * np.ones(d), objective, metrics,
        f_ref, f_base, truth,
        {"matrix_shape": [m, n], "rank": rank, "sample_count": sample_count},
        specialized_reference,
    )


def make_matrix_factorization(instance: int) -> ScientificTask:
    return _make_matrix_factorization(instance, False)


def make_large_matrix_factorization(instance: int) -> ScientificTask:
    return _make_matrix_factorization(instance, True)


def _burgers_final(control: Array, n_space: int = 48, n_steps: int = 32) -> Array:
    control = np.asarray(control, dtype=float)
    x = np.linspace(0.0, 1.0, n_space + 2)
    dx = 1.0 / (n_space + 1)
    dt = 0.18 / n_steps
    viscosity = 0.02
    u = 0.45 * np.sin(math.pi * x)
    shape = np.sin(math.pi * x)
    for step in range(n_steps):
        position = step * (len(control) - 1) / max(n_steps - 1, 1)
        lo = int(math.floor(position))
        hi = min(lo + 1, len(control) - 1)
        weight = position - lo
        c = (1.0 - weight) * control[lo] + weight * control[hi]
        interior = u[1:-1]
        left = u[:-2]
        right = u[2:]
        convection = interior * (interior - left) / dx
        diffusion = viscosity * (right - 2.0 * interior + left) / (dx * dx)
        next_u = interior + dt * (-convection + diffusion + c * shape[1:-1])
        u[1:-1] = np.clip(next_u, -5.0, 5.0)
        u[0] = 0.0
        u[-1] = 0.0
    return u


def make_burgers_control(instance: int) -> ScientificTask:
    dimensions = {11: 8, 12: 12, 13: 16, 1: 8, 2: 12}
    d = dimensions[instance]
    rng = np.random.default_rng(170_000 + instance)
    grid = np.linspace(0.0, 1.0, d)
    truth = 0.65 * np.sin(2 * math.pi * grid) + 0.25 * np.cos(3 * math.pi * grid)
    truth += rng.normal(0.0, 0.03, d)
    target = _burgers_final(truth)
    state_scale = max(float(np.mean(target ** 2)), 1e-10)

    def objective(control: Array) -> float:
        final = _burgers_final(control)
        return float(np.mean((final - target) ** 2) / state_scale + 5e-5 * np.mean(np.asarray(control) ** 2))

    def metrics(control: Array) -> dict[str, float]:
        final = _burgers_final(control)
        return {
            "final_state_relative_error": float(np.linalg.norm(final - target) / max(np.linalg.norm(target), 1e-12)),
            "control_energy": float(np.mean(np.asarray(control) ** 2)),
        }

    base = np.zeros(d)
    f_ref, f_base = _safe_reference(objective, truth, base)
    return ScientificTask(
        "c1", "burgers_control", instance, d, 200,
        -1.5 * np.ones(d), 1.5 * np.ones(d), objective, metrics,
        f_ref, f_base, truth, {"space_points": 50, "time_steps": 32},
    )


def make_allen_cahn(instance: int) -> ScientificTask:
    dimensions = {11: 32, 12: 64, 13: 96, 1: 32, 2: 64}
    d = dimensions[instance]
    epsilon = 0.06 + 0.005 * (instance % 3)
    x = np.linspace(-1.0, 1.0, d + 2)
    shift = 0.06 * (instance - 12) if instance >= 11 else 0.0
    truth_full = np.tanh((x - shift) / (math.sqrt(2.0) * epsilon))
    truth_full[0], truth_full[-1] = -1.0, 1.0
    truth = truth_full[1:-1]
    dx = 2.0 / (d + 1)

    def energy(interior: Array) -> float:
        full = np.concatenate(([-1.0], np.asarray(interior, dtype=float), [1.0]))
        grad = np.diff(full) / dx
        potential = (full[1:-1] ** 2 - 1.0) ** 2
        return float(0.5 * epsilon * np.sum(grad ** 2) * dx + np.sum(potential) * dx / (4.0 * epsilon))

    def metrics(interior: Array) -> dict[str, float]:
        interior = np.asarray(interior, dtype=float)
        crossing = int(np.argmin(np.abs(interior)))
        truth_crossing = int(np.argmin(np.abs(truth)))
        return {
            "field_relative_error": float(np.linalg.norm(interior - truth) / max(np.linalg.norm(truth), 1e-12)),
            "interface_index_error": float(abs(crossing - truth_crossing)),
            "energy": energy(interior),
        }

    base = np.linspace(-1.0, 1.0, d + 2)[1:-1]
    f_ref, f_base = _safe_reference(energy, truth, base)
    return ScientificTask(
        "c1", "allen_cahn_energy", instance, d, 100,
        -1.2 * np.ones(d), 1.2 * np.ones(d), energy, metrics,
        f_ref, f_base, truth, {"epsilon": epsilon},
    )


def make_sparse_inverse(instance: int) -> ScientificTask:
    dimensions = {11: 20, 12: 40, 13: 80, 1: 20, 2: 40}
    d = dimensions[instance]
    rng = np.random.default_rng(190_000 + instance)
    truth = np.zeros(d)
    active = rng.choice(d, min(6, max(3, d // 10)), replace=False)
    truth[active] = rng.uniform(-1.4, 1.4, len(active))

    def forward(vector: Array) -> Array:
        vector = np.asarray(vector, dtype=float)
        return (
            np.sin(vector + 0.4 * np.roll(vector, 1))
            + 0.15 * (vector - np.roll(vector, 2)) ** 2
            + 0.08 * (np.roll(vector, -1) + vector + np.roll(vector, 1))
        )

    clean = forward(truth)
    noise = rng.normal(0.0, 0.005 * max(np.std(clean), 1e-8), d)
    observed = clean + noise
    scale = max(float(np.mean(observed ** 2)), 1e-12)
    lam = 2e-4

    def objective(vector: Array) -> float:
        vector = np.asarray(vector, dtype=float)
        data = np.mean((forward(vector) - observed) ** 2) / scale
        sparse = np.mean(np.sqrt(vector ** 2 + 1e-8))
        return float(data + lam * sparse)

    def metrics(vector: Array) -> dict[str, float]:
        vector = np.asarray(vector, dtype=float)
        return {
            "parameter_relative_error": float(np.linalg.norm(vector - truth) / max(np.linalg.norm(truth), 1e-12)),
            "forward_relative_error": float(np.linalg.norm(forward(vector) - clean) / max(np.linalg.norm(clean), 1e-12)),
            "estimated_l1": float(np.sum(np.abs(vector))),
        }

    base = np.zeros(d)
    f_ref, f_base = _safe_reference(objective, truth, base)
    return ScientificTask(
        "c1", "sparse_nonlinear_inverse", instance, d, 150,
        -2.0 * np.ones(d), 2.0 * np.ones(d), objective, metrics,
        f_ref, f_base, truth, {"active_coefficients": len(active)},
    )


FACTORIES: dict[str, Callable[[int], ScientificTask]] = {
    "elliptic_pde_inverse": make_elliptic_pde,
    "lorenz63_calibration": make_lorenz63,
    "phase_retrieval": make_phase_retrieval,
    "noisy_phase_retrieval": make_noisy_phase_retrieval,
    "matrix_factorization": make_matrix_factorization,
    "large_matrix_factorization": make_large_matrix_factorization,
    "burgers_control": make_burgers_control,
    "allen_cahn_energy": make_allen_cahn,
    "sparse_nonlinear_inverse": make_sparse_inverse,
}


def make_c1_task(family: str, instance: int) -> ScientificTask:
    if family not in FACTORIES:
        raise KeyError(f"Unknown C1 family: {family}")
    return FACTORIES[family](int(instance))
