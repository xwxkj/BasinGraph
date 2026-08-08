"""Budget-exact modern baseline implementations for B21 Track B.

The module separates provenance from execution:

* CMA-ES and BIPOP-CMA-ES use the public ``cma`` package interface.
* DIRECT-L uses ``scipy.optimize.direct`` with ``locally_biased=True``.
* L-SHADE 1.0.1, jSO and L-SRTDE are transparent Python ports whose frozen
  parameters and source basis are recorded in the Track B protocol.
* Multi-start L-BFGS-B uses SciPy with repeated starts until the prescribed
  objective-evaluation budget is exhausted.

Every objective call passes through :class:`BudgetedObjective`; no optimizer can
silently exceed the registered budget.  The returned history stores the
best-so-far value after every objective call.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from scipy.optimize import Bounds, direct, minimize


class BudgetExhausted(RuntimeError):
    """Raised internally when an optimizer attempts to exceed its FE budget."""


@dataclass
class BudgetedObjective:
    objective: Callable[[np.ndarray], float]
    lb: np.ndarray
    ub: np.ndarray
    max_evals: int

    def __post_init__(self) -> None:
        self.lb = np.asarray(self.lb, dtype=float).reshape(-1)
        self.ub = np.asarray(self.ub, dtype=float).reshape(-1)
        self.max_evals = int(self.max_evals)
        if self.lb.shape != self.ub.shape:
            raise ValueError("Lower and upper bounds have different shapes.")
        if self.max_evals < 1:
            raise ValueError("max_evals must be positive.")
        if np.any(~np.isfinite(self.lb)) or np.any(~np.isfinite(self.ub)):
            raise ValueError("Track B requires finite bounds.")
        if np.any(self.ub <= self.lb):
            raise ValueError("Every upper bound must exceed its lower bound.")
        self.nfe = 0
        self.xbest: np.ndarray | None = None
        self.fbest = float("inf")
        self.history: list[tuple[int, float]] = []

    @property
    def dimension(self) -> int:
        return int(self.lb.size)

    @property
    def remaining(self) -> int:
        return max(0, self.max_evals - self.nfe)

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=float).reshape(-1), self.lb, self.ub)

    def __call__(self, x: np.ndarray) -> float:
        if self.nfe >= self.max_evals:
            raise BudgetExhausted()
        z = self.project(x)
        value = float(self.objective(z))
        self.nfe += 1
        if np.isfinite(value) and value < self.fbest:
            self.fbest = value
            self.xbest = z.copy()
        self.history.append((self.nfe, self.fbest))
        return value


def _result(
    bo: BudgetedObjective,
    *,
    algorithm: str,
    message: str,
    implementation: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if bo.xbest is None:
        bo.xbest = 0.5 * (bo.lb + bo.ub)
    return {
        "algorithm": algorithm,
        "implementation": implementation,
        "xbest": bo.xbest.copy(),
        "fbest": float(bo.fbest),
        "nfe": int(bo.nfe),
        "history": list(bo.history),
        "message": str(message),
        "metadata": dict(metadata or {}),
    }


def _sample_cauchy_positive(
    rng: np.random.Generator,
    location: float,
    scale: float = 0.1,
) -> float:
    for _ in range(100):
        value = location + scale * math.tan(math.pi * (rng.random() - 0.5))
        if value > 0:
            return min(value, 1.0)
    return 0.5


def _weighted_lehmer(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    denominator = float(np.sum(weights * values))
    if abs(denominator) <= 1e-300:
        return 0.0
    return float(np.sum(weights * values * values) / denominator)


def _round_half_up_positive(value: float) -> int:
    """Match C/C++ round for non-negative algorithmic counts."""

    value = float(value)
    if value < 0:
        raise ValueError("value must be non-negative")
    return int(math.floor(value + 0.5))


def _sample_cr_from_memory(
    rng: np.random.Generator,
    location: float,
    scale: float = 0.1,
) -> float:
    """Sample CR, preserving the negative L-SHADE/jSO sentinel."""

    if float(location) < 0.0:
        return 0.0
    return float(np.clip(rng.normal(float(location), scale), 0.0, 1.0))


def _success_history_cr(
    values: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return the weighted Lehmer CR memory or the -1 sentinel."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if float(np.sum(weights * values)) <= 0.0:
        return -1.0
    return _weighted_lehmer(values, weights)


def _jso_blend_memory(old_value: float, new_value: float) -> float:
    """Apply the jSO half-old, half-new memory update."""

    return 0.5 * (float(old_value) + float(new_value))


def _repair_midpoint(
    trial: np.ndarray,
    parent: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    repaired = np.asarray(trial, dtype=float).copy()
    below = repaired < lb
    above = repaired > ub
    repaired[below] = 0.5 * (lb[below] + parent[below])
    repaired[above] = 0.5 * (ub[above] + parent[above])
    return np.clip(repaired, lb, ub)


def _archive_insert(
    archive: list[np.ndarray],
    points: list[np.ndarray],
    capacity: int,
    rng: np.random.Generator,
) -> None:
    """Update the external archive using source-style random replacement.

    New successful parents fill unused slots and then replace uniformly random
    active slots.  When population reduction lowers capacity, the active prefix
    is truncated, matching the C/C++ archive-count update.
    """

    capacity = int(capacity)
    if capacity <= 0:
        archive.clear()
        return
    if len(archive) > capacity:
        del archive[capacity:]
    for point in points:
        candidate = np.asarray(point, dtype=float).copy()
        if len(archive) < capacity:
            archive.append(candidate)
        else:
            archive[int(rng.integers(capacity))] = candidate


def _initial_population(
    bo: BudgetedObjective,
    rng: np.random.Generator,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    size = max(4, min(int(size), bo.remaining))
    population = bo.lb + rng.random((size, bo.dimension)) * (bo.ub - bo.lb)
    fitness = np.full(size, np.inf, dtype=float)
    for index in range(size):
        fitness[index] = bo(population[index])
    return population, fitness


def _jso_initial_population_size(dimension: int) -> int:
    """Return the registered jSO initial population size.

    The published jSO specification uses N_init = 25 ln(D) sqrt(D).
    """

    dimension = int(dimension)
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return max(4, _round_half_up_positive(25.0 * math.log(dimension) * math.sqrt(dimension)))


def optimize_lshade_101(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Corrected L-SHADE 1.0.1 source-aligned Python port.

    Frozen parameters follow the corrected public release: initial population
    ``18D``, minimum population 4, history size 6, archive rate 2.6 and
    p-best rate 0.11.  The port retains the corrected random archive replacement
    rather than the defective 1.0.0 archive update.
    """

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    dimension = bo.dimension
    initial_size = max(4, 18 * dimension)
    minimum_size = 4
    memory_size = 6
    archive_rate = 2.6
    p_rate = 0.11

    try:
        population, fitness = _initial_population(bo, rng, initial_size)
    except BudgetExhausted:
        return _result(
            bo,
            algorithm="L_SHADE_1_0_1",
            message="budget_exhausted_during_initialization",
            implementation="source_aligned_python_port_corrected_1.0.1",
        )

    initial_size = len(population)
    memory_f = np.full(memory_size, 0.5, dtype=float)
    memory_cr = np.full(memory_size, 0.5, dtype=float)
    memory_index = 0
    archive: list[np.ndarray] = []

    try:
        while bo.remaining > 0 and len(population) >= 4:
            population_size = len(population)
            order = np.argsort(fitness)
            successful_f: list[float] = []
            successful_cr: list[float] = []
            improvements: list[float] = []
            replaced_parents: list[np.ndarray] = []
            next_population = population.copy()
            next_fitness = fitness.copy()

            for i in range(population_size):
                if bo.remaining <= 0:
                    break
                memory_slot = int(rng.integers(memory_size))
                f_scale = _sample_cauchy_positive(rng, memory_f[memory_slot])
                cr = _sample_cr_from_memory(rng, memory_cr[memory_slot])

                p_count = max(2, _round_half_up_positive(p_rate * population_size))
                p_count = min(p_count, population_size)
                pbest = int(order[int(rng.integers(p_count))])

                candidates = [index for index in range(population_size) if index != i]
                r1 = int(rng.choice(candidates))
                union_size = population_size + len(archive)
                while True:
                    r2_union = int(rng.integers(union_size))
                    if r2_union < population_size:
                        if r2_union not in {i, r1}:
                            r2_vector = population[r2_union]
                            break
                    else:
                        r2_vector = archive[r2_union - population_size]
                        break

                mutant = (
                    population[i]
                    + f_scale * (population[pbest] - population[i])
                    + f_scale * (population[r1] - r2_vector)
                )
                mutant = _repair_midpoint(mutant, population[i], bo.lb, bo.ub)
                crossover = rng.random(dimension) <= cr
                crossover[int(rng.integers(dimension))] = True
                trial = np.where(crossover, mutant, population[i])
                trial_value = bo(trial)

                if trial_value <= fitness[i]:
                    next_population[i] = trial
                    next_fitness[i] = trial_value
                    if trial_value < fitness[i]:
                        replaced_parents.append(population[i].copy())
                        successful_f.append(f_scale)
                        successful_cr.append(cr)
                        improvements.append(fitness[i] - trial_value)

            population = next_population
            fitness = next_fitness
            archive_capacity = _round_half_up_positive(archive_rate * len(population))
            _archive_insert(archive, replaced_parents, archive_capacity, rng)

            if successful_f:
                weights = np.asarray(improvements, dtype=float)
                if weights.sum() <= 0:
                    weights = np.full(len(weights), 1.0 / len(weights))
                else:
                    weights /= weights.sum()
                memory_f[memory_index] = _weighted_lehmer(
                    np.asarray(successful_f), weights
                )
                successful_cr_array = np.asarray(successful_cr)
                memory_cr[memory_index] = _success_history_cr(
                    successful_cr_array, weights
                )
                memory_index = (memory_index + 1) % memory_size

            target_size = _round_half_up_positive(
                initial_size
                + (minimum_size - initial_size)
                * (bo.nfe / bo.max_evals)
            )
            target_size = max(minimum_size, min(target_size, len(population)))
            if len(population) > target_size:
                keep = np.argsort(fitness)[:target_size]
                population = population[keep]
                fitness = fitness[keep]
                archive_capacity = _round_half_up_positive(archive_rate * target_size)
                _archive_insert(archive, [], archive_capacity, rng)

    except BudgetExhausted:
        pass

    return _result(
        bo,
        algorithm="L_SHADE_1_0_1",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="source_aligned_python_port_corrected_1.0.1",
        metadata={
            "initial_population": 18 * dimension,
            "minimum_population": minimum_size,
            "memory_size": memory_size,
            "archive_rate": archive_rate,
            "p_rate": p_rate,
        },
    )


def optimize_jso(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Source-aligned jSO Python port with frozen CEC-2017 parameters.

    The registered initial population is round(25 ln(D) sqrt(D)),
    with a minimum population of four.
    """

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    dimension = bo.dimension
    registered_initial_size = _jso_initial_population_size(dimension)
    initial_size = registered_initial_size
    minimum_size = 4
    memory_size = 5
    archive_rate = 1.0
    p_max = 0.25
    p_min = 0.125

    try:
        population, fitness = _initial_population(bo, rng, initial_size)
    except BudgetExhausted:
        return _result(
            bo,
            algorithm="jSO",
            message="budget_exhausted_during_initialization",
            implementation="source_aligned_python_port_CEC2017",
        )

    initial_size = len(population)
    memory_f = np.full(memory_size, 0.3, dtype=float)
    memory_cr = np.full(memory_size, 0.8, dtype=float)
    memory_index = 0
    p_rate = p_max
    archive: list[np.ndarray] = []

    try:
        while bo.remaining > 0 and len(population) >= 4:
            progress = bo.nfe / bo.max_evals
            population_size = len(population)
            order = np.argsort(fitness)
            successful_f: list[float] = []
            successful_cr: list[float] = []
            improvements: list[float] = []
            replaced_parents: list[np.ndarray] = []
            next_population = population.copy()
            next_fitness = fitness.copy()

            for i in range(population_size):
                if bo.remaining <= 0:
                    break
                slot = int(rng.integers(memory_size))
                if slot == memory_size - 1:
                    mu_f = 0.9
                    mu_cr = 0.9
                else:
                    mu_f = memory_f[slot]
                    mu_cr = memory_cr[slot]
                f_scale = _sample_cauchy_positive(rng, mu_f)
                if progress < 0.6:
                    f_scale = min(f_scale, 0.7)
                cr = _sample_cr_from_memory(rng, mu_cr)
                if progress < 0.25:
                    cr = max(cr, 0.7)
                elif progress < 0.50:
                    cr = max(cr, 0.6)

                if progress < 0.20:
                    weighted_f = 0.7 * f_scale
                elif progress < 0.40:
                    weighted_f = 0.8 * f_scale
                else:
                    weighted_f = 1.2 * f_scale

                p_count = max(2, _round_half_up_positive(p_rate * population_size))
                p_count = min(p_count, population_size)
                pbest = int(order[int(rng.integers(p_count))])
                while progress < 0.50 and pbest == i:
                    pbest = int(order[int(rng.integers(p_count))])
                candidates = [index for index in range(population_size) if index != i]
                r1 = int(rng.choice(candidates))
                union_size = population_size + len(archive)
                while True:
                    r2_union = int(rng.integers(union_size))
                    if r2_union < population_size:
                        if r2_union not in {i, r1}:
                            r2_vector = population[r2_union]
                            break
                    else:
                        r2_vector = archive[r2_union - population_size]
                        break

                mutant = (
                    population[i]
                    + weighted_f * (population[pbest] - population[i])
                    + f_scale * (population[r1] - r2_vector)
                )
                mutant = _repair_midpoint(mutant, population[i], bo.lb, bo.ub)
                crossover = rng.random(dimension) <= cr
                crossover[int(rng.integers(dimension))] = True
                trial = np.where(crossover, mutant, population[i])
                trial_value = bo(trial)

                if trial_value <= fitness[i]:
                    next_population[i] = trial
                    next_fitness[i] = trial_value
                    if trial_value < fitness[i]:
                        replaced_parents.append(population[i].copy())
                        successful_f.append(f_scale)
                        successful_cr.append(cr)
                        improvements.append(fitness[i] - trial_value)

            population = next_population
            fitness = next_fitness
            _archive_insert(
                archive,
                replaced_parents,
                _round_half_up_positive(archive_rate * len(population)),
                rng,
            )

            if successful_f:
                weights = np.asarray(improvements, dtype=float)
                if weights.sum() <= 0:
                    weights = np.full(len(weights), 1.0 / len(weights))
                else:
                    weights /= weights.sum()
                old_f = memory_f[memory_index]
                old_cr = memory_cr[memory_index]
                new_f = _weighted_lehmer(np.asarray(successful_f), weights)
                new_cr = _success_history_cr(
                    np.asarray(successful_cr), weights
                )
                memory_f[memory_index] = _jso_blend_memory(old_f, new_f)
                memory_cr[memory_index] = _jso_blend_memory(old_cr, new_cr)
                memory_index = (memory_index + 1) % memory_size

            target_size = _round_half_up_positive(
                initial_size
                + (minimum_size - initial_size)
                * (bo.nfe / bo.max_evals)
            )
            target_size = max(minimum_size, min(target_size, len(population)))
            if len(population) > target_size:
                keep = np.argsort(fitness)[:target_size]
                population = population[keep]
                fitness = fitness[keep]
                _archive_insert(
                    archive,
                    [],
                    _round_half_up_positive(archive_rate * target_size),
                    rng,
                )
                p_rate = p_max * (1.0 - 0.5 * bo.nfe / bo.max_evals)

    except BudgetExhausted:
        pass

    return _result(
        bo,
        algorithm="jSO",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="source_aligned_python_port_CEC2017",
        metadata={
            "initial_population": registered_initial_size,
            "initial_population_formula": "round(25*ln(D)*sqrt(D))",
            "minimum_population": minimum_size,
            "memory_size": memory_size,
            "archive_rate": archive_rate,
            "p_range": [p_min, p_max],
            "memory_update": "half_old_plus_half_new_weighted_lehmer",
            "fixed_sampling_slot": 0.9,
            "early_pbest_excludes_target_until_fraction": 0.50,
        },
    )


def optimize_lsrtde(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Transparent Python port of the public L-SRTDE CEC-2024 core.

    Mutation, success-rate adaptation, crossover-memory update and linear
    population reduction follow the public C++ core.  Registered deviations are
    the NumPy random generator, arbitrary finite bounds/external objective,
    pre-replacement success-delta measurement, stable elite reduction and
    front-index normalization after population reduction.  This is not the GPL
    C++ executable.
    """

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    dimension = bo.dimension
    initial_size = max(4, 20 * dimension)
    minimum_size = 4
    memory_size = 5

    try:
        population, fitness = _initial_population(bo, rng, initial_size)
    except BudgetExhausted:
        return _result(
            bo,
            algorithm="L_SRTDE",
            message="budget_exhausted_during_initialization",
            implementation="transparent_python_port_public_CEC2024_core",
        )

    initial_size = len(population)
    front = population[np.argsort(fitness)].copy()
    front_fitness = np.sort(fitness).copy()
    memory_cr = np.ones(memory_size, dtype=float)
    memory_index = 0
    success_rate = 0.5
    front_replace_index = 0

    try:
        while bo.remaining > 0 and len(front) >= 4:
            population_size = len(front)
            current_order = np.argsort(fitness)
            front_order = np.argsort(front_fitness)
            mean_f = 0.4 + math.tanh(success_rate * 5.0) * 0.25
            sigma_f = 0.02
            p_count = max(
                2,
                int(population_size * 0.7 * math.exp(-success_rate * 7.0)),
            )
            p_count = min(p_count, population_size)
            rank_weights = np.exp(
                -np.arange(population_size, dtype=float)
                / population_size
                * 3.0
            )
            rank_weights /= rank_weights.sum()

            successes: list[np.ndarray] = []
            success_fitness: list[float] = []
            success_cr: list[float] = []
            success_delta: list[float] = []

            for _ in range(population_size):
                if bo.remaining <= 0:
                    break
                target = int(rng.integers(population_size))
                slot = int(rng.integers(memory_size))
                pbest = int(current_order[int(rng.integers(p_count))])
                r1_rank = int(rng.choice(population_size, p=rank_weights))
                r1 = int(front_order[r1_rank])
                while r1 == pbest:
                    r1_rank = int(rng.choice(population_size, p=rank_weights))
                    r1 = int(front_order[r1_rank])
                r2 = int(rng.integers(population_size))
                while r2 in {pbest, r1}:
                    r2 = int(rng.integers(population_size))

                f_scale = float(rng.normal(mean_f, sigma_f))
                while f_scale <= 0.0 or f_scale > 1.0:
                    f_scale = float(rng.normal(mean_f, sigma_f))
                cr = float(np.clip(rng.normal(memory_cr[slot], 0.05), 0.0, 1.0))
                force = int(rng.integers(dimension))
                mask = rng.random(dimension) < cr
                mask[force] = True
                trial = front[target].copy()
                donor = (
                    front[target]
                    + f_scale * (population[pbest] - front[target])
                    + f_scale * (front[r1] - population[r2])
                )
                trial[mask] = donor[mask]
                out = (trial < bo.lb) | (trial > bo.ub)
                if np.any(out):
                    trial[out] = bo.lb[out] + rng.random(np.sum(out)) * (
                        bo.ub[out] - bo.lb[out]
                    )
                trial_value = bo(trial)

                if trial_value <= front_fitness[target]:
                    delta = abs(front_fitness[target] - trial_value)
                    front[front_replace_index] = trial
                    front_fitness[front_replace_index] = trial_value
                    front_replace_index = (front_replace_index + 1) % population_size
                    successes.append(trial.copy())
                    success_fitness.append(trial_value)
                    success_cr.append(float(np.mean(mask)))
                    success_delta.append(delta)

            success_count = len(successes)
            success_rate = success_count / max(1, population_size)
            if success_count:
                weights = np.asarray(success_delta, dtype=float)
                if weights.sum() <= 0:
                    weights = np.full(success_count, 1.0 / success_count)
                else:
                    weights /= weights.sum()
                memory_cr[memory_index] = 0.5 * (
                    _weighted_lehmer(np.asarray(success_cr), weights)
                    + memory_cr[memory_index]
                )
                memory_index = (memory_index + 1) % memory_size

            if successes:
                population = np.vstack([population, np.asarray(successes)])
                fitness = np.concatenate([fitness, np.asarray(success_fitness)])

            target_size = int(
                (minimum_size - initial_size) / bo.max_evals * bo.nfe
                + initial_size
            )
            target_size = max(minimum_size, min(target_size, len(front)))
            if len(front) > target_size:
                keep_front = np.argsort(front_fitness)[:target_size]
                front = front[keep_front]
                front_fitness = front_fitness[keep_front]
                front_replace_index %= target_size

            keep_population = np.argsort(fitness)[:target_size]
            population = population[keep_population]
            fitness = fitness[keep_population]

    except BudgetExhausted:
        pass

    return _result(
        bo,
        algorithm="L_SRTDE",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="transparent_python_port_public_CEC2024_core",
        metadata={
            "official_source_commit": "fa7291054a83ce5f46132c4045c6a7878e9611e9",
            "initial_population": 20 * dimension,
            "minimum_population": minimum_size,
            "memory_size": memory_size,
            "registered_deviations": [
                "NumPy Generator rather than std::mt19937/distributions",
                "arbitrary finite per-coordinate bounds and external objective",
                "success delta measured before cyclic-front replacement",
                "stable elite reduction for front and population",
                "front replacement index normalized after reduction",
            ],
        },
    )


def optimize_cmaes(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """pycma CMA-ES with fixed population and repeated native restarts."""

    import cma

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    dimension = bo.dimension
    restart = 0

    try:
        while bo.remaining > 0:
            x0 = (
                0.5 * (bo.lb + bo.ub)
                if restart == 0
                else bo.lb + rng.random(dimension) * (bo.ub - bo.lb)
            )
            sigma0 = 0.25 * float(np.mean(bo.ub - bo.lb))
            options = {
                "bounds": [bo.lb.tolist(), bo.ub.tolist()],
                "seed": int(seed + 1009 * restart),
                "verbose": -9,
                "verb_log": 0,
                "verb_disp": 0,
                "tolfun": 0,
                "tolx": 0,
                "tolstagnation": 0,
                "maxfevals": int(bo.remaining),
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, options)
            while bo.remaining > 0 and not es.stop():
                points = es.ask()
                count = min(len(points), bo.remaining)
                values = [bo(point) for point in points[:count]]
                if count == len(points):
                    es.tell(points, values)
                else:
                    # The final partial generation is evaluated for strict
                    # fixed-budget evidence but is not fed back to pycma.
                    break
            restart += 1
    except BudgetExhausted:
        pass
    except Exception as exc:  # pragma: no cover - retained in result metadata
        return _result(
            bo,
            algorithm="CMA_ES",
            message=f"exception:{type(exc).__name__}",
            implementation="pycma_CMAEvolutionStrategy",
            metadata={"restarts_completed": restart},
        )

    return _result(
        bo,
        algorithm="CMA_ES",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="pycma_CMAEvolutionStrategy",
        metadata={"restarts_completed": restart},
    )


def optimize_bipop_cmaes(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """pycma BIPOP-CMA-ES through the public ``fmin2(..., bipop=True)`` API."""

    import cma

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    dimension = bo.dimension
    call_index = 0

    def x0_factory() -> np.ndarray:
        return bo.lb + rng.random(dimension) * (bo.ub - bo.lb)

    try:
        while bo.remaining > 0:
            before = bo.nfe
            sigma0 = 0.25 * float(np.mean(bo.ub - bo.lb))
            cma.fmin2(
                bo,
                x0_factory,
                sigma0,
                {
                    "bounds": [bo.lb.tolist(), bo.ub.tolist()],
                    "seed": int(seed + 2029 * call_index),
                    "maxfevals": int(bo.remaining),
                    "verbose": -9,
                    "verb_log": 0,
                    "verb_disp": 0,
                    "verb_time": 0,
                    "tolfun": 0,
                    "tolx": 0,
                },
                restarts=9,
                bipop=True,
            )
            call_index += 1
            if bo.nfe == before:
                break
    except BudgetExhausted:
        pass
    except Exception as exc:  # pragma: no cover
        return _result(
            bo,
            algorithm="BIPOP_CMA_ES",
            message=f"exception:{type(exc).__name__}",
            implementation="pycma_fmin2_bipop",
            metadata={"fmin2_calls": call_index},
        )

    return _result(
        bo,
        algorithm="BIPOP_CMA_ES",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="pycma_fmin2_bipop",
        metadata={"fmin2_calls": call_index},
    )


def optimize_direct_l(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """SciPy DIRECT-L C implementation with strict external FE accounting."""

    del seed  # DIRECT-L is deterministic for a fixed objective and bounds.
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    try:
        direct(
            bo,
            Bounds(bo.lb, bo.ub),
            maxfun=int(max_evals),
            maxiter=max(1000, int(max_evals)),
            locally_biased=True,
            vol_tol=0.0,
            len_tol=0.0,
        )
    except BudgetExhausted:
        pass
    except Exception as exc:  # pragma: no cover
        if bo.remaining > 0:
            return _result(
                bo,
                algorithm="DIRECT_L",
                message=f"exception:{type(exc).__name__}",
                implementation="scipy.optimize.direct_locally_biased",
            )
    return _result(
        bo,
        algorithm="DIRECT_L",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="scipy.optimize.direct_locally_biased",
    )


def optimize_multistart_lbfgsb(
    objective: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    max_evals: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Repeated bounded L-BFGS-B starts until the full FE budget is used."""

    rng = np.random.default_rng(seed)
    bo = BudgetedObjective(objective, lb, ub, max_evals)
    start_index = 0
    try:
        while bo.remaining > 0:
            if start_index == 0:
                x0 = 0.5 * (bo.lb + bo.ub)
            elif bo.xbest is not None and start_index % 2 == 0:
                scale = 0.10 * (bo.ub - bo.lb)
                x0 = bo.project(bo.xbest + rng.standard_t(2.0, bo.dimension) * scale)
            else:
                x0 = bo.lb + rng.random(bo.dimension) * (bo.ub - bo.lb)
            before = bo.nfe
            try:
                minimize(
                    bo,
                    x0,
                    method="L-BFGS-B",
                    bounds=list(zip(bo.lb, bo.ub)),
                    options={
                        "maxfun": int(bo.remaining),
                        "maxiter": int(bo.remaining),
                        "ftol": 1e-12,
                        "gtol": 1e-8,
                    },
                )
            except BudgetExhausted:
                break
            if bo.nfe == before:
                bo(x0)
            start_index += 1
    except BudgetExhausted:
        pass

    return _result(
        bo,
        algorithm="MS_LBFGSB",
        message="budget_exhausted" if bo.remaining == 0 else "completed",
        implementation="scipy_minimize_repeated_L-BFGS-B",
        metadata={"starts": start_index + 1},
    )


OPTIMIZERS: dict[str, Callable[..., dict[str, Any]]] = {
    "CMA_ES": optimize_cmaes,
    "BIPOP_CMA_ES": optimize_bipop_cmaes,
    "L_SHADE_1_0_1": optimize_lshade_101,
    "jSO": optimize_jso,
    "L_SRTDE": optimize_lsrtde,
    "DIRECT_L": optimize_direct_l,
    "MS_LBFGSB": optimize_multistart_lbfgsb,
}
