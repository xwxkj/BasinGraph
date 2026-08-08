"""Frozen contract tests for B21 Track B."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basingraph_v2.optimizer import BasinGraphOptions
from evidence_extension_v1.track_b.common import (
    ALGORITHMS,
    CANDIDATE_COMMIT,
    EXPECTED_FULL_HASH,
    MODE_CONFIG,
    seed_for,
)
from evidence_extension_v1.track_b.modern_baselines import (
    OPTIMIZERS,
    _jso_initial_population_size,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "protocols" / "evidence_extension_v1" / "track_b"


def sphere(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=float)
    return float(np.sum((z - 0.37) ** 2))


def assert_budgeted_result(result: dict, budget: int) -> None:
    assert int(result["nfe"]) == budget
    history = [(int(nfe), float(value)) for nfe, value in result["history"]]
    assert len(history) == budget
    assert history[-1][0] == budget
    assert all(history[i][0] == i + 1 for i in range(len(history)))
    assert all(
        history[i][1] <= history[i - 1][1]
        for i in range(1, len(history))
    )
    assert np.isfinite(float(result["fbest"]))


def test_track_b_partition_and_matrix_contract() -> None:
    smoke = MODE_CONFIG["smoke"]
    confirmatory = MODE_CONFIG["confirmatory"]
    assert smoke["expected_rows"] == 80
    assert confirmatory["expected_rows"] == 3840
    assert confirmatory["instances"] == list(range(21, 31))
    assert set(confirmatory["instances"]).isdisjoint({1, 2, 3})
    assert set(confirmatory["instances"]).isdisjoint(range(4, 16))
    assert set(confirmatory["instances"]).isdisjoint(range(16, 21))
    assert len(ALGORITHMS) == 8
    assert ALGORITHMS[0] == "BasinGraph"


def test_track_b_frozen_basingraph_identity() -> None:
    assert CANDIDATE_COMMIT == "adbc0ecdf1153044188f0508321c47001ad9bdb0"
    assert BasinGraphOptions().stable_hash() == EXPECTED_FULL_HASH


def test_track_b_protocol_files_agree() -> None:
    lock = json.loads(
        (PROTOCOL_ROOT / "TRACK_B_CONFIRMATORY_LOCK.json").read_text()
    )
    identity = json.loads(
        (PROTOCOL_ROOT / "TRACK_B_INSTANCE_PARTITION.json").read_text()
    )
    algorithms = json.loads(
        (PROTOCOL_ROOT / "TRACK_B_ALGORITHM_SPECIFICATIONS.json").read_text()
    )
    assert lock["status"] == "TRACK_B_CONFIRMATORY_PROTOCOL_FROZEN"
    assert lock["instances"] == list(range(21, 31))
    assert lock["expected_rows"] == 3840
    assert lock["algorithms"] == ALGORITHMS
    assert identity["track_b_confirmatory"] == lock["instances"]
    assert identity["pairwise_disjoint"] is True
    assert [item["id"] for item in algorithms["algorithms"]] == ALGORITHMS
    assert algorithms["no_post_freeze_tuning"] is True


def test_jso_registered_initial_population_formula() -> None:
    assert _jso_initial_population_size(5) == 90
    assert _jso_initial_population_size(20) == 335



def test_track_b_seed_formula() -> None:
    assert seed_for(20260808, 1, 5, 21) == 20365829
    assert seed_for(20260808, 24, 20, 30) == 22680838


@pytest.mark.parametrize(
    "algorithm",
    [
        "L_SHADE_1_0_1",
        "jSO",
        "L_SRTDE",
        "DIRECT_L",
        "MS_LBFGSB",
    ],
)
def test_non_cma_modern_baselines_use_exact_budget(algorithm: str) -> None:
    budget = 140
    result = OPTIMIZERS[algorithm](
        objective=sphere,
        lb=-5.0 * np.ones(3),
        ub=5.0 * np.ones(3),
        max_evals=budget,
        seed=709,
    )
    assert_budgeted_result(result, budget)


@pytest.mark.parametrize("algorithm", ["CMA_ES", "BIPOP_CMA_ES"])
def test_pycma_baselines_use_exact_budget(algorithm: str) -> None:
    pytest.importorskip("cma")
    budget = 160
    result = OPTIMIZERS[algorithm](
        objective=sphere,
        lb=-5.0 * np.ones(3),
        ub=5.0 * np.ones(3),
        max_evals=budget,
        seed=811,
    )
    assert_budgeted_result(result, budget)
