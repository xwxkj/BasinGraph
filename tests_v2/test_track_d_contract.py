"""Frozen contract tests for B21 Track D."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from basingraph_v2.optimizer import BasinGraphOptions, minimize_basingraph_v2
from evidence_extension_v1.track_b.modern_baselines import OPTIMIZERS
from evidence_extension_v1.track_d.common import (
    ALGORITHMS,
    EXPECTED_FULL_HASH,
    MODE_CONFIG,
    SUITE_NAME,
    checkpoint_values,
    normalize_ru_maxrss_mb,
    parse_problem_id,
    seed_for,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = ROOT / "protocols" / "evidence_extension_v1" / "track_d"


def sphere(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=float)
    return float(np.sum((z - 0.37) ** 2))


def assert_budgeted_result(result: dict, budget: int) -> None:
    assert int(result["nfe"]) == budget
    history = [(int(nfe), float(value)) for nfe, value in result["history"]]
    assert len(history) == budget
    assert history[-1][0] == budget
    assert all(history[index][0] == index + 1 for index in range(len(history)))
    assert all(
        history[index][1] <= history[index - 1][1]
        for index in range(1, len(history))
    )
    assert np.isfinite(float(result["fbest"]))


def test_track_d_partition_and_matrix_contract() -> None:
    assert SUITE_NAME == "bbob-largescale"
    assert ALGORITHMS == [
        "BasinGraph",
        "CMA_ES",
        "L_SHADE_1_0_1",
        "L_SRTDE",
        "MS_LBFGSB",
    ]

    smoke = MODE_CONFIG["smoke"]
    overhead = MODE_CONFIG["overhead"]
    confirmatory = MODE_CONFIG["confirmatory"]

    assert smoke["expected_rows"] == 30
    assert overhead["expected_rows"] == 60
    assert confirmatory["expected_rows"] == 1440
    assert overhead["primary_overhead"] is True
    assert confirmatory["functions"] == list(range(1, 25))
    assert confirmatory["dimensions"] == [40, 80, 160, 320]
    assert confirmatory["instances"] == [16, 17, 18]
    assert confirmatory["budget_multiplier"] == 200
    assert set(confirmatory["instances"]).isdisjoint({1, 2, 3})


def test_track_d_frozen_basingraph_identity() -> None:
    assert BasinGraphOptions().stable_hash() == EXPECTED_FULL_HASH


def test_track_d_protocol_files_agree() -> None:
    lock = json.loads(
        (PROTOCOL_ROOT / "TRACK_D_CONFIRMATORY_LOCK.json").read_text()
    )
    partition = json.loads(
        (PROTOCOL_ROOT / "TRACK_D_INSTANCE_PARTITION.json").read_text()
    )
    algorithms = json.loads(
        (PROTOCOL_ROOT / "TRACK_D_ALGORITHM_SPECIFICATIONS.json").read_text()
    )
    identity = json.loads(
        (PROTOCOL_ROOT / "TRACK_D_SOURCE_IDENTITY.json").read_text()
    )

    assert lock["status"] == "TRACK_D_CONFIRMATORY_PROTOCOL_FROZEN"
    assert lock["suite"] == SUITE_NAME
    assert lock["functions"] == list(range(1, 25))
    assert lock["dimensions"] == [40, 80, 160, 320]
    assert lock["instances"] == [16, 17, 18]
    assert lock["expected_rows"] == 1440
    assert lock["algorithms"] == ALGORITHMS
    assert partition["confirmatory"]["instances"] == [16, 17, 18]
    assert partition["registered_overhead"]["instances"] == [1]
    assert algorithms["no_post_freeze_tuning"] is True
    assert [item["id"] for item in algorithms["algorithms"]] == ALGORITHMS
    assert identity["status"] == "TRACK_D_SOURCE_IDENTITY_FROZEN"
    assert identity["suite"] == SUITE_NAME


def test_track_d_problem_id_parser() -> None:
    assert parse_problem_id("bbob-largescale_f24_i18_d320") == (24, 18, 320)
    assert parse_problem_id("bbob_f1_i1_d40") == (1, 1, 40)
    with pytest.raises(ValueError):
        parse_problem_id("not_a_bbob_problem")


def test_track_d_seed_formula() -> None:
    assert seed_for(20260808, 1, 40, 16) == 20400824
    assert seed_for(20260808, 24, 320, 18) == 22980826


def test_track_d_rss_normalization() -> None:
    assert normalize_ru_maxrss_mb(1024.0 * 1024.0, "Darwin") == pytest.approx(1.0)
    assert normalize_ru_maxrss_mb(1024.0, "Linux") == pytest.approx(1.0)


def test_track_d_checkpoint_values() -> None:
    history = [(index, float(100 - index)) for index in range(1, 201)]
    values = checkpoint_values(
        history,
        mode="smoke",
        dimension=4,
        budget=200,
    )
    assert values == {
        "1": 96.0,
        "3": 88.0,
        "10": 60.0,
        "30": -20.0,
        "50": -100.0,
    }


def test_basingraph_uses_exact_budget() -> None:
    budget = 140
    result = minimize_basingraph_v2(
        objective=sphere,
        lb=-5.0 * np.ones(3),
        ub=5.0 * np.ones(3),
        max_evals=budget,
        seed=607,
    )
    assert result.nfe == budget
    assert len(result.history) == budget
    assert sum(result.phase_evaluations.values()) == budget
    assert 1 <= len(result.archive) <= 80


@pytest.mark.parametrize(
    "algorithm",
    ["CMA_ES", "L_SHADE_1_0_1", "L_SRTDE", "MS_LBFGSB"],
)
def test_track_d_baselines_use_exact_budget(algorithm: str) -> None:
    if algorithm == "CMA_ES":
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
