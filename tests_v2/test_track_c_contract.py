from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION
from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    C1_INSTANCES,
    EXPECTED_OPTIONS_HASH,
    NIST_DATASETS,
    PAIRED_SEEDS,
    TARGET_RATIOS,
    seed_for_c1,
    seed_for_nist,
)
from evidence_extension_v1.track_c.nist import load_nist_data, make_nist_task
from evidence_extension_v1.track_c.tasks import make_c1_task

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_candidate_identity() -> None:
    assert IMPLEMENTATION_VERSION == "2.0.0-rc1"
    assert BasinGraphOptions().stable_hash() == EXPECTED_OPTIONS_HASH


def test_registered_matrix_size() -> None:
    assert len(C1_FAMILIES) == 9
    assert len(NIST_DATASETS) == 6
    assert C1_INSTANCES == [11, 12, 13]
    assert PAIRED_SEEDS == list(range(10))
    assert len(ALGORITHMS) == 7
    assert 9 * 3 * 10 * 7 + 6 * 10 * 7 == 2310
    assert "DIRECT_L" not in ALGORITHMS


def test_seed_mapping_is_paired_and_unique() -> None:
    values = set()
    for family in C1_FAMILIES:
        for instance in C1_INSTANCES:
            for paired_seed in PAIRED_SEEDS:
                value = seed_for_c1(family, instance, paired_seed)
                assert value not in values
                values.add(value)
    for dataset in NIST_DATASETS:
        for paired_seed in PAIRED_SEEDS:
            value = seed_for_nist(dataset, paired_seed)
            assert value not in values
            values.add(value)


def test_target_ratios_frozen() -> None:
    assert TARGET_RATIOS == [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def test_development_scientific_tasks_are_finite() -> None:
    for family in [
        "elliptic_pde_inverse",
        "phase_retrieval",
        "allen_cahn_energy",
        "lorenz63_calibration",
        "matrix_factorization",
        "burgers_control",
        "sparse_nonlinear_inverse",
    ]:
        task = make_c1_task(family, 1)
        assert task.dimension > 0
        assert task.budget > task.dimension
        assert np.isfinite(task.f_ref)
        assert np.isfinite(task.f_base)
        assert task.f_base > task.f_ref
        assert np.all(task.reference_x >= task.lb)
        assert np.all(task.reference_x <= task.ub)


def test_nist_files_and_models() -> None:
    for dataset in NIST_DATASETS:
        y, x = load_nist_data(dataset)
        assert len(y) == len(x) > 0
        task = make_nist_task(dataset)
        assert np.isfinite(task.f_ref)
        assert np.isfinite(task.f_base)
        assert task.f_base > task.f_ref
        assert np.all(task.reference_x >= task.lb)
        assert np.all(task.reference_x <= task.ub)
        assert np.isfinite(task.objective(task.reference_x))


def test_protocol_lock() -> None:
    lock = json.loads(
        (ROOT / "protocols/evidence_extension_v1/track_c/TRACK_C_CONFIRMATORY_LOCK.json").read_text()
    )
    assert lock["status"] == "TRACK_C_CONFIRMATORY_LOCKED"
    assert lock["total_runs"] == 2310
    assert lock["confirmatory_objective_evaluations_before_freeze"] == 0


def test_source_identity() -> None:
    identity = json.loads(
        (ROOT / "protocols/evidence_extension_v1/track_c/TRACK_C_SOURCE_IDENTITY.json").read_text()
    )
    assert identity["status"] == "TRACK_C_SOURCE_IDENTITY_FROZEN"
    assert identity["confirmatory_objective_evaluations_before_freeze"] == 0
    assert identity["full_options_hash"] == EXPECTED_OPTIONS_HASH
    assert len(identity["nist_notice"]["files"]) == 6


def test_elliptic_pde_interface_discretization() -> None:
    import numpy as np
    from evidence_extension_v1.track_c.tasks import make_c1_task

    task = make_c1_task("elliptic_pde_inverse", 11)
    assert task.dimension == 6
    assert np.isfinite(task.objective(task.reference_x))
