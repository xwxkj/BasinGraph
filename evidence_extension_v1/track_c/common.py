"""Frozen constants and identity helpers for B21 Track C."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_COMMIT = "adbc0ecdf1153044188f0508321c47001ad9bdb0"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
TRACK_D_FREEZE = "39d96cb3543035680dcc96860d527aa15e29c97b"

ALGORITHMS = [
    "BasinGraph",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "L_SHADE_1_0_1",
    "jSO",
    "L_SRTDE",
    "MS_LBFGSB",
]
DISPLAY_NAMES = {
    "BasinGraph": "BasinGraph",
    "CMA_ES": "CMA-ES",
    "BIPOP_CMA_ES": "BIPOP-CMA-ES",
    "L_SHADE_1_0_1": "L-SHADE 1.0.1",
    "jSO": "jSO",
    "L_SRTDE": "L-SRTDE",
    "MS_LBFGSB": "Multi-start L-BFGS-B",
}

C1_FAMILIES = [
    "elliptic_pde_inverse",
    "lorenz63_calibration",
    "phase_retrieval",
    "noisy_phase_retrieval",
    "matrix_factorization",
    "large_matrix_factorization",
    "burgers_control",
    "allen_cahn_energy",
    "sparse_nonlinear_inverse",
]
NIST_DATASETS = [
    "Chwirut1",
    "Roszman1",
    "ENSO",
    "Eckerle4",
    "Bennett5",
    "BoxBOD",
]

C1_INSTANCES = [11, 12, 13]
DEVELOPMENT_INSTANCE = 1
PAIRED_SEEDS = list(range(10))
BASE_SEED = 20260808
TARGET_RATIOS = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
CHECKPOINTS_PER_DIMENSION = [1, 3, 10, 30, 100, 300]

SMOKE_TASKS = [
    ("c1", "elliptic_pde_inverse"),
    ("c1", "phase_retrieval"),
    ("c1", "allen_cahn_energy"),
    ("c2", "BoxBOD"),
    ("c2", "ENSO"),
]

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

SOURCE_IDENTITY_PATH = (
    ROOT
    / "protocols"
    / "evidence_extension_v1"
    / "track_c"
    / "TRACK_C_SOURCE_IDENTITY.json"
)


def set_single_thread_environment() -> None:
    for key in THREAD_ENV:
        os.environ[key] = "1"


def stable_digest(payload: Any) -> str:
    data = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def git_blob(path: str) -> str:
    return git_output("rev-parse", f"HEAD:{path}")


def seed_for_c1(family: str, instance: int, paired_seed: int) -> int:
    family_index = C1_FAMILIES.index(family) + 1
    return int(BASE_SEED + 10_000_000 * family_index + 10_000 * instance + paired_seed)


def seed_for_nist(dataset: str, paired_seed: int) -> int:
    dataset_index = NIST_DATASETS.index(dataset) + 1
    return int(BASE_SEED + 500_000_000 + 10_000_000 * dataset_index + paired_seed)


def normalize_gap(value: float, f_ref: float, f_base: float) -> float:
    denominator = max(float(f_base) - float(f_ref), 1e-15)
    return max(0.0, (float(value) - float(f_ref)) / denominator)


def target_values(f_ref: float, f_base: float) -> list[float]:
    scale = max(float(f_base) - float(f_ref), 1e-15)
    return [float(f_ref + ratio * scale) for ratio in TARGET_RATIOS]


def checkpoint_multipliers(budget_multiplier: int) -> list[int]:
    available = [
        value
        for value in CHECKPOINTS_PER_DIMENSION
        if value <= int(budget_multiplier)
    ]
    return sorted(set([*available, int(budget_multiplier)]))


def verify_source_identity(*, require_clean: bool = True) -> dict[str, Any]:
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )
    observed_options = BasinGraphOptions().stable_hash()
    if observed_options != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Frozen BasinGraph options hash mismatch.")
    if not SOURCE_IDENTITY_PATH.is_file():
        raise RuntimeError(f"Missing source identity: {SOURCE_IDENTITY_PATH}")
    identity = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding="utf-8"))
    if identity.get("status") != "TRACK_C_SOURCE_IDENTITY_FROZEN":
        raise RuntimeError("Track C source identity is not frozen.")
    if identity.get("candidate_commit") != CANDIDATE_COMMIT:
        raise RuntimeError("Candidate commit mismatch in Track C identity.")
    if identity.get("full_options_hash") != EXPECTED_OPTIONS_HASH:
        raise RuntimeError("Options hash mismatch in Track C identity.")
    if int(identity.get("confirmatory_objective_evaluations_before_freeze", -1)) != 0:
        raise RuntimeError("Track C identity reports pre-freeze confirmatory evaluations.")
    for path, expected in identity["git_blobs"].items():
        observed = git_blob(path)
        if observed != expected:
            raise RuntimeError(
                f"Track C source blob mismatch for {path}: {observed} != {expected}"
            )
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TRACK_D_FREEZE, "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("Track C HEAD does not descend from the Track D freeze.")
    if require_clean:
        changed = git_output("status", "--porcelain", "--untracked-files=no")
        if changed:
            raise RuntimeError("Tracked source modifications are present:\n" + changed)
    return identity
