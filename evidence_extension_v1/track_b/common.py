"""Shared frozen constants and identity checks for B21 Track B."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION


ROOT = Path(__file__).resolve().parents[2]
TRACK_A_MERGE_COMMIT = "c955bdba08a6a50316248b6c1dd9ff61f1a4840b"
CANDIDATE_TAG = "route-b-v2.0.0-rc1-selected-final-candidate"
CANDIDATE_COMMIT = "adbc0ecdf1153044188f0508321c47001ad9bdb0"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_FULL_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)

ALGORITHMS = [
    "BasinGraph",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "L_SHADE_1_0_1",
    "jSO",
    "L_SRTDE",
    "DIRECT_L",
    "MS_LBFGSB",
]

DISPLAY_NAMES = {
    "BasinGraph": "BasinGraph",
    "CMA_ES": "CMA-ES",
    "BIPOP_CMA_ES": "BIPOP-CMA-ES",
    "L_SHADE_1_0_1": "L-SHADE 1.0.1",
    "jSO": "jSO",
    "L_SRTDE": "L-SRTDE",
    "DIRECT_L": "DIRECT-L",
    "MS_LBFGSB": "Multi-start L-BFGS-B",
}

OBSERVER_NAMES = {
    name: f"TrackB_{name}" for name in ALGORITHMS
}

FUNCTION_GROUPS = {
    **{index: ("G1", "separable") for index in range(1, 6)},
    **{
        index: ("G2", "low_or_moderate_conditioning")
        for index in range(6, 10)
    },
    **{
        index: ("G3", "highly_conditioned_unimodal")
        for index in range(10, 15)
    },
    **{
        index: ("G4", "multimodal_adequate_global_structure")
        for index in range(15, 20)
    },
    **{
        index: ("G5", "multimodal_weak_global_structure")
        for index in range(20, 25)
    },
}

MODE_CONFIG = {
    "smoke": {
        "partition": "development_smoke",
        "functions": [1, 6, 10, 15, 20],
        "dimensions": [5, 20],
        "instances": [1],
        "budget_multiplier": 100,
        "expected_rows": 80,
        "confirmatory": False,
    },
    "confirmatory": {
        "partition": "modern_baseline_confirmatory",
        "functions": list(range(1, 25)),
        "dimensions": [5, 20],
        "instances": list(range(21, 31)),
        "budget_multiplier": 1000,
        "expected_rows": 3840,
        "confirmatory": True,
    },
}

TRACE_FUNCTIONS = {1, 8, 15, 20, 24}
TRACE_INSTANCE = 21
CHECKPOINTS_PER_DIMENSION = {
    "smoke": [1, 3, 10, 30, 100],
    "confirmatory": [1, 3, 10, 30, 100, 300, 1000],
}

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
    / "track_b"
    / "TRACK_B_SOURCE_IDENTITY.json"
)


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
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_blob(path: str) -> str:
    return git_output("rev-parse", f"HEAD:{path}")


def set_single_thread_environment() -> None:
    for key in THREAD_ENV:
        os.environ[key] = "1"


def parse_problem_id(problem_id: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"bbob_f(\d+)_i(\d+)_d(\d+)", problem_id)
    if not match:
        raise ValueError(f"Unexpected COCO problem id: {problem_id}")
    return tuple(int(value) for value in match.groups())


def seed_for(
    base_seed: int,
    function_index: int,
    dimension: int,
    instance_index: int,
) -> int:
    return int(
        base_seed
        + 100_000 * function_index
        + 1_000 * dimension
        + instance_index
    )


def suite_options(mode: str, dimension: int) -> tuple[str, str]:
    config = MODE_CONFIG[mode]
    instances = config["instances"]
    functions = config["functions"]
    instance_text = (
        str(instances[0])
        if len(instances) == 1
        else f"{instances[0]}-{instances[-1]}"
    )
    function_text = ",".join(str(value) for value in functions)
    return (
        f"instances: {instance_text}",
        f"dimensions: {dimension} function_indices: {function_text}",
    )


def should_store_trace(
    mode: str,
    function_index: int,
    instance_index: int,
) -> bool:
    if mode == "smoke":
        return True
    return function_index in TRACE_FUNCTIONS and instance_index == TRACE_INSTANCE


def checkpoint_values(
    history: list[tuple[int, float]],
    *,
    mode: str,
    dimension: int,
    budget: int,
) -> dict[str, float]:
    ordered = sorted((int(nfe), float(value)) for nfe, value in history)
    output: dict[str, float] = {}
    cursor = 0
    current = float("inf")
    for multiplier in CHECKPOINTS_PER_DIMENSION[mode]:
        target = min(int(multiplier * dimension), int(budget))
        while cursor < len(ordered) and ordered[cursor][0] <= target:
            current = min(current, ordered[cursor][1])
            cursor += 1
        output[str(multiplier)] = float(current)
    return output


def verify_source_identity(*, require_clean: bool = True) -> dict[str, Any]:
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )
    options_hash = BasinGraphOptions().stable_hash()
    if options_hash != EXPECTED_FULL_HASH:
        raise RuntimeError(
            "Frozen BasinGraph options hash mismatch:\n"
            f"expected={EXPECTED_FULL_HASH}\nobserved={options_hash}"
        )
    if not SOURCE_IDENTITY_PATH.is_file():
        raise RuntimeError(f"Missing source identity: {SOURCE_IDENTITY_PATH}")
    identity = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding="utf-8"))
    if identity.get("status") != "TRACK_B_SOURCE_IDENTITY_FROZEN":
        raise RuntimeError("Track B source identity is not frozen.")
    if identity.get("candidate_commit") != CANDIDATE_COMMIT:
        raise RuntimeError("Candidate commit mismatch in Track B identity.")
    if identity.get("full_options_hash") != EXPECTED_FULL_HASH:
        raise RuntimeError("Full options hash mismatch in Track B identity.")

    observed_blobs = {}
    for path, expected in identity["git_blobs"].items():
        observed = git_blob(path)
        observed_blobs[path] = observed
        if observed != expected:
            raise RuntimeError(
                f"Track B source blob mismatch: {path}\n"
                f"expected={expected}\nobserved={observed}"
            )

    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", TRACK_A_MERGE_COMMIT, "HEAD"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("Track B HEAD does not descend from Track A merge.")

    if require_clean:
        changed = git_output("status", "--porcelain", "--untracked-files=no")
        if changed:
            raise RuntimeError("Tracked working-tree changes exist:\n" + changed)

    return {
        "status": "TRACK_B_SOURCE_IDENTITY_OK",
        "head_commit": git_output("rev-parse", "HEAD"),
        "source_identity_sha256": sha256_file(SOURCE_IDENTITY_PATH),
        "candidate_commit": CANDIDATE_COMMIT,
        "implementation_version": IMPLEMENTATION_VERSION,
        "full_options_hash": options_hash,
        "git_blobs": observed_blobs,
    }
