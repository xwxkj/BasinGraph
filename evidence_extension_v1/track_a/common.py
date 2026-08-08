"""Shared constants and identity checks for B21 Track A."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
)

ROOT = Path(__file__).resolve().parents[2]

STEP0_MERGE_COMMIT = "0457d98f3f7041bee491f3ea9998db5e8c656dba"
STEP0_FREEZE_REF = "b21-step0-identity-freeze"
CANDIDATE_TAG = "route-b-v2.0.0-rc1-selected-final-candidate"
CANDIDATE_COMMIT = "adbc0ecdf1153044188f0508321c47001ad9bdb0"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_FULL_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)

VARIANT_BUILDERS = {
    "Full": lambda base: base,
    "NoGraphGuidance": lambda base: replace(
        base,
        enable_graph_guidance=False,
    ),
    "SingleBracket": lambda base: replace(
        base,
        enable_multibracket=False,
    ),
    "NoFarBasin": lambda base: replace(
        base,
        enable_far_basin=False,
    ),
    "NoGeometryController": lambda base: replace(
        base,
        enable_geometry_controller=False,
    ),
    "NoArchiveFallback": lambda base: replace(
        base,
        enable_archive_fallback=False,
    ),
    "NoFinalPolish": lambda base: replace(
        base,
        enable_final_polish=False,
    ),
    "NoCenterLocal": lambda base: replace(
        base,
        center_local_max_dim=-1,
        local_mode_min_score=float("inf"),
    ),
}
VARIANT_ORDER = list(VARIANT_BUILDERS)
VARIANT_OBSERVER_NAMES = {
    "Full": "BasinGraph_Full",
    "NoGraphGuidance": "BasinGraph_NoGraphGuidance",
    "SingleBracket": "BasinGraph_SingleBracket",
    "NoFarBasin": "BasinGraph_NoFarBasin",
    "NoGeometryController": "BasinGraph_NoGeometryController",
    "NoArchiveFallback": "BasinGraph_NoArchiveFallback",
    "NoFinalPolish": "BasinGraph_NoFinalPolish",
    "NoCenterLocal": "BasinGraph_NoCenterLocal",
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
        "partition": "evidence_extension_confirmatory",
        "functions": list(range(1, 25)),
        "dimensions": [5, 20],
        "instances": list(range(16, 21)),
        "budget_multiplier": 1000,
        "expected_rows": 1920,
        "confirmatory": True,
    },
}

TRACE_FUNCTIONS = {1, 8, 15, 20, 24}
TRACE_INSTANCE = 16
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
    / "track_a"
    / "TRACK_A_SOURCE_IDENTITY.json"
)


def options_for_variant(name: str) -> BasinGraphOptions:
    if name not in VARIANT_BUILDERS:
        raise KeyError(f"Unknown Track A variant: {name}")
    return VARIANT_BUILDERS[name](BasinGraphOptions())


def expected_variant_hashes() -> dict[str, str]:
    return {
        name: options_for_variant(name).stable_hash()
        for name in VARIANT_ORDER
    }


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
    return (
        function_index in TRACE_FUNCTIONS
        and instance_index == TRACE_INSTANCE
    )


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
    observed_hashes = expected_variant_hashes()
    if observed_hashes["Full"] != EXPECTED_FULL_HASH:
        raise RuntimeError(
            "Frozen Full options hash mismatch: "
            f"{observed_hashes['Full']}"
        )

    identity = json.loads(SOURCE_IDENTITY_PATH.read_text(encoding="utf-8"))
    if identity["status"] != "TRACK_A_SOURCE_IDENTITY_FROZEN":
        raise RuntimeError("Track A source identity is not frozen.")
    if identity["step0_merge_commit"] != STEP0_MERGE_COMMIT:
        raise RuntimeError("Step 0 merge commit mismatch.")
    if identity["candidate_commit"] != CANDIDATE_COMMIT:
        raise RuntimeError("Candidate commit mismatch.")
    if identity["variant_hashes"] != observed_hashes:
        raise RuntimeError("Track A variant hash mismatch.")

    subprocess.run(
        ["git", "merge-base", "--is-ancestor", STEP0_MERGE_COMMIT, "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    mismatches = {}
    for path, expected_blob in identity["git_blobs"].items():
        observed_blob = git_blob(path)
        if observed_blob != expected_blob:
            mismatches[path] = {
                "expected": expected_blob,
                "observed": observed_blob,
            }
    if mismatches:
        raise RuntimeError(
            "Track A source-identity mismatch:\n"
            + json.dumps(mismatches, indent=2)
        )

    if require_clean:
        protected_paths = [
            "basingraph_v2",
            "evidence_extension_v1/track_a",
            "protocols/evidence_extension_v1/track_a",
        ]
        for staged in (False, True):
            command = ["git", "diff", "--quiet"]
            if staged:
                command.append("--cached")
            command.extend(["--", *protected_paths])
            if subprocess.run(command, cwd=ROOT).returncode != 0:
                raise RuntimeError(
                    "Uncommitted changes exist in protected Track A paths."
                )

    return {
        "status": "TRACK_A_SOURCE_IDENTITY_OK",
        "head_commit": git_output("rev-parse", "HEAD"),
        "implementation_version": IMPLEMENTATION_VERSION,
        "variant_hashes": observed_hashes,
        "source_identity_sha256": sha256_file(SOURCE_IDENTITY_PATH),
    }
