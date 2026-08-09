#!/usr/bin/env python3
"""Materialize frozen Track C data provenance, run matrix and source identity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_c.common import (
    ALGORITHMS,
    C1_FAMILIES,
    C1_INSTANCES,
    NIST_DATASETS,
    PAIRED_SEEDS,
    seed_for_c1,
    seed_for_nist,
    sha256_file,
)
from evidence_extension_v1.track_c.nist import SPECS, load_nist_data, make_nist_task
from evidence_extension_v1.track_c.tasks import make_c1_task

NIST_URL = "https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/{name}.dat"


def git_hash(path: Path) -> str:
    return subprocess.check_output(["git", "hash-object", str(path)], cwd=ROOT, text=True).strip()


def main() -> None:
    protocol_root = ROOT / "protocols/evidence_extension_v1/track_c"
    data_root = ROOT / "data/nist"
    protocol_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    provenance = []
    sha_lines = []
    for name in NIST_DATASETS:
        path = data_root / f"{name}.dat"
        if not path.is_file():
            raise RuntimeError(f"Missing downloaded NIST file: {path}")
        y, x = load_nist_data(name)
        digest = sha256_file(path)
        provenance.append(
            {
                "dataset": name,
                "official_url": NIST_URL.format(name=name),
                "retrieval_mirror": "https://raw.githubusercontent.com/lmfit/lmfit-py/fe389bbbd1fe936cd73742bd81fc6fce7ac92858/NIST_STRD/{name}.dat".format(name=name),
                "mirror_commit": "fe389bbbd1fe936cd73742bd81fc6fce7ac92858",
                "repository_path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "retrieved_utc": retrieved,
                "observations": len(y),
                "parameters": len(SPECS[name].certified),
                "difficulty": SPECS[name].difficulty,
                "observed_data": True,
                "certified_rss": SPECS[name].certified_rss,
            }
        )
        sha_lines.append(f"{digest}  {path.name}")
    with (protocol_root / "TRACK_C_NIST_PROVENANCE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
        writer.writeheader()
        writer.writerows(provenance)
    (data_root / "SHA256SUMS.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    matrix = []
    for family in C1_FAMILIES:
        for instance in C1_INSTANCES:
            task = make_c1_task(family, instance)
            for paired_seed in PAIRED_SEEDS:
                seed = seed_for_c1(family, instance, paired_seed)
                for algorithm in ALGORITHMS:
                    matrix.append(
                        {
                            "domain": "c1",
                            "task_name": family,
                            "task_id": task.task_id,
                            "instance": instance,
                            "dimension": task.dimension,
                            "budget_multiplier": task.budget_multiplier,
                            "budget": task.budget,
                            "paired_seed": paired_seed,
                            "seed": seed,
                            "algorithm": algorithm,
                        }
                    )
    for dataset in NIST_DATASETS:
        task = make_nist_task(dataset)
        for paired_seed in PAIRED_SEEDS:
            seed = seed_for_nist(dataset, paired_seed)
            for algorithm in ALGORITHMS:
                matrix.append(
                    {
                        "domain": "c2",
                        "task_name": dataset,
                        "task_id": task.task_id,
                        "instance": 0,
                        "dimension": task.dimension,
                        "budget_multiplier": task.budget_multiplier,
                        "budget": task.budget,
                        "paired_seed": paired_seed,
                        "seed": seed,
                        "algorithm": algorithm,
                    }
                )
    if len(matrix) != 2310:
        raise RuntimeError(f"Track C matrix size mismatch: {len(matrix)}")
    with (protocol_root / "TRACK_C_EXPECTED_RUN_MATRIX.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
        writer.writeheader()
        writer.writerows(matrix)

    references = []
    for family in C1_FAMILIES:
        for instance in C1_INSTANCES:
            task = make_c1_task(family, instance)
            references.append(
                {
                    "domain": "c1",
                    "task_name": family,
                    "task_id": task.task_id,
                    "dimension": task.dimension,
                    "f_ref": task.f_ref,
                    "f_base": task.f_base,
                    "reference_type": "registered generative/reference construction",
                }
            )
    for dataset in NIST_DATASETS:
        task = make_nist_task(dataset)
        references.append(
            {
                "domain": "c2",
                "task_name": dataset,
                "task_id": task.task_id,
                "dimension": task.dimension,
                "f_ref": task.f_ref,
                "f_base": task.f_base,
                "reference_type": "NIST certified parameter vector and official far start",
            }
        )
    with (protocol_root / "TRACK_C_REFERENCE_VALUES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]))
        writer.writeheader()
        writer.writerows(references)

    paths: list[Path] = []
    for base in [
        ROOT / "basingraph_v2",
        ROOT / "evidence_extension_v1/track_c",
        protocol_root,
        data_root,
    ]:
        for path in sorted(base.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
                and path.name not in {"TRACK_C_SOURCE_IDENTITY.json", "MANIFEST_SHA256.csv"}
            ):
                paths.append(path)
    paths.extend(
        [
            ROOT / "evidence_extension_v1/track_b/modern_baselines.py",
            ROOT / "START_B21_TRACK_C_SMOKE.command",
            ROOT / "START_B21_TRACK_C_CONFIRMATORY.command",
            ROOT / "tests_v2/test_track_c_contract.py",
        ]
    )
    unique = []
    seen = set()
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if relative not in seen:
            unique.append(path)
            seen.add(relative)
    parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    identity = {
        "status": "TRACK_C_SOURCE_IDENTITY_FROZEN",
        "identity_version": "1.1.0",
        "date": retrieved,
        "identity_revision": "portable identity, available-checkpoint semantics, paired primary inference and registered descriptive task-specific references",
        "repository": "xwxkj/BasinGraph",
        "branch": "ncs-evidence-v1-track-c",
        "materialization_parent_commit": parent,
        "track_d_frozen_source": "39d96cb3543035680dcc96860d527aa15e29c97b",
        "candidate_commit": "adbc0ecdf1153044188f0508321c47001ad9bdb0",
        "implementation_version": "2.0.0-rc1",
        "full_options_hash": "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69",
        "confirmatory_partition": {
            "c1_families": C1_FAMILIES,
            "c1_instances": C1_INSTANCES,
            "nist_datasets": NIST_DATASETS,
            "paired_seeds": PAIRED_SEEDS,
            "algorithms": ALGORITHMS,
            "expected_rows": 2310,
        },
        "implementation_notice": {
            "L_SHADE_1_0_1": "Source-aligned transparent Python port; not a byte-identical official executable.",
            "jSO": "Source-aligned transparent Python port; not a byte-identical official executable.",
            "L_SRTDE": "Transparent source-guided Python port with registered deviations; not the GPL C++ executable.",
        },
        "nist_notice": {
            "source": "NIST Statistical Reference Datasets (StRD)",
            "doi": "10.18434/T43G6C",
            "files": {row["dataset"]: row["sha256"] for row in provenance},
        },
        "git_blobs": {path.relative_to(ROOT).as_posix(): git_hash(path) for path in unique},
        "excluded_from_self_identity": [
            "protocols/evidence_extension_v1/track_c/TRACK_C_SOURCE_IDENTITY.json",
            "protocols/evidence_extension_v1/track_c/MANIFEST_SHA256.csv",
        ],
        "algorithm_source_changes_allowed": False,
        "post_freeze_runner_changes_require_new_identity": True,
        "confirmatory_objective_evaluations_before_freeze": 0,
        "reference_construction_tasks": len(references),
        "task_specific_reference_tasks": 18,
        "checkpoint_semantics": "available checkpoints plus explicit registered final budget",
    }
    (protocol_root / "TRACK_C_SOURCE_IDENTITY.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, "evidence_extension_v1/track_c/generate_protocol_manifest.py"], cwd=ROOT, check=True)
    print("TRACK_C_MATERIALIZATION_OK")
    print(json.dumps({"nist_files": 6, "expected_rows": len(matrix), "references": len(references)}, indent=2))


if __name__ == "__main__":
    main()
