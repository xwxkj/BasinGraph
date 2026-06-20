#!/usr/bin/env python3
"""
Route B Step B6A: freeze the formal v2 development/holdout experiment protocol.

This script performs no optimizer runs. It:

1. verifies the frozen BasinGraph v2.0.0-rc1 implementation and options hash;
2. freezes COCO/BBOB development, holdout, and final-report partitions;
3. freezes a prospective 24-instance CUTEst holdout from the previously
   inventoried valid-but-not-selected pool;
4. freezes the final ablation design on development data only;
5. writes baseline provenance and claim-evidence maps;
6. creates SHA-256 manifests and a machine-readable protocol lock.

The holdout sets must not be inspected before the final v2.0.0 algorithm tag.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
SELECTION_SEED = "BasinGraph-v2-holdout-20260619"

CUTEST_DEVELOPMENT_LIST = (
    PROJECT_ROOT / "protocols" / "cutest_pre_registered_problem_list_v2.csv"
)
CUTEST_VALID_NOT_SELECTED = (
    PROJECT_ROOT / "protocols" / "cutest_valid_not_selected_v2.csv"
)
CUTEST_HOLDOUT_QUOTAS = {
    "small_2_20": 11,
    "medium_21_100": 7,
    "large_101_500": 6,
}

COCO_FUNCTIONS = list(range(1, 25))
COCO_DIMENSIONS_FINAL = [2, 3, 5, 10, 20]
COCO_DIMENSIONS_DEVELOPMENT = [2, 5, 10]
COCO_DEVELOPMENT_INSTANCES = [1, 2, 3]
COCO_HOLDOUT_INSTANCES = list(range(4, 16))
COCO_ALL_INSTANCES = list(range(1, 16))
COCO_BUDGET_MULTIPLIER = 1000

ALGORITHMS = [
    "BasinGraph_v2",
    "CMA_ES",
    "BIPOP_CMA_ES",
    "DE",
    "MS_LBFGSB",
    "LHS",
    "Random",
]

ABLATION_VARIANTS = [
    "Full",
    "NoGraphGuidance",
    "SingleBracket",
    "NoFarBasin",
    "NoGeometryController",
    "NoArchiveFallback",
    "NoFinalPolish",
]

ABLATION_COCO_FUNCTIONS = [1, 3, 5, 6, 8, 10, 12, 15, 17, 20, 21, 24]
ABLATION_COCO_DIMENSIONS = [5, 10]
ABLATION_COCO_INSTANCES = [1, 2, 3]
ABLATION_COCO_SEEDS = [20260619 + i for i in range(5)]
ABLATION_CUTEST_SEEDS = [20260619 + i for i in range(10)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(
        f"{SELECTION_SEED}|{text}".encode("utf-8")
    ).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def normalize_instance_id(row: pd.Series) -> str:
    if "instance_id" in row and pd.notna(row["instance_id"]):
        return str(row["instance_id"])
    problem = str(row.get("problem_name", "unknown"))
    sif_n = row.get("sif_N")
    if pd.notna(sif_n):
        return f"{problem}[N={int(float(sif_n))}]"
    return problem


def select_cutest_holdout(
    pool: pd.DataFrame,
    development: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pool.copy()
    development = development.copy()

    if "instance_id" not in pool.columns:
        pool["instance_id"] = pool.apply(normalize_instance_id, axis=1)
    if "instance_id" not in development.columns:
        development["instance_id"] = development.apply(
            normalize_instance_id,
            axis=1,
        )

    required = {
        "problem_name",
        "instance_id",
        "dimension",
        "dimension_group",
        "source_type",
    }
    missing = required.difference(pool.columns)
    if missing:
        raise RuntimeError(
            f"CUTEst valid-not-selected table lacks columns: {sorted(missing)}"
        )

    development_instances = set(development["instance_id"].astype(str))
    development_bases = set(development["problem_name"].astype(str))

    pool = pool[
        ~pool["instance_id"].astype(str).isin(development_instances)
    ].copy()

    pool["holdout_selection_hash"] = pool["instance_id"].map(stable_hash)
    pool["base_overlaps_development"] = (
        pool["problem_name"].astype(str).isin(development_bases)
    )

    selected_rows: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []

    for group, quota in CUTEST_HOLDOUT_QUOTAS.items():
        candidates = pool[
            pool["dimension_group"].astype(str) == group
        ].copy()

        # Prefer new base problems, then deterministic hash.
        candidates.sort_values(
            [
                "base_overlaps_development",
                "holdout_selection_hash",
                "instance_id",
            ],
            inplace=True,
        )

        used_bases: set[str] = set()
        group_selected: list[pd.Series] = []

        # First pass: unique base names.
        for _, row in candidates.iterrows():
            base = str(row["problem_name"])
            if base in used_bases:
                continue
            group_selected.append(row)
            used_bases.add(base)
            if len(group_selected) >= quota:
                break

        # Second pass: additional scales only if needed.
        if len(group_selected) < quota:
            chosen_ids = {str(row["instance_id"]) for row in group_selected}
            for _, row in candidates.iterrows():
                if str(row["instance_id"]) in chosen_ids:
                    continue
                group_selected.append(row)
                if len(group_selected) >= quota:
                    break

        if len(group_selected) < quota:
            raise RuntimeError(
                f"Insufficient CUTEst holdout instances in {group}: "
                f"needed {quota}, found {len(group_selected)}"
            )

        for order, row in enumerate(group_selected, start=1):
            selected_rows.append(row)
            audit_rows.append(
                {
                    "dimension_group": group,
                    "selection_order_within_group": order,
                    "quota": quota,
                    "instance_id": str(row["instance_id"]),
                    "problem_name": str(row["problem_name"]),
                    "dimension": int(row["dimension"]),
                    "source_type": str(row["source_type"]),
                    "base_overlaps_development": bool(
                        row["base_overlaps_development"]
                    ),
                    "holdout_selection_hash": str(
                        row["holdout_selection_hash"]
                    ),
                }
            )

    selected = pd.DataFrame(selected_rows).copy()
    group_order = {
        "small_2_20": 1,
        "medium_21_100": 2,
        "large_101_500": 3,
    }
    selected["_group_order"] = selected["dimension_group"].map(group_order)
    selected.sort_values(
        ["_group_order", "holdout_selection_hash", "instance_id"],
        inplace=True,
    )
    selected.drop(columns=["_group_order"], inplace=True)
    selected.insert(
        0,
        "global_holdout_order",
        np.arange(1, len(selected) + 1),
    )

    audit = pd.DataFrame(audit_rows)
    return selected, audit


def select_cutest_ablation_development(
    development: pd.DataFrame,
) -> pd.DataFrame:
    chosen = []
    for group in ["small_2_20", "medium_21_100", "large_101_500"]:
        sub = development[
            development["dimension_group"].astype(str) == group
        ].copy()
        sort_columns = [
            col
            for col in [
                "global_protocol_order",
                "selection_hash",
                "instance_id",
            ]
            if col in sub.columns
        ]
        if sort_columns:
            sub.sort_values(sort_columns, inplace=True)
        chosen.append(sub.head(5))
    result = pd.concat(chosen, ignore_index=True)
    result.insert(
        0,
        "ablation_problem_order",
        np.arange(1, len(result) + 1),
    )
    return result


def write_coco_partition_csv(path: Path) -> None:
    rows = []
    partitions = [
        (
            "development",
            COCO_DIMENSIONS_DEVELOPMENT,
            COCO_DEVELOPMENT_INSTANCES,
            "May be used for engineering and parameter development.",
        ),
        (
            "prospective_holdout",
            COCO_DIMENSIONS_FINAL,
            COCO_HOLDOUT_INSTANCES,
            "Must remain uninspected until final v2.0.0 code tag.",
        ),
        (
            "final_full_report",
            COCO_DIMENSIONS_FINAL,
            COCO_ALL_INSTANCES,
            "Run only after final code freeze; report development and holdout separately.",
        ),
    ]

    for partition, dimensions, instances, rule in partitions:
        for dimension in dimensions:
            rows.append(
                {
                    "partition": partition,
                    "suite": "bbob",
                    "function_indices": "1-24",
                    "dimension": dimension,
                    "instance_indices": ",".join(map(str, instances)),
                    "budget_multiplier": COCO_BUDGET_MULTIPLIER,
                    "algorithms": ";".join(ALGORITHMS),
                    "inspection_rule": rule,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    import sys

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from basingraph_v2.optimizer import (
        BasinGraphOptions,
        IMPLEMENTATION_VERSION,
    )

    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(
            f"Expected {EXPECTED_IMPLEMENTATION}, got {IMPLEMENTATION_VERSION}"
        )

    options = BasinGraphOptions()
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(
            "Frozen options hash mismatch:\n"
            f"expected={EXPECTED_OPTIONS_HASH}\n"
            f"observed={options.stable_hash()}"
        )

    branch = git_output("branch", "--show-current")
    if branch != "route-b/full-basingraph-v2":
        raise RuntimeError(f"Wrong branch: {branch}")

    commit = git_output("rev-parse", "HEAD")
    ablation_tag_commit = git_output(
        "rev-list",
        "-n",
        "1",
        "route-b-v2.0.0-rc1-ablationfreeze",
    )

    output = PROJECT_ROOT / "protocols" / "route_b" / "formal_v2_protocol"
    output.mkdir(parents=True, exist_ok=True)

    development = pd.read_csv(CUTEST_DEVELOPMENT_LIST)
    valid_not_selected = pd.read_csv(CUTEST_VALID_NOT_SELECTED)

    holdout, holdout_audit = select_cutest_holdout(
        valid_not_selected,
        development,
    )
    ablation_cutest = select_cutest_ablation_development(development)

    holdout_path = output / "CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv"
    holdout_audit_path = output / "CUTEST_V2_HOLDOUT_SELECTION_AUDIT.csv"
    ablation_cutest_path = output / "CUTEST_V2_ABLATION_DEVELOPMENT_15.csv"
    coco_path = output / "COCO_V2_PARTITIONS.csv"

    holdout.to_csv(holdout_path, index=False)
    holdout_audit.to_csv(holdout_audit_path, index=False)
    ablation_cutest.to_csv(ablation_cutest_path, index=False)
    write_coco_partition_csv(coco_path)

    ablation_design = {
        "status": "frozen_before_formal_v2_runs",
        "variants": ABLATION_VARIANTS,
        "coco": {
            "functions": ABLATION_COCO_FUNCTIONS,
            "dimensions": ABLATION_COCO_DIMENSIONS,
            "instances": ABLATION_COCO_INSTANCES,
            "seeds": ABLATION_COCO_SEEDS,
            "budget_multiplier": COCO_BUDGET_MULTIPLIER,
        },
        "cutest": {
            "development_list": str(
                ablation_cutest_path.relative_to(PROJECT_ROOT)
            ),
            "seeds": ABLATION_CUTEST_SEEDS,
            "budget": "min(20000, max(1000, 50*n))",
        },
    }
    ablation_path = output / "V2_FINAL_ABLATION_DESIGN.json"
    ablation_path.write_text(json.dumps(ablation_design, indent=2))

    baseline_rows = [
        {
            "algorithm": "BasinGraph_v2",
            "source": "this work",
            "implementation": "basingraph_v2",
            "version_policy": "final v2.0.0 tag",
            "modification": "not applicable",
        },
        {
            "algorithm": "CMA_ES",
            "source": "pycma",
            "implementation": "cma",
            "version_policy": "frozen environment lock",
            "modification": "objective/budget wrapper only",
        },
        {
            "algorithm": "BIPOP_CMA_ES",
            "source": "pycma fmin2",
            "implementation": "restarts=9, bipop=True",
            "version_policy": "frozen environment lock",
            "modification": "objective/budget wrapper only",
        },
        {
            "algorithm": "DE",
            "source": "self-contained reference implementation",
            "implementation": "repository baseline",
            "version_policy": "final v2.0.0 source snapshot",
            "modification": "none",
        },
        {
            "algorithm": "MS_LBFGSB",
            "source": "SciPy",
            "implementation": "scipy.optimize.minimize L-BFGS-B",
            "version_policy": "frozen environment lock",
            "modification": "multistart/budget wrapper",
        },
        {
            "algorithm": "LHS",
            "source": "self-contained",
            "implementation": "repository baseline",
            "version_policy": "final v2.0.0 source snapshot",
            "modification": "none",
        },
        {
            "algorithm": "Random",
            "source": "self-contained",
            "implementation": "repository baseline",
            "version_policy": "final v2.0.0 source snapshot",
            "modification": "none",
        },
    ]
    baseline_path = output / "V2_BASELINE_PROVENANCE_PLAN.csv"
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)

    claim_rows = [
        {
            "claim": "The implementation explicitly maintains basin nodes and directed transition edges.",
            "required_evidence": "serialized archive/graph outputs and integrity tests",
            "source": "unit tests; final run JSON; graph integrity logs",
        },
        {
            "claim": "Graph guidance affects search decisions.",
            "required_evidence": "NoGraphGuidance ablation under identical paired protocol",
            "source": "final ablation design",
        },
        {
            "claim": "BasinGraph v2 is competitive on standard black-box optimization.",
            "required_evidence": "prospective COCO holdout instances 4-15",
            "source": "official cocopp ECDF/ERT",
        },
        {
            "claim": "BasinGraph v2 generalizes to nonlinear optimization problems.",
            "required_evidence": "prospective CUTEst holdout 24 plus frozen development 50",
            "source": "performance/data profiles and problem-level statistics",
        },
        {
            "claim": "Module-level interpretations are supported.",
            "required_evidence": "frozen paired ablation on development data only",
            "source": "COCO/CUTEst ablation tables",
        },
    ]
    claim_path = output / "V2_CLAIM_EVIDENCE_MAP.csv"
    pd.DataFrame(claim_rows).to_csv(claim_path, index=False)

    holdout_summary = {
        "selection_seed": SELECTION_SEED,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_runs_performed_by_this_script": 0,
        "development_cutest_instances": int(len(development)),
        "prospective_holdout_instances": int(len(holdout)),
        "holdout_groups": (
            holdout["dimension_group"].value_counts().sort_index().to_dict()
        ),
        "holdout_unique_base_problems": int(
            holdout["problem_name"].nunique()
        ),
        "holdout_base_overlaps_development": int(
            holdout["problem_name"].astype(str).isin(
                set(development["problem_name"].astype(str))
            ).sum()
        ),
    }
    summary_path = output / "CUTEST_V2_HOLDOUT_SUMMARY.json"
    summary_path.write_text(json.dumps(holdout_summary, indent=2))

    protocol_text = f"""# BasinGraph v2 formal experiment protocol

## Frozen algorithm

- implementation: `{EXPECTED_IMPLEMENTATION}`
- full options hash: `{EXPECTED_OPTIONS_HASH}`
- branch: `{branch}`
- protocol-freeze commit: `{commit}`
- rc1 ablation-freeze tag commit: `{ablation_tag_commit}`

## Non-negotiable inspection rule

No result from a prospective holdout partition may be inspected before the
final BasinGraph v2.0.0 code tag is created. Engineering changes and parameter
choices may use development partitions only.

## Official COCO/BBOB

### Development partition

- functions 1-24;
- dimensions 2, 5 and 10;
- instances 1-3;
- budget 1000d;
- seven algorithms.

### Prospective holdout

- functions 1-24;
- dimensions 2, 3, 5, 10 and 20;
- instances 4-15;
- budget 1000d;
- seven algorithms.

### Final report

After the final code tag, run instances 1-15 and report development (1-3) and
prospective holdout (4-15) separately before any pooled summary.

## CUTEst

### Development/comparability suite

The existing frozen 50-instance list is retained for development and
comparability with the historical implementation. Because prior results on
these problems have already been observed, it is not treated as a prospective
v2 holdout.

### Prospective holdout

Twenty-four instances are selected from the valid-but-not-selected technical
inventory with the deterministic seed `{SELECTION_SEED}`:

- 11 small;
- 7 medium;
- 6 large.

No optimizer result is used in selection. The holdout list must remain
uninspected until the final v2.0.0 code tag.

### Budget and seeds

- 30 paired seeds;
- budget `min(20000, max(1000, 50*n))`;
- seven algorithms.

## Ablation

Ablation uses development data only. It must not use the prospective COCO or
CUTEst holdouts. The seven frozen variants are:

{chr(10).join(f"- {name}" for name in ABLATION_VARIANTS)}

## Result identity requirements

Every BasinGraph v2 formal result must contain:

- implementation version;
- options hash;
- exact Git commit;
- phase evaluation counts;
- explicit archive nodes;
- graph edges with referential integrity;
- diagnostics;
- event log;
- seed and budget;
- protocol manifest hash.

## Manuscript rule

Only results generated after the final v2.0.0 code tag and under this protocol
may replace the historical v1.0.0 evidence in the manuscript.
"""
    protocol_path = output / "V2_FORMAL_EXPERIMENT_PROTOCOL.md"
    protocol_path.write_text(protocol_text)

    lock = {
        "status": "V2_FORMAL_PROTOCOL_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "branch": branch,
        "protocol_freeze_commit": commit,
        "ablation_freeze_tag_commit": ablation_tag_commit,
        "coco": {
            "development_dimensions": COCO_DIMENSIONS_DEVELOPMENT,
            "final_dimensions": COCO_DIMENSIONS_FINAL,
            "functions": COCO_FUNCTIONS,
            "development_instances": COCO_DEVELOPMENT_INSTANCES,
            "holdout_instances": COCO_HOLDOUT_INSTANCES,
            "budget_multiplier": COCO_BUDGET_MULTIPLIER,
            "algorithms": ALGORITHMS,
        },
        "cutest": {
            "development_instances": len(development),
            "prospective_holdout_instances": len(holdout),
            "holdout_quotas": CUTEST_HOLDOUT_QUOTAS,
            "selection_seed": SELECTION_SEED,
            "seeds": 30,
            "budget": "min(20000, max(1000, 50*n))",
            "algorithms": ALGORITHMS,
        },
        "ablation_design": ablation_design,
    }
    lock_path = output / "V2_FORMAL_PROTOCOL_LOCK.json"
    lock_path.write_text(json.dumps(lock, indent=2))

    manifest_path = output / "V2_FORMAL_PROTOCOL_MANIFEST_SHA256.csv"
    manifest_rows = []
    for path in sorted(output.iterdir()):
        if not path.is_file() or path == manifest_path:
            continue
        manifest_rows.append(
            {
                "filename": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    print("V2_FORMAL_PROTOCOL_FROZEN")
    print("implementation:", EXPECTED_IMPLEMENTATION)
    print("options hash:", EXPECTED_OPTIONS_HASH)
    print("protocol freeze commit:", commit)
    print("COCO development instances:", COCO_DEVELOPMENT_INSTANCES)
    print("COCO holdout instances:", COCO_HOLDOUT_INSTANCES)
    print("CUTEst development instances:", len(development))
    print("CUTEst holdout instances:", len(holdout))
    print(
        "CUTEst holdout groups:",
        holdout_summary["holdout_groups"],
    )
    print(
        "CUTEst holdout base overlaps development:",
        holdout_summary["holdout_base_overlaps_development"],
    )
    print("output:", output)


if __name__ == "__main__":
    main()
