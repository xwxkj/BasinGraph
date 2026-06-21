#!/usr/bin/env python3
# Freeze the exact paper-code semantic contract for the selected rc1 code.

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import BasinGraphOptions, IMPLEMENTATION_VERSION

EXPECTED_BRANCH = "route-b/finalize-rc1-consistent"
EXPECTED_IMPLEMENTATION = "2.0.0-rc1"
EXPECTED_OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69"
)
RC1_CODE_FREEZE_TAG = "route-b-v2.0.0-rc1-codefreeze"
RC1_ABLATION_FREEZE_TAG = "route-b-v2.0.0-rc1-ablationfreeze"
RC1_PROTOCOL_FREEZE_TAG = "route-b-v2.0.0-rc1-protocolfreeze"
RC2_REJECTION_TAG = "route-b-v2.0.0-rc2-rejected"
OUT = ROOT / "protocols" / "route_b" / "final_rc1"


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_show(tag: str, relative_path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{tag}:{relative_path}"],
        cwd=ROOT,
        text=True,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    branch = git_output("branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Run on {EXPECTED_BRANCH}, not {branch}")

    options = BasinGraphOptions()
    if IMPLEMENTATION_VERSION != EXPECTED_IMPLEMENTATION:
        raise RuntimeError(IMPLEMENTATION_VERSION)
    if options.stable_hash() != EXPECTED_OPTIONS_HASH:
        raise RuntimeError(
            f"Options hash mismatch: {options.stable_hash()}"
        )

    # No algorithmic source change is permitted.
    if subprocess.run(
        ["git", "diff", "--quiet", RC1_CODE_FREEZE_TAG, "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("basingraph_v2 differs from the rc1 code-freeze tag")

    if subprocess.run(
        ["git", "diff", "--quiet", "--", "basingraph_v2"],
        cwd=ROOT,
    ).returncode != 0:
        raise RuntimeError("Uncommitted algorithm changes exist")

    rc2_rejection = json.loads(
        git_show(
            RC2_REJECTION_TAG,
            "protocols/route_b/final_selection/RC2_REJECTION_DECISION.json",
        )
    )
    rc2_rejection_md = git_show(
        RC2_REJECTION_TAG,
        "protocols/route_b/final_selection/RC2_REJECTION_DECISION.md",
    )
    assert rc2_rejection["holdout_accessed"] is False
    assert rc2_rejection["rc3_authorized"] is False

    formal_lock_path = (
        ROOT
        / "protocols"
        / "route_b"
        / "formal_v2_protocol"
        / "V2_FORMAL_PROTOCOL_LOCK.json"
    )
    rc1_machine_path = (
        ROOT / "protocols" / "route_b" / "V2_RC1_MACHINE_SPEC.json"
    )
    development_root = (
        ROOT / "results_v2" / "formal_development" / "coco_rc1"
    )
    run_id = (development_root / "LAST_RUN_ID.txt").read_text().strip()
    validation_path = development_root / run_id / "validation_report.json"
    diagnostic_path = (
        development_root
        / run_id
        / "development_diagnostics"
        / "development_diagnostic_report.json"
    )

    for path in [
        formal_lock_path,
        rc1_machine_path,
        validation_path,
        diagnostic_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    formal_lock = json.loads(formal_lock_path.read_text())
    validation = json.loads(validation_path.read_text())
    diagnostic = json.loads(diagnostic_path.read_text())

    assert formal_lock["status"] == "V2_FORMAL_PROTOCOL_FROZEN"
    assert validation["holdout_leakage"] is False
    assert diagnostic["holdout_accessed"] is False

    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "HISTORICAL_RC2_REJECTION_DECISION.json").write_text(
        json.dumps(rc2_rejection, indent=2)
    )
    (OUT / "HISTORICAL_RC2_REJECTION_DECISION.md").write_text(
        rc2_rejection_md
    )

    code_freeze_commit = git_output(
        "rev-list", "-n", "1", RC1_CODE_FREEZE_TAG
    )
    selection = {
        "status": "RC1_SELECTED_AS_FINAL_HOLDOUT_CANDIDATE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_implementation": EXPECTED_IMPLEMENTATION,
        "selected_options_hash": EXPECTED_OPTIONS_HASH,
        "selected_code_freeze_tag": RC1_CODE_FREEZE_TAG,
        "selected_code_freeze_commit": code_freeze_commit,
        "ablation_freeze_tag": RC1_ABLATION_FREEZE_TAG,
        "protocol_freeze_tag": RC1_PROTOCOL_FREEZE_TAG,
        "current_branch": branch,
        "branch_parent_commit": git_output("rev-parse", "HEAD"),
        "development_run_id": run_id,
        "development_rows": validation["rows"],
        "development_problems": validation["problems"],
        "development_mean_rank": diagnostic["basingraph_overall_mean_rank"],
        "development_wins": diagnostic["basingraph_overall_wins"],
        "rc2_rejected": True,
        "rc3_authorized": False,
        "holdout_accessed": False,
        "release_policy": (
            "The public release may later be tagged v2.0.0, but all result "
            "records and Methods text must identify implementation 2.0.0-rc1, "
            "the options hash and the code-freeze tag."
        ),
    }
    (OUT / "RC1_FINAL_SELECTION_DECISION.json").write_text(
        json.dumps(selection, indent=2)
    )

    selection_md = f'''# Selection of BasinGraph 2.0.0-rc1 as final holdout candidate

The byte-frozen implementation `{EXPECTED_IMPLEMENTATION}` is selected for
prospective holdout evaluation.

- options hash: `{EXPECTED_OPTIONS_HASH}`;
- code-freeze tag: `{RC1_CODE_FREEZE_TAG}`;
- code-freeze commit: `{code_freeze_commit}`;
- development run: `{run_id}`;
- rc2 was rejected under its predeclared development gate;
- rc3 was not authorized;
- prospective holdouts remain unopened.

Selection is conditional on using the exact terminology and definitions in
`RC1_EXACT_ALGORITHM_SEMANTICS.md`.
'''
    (OUT / "RC1_FINAL_SELECTION_DECISION.md").write_text(selection_md)

    semantics = '''# BasinGraph 2.0.0-rc1 exact algorithm semantics

This is the controlling manuscript-code contract. It describes the selected
code exactly and deliberately avoids stronger, unsupported interpretations.

## Operational basin-state node

A `BasinNode` is an operational search-state representative in the active
archive. It is not guaranteed to be a certified local minimum, a stationary
point, or an exact attraction basin.

Each node stores an identifier, representative point, objective value,
operational radius, curvature proxy, visit count, creation/update evaluation
indices, novelty and source mode.

Finite evaluations from the initial design, center-local search, coordinate
multi-bracket search, far-basin exploration, archive fallback and budget
completion may create or update nodes.

## Node merge and capacity rules

Let `D = ||ub-lb||_2`. A candidate merges with its nearest node when

`distance <= max(candidate_radius, merge_radius_factor * D)`.

Otherwise it creates a new node. If capacity is exceeded, the active node with
the largest objective value is evicted, and all incident graph edges are
removed. This is an operational archive rule, not a proof of exact basin
identity.

## Directed observed-transition graph

A directed edge records an observed algorithmic transition between two
distinct active nodes. It stores cumulative attributed evaluations, best
observed improvement, minimum barrier proxy, maximum accessibility, attempt
count, source mode and creation/update evaluation indices.

For a newly observed edge,

`accessibility = 1 / (1 + barrier_proxy + evaluations)`.

The barrier is a sampled algorithmic proxy, not an exact landscape barrier.

## Graph guidance

Archive fallback and budget completion use

`0.55 * quality + 0.25 * novelty + 0.20 * accessibility`.

Quality is reverse objective rank among active nodes. Accessibility is the
maximum incoming edge accessibility, with a neutral value for nodes without
incoming edges.

## Initial design

The design uses the box centre, lower/upper corners, two alternating corners,
coordinate lower-centre-upper anchors and 24 Latin-hypercube points, subject
to the phase evaluation limit.

## Geometry diagnostics

- mean scale: mean box width;
- maximum scale: maximum box width;
- anisotropy: maximum box width divided by minimum box width;
- boundary signal: normalized median interior-minus-boundary objective;
- ruggedness: normalized median absolute deviation of anchor values;
- sign-change rate: fraction of valid coordinate lower-centre-upper slope
  pairs with opposite signs;
- finite-anchor fraction.

The rc1 quantity named `anisotropy` is domain anisotropy, not objective
curvature anisotropy.

The local-mode score is

`1 / (1 + ruggedness + sign_change_rate + log(1 + max(anisotropy-1,0)))`.

## Frozen phase order and nominal fractions

1. initial design: 0.10;
2. center-local contraction: 0.15;
3. coordinate multi-bracket sweep: 0.30;
4. far-basin exploration: 0.15;
5. archive fallback: 0.10;
6. final polishing: 0.05;
7. graph-aware budget completion: all unused and remaining evaluations.

Center-local contraction is enabled when dimension <= 20 and local-mode score
>= 0.12. Far-basin exploration is enabled when ruggedness >= 0.10, boundary
signal >= 0.05, or maximum box width >= 100.

## Coordinate multi-bracket search

For each selected start and coordinate, the implementation samples 11 equally
spaced values, detects sampled local-minimum brackets, retains up to three
brackets, and refines each using bounded scalar minimization for up to 20
iterations.

## Other search modes

Far-basin exploration uses heavy-tailed directions around the box centre.
Archive fallback and final polishing use bounded L-BFGS-B searches from
selected archive representatives. Budget completion uses graph-guided or
elite perturbations, uniform probes and stall-triggered local polishing.

## Returned record

Each run serializes the best point/value, total and per-phase evaluation
counts, best-value history, active archive nodes, directed graph edges,
diagnostics, event log, implementation version, options, options hash and
termination message.

## Interpretation boundary

The implementation supports claims about an explicit, auditable graph over
operational search-region representatives and observed transitions. It does
not establish an exact topological decomposition or certify every node as a
true local attraction basin.
'''
    (OUT / "RC1_EXACT_ALGORITHM_SEMANTICS.md").write_text(semantics)

    language_rows = [
        {
            "category": "allowed",
            "language": "operational basin-state node",
            "reason": "Exact meaning of rc1 BasinNode.",
        },
        {
            "category": "allowed",
            "language": "candidate basin representative",
            "reason": "Does not imply local-minimum certification.",
        },
        {
            "category": "allowed",
            "language": "directed graph of observed search transitions",
            "reason": "Matches edge creation.",
        },
        {
            "category": "allowed",
            "language": "barrier proxy",
            "reason": "Matches the stored sampled proxy.",
        },
        {
            "category": "allowed",
            "language": "domain anisotropy",
            "reason": "Matches max/min box width.",
        },
        {
            "category": "prohibited",
            "language": "certified basin node",
            "reason": "Certification exists only in rejected rc2.",
        },
        {
            "category": "prohibited",
            "language": "every node is a local minimum",
            "reason": "Raw finite evaluations can enter the rc1 archive.",
        },
        {
            "category": "prohibited",
            "language": "exact attraction-basin decomposition",
            "reason": "Not established by rc1.",
        },
        {
            "category": "prohibited",
            "language": "exact energy barrier",
            "reason": "Only a barrier proxy is stored.",
        },
        {
            "category": "prohibited",
            "language": "objective curvature anisotropy",
            "reason": "rc1 stores domain anisotropy.",
        },
    ]
    write_csv(OUT / "RC1_MANUSCRIPT_LANGUAGE_RULES.csv", language_rows)

    field_rows = [
        {
            "paper_object": "operational basin-state node",
            "code_object": "BasinNode",
            "serialized_field": "archive[]",
            "exact_interpretation": "active archive representative; not certified",
        },
        {
            "paper_object": "representative point",
            "code_object": "BasinNode.center",
            "serialized_field": "archive[].center",
            "exact_interpretation": "best stored point for the node",
        },
        {
            "paper_object": "observed transition",
            "code_object": "TransitionEdge",
            "serialized_field": "graph_edges[]",
            "exact_interpretation": "directed algorithmic transition",
        },
        {
            "paper_object": "barrier proxy",
            "code_object": "TransitionEdge.barrier_proxy",
            "serialized_field": "graph_edges[].barrier_proxy",
            "exact_interpretation": "sample-derived proxy",
        },
        {
            "paper_object": "accessibility",
            "code_object": "TransitionEdge.accessibility",
            "serialized_field": "graph_edges[].accessibility",
            "exact_interpretation": "1/(1+barrier proxy+evaluations) on creation",
        },
        {
            "paper_object": "domain anisotropy",
            "code_object": "GeometryDiagnostics.anisotropy",
            "serialized_field": "diagnostics.anisotropy",
            "exact_interpretation": "max box width / min box width",
        },
        {
            "paper_object": "phase FE accounting",
            "code_object": "EvaluationLedger.phase_evaluations",
            "serialized_field": "phase_evaluations",
            "exact_interpretation": "exact objective calls by phase",
        },
        {
            "paper_object": "best-value trajectory",
            "code_object": "EvaluationLedger.history",
            "serialized_field": "history",
            "exact_interpretation": "best-so-far after each evaluation",
        },
    ]
    write_csv(OUT / "RC1_PAPER_CODE_FIELD_MAP.csv", field_rows)

    parameter_rows = [
        {
            "parameter": key,
            "value": json.dumps(value),
            "options_hash": EXPECTED_OPTIONS_HASH,
        }
        for key, value in options.to_jsonable().items()
    ]
    write_csv(OUT / "RC1_FINAL_PARAMETER_TABLE.csv", parameter_rows)

    checklist = '''# Manuscript rewrite checklist for selected rc1

- Define nodes as operational basin-state representatives.
- State that nodes are not guaranteed local minima.
- Describe edges as observed algorithmic transitions.
- Use “barrier proxy”, not exact barrier.
- Use “domain anisotropy”, not curvature anisotropy.
- State the exact merge, eviction, accessibility and graph-guidance formulas.
- Use the exact phase order and frozen parameter table.
- Label Figure 1 as conceptual rather than a literal run reconstruction.
- Do not transfer any rc2 result or rc2 certified-node terminology.
- Report future COCO development and holdout partitions separately.
- Report future CUTEst development/comparability and holdout partitions
  separately.
'''
    (OUT / "RC1_MANUSCRIPT_REWRITE_CHECKLIST.md").write_text(checklist)

    claim_rows = [
        {
            "claim": "The optimizer maintains an explicit graph during search.",
            "status": "supported",
            "required_evidence": "serialized archive/edges and integrity tests",
        },
        {
            "claim": "Graph guidance changes search behavior.",
            "status": "supported",
            "required_evidence": "NoGraphGuidance ablation",
        },
        {
            "claim": "Every node is a certified local minimum.",
            "status": "prohibited",
            "required_evidence": "not available in rc1",
        },
        {
            "claim": "The implementation exactly reconstructs basin topology.",
            "status": "prohibited",
            "required_evidence": "not established",
        },
        {
            "claim": "Generalization beyond development problems.",
            "status": "pending_holdout",
            "required_evidence": "prospective COCO and CUTEst holdout",
        },
    ]
    write_csv(OUT / "RC1_FINAL_CLAIM_EVIDENCE_MAP.csv", claim_rows)

    contract = {
        "status": "RC1_PAPER_CODE_CONTRACT_FROZEN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "implementation_version": EXPECTED_IMPLEMENTATION,
        "options_hash": EXPECTED_OPTIONS_HASH,
        "branch": branch,
        "branch_parent_commit": git_output("rev-parse", "HEAD"),
        "code_freeze_tag": RC1_CODE_FREEZE_TAG,
        "code_freeze_commit": code_freeze_commit,
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "basingraph_v2").glob("*.py"))
        },
        "options": options.to_jsonable(),
        "formal_protocol_lock_sha256": sha256_file(formal_lock_path),
        "rc1_machine_spec_sha256": sha256_file(rc1_machine_path),
        "development_validation_sha256": sha256_file(validation_path),
        "development_diagnostic_sha256": sha256_file(diagnostic_path),
        "holdout_accessed": False,
    }
    (OUT / "RC1_PAPER_CODE_CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True)
    )

    manifest_rows = []
    manifest_path = OUT / "MANIFEST_SHA256.csv"
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path != manifest_path:
            manifest_rows.append(
                {
                    "filename": path.name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    write_csv(manifest_path, manifest_rows)

    print("RC1_PAPER_CODE_CONTRACT_FROZEN")
    print("branch:", branch)
    print("implementation:", EXPECTED_IMPLEMENTATION)
    print("options hash:", EXPECTED_OPTIONS_HASH)
    print("code-freeze commit:", code_freeze_commit)
    print("development run:", run_id)
    print("holdout accessed:", False)
    print("output:", OUT)


if __name__ == "__main__":
    main()
