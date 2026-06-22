#!/usr/bin/env python3
"""Remove obsolete MATLAB/AppliedMath public statements and prepare v2.0.1 docs-only metadata.

This script changes documentation and release metadata only. It refuses to
modify the selected algorithm and checks byte identity against the rc1 freeze.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OPTIONS_HASH = (
    "031b9c3df716889e48e2db753c73ec960"
    "b96a0239173ce791b4ed1ee63ed0f69"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-doi", required=True)
    parser.add_argument("--release-date", default="2026-06-22")
    return parser.parse_args()


def validate_doi(value: str) -> None:
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", value):
        raise ValueError(f"Invalid Zenodo DOI: {value}")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    validate_doi(args.dataset_doi)

    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    if branch != "route-b/finalize-rc1-consistent":
        raise RuntimeError(f"Wrong branch: {branch}")

    if subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "route-b-v2.0.0-rc1-codefreeze",
            "--",
            "basingraph_v2",
        ],
        cwd=ROOT,
    ).returncode:
        raise RuntimeError("basingraph_v2 differs from the frozen rc1 source")

    readme = f"""# BasinGraph [![DOI](https://zenodo.org/badge/1274531480.svg)](https://doi.org/10.5281/zenodo.20765883)

BasinGraph is a geometry-controlled optimization framework that maintains a
fixed-capacity archive of operational basin-state representatives and a
directed graph of observed search transitions.

## Selected result-bearing implementation

- Implementation: `BasinGraph 2.0.0-rc1`
- Selected-candidate tag: `route-b-v2.0.0-rc1-selected-final-candidate`
- Options hash: `{OPTIONS_HASH}`

Software release `v2.0.1` is a documentation-only cleanup. It does not change
the selected algorithm, benchmark code, frozen options, raw results or
numerical conclusions.

## Prospective validation

- Official noiseless COCO/BBOB holdout: 24 functions; dimensions 2, 3, 5, 10
  and 20; actual instances 4-15; seven algorithms; budget 1,000d; 10,080
  records.
- Prospective CUTEst holdout: 24 performance-independently selected instances;
  30 paired seeds; seven algorithms; 5,040 records.

COCO instances 1-3 and the 50-instance CUTEst development/comparability set
are retained separately and are not pooled with the prospective holdouts.

## Repository contents

- `basingraph_v2/`: selected optimizer implementation.
- `baselines/`: frozen comparator implementations and wrappers.
- `experiments_v2/`: COCO/BBOB and CUTEst runners.
- `analysis_v2/`: validation and analysis scripts.
- `scripts_v2/`: reproducibility and release workflows.
- `protocols/`: frozen protocols, machine contracts and manifests.
- `results_v2/final_analysis/`: lightweight final analysis tables and text.
- `environments/`: dependency and CUTEst toolchain records.

Large observer logs, atomic histories and raw archives are deposited on Zenodo.

## Installation

```bash
python -m pip install numpy scipy pandas matplotlib cma coco-experiment cocopp
```

CUTEst additionally requires SIFDecode, CUTEst, MASTSIF and PyCUTEst. See
`protocols/` and `environments/` for the frozen toolchain records.

## Reproducibility data

- Dataset version 2.0.1 DOI: https://doi.org/{args.dataset_doi}
- Dataset all-versions DOI: https://doi.org/10.5281/zenodo.20765046

The v2.0.1 dataset contains the prospective COCO/BBOB and CUTEst evidence,
Source Data, protocols, manuscript files and integrity records. Obsolete
legacy implementation materials are intentionally excluded.

## Software archive

- Software all-versions DOI: https://doi.org/10.5281/zenodo.20765883
- Version 2.0.0 DOI: https://doi.org/10.5281/zenodo.20789002

The version-specific DOI for software release v2.0.1 is added to the current
branch after Zenodo completes the GitHub release archive.

## Citation

See `CITATION.cff`.

## License

BasinGraph is released under the BSD-3-Clause license. See `LICENSE`.
"""
    write(ROOT / "README.md", readme)

    write(
        ROOT / "DATA_AVAILABILITY_TEMPLATE.md",
        f"""# Data availability

The official prospective COCO/BBOB observer logs and cocopp outputs, the
frozen prospective CUTEst results and improvement histories, figure Source
Data, analysis plans, frozen protocols and SHA-256 manifests are openly
available in the BasinGraph reproducibility dataset, version 2.0.1, at
https://doi.org/{args.dataset_doi}. The all-versions DOI is
https://doi.org/10.5281/zenodo.20765046.
""",
    )

    write(
        ROOT / "data/README.md",
        f"""# Large data archives

Large raw benchmark outputs are intentionally excluded from GitHub. The
reproducibility dataset version 2.0.1 at
https://doi.org/{args.dataset_doi} contains:

1. prospective official COCO/BBOB observer logs and cocopp output;
2. prospective CUTEst raw records and all atomic improvement histories;
3. Source Data and figure sources;
4. frozen protocols, analysis plans and excluded-attempt audits;
5. SHA-256 manifests and environment metadata.

The historical dataset version 2.0.0 remains in the Zenodo version chain but
is not the version cited by the final manuscript.
""",
    )

    write(
        ROOT / "docs/route_b/ROUTE_B_FREEZE.md",
        f"""# Route B freeze: selected BasinGraph rc1

## Final selected candidate

The result-bearing implementation is BasinGraph 2.0.0-rc1 with options hash
`{OPTIONS_HASH}` and tag
`route-b-v2.0.0-rc1-selected-final-candidate`.

## Final prospective evidence

- COCO/BBOB: actual instances 4-15, 24 functions, dimensions 2, 3, 5, 10 and
  20, seven algorithms and budget 1,000d.
- CUTEst: 24 held-out instances, 30 paired seeds and seven algorithms.

Development records, rejected rc2 records and excluded execution attempts are
retained as audit evidence and are not pooled with the final holdouts.

## Public release policy

Software v2.0.1 is a documentation-only cleanup. It removes obsolete legacy
statements and does not modify the selected algorithm or any reported result.
""",
    )

    write(
        ROOT / "protocols/route_b/V2_CONSISTENCY_CONTRACT.md",
        f"""# BasinGraph final consistency contract

## Non-negotiable rule

No manuscript claim, figure or table may be attributed to the selected
BasinGraph candidate unless it is supported by the byte-frozen
`basingraph_v2` implementation, options hash `{OPTIONS_HASH}`, and the frozen
prospective COCO/BBOB or CUTEst evidence.

## Required public objects

- BasinNode
- BasinArchive
- TransitionEdge
- BasinTransitionGraph
- GeometryDiagnostics
- BasinGraphResult

## Required optimizer outputs

- xbest
- fbest
- nfe
- history
- archive
- graph_edges
- diagnostics
- event_log
- message

## Final evidence boundary

Primary external evidence consists only of the prospective COCO/BBOB and
CUTEst holdouts. Development records, rejected candidate records and legacy
archives remain historical/audit material and must not be presented as final
performance evidence.
""",
    )

    builder = ROOT / "scripts/step14b_build_github_repo.py"
    if builder.exists():
        text = builder.read_text()
        text = text.replace(
            'for name in ["figure_source", "matlab_validation", "analysis"]:',
            'for name in ["figure_source", "analysis"]:',
        )
        text = text.replace(
            "# Figure source and independent MATLAB validation if present.",
            "# Figure source and final analysis products if present.",
        )
        text = text.replace(
            "- Independent MATLAB full-parallel validation and applied-mathematics tests.",
            "",
        )
        text = text.replace(
            "3. MATLAB full-parallel raw results;\n"
            "4. Source Data for each manuscript figure;\n"
            "5. SHA-256 manifests and environment metadata.",
            "3. Source Data for each manuscript figure;\n"
            "4. SHA-256 manifests and environment metadata.",
        )
        builder.write_text(text)

    zenodo_path = ROOT / ".zenodo.json"
    metadata = json.loads(zenodo_path.read_text()) if zenodo_path.exists() else {}
    metadata["version"] = "2.0.1"
    metadata["publication_date"] = args.release_date
    metadata["description"] = (
        "Documentation-only cleanup for the selected BasinGraph 2.0.0-rc1 "
        "implementation. Obsolete legacy-validation statements were removed; "
        "algorithm source, frozen options and reported results are unchanged."
    )
    related = [
        item
        for item in metadata.get("related_identifiers", [])
        if item.get("identifier") != args.dataset_doi
    ]
    related.append(
        {
            "identifier": args.dataset_doi,
            "relation": "isSupplementedBy",
            "scheme": "doi",
            "resource_type": "dataset",
        }
    )
    metadata["related_identifiers"] = related
    zenodo_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")

    cff_path = ROOT / "CITATION.cff"
    if cff_path.exists():
        text = cff_path.read_text()
        text = re.sub(r"(?m)^version:\s*.*$", "version: 2.0.1", text)
        text = re.sub(
            r"(?m)^date-released:\s*.*$",
            f"date-released: {args.release_date}",
            text,
        )
        cff_path.write_text(text)

    release_notes = f"""# BasinGraph v2.0.1

This is a documentation-only cleanup release for the selected result-bearing
implementation BasinGraph 2.0.0-rc1.

## Changes

- removed obsolete legacy-validation statements from public documentation;
- updated README and data-availability text to the prospective COCO/BBOB and
  CUTEst holdouts;
- clarified that the operational archive is not a complete basin enumeration;
- linked the cleaned reproducibility dataset v2.0.1:
  https://doi.org/{args.dataset_doi}

## Unchanged

- no algorithm source change;
- no option or options-hash change;
- no benchmark rerun;
- no raw result or numerical conclusion change.

Selected implementation: `2.0.0-rc1`  
Options hash: `{OPTIONS_HASH}`
"""
    write(ROOT / "RELEASE_NOTES_v2.0.1.md", release_notes)

    report = {
        "status": "B11_MATLAB_FREE_REPO_PATCH_OK",
        "branch": branch,
        "dataset_doi": args.dataset_doi,
        "release_date": args.release_date,
        "algorithm_changed": False,
        "options_hash": OPTIONS_HASH,
    }
    report_path = (
        ROOT
        / "protocols/route_b/final_rc1/B11_MATLAB_FREE_PUBLIC_RELEASE_PREP.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    audit_targets = [
        ROOT / "README.md",
        ROOT / "DATA_AVAILABILITY_TEMPLATE.md",
        ROOT / "data/README.md",
        ROOT / "docs/route_b/ROUTE_B_FREEZE.md",
        ROOT / "protocols/route_b/V2_CONSISTENCY_CONTRACT.md",
        ROOT / ".zenodo.json",
        ROOT / "RELEASE_NOTES_v2.0.1.md",
        ROOT / "scripts/step14b_build_github_repo.py",
    ]
    for path in audit_targets:
        if re.search(
            r"MATLAB|full[- ]parallel|applied[- ]mathematics",
            path.read_text(),
            re.I,
        ):
            raise RuntimeError(f"Residual obsolete statement: {path}")

    print("B11_MATLAB_FREE_REPO_PATCH_OK")
    print("Dataset DOI:", args.dataset_doi)
    print("Algorithm changed: False")


if __name__ == "__main__":
    main()
