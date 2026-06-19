#!/usr/bin/env python3
"""
Build a GitHub-ready BasinGraph repository from the local research project.

The script deliberately excludes large raw result directories from GitHub.
Those files should be deposited on Zenodo and referenced by DOI.

Usage
-----
cd ~/Documents/BasinGraph202606
python scripts/step14b_build_github_repo.py

Optional:
python scripts/step14b_build_github_repo.py --force --init-git
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path.home() / "Documents" / "BasinGraph202606"
REPO_ROOT = PROJECT_ROOT / "release" / "BasinGraph"

VERSION = "1.0.0"
RELEASE_DATE = "2026-06-19"

AUTHORS = [
    {
        "given": "Kaijie",
        "family": "Xu",
        "email": "kjxu@xidian.edu.cn",
        "orcid": "https://orcid.org/0000-0003-4408-9070",
        "affiliation": "School of Information Mechanics and Sensing Engineering, Xidian University, Xi'an 710071, China; Department of Chemical and Biological Engineering, University of British Columbia, Vancouver, BC V6T 1Z3, Canada",
    },
    {
        "given": "Xiaoan",
        "family": "Tang",
        "email": "tangxa@hfut.edu.cn",
        "orcid": "https://orcid.org/0000-0002-4624-3687",
        "affiliation": "Key Laboratory of Process Optimization and Intelligent Decision-making, Ministry of Education, and School of Management, Hefei University of Technology, Hefei 230009, Anhui, China",
    },
    {
        "given": "Weike",
        "family": "Nie",
        "email": "weikenie@nwu.edu.cn",
        "orcid": "https://orcid.org/0000-0003-2092-3083",
        "affiliation": "College of Computer Science, Northwest University, Xi'an 710127, China",
    },
    {
        "given": "Ye",
        "family": "Cui",
        "email": "ycui7@ualberta.ca",
        "orcid": "https://orcid.org/0000-0003-4402-5341",
        "affiliation": "Department of Electrical and Computer Engineering, University of Alberta, Edmonton, AB T6G 2R3, Canada",
    },
    {
        "given": "Hanyu",
        "family": "E",
        "email": "hanyu6@ualberta.ca",
        "orcid": "https://orcid.org/0000-0002-5567-6126",
        "affiliation": "Department of Electrical and Computer Engineering, University of Alberta, Edmonton, AB T6G 2R3, Canada",
    },
    {
        "given": "Xukun",
        "family": "Yin",
        "email": "xkyin@xidian.edu.cn",
        "orcid": "https://orcid.org/0000-0003-4840-9529",
        "affiliation": "School of Information Mechanics and Sensing Engineering, Xidian University, Xi'an 710071, China; Hangzhou Institute of Technology, Xidian University, Hangzhou, China",
    },
    {
        "given": "Qiang",
        "family": "Zhang",
        "email": "zhangqiangonline@163.com",
        "orcid": "https://orcid.org/0000-0003-2947-1351",
        "affiliation": "Key Laboratory of Process Optimization and Intelligent Decision-making, Ministry of Education, and School of Management, Hefei University of Technology, Hefei 230009, Anhui, China",
    },
    {
        "given": "Yinghui",
        "family": "Quan",
        "email": "quanyinghui@126.com",
        "orcid": "https://orcid.org/0000-0001-6541-9441",
        "affiliation": "School of Information Mechanics and Sensing Engineering, Xidian University, Xi'an 710071, China",
    },
    {
        "given": "Dazheng",
        "family": "Feng",
        "email": "dzfeng@xidian.edu.cn",
        "orcid": "https://orcid.org/0000-0002-0168-8340",
        "affiliation": "School of Electronic Engineering, Xidian University, Xi'an 710071, China",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--init-git", action="store_true")
    return parser.parse_args()


def copy_tree_if_present(source: Path, destination: Path, ignore=None):
    if not source.exists():
        return False
    shutil.copytree(source, destination, dirs_exist_ok=True, ignore=ignore)
    return True


def copy_file_if_present(source: Path, destination: Path):
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def latest_matching(directory: Path, pattern: str):
    files = list(directory.glob(pattern)) if directory.exists() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def yaml_quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_citation_cff(path: Path):
    lines = [
        "cff-version: 1.2.0",
        'message: "If you use BasinGraph, please cite the software and the associated article."',
        'title: "BasinGraph: geometry-controlled basin-graph optimization"',
        f"version: {VERSION}",
        f"date-released: {RELEASE_DATE}",
        "type: software",
        "authors:",
    ]
    for author in AUTHORS:
        lines.extend([
            f"  - family-names: {yaml_quote(author['family'])}",
            f"    given-names: {yaml_quote(author['given'])}",
            f"    email: {yaml_quote(author['email'])}",
            f"    affiliation: {yaml_quote(author['affiliation'])}",
            f"    orcid: {yaml_quote(author['orcid'])}",
        ])
    lines.extend([
        'abstract: "BasinGraph is a geometry-controlled global optimization framework that represents discovered attraction basins and transition pathways explicitly."',
        'keywords:',
        '  - global optimization',
        '  - derivative-free optimization',
        '  - nonconvex optimization',
        '  - basin graph',
        '  - COCO/BBOB',
        '  - CUTEst',
        'repository-code: "https://github.com/TO_BE_CONFIRMED/BasinGraph"',
        'license: "TO_BE_CONFIRMED"',
    ])
    path.write_text("\n".join(lines) + "\n")


def write_zenodo_json(path: Path):
    creators = []
    for author in AUTHORS:
        creators.append({
            "name": f"{author['family']}, {author['given']}",
            "affiliation": author["affiliation"],
            "orcid": author["orcid"].replace("https://orcid.org/", ""),
        })
    metadata = {
        "title": "BasinGraph: geometry-controlled basin-graph optimization",
        "description": (
            "Source code, benchmark protocols, figure-generation scripts, "
            "and reproducibility metadata for BasinGraph."
        ),
        "upload_type": "software",
        "creators": creators,
        "keywords": [
            "global optimization",
            "derivative-free optimization",
            "nonconvex optimization",
            "COCO/BBOB",
            "CUTEst",
        ],
        "version": VERSION,
        "publication_date": RELEASE_DATE,
        "related_identifiers": [],
        "notes": (
            "Select the final software license and replace all TO_BE_CONFIRMED "
            "fields before creating the public release."
        ),
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def main():
    args = parse_args()

    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(PROJECT_ROOT)

    if REPO_ROOT.exists():
        if not args.force:
            raise RuntimeError(
                f"{REPO_ROOT} already exists. Re-run with --force to rebuild it."
            )
        shutil.rmtree(REPO_ROOT)

    REPO_ROOT.mkdir(parents=True)
    copied = []

    # Core source.
    for name in ["basingraph", "baselines", "experiments", "scripts"]:
        if copy_tree_if_present(
            PROJECT_ROOT / name,
            REPO_ROOT / name,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".DS_Store", "*.log"
            ),
        ):
            copied.append(name)

    # Frozen benchmark protocols, but not local compiler installations.
    if copy_tree_if_present(
        PROJECT_ROOT / "protocols",
        REPO_ROOT / "protocols",
        ignore=shutil.ignore_patterns(
            "*.log", "requirements-lock.txt", ".DS_Store"
        ),
    ):
        copied.append("protocols")

    # Figure source and independent MATLAB validation if present.
    for name in ["figure_source", "matlab_validation", "analysis"]:
        if copy_tree_if_present(
            PROJECT_ROOT / name,
            REPO_ROOT / name,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".DS_Store", "*.log"
            ),
        ):
            copied.append(name)

    # Lightweight final analysis products and Source Data.
    cutest_analysis = (
        PROJECT_ROOT
        / "cutest_results"
        / "protocol_v2"
        / "analysis_final_v1"
    )
    if cutest_analysis.exists():
        for sub in ["source_data", "tables", "manuscript_text", "figures"]:
            copy_tree_if_present(
                cutest_analysis / sub,
                REPO_ROOT / "reproducibility" / "cutest" / sub,
            )
        copied.append("CUTEst analysis products")

    # Final COCO provenance and frozen source snapshot.
    final_run_id_file = (
        PROJECT_ROOT / "official_results" / "OFFICIAL_FINAL_CORE_RUN_ID.txt"
    )
    if final_run_id_file.exists():
        run_id = final_run_id_file.read_text().strip()
        frozen = PROJECT_ROOT / "official_results" / f"{run_id}_FINAL"
        for sub in ["protocols", "source_snapshot", "tables"]:
            copy_tree_if_present(
                frozen / sub,
                REPO_ROOT / "reproducibility" / "coco_bbob" / sub,
            )
        copy_file_if_present(
            frozen / "README.md",
            REPO_ROOT / "reproducibility" / "coco_bbob" / "README.md",
        )
        copy_file_if_present(
            frozen / "MANIFEST_SHA256.csv",
            REPO_ROOT / "reproducibility" / "coco_bbob" / "MANIFEST_SHA256.csv",
        )
        copied.append("official COCO/BBOB metadata")

    # Environment locks.
    env_dir = REPO_ROOT / "environments"
    env_dir.mkdir()
    lock_files = sorted((PROJECT_ROOT / "protocols").glob("*requirements-lock*.txt"))
    for lock in lock_files:
        shutil.copy2(lock, env_dir / lock.name)
    copy_file_if_present(
        PROJECT_ROOT / "protocols" / "cutest_env.sh",
        env_dir / "cutest_env_template.sh",
    )
    copy_file_if_present(
        PROJECT_ROOT / "protocols" / "step13b_cutest_stack_versions.txt",
        env_dir / "cutest_stack_versions.txt",
    )

    # Metadata and repository documents.
    write_citation_cff(REPO_ROOT / "CITATION.cff")
    write_zenodo_json(REPO_ROOT / ".zenodo.json")

    (REPO_ROOT / "README.md").write_text(
        """# BasinGraph

BasinGraph is a geometry-controlled basin-graph optimization framework for
nonconvex mathematical models.

## Repository contents

- `basingraph/`: proposed optimizer.
- `baselines/`: standardized benchmark implementations and wrappers.
- `experiments/`: COCO/BBOB and benchmark runners.
- `scripts/`: reproducible experiment and analysis entry points.
- `protocols/`: frozen benchmark protocols and manifests.
- `reproducibility/`: lightweight Source Data, summary tables, and metadata.
- `environments/`: dependency locks and CUTEst environment information.

Large COCO observer logs, CUTEst run-level histories, and full raw archives are
not committed to GitHub. They will be deposited on Zenodo and linked by DOI.

## Main validation

- Official noiseless COCO/BBOB: 24 functions, dimensions 2, 3, 5, 10 and 20,
  15 instances, and a budget of 1,000d evaluations.
- Pre-registered CUTEst: 50 instances, seven algorithms, and 30 paired seeds.
- Independent MATLAB full-parallel validation and applied-mathematics tests.

## Installation

Create the COCO environment from the archived dependency lock or install:

```bash
python -m pip install numpy scipy pandas matplotlib cma coco-experiment cocopp
```

CUTEst requires SIFDecode, CUTEst, MASTSIF and PyCUTEst. See
`protocols/` and `environments/` for the frozen toolchain information.

## Reproduction

Experiment entry points are documented in `scripts/`. The final public release
will contain a DOI-backed Zenodo archive with raw logs and convergence histories.

## Citation

See `CITATION.cff`.

## License

**To be confirmed by all authors before public release.**
"""
    )

    (REPO_ROOT / ".gitignore").write_text(
        """# Python
__pycache__/
*.py[cod]
*.so
*.dylib
.pytest_cache/
.mypy_cache/
.venv/
venv/

# macOS / editors
.DS_Store
.vscode/
.idea/

# Local build and CUTEst stack
cutest_stack/
pycutest_cache/
build/
dist/
*.egg-info/

# Large benchmark outputs
exdata/
ppdata/
official_results/
cutest_results/protocol_v2/job_records/
cutest_results/protocol_v2/histories/
cutest_results/protocol_v2/job_failures/
logs/
processed_results/
archive/
manuscript/

# Secrets / local configuration
.env
*.token
"""
    )

    (REPO_ROOT / "LICENSE_PENDING.md").write_text(
        """# License pending

The authors must select the final software license before the first public
release. A permissive scientific-software license such as BSD-3-Clause or MIT
may be considered. Remove this file and add the final `LICENSE` file before
creating release v1.0.0.
"""
    )

    (REPO_ROOT / "DATA_AVAILABILITY_TEMPLATE.md").write_text(
        """# Data availability template

The official COCO/BBOB observer logs, cocopp outputs, CUTEst run-level results,
convergence histories, Source Data tables, and SHA-256 manifests supporting this
study are available in the Zenodo archive at **[DOI TO BE INSERTED]**.
"""
    )

    (REPO_ROOT / "CODE_AVAILABILITY_TEMPLATE.md").write_text(
        """# Code availability template

The BasinGraph source code, benchmark runners, frozen protocols, analysis
scripts, and figure-generation scripts are available at
**[GITHUB URL TO BE INSERTED]**. Version 1.0.0 is archived on Zenodo at
**[DOI TO BE INSERTED]**.
"""
    )

    (REPO_ROOT / "CONTRIBUTING.md").write_text(
        """# Contributing

Before public release, contributions should be coordinated with the
corresponding authors. Please report reproducibility issues using a GitHub issue
and include the platform, Python version, dependency versions, protocol hash,
and complete error log.
"""
    )

    (REPO_ROOT / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "basingraph-optimizer"
version = "1.0.0"
description = "Geometry-controlled basin-graph optimization"
readme = "README.md"
requires-python = ">=3.11"
authors = [
  {name = "Kaijie Xu", email = "kjxu@xidian.edu.cn"},
  {name = "Xiaoan Tang", email = "tangxa@hfut.edu.cn"},
  {name = "Xukun Yin", email = "xkyin@xidian.edu.cn"},
]
dependencies = [
  "numpy",
  "scipy",
  "pandas",
  "matplotlib",
]

[project.optional-dependencies]
coco = ["cma", "coco-experiment", "cocopp"]
cutest = ["pycutest==1.8.2"]

[tool.setuptools.packages.find]
include = ["basingraph*"]
"""
    )

    # Large-data note.
    data_dir = REPO_ROOT / "data"
    data_dir.mkdir()
    (data_dir / "README.md").write_text(
        """# Large data archives

Large raw benchmark outputs are intentionally excluded from GitHub.

The Zenodo release should contain:

1. final official COCO/BBOB observer logs and cocopp output;
2. CUTEst 10,500-row raw result table and all convergence histories;
3. MATLAB full-parallel raw results;
4. Source Data for each manuscript figure;
5. SHA-256 manifests and environment metadata.

Insert the Zenodo DOI here after release.
"""
    )

    # Detect manuscript status, but do not copy unpublished manuscript into repo.
    current_manuscripts = list(
        (PROJECT_ROOT / "manuscript" / "current").glob("*.docx")
    ) if (PROJECT_ROOT / "manuscript" / "current").exists() else []

    # Absolute path audit.
    absolute_hits = []
    patterns = [str(Path.home()), "/Users/", "Downloads"]
    text_suffixes = {
        ".py", ".sh", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml", ".cff"
    }
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(errors="ignore")
        except Exception:
            continue
        for pattern in patterns:
            if pattern in text:
                absolute_hits.append({
                    "file": str(path.relative_to(REPO_ROOT)),
                    "pattern": pattern,
                })

    # Manifest.
    manifest_rows = []
    total_size = 0
    large_files = []
    for path in sorted(REPO_ROOT.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        total_size += size
        if size > 50 * 1024 * 1024:
            large_files.append(str(path.relative_to(REPO_ROOT)))
        manifest_rows.append({
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": size,
        })

    with (REPO_ROOT / "MANIFEST_SHA256.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["relative_path", "sha256", "size_bytes"]
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    report = {
        "repository": str(REPO_ROOT),
        "version": VERSION,
        "copied_components": copied,
        "file_count": len(manifest_rows),
        "total_size_bytes_before_manifest": total_size,
        "files_over_50_MB": large_files,
        "absolute_path_hits": absolute_hits,
        "manuscript_current_docx_count": len(current_manuscripts),
        "manuscript_current_files": [p.name for p in current_manuscripts],
        "license_status": "pending",
        "github_url_status": "pending",
        "zenodo_doi_status": "pending",
    }
    (REPO_ROOT / "STEP14B_REPO_BUILD_REPORT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    if args.init_git:
        subprocess.run(["git", "init", "-b", "main"], cwd=REPO_ROOT, check=True)
        subprocess.run(["git", "add", "."], cwd=REPO_ROOT, check=True)

    print("STEP_14B_OK")
    print("Repository:", REPO_ROOT)
    print("Files:", len(manifest_rows))
    print("Size (MB):", round(total_size / 1024 / 1024, 2))
    print("Files over 50 MB:", len(large_files))
    print("Absolute-path hits:", len(absolute_hits))
    print("Current manuscript DOCX files:", len(current_manuscripts))
    print("License: pending author decision")
    if args.init_git:
        print("Git initialized and files staged; no commit was created.")


if __name__ == "__main__":
    main()
