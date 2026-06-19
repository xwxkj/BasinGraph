#!/usr/bin/env python3
"""Validate the GitHub-ready BasinGraph repository."""

from pathlib import Path
import json
import re
import sys

repo = Path.home() / "Documents" / "BasinGraph202606" / "release" / "BasinGraph"
if not repo.exists():
    raise SystemExit(f"Repository does not exist: {repo}")

required = [
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    ".gitignore",
    "pyproject.toml",
    "MANIFEST_SHA256.csv",
    "basingraph",
    "baselines",
    "experiments",
    "scripts",
    "protocols",
]

missing = [name for name in required if not (repo / name).exists()]
if missing:
    raise SystemExit(f"Missing required repository items: {missing}")

report_path = repo / "STEP14B_REPO_BUILD_REPORT.json"
report = json.loads(report_path.read_text())

print("Repository:", repo)
print("Files:", report["file_count"])
print("Files over 50 MB:", report["files_over_50_MB"])
print("Absolute path hits:", len(report["absolute_path_hits"]))
print("Current manuscript DOCX files:", report["manuscript_current_files"])
print("License status:", report["license_status"])
print("GitHub URL status:", report["github_url_status"])
print("Zenodo DOI status:", report["zenodo_doi_status"])

if report["files_over_50_MB"]:
    raise SystemExit("Repository contains files over 50 MB.")

if report["absolute_path_hits"]:
    print("\nWARNING: local absolute paths remain:")
    for item in report["absolute_path_hits"][:30]:
        print(" ", item)

citation = (repo / "CITATION.cff").read_text()
for token in ["TO_BE_CONFIRMED", "github.com/TO_BE_CONFIRMED"]:
    if token in citation:
        print(f"WARNING: CITATION.cff still contains {token}")

print("STEP_14B_VALIDATION_OK")
