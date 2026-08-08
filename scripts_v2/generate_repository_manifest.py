#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for tracked repository files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = "MANIFEST_SHA256.csv"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    ".venv-b21",
}
EXCLUDED_PREFIXES = (
    "results_b21/",
    "results_b21_smoke/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="Walk the working tree instead of using git ls-files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            stderr=subprocess.DEVNULL,
        )
        names = [item for item in output.decode("utf-8").split("\0") if item]
        return [ROOT / name for name in names]
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return []


def walked_files() -> list[Path]:
    paths = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    output_path = (ROOT / args.output).resolve()

    candidates = walked_files() if args.include_untracked else tracked_files()
    if not candidates:
        candidates = walked_files()

    rows = []
    for path in sorted(candidates, key=lambda item: item.relative_to(ROOT).as_posix()):
        if not path.is_file():
            continue
        if path.resolve() == output_path:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(EXCLUDED_PREFIXES):
            continue
        if any(part in EXCLUDED_PARTS for part in Path(relative).parts):
            continue
        rows.append(
            {
                "relative_path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    scope_path = output_path.with_suffix(".scope.json")
    scope = {
        "status": "REPOSITORY_MANIFEST_GENERATED",
        "root": str(ROOT),
        "manifest": str(output_path.relative_to(ROOT)),
        "manifest_self_excluded": True,
        "tracked_files_preferred": not args.include_untracked,
        "excluded_parts": sorted(EXCLUDED_PARTS),
        "excluded_prefixes": list(EXCLUDED_PREFIXES),
        "files": len(rows),
    }
    scope_path.write_text(json.dumps(scope, indent=2), encoding="utf-8")

    print("REPOSITORY_MANIFEST_GENERATED")
    print(f"Files: {len(rows)}")
    print(f"Manifest: {output_path}")
    print(f"Scope: {scope_path}")


if __name__ == "__main__":
    main()
