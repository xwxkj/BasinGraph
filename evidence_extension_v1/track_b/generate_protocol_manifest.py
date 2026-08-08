#!/usr/bin/env python3
"""Generate the deterministic Track B protocol SHA-256 manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = (
    ROOT / "protocols" / "evidence_extension_v1" / "track_b"
)
MANIFEST = PROTOCOL_ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not PROTOCOL_ROOT.is_dir():
        raise RuntimeError(f"Missing Track B protocol root: {PROTOCOL_ROOT}")

    rows = []
    for path in sorted(PROTOCOL_ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(PROTOCOL_ROOT).as_posix(),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("TRACK_B_PROTOCOL_MANIFEST_GENERATED")
    print(f"Files: {len(rows)}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
