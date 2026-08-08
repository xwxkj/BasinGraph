#!/usr/bin/env python3
"""Generate the deterministic Track D protocol manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_ROOT = ROOT / "protocols" / "evidence_extension_v1" / "track_d"
MANIFEST = PROTOCOL_ROOT / "MANIFEST_SHA256.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(PROTOCOL_ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "sha256", "size_bytes"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print("TRACK_D_PROTOCOL_MANIFEST_GENERATED")
    print(f"Files: {len(rows)}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
