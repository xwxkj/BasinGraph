#!/usr/bin/env python3
"""Generate deterministic SHA-256 manifest for Track C protocol files."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_c.common import sha256_file


def main() -> None:
    protocol_root = ROOT / "protocols/evidence_extension_v1/track_c"
    manifest = protocol_root / "MANIFEST_SHA256.csv"
    rows = []
    for path in sorted(protocol_root.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(
                {
                    "relative_path": path.relative_to(protocol_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print("TRACK_C_PROTOCOL_MANIFEST_GENERATED")
    print(f"Files: {len(rows)}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
