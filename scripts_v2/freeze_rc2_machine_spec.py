#!/usr/bin/env python3
"""Freeze the rc2 implementation/options identity after tests pass."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if IMPLEMENTATION_VERSION != "2.0.0-rc2":
    raise RuntimeError(IMPLEMENTATION_VERSION)

options = BasinGraphOptions()
out = ROOT / "protocols" / "route_b" / "rc2_revision"
out.mkdir(parents=True, exist_ok=True)

spec = {
    "status": "RC2_MACHINE_SPEC_FROZEN",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "implementation_version": IMPLEMENTATION_VERSION,
    "options_hash": options.stable_hash(),
    "options": options.to_jsonable(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip(),
    "plan_freeze_tag": "route-b-v2.0.0-rc2-planfreeze",
    "source_hashes": {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in sorted(
            (ROOT / "basingraph_v2").glob("*.py")
        )
    },
}

path = out / "RC2_MACHINE_SPEC.json"
path.write_text(
    json.dumps(spec, indent=2, sort_keys=True)
)

print("RC2_MACHINE_SPEC_FROZEN")
print("implementation:", IMPLEMENTATION_VERSION)
print("options hash:", options.stable_hash())
print("path:", path)
