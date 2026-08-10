#!/usr/bin/env python3
"""Run Track E engineering pilot 3 with the phase-boundary state probe."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evidence_extension_v1.track_e import run_matched_trace_pilot as base  # noqa: E402
from evidence_extension_v1.track_e.state_probe_pilot import run_block  # noqa: E402


if __name__ == "__main__":
    base.run_block = run_block
    base.main()
