#!/usr/bin/env python3
"""Run Track E engineering pilot 2 with an exact mid-run snapshot."""

from __future__ import annotations

from evidence_extension_v1.track_e import run_matched_trace_pilot as base
from evidence_extension_v1.track_e.snapshot_pilot import (
    SNAPSHOT_CHECKPOINT_MULTIPLIER,
    run_block,
)


if __name__ == "__main__":
    base.run_block = run_block
    base.PREFIX_BUDGET_MULTIPLIER = SNAPSHOT_CHECKPOINT_MULTIPLIER
    base.main()
