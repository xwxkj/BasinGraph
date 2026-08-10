"""Referentially sanitized snapshot wrapper for Track E pilot 3."""

from __future__ import annotations

from evidence_extension_v1.track_e import state_probe_pilot as probe


_original_capture = probe.capture_phase_boundary_snapshot
_last_dropped_transient_edges = 0


def sanitized_capture(*args, **kwargs):
    global _last_dropped_transient_edges
    snapshot = _original_capture(*args, **kwargs)
    active_ids = {int(node.node_id) for node in snapshot.archive}
    original_edges = list(snapshot.graph_edges)
    snapshot.graph_edges = [
        edge
        for edge in original_edges
        if int(edge.source_id) in active_ids
        and int(edge.target_id) in active_ids
    ]
    _last_dropped_transient_edges = int(
        len(original_edges) - len(snapshot.graph_edges)
    )
    return snapshot


def run_block(family: str, dimension: int, instance: int):
    global _last_dropped_transient_edges
    _last_dropped_transient_edges = 0
    probe.capture_phase_boundary_snapshot = sanitized_capture
    rows = probe.run_block(family, dimension, instance)
    for row in rows:
        row["snapshot_dropped_transient_edges"] = int(
            _last_dropped_transient_edges
        )
    return rows
