"""Compatibility entry point for the manuscript result-bearing BasinGraph.

The historical simplified implementation that formerly occupied this module is
not the implementation used for the prospective COCO/BBOB and CUTEst results.
The result-bearing source is frozen in :mod:`basingraph_v2.optimizer` at tag
``route-b-v2.0.0-rc1-selected-final-candidate`` and commit
``adbc0ecdf1153044188f0508321c47001ad9bdb0``.

This module re-exports that implementation without changing the objective-call
ledger, phase budgets, archive, graph, options or return object.
"""

from basingraph_v2.optimizer import (
    BasinGraphOptions,
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)
from basingraph_v2.types import BasinGraphResult

minimize_basingraph = minimize_basingraph_v2

__all__ = [
    "BasinGraphOptions",
    "BasinGraphResult",
    "IMPLEMENTATION_VERSION",
    "minimize_basingraph",
    "minimize_basingraph_v2",
]
