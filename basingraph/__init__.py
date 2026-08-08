"""Public BasinGraph API.

The manuscript result-bearing implementation is frozen in :mod:`basingraph_v2`.
This compatibility package exposes that implementation under the concise
``basingraph`` import path without changing its source, defaults or options
hash.
"""

from basingraph_v2 import BasinGraphOptions, BasinGraphResult
from basingraph_v2.optimizer import (
    IMPLEMENTATION_VERSION,
    minimize_basingraph_v2,
)

# Public manuscript-facing name. The alias is intentional: no wrapper changes
# evaluation accounting, return fields or frozen option semantics.
minimize_basingraph = minimize_basingraph_v2

__all__ = [
    "BasinGraphOptions",
    "BasinGraphResult",
    "IMPLEMENTATION_VERSION",
    "minimize_basingraph",
    "minimize_basingraph_v2",
]
