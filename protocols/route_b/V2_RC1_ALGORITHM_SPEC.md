# BasinGraph v2.0.0-rc1 algorithm specification

This document is the controlling paper-code specification for the Route B
release candidate.

## Phase order

1. Initial design and exact function-evaluation accounting.
2. Geometry diagnostics.
3. Controller decision.
4. Center-local contraction.
5. Coordinate coarse sampling and multi-bracket bounded refinement.
6. Far-basin exploration.
7. Graph-guided archive fallback.
8. Final elite polishing.
9. Graph-aware budget completion.

## Basin identity

A candidate is merged into its nearest active basin node when the Euclidean
distance is not larger than

`max(candidate_radius, merge_radius_factor * ||ub - lb||_2)`.

Otherwise a new node is created. If capacity is exceeded, the active node with
the largest objective value is evicted, and all incident graph edges are
removed.

## Transition graph

A directed edge is added only when a search action moves from one active basin
node to a distinct active basin node. Each edge records:

- total evaluations;
- best objective improvement;
- minimum observed barrier proxy;
- accessibility;
- attempts;
- source mode.

Accessibility is

`1 / (1 + barrier_proxy + evaluations)`.

## Graph guidance

Archive fallback and budget completion rank or sample active nodes using

`0.55 * quality + 0.25 * novelty + 0.20 * accessibility`.

The weights and all phase fractions are frozen in
`V2_RC1_FROZEN_PARAMETER_TABLE.csv`.

## Geometry diagnostics

- anisotropy: max box width / min box width;
- boundary signal: normalized median interior-minus-boundary objective;
- ruggedness: normalized median absolute deviation of anchor values;
- sign-change rate: fraction of coordinate lower-centre-upper slope pairs
  whose signs change;
- finite-anchor fraction.

## Experiment consistency rule

No v2.0.0-rc1 result may be reported unless its result JSON contains:

- implementation_version = `2.0.0-rc1`;
- the frozen options hash;
- phase_evaluations summing to nfe;
- explicit archive nodes;
- graph edges with active node references only;
- diagnostics and event log.
