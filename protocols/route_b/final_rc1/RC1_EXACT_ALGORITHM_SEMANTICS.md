# BasinGraph 2.0.0-rc1 exact algorithm semantics

This is the controlling manuscript-code contract. It describes the selected
code exactly and deliberately avoids stronger, unsupported interpretations.

## Operational basin-state node

A `BasinNode` is an operational search-state representative in the active
archive. It is not guaranteed to be a certified local minimum, a stationary
point, or an exact attraction basin.

Each node stores an identifier, representative point, objective value,
operational radius, curvature proxy, visit count, creation/update evaluation
indices, novelty and source mode.

Finite evaluations from the initial design, center-local search, coordinate
multi-bracket search, far-basin exploration, archive fallback and budget
completion may create or update nodes.

## Node merge and capacity rules

Let `D = ||ub-lb||_2`. A candidate merges with its nearest node when

`distance <= max(candidate_radius, merge_radius_factor * D)`.

Otherwise it creates a new node. If capacity is exceeded, the active node with
the largest objective value is evicted, and all incident graph edges are
removed. This is an operational archive rule, not a proof of exact basin
identity.

## Directed observed-transition graph

A directed edge records an observed algorithmic transition between two
distinct active nodes. It stores cumulative attributed evaluations, best
observed improvement, minimum barrier proxy, maximum accessibility, attempt
count, source mode and creation/update evaluation indices.

For a newly observed edge,

`accessibility = 1 / (1 + barrier_proxy + evaluations)`.

The barrier is a sampled algorithmic proxy, not an exact landscape barrier.

## Graph guidance

Archive fallback and budget completion use

`0.55 * quality + 0.25 * novelty + 0.20 * accessibility`.

Quality is reverse objective rank among active nodes. Accessibility is the
maximum incoming edge accessibility, with a neutral value for nodes without
incoming edges.

## Initial design

The design uses the box centre, lower/upper corners, two alternating corners,
coordinate lower-centre-upper anchors and 24 Latin-hypercube points, subject
to the phase evaluation limit.

## Geometry diagnostics

- mean scale: mean box width;
- maximum scale: maximum box width;
- anisotropy: maximum box width divided by minimum box width;
- boundary signal: normalized median interior-minus-boundary objective;
- ruggedness: normalized median absolute deviation of anchor values;
- sign-change rate: fraction of valid coordinate lower-centre-upper slope
  pairs with opposite signs;
- finite-anchor fraction.

The rc1 quantity named `anisotropy` is domain anisotropy, not objective
curvature anisotropy.

The local-mode score is

`1 / (1 + ruggedness + sign_change_rate + log(1 + max(anisotropy-1,0)))`.

## Frozen phase order and nominal fractions

1. initial design: 0.10;
2. center-local contraction: 0.15;
3. coordinate multi-bracket sweep: 0.30;
4. far-basin exploration: 0.15;
5. archive fallback: 0.10;
6. final polishing: 0.05;
7. graph-aware budget completion: all unused and remaining evaluations.

Center-local contraction is enabled when dimension <= 20 and local-mode score
>= 0.12. Far-basin exploration is enabled when ruggedness >= 0.10, boundary
signal >= 0.05, or maximum box width >= 100.

## Coordinate multi-bracket search

For each selected start and coordinate, the implementation samples 11 equally
spaced values, detects sampled local-minimum brackets, retains up to three
brackets, and refines each using bounded scalar minimization for up to 20
iterations.

## Other search modes

Far-basin exploration uses heavy-tailed directions around the box centre.
Archive fallback and final polishing use bounded L-BFGS-B searches from
selected archive representatives. Budget completion uses graph-guided or
elite perturbations, uniform probes and stall-triggered local polishing.

## Returned record

Each run serializes the best point/value, total and per-phase evaluation
counts, best-value history, active archive nodes, directed graph edges,
diagnostics, event log, implementation version, options, options hash and
termination message.

## Interpretation boundary

The implementation supports claims about an explicit, auditable graph over
operational search-region representatives and observed transitions. It does
not establish an exact topological decomposition or certify every node as a
true local attraction basin.
