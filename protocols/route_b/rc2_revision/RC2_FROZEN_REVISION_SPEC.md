# BasinGraph v2.0.0-rc2 frozen revision specification

## 1. Certified basin semantics

`ProbeRecord` and `BasinNode` become distinct objects.

Raw points from initial design, far-basin probing and budget completion are
stored only as probes. A point may enter the basin archive only after a
certification operation:

- local polishing;
- coordinate multi-bracket refinement;
- principal-direction multi-bracket refinement;
- archive-fallback polishing;
- stall-triggered polishing.

Each active node records `certified=True`, `certification_mode`, parent probe
identifier, refinement evaluations and local-support metadata.

## 2. Adaptive basin identity

The merge threshold is

`r_merge = min(r_max, r_base * (1 + 2 * occupancy^2)) * ||ub-lb||_2`.

Frozen constants:

- `r_base = 0.025`;
- `r_max = 0.080`;
- `occupancy = active_nodes / archive_capacity`.

## 3. Quality-diversity-accessibility retention

The global-best node is protected. When capacity is exceeded, all other nodes
are ranked by

`0.50 * quality + 0.30 * diversity + 0.20 * accessibility`.

Quality is reverse objective rank, diversity is normalized nearest-neighbour
distance, and accessibility is the maximum incoming edge accessibility.
The lowest-scoring unprotected node is evicted, with all incident edges.

## 4. Landscape curvature anisotropy

Replace box-width anisotropy by a derivative-free curvature proxy. For each
valid lower-centre-upper coordinate triplet,

`kappa_j = |f(x_j^-) - 2 f(c) + f(x_j^+)| / h_j^2`.

Curvature anisotropy is the robust ratio `q90(kappa)/max(q10(kappa), eps)`,
capped at `1e6`. The box-width ratio remains available as
`domain_anisotropy` but must not be described as landscape anisotropy.

## 5. Principal-direction refinement

Add directional coarse sampling plus top-K bounded refinement along:

- leading eigenvectors of elite-node covariance;
- normalized successful transition directions;
- coordinate directions as deterministic fallback.

The directional line-search routine must share the same bracket semantics as
the coordinate multi-bracket routine.

## 6. Probe-refine-commit exploration

Far-basin and budget-completion phases use batches:

1. generate probes;
2. score probes by objective quality and distance from certified nodes;
3. refine only the best/most novel probes;
4. commit only certified refined results;
5. create graph edges only between certified nodes.

No raw probe may become a graph node.

## 7. Graph sparsification

For each active node retain at most:

- three outgoing edges;
- three incoming edges.

Edge retention score combines best improvement, accessibility and recency.
Graph pruning occurs after each node or edge update.

## 8. Controller and phase allocation

Frozen rc2 phase fractions:

- initial design: 0.10;
- center-local: 0.08;
- coordinate multi-bracket: 0.25;
- principal-direction refinement: 0.15;
- far-basin probe/refine: 0.12;
- archive fallback: 0.10;
- final polishing: 0.08;
- remaining/unused budget: graph-aware completion.

Center-local contraction is enabled only when:

- dimension <= 20;
- local-mode score >= 0.20;
- curvature anisotropy <= 100;
- ruggedness score <= 0.10.

## 9. Required result contract

Every rc2 result must serialize:

- implementation version and options hash;
- probes and certified archive nodes separately;
- certification metadata for every node;
- sparse transition graph;
- curvature and domain anisotropy separately;
- principal-direction diagnostics;
- exact phase evaluation counts;
- event log;
- graph/archive referential integrity.

## 10. Change budget

No other algorithmic mechanism or parameter search is permitted before the
first rc2 development evaluation. Any correction necessary for a software bug
must be documented separately and must not use holdout evidence.
