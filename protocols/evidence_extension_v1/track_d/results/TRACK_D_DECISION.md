# Track D confirmatory decision

Status: **audited and frozen**  
Run IDs: `b21_track_d_overhead_20260808T202425Z` and `b21_track_d_confirmatory_20260808T202425Z`  
Frozen source commit: `39d96cb3543035680dcc96860d527aa15e29c97b`  
Decision: **GO to Track C scientific applications, with a bounded-state scalability claim and no claim of universal high-dimensional superiority.**

## Integrity

The registered evidence completed without failed runs:

- sequential overhead probe: 60/60 runs;
- large-scale confirmatory matrix: 1,440/1,440 runs;
- official `bbob-largescale` functions 1–24;
- dimensions 40, 80, 160 and 320;
- confirmatory instances 16–18;
- five frozen algorithms;
- budget `200d` for confirmatory performance and `50d` for the overhead probe;
- exact internal, timed-objective and COCO observer evaluation accounting;
- all 20 overhead and 20 confirmatory shards completed on attempt 1;
- official cocopp post-processing completed;
- BasinGraph phase, archive and graph integrity checks passed.

Independently verified archive identities:

- result ZIP SHA-256: `96bd38676d2b43f902c5e09396ab1f6f6883012e219a4cde689799a19a5d242f`;
- observer archive SHA-256: `a496a9fd292932c628f42e20166c0d12d3fd2e9f7fd5579251d81a0d7e2c9526`.

## Primary target-runtime result

Across the standard 51-target grid, BasinGraph attained the following pooled fractions of function–instance–target pairs:

| evaluations per dimension | fraction | descriptive rank |
|---:|---:|---:|
| 1 | 0.0264 | 1 |
| 3 | 0.0276 | 2 |
| 10 | 0.0283 | 2 |
| 30 | 0.0287 | 3 |
| 100 | 0.1778 | 2 |
| 200 | 0.2239 | 2 |

At `200d`, multi-start L-BFGS-B led with 0.2443, BasinGraph followed with 0.2239, CMA-ES attained 0.1853, L-SHADE 1.0.1 attained 0.0549 and L-SRTDE attained 0.0353.

Paired per-problem target fractions at `200d` showed that BasinGraph was significantly better than L-SHADE 1.0.1 and L-SRTDE after Holm correction, significantly worse than multi-start L-BFGS-B, and statistically indistinguishable from CMA-ES. Dimension-specific final-checkpoint ranks were second in 40D, second in 80D, first in 160D and second in 320D.

## Secondary final-value result

The final-value Friedman test was significant (`p = 6.15e-69`). Mean final ranks were:

1. CMA-ES: 2.240;
2. multi-start L-BFGS-B: 2.500;
3. BasinGraph: 2.580;
4. L-SHADE 1.0.1: 3.398;
5. L-SRTDE: 4.283.

BasinGraph final values were statistically indistinguishable from CMA-ES and multi-start L-BFGS-B after Holm correction, and significantly better than L-SHADE 1.0.1 and L-SRTDE. Final-value evidence remains secondary to the official target-runtime analysis.

## Registered sequential overhead result

The single-worker overhead probe measured optimizer-side time separately from observed objective-call time.

For BasinGraph:

- median optimizer overhead rose from 21.35 microseconds per evaluation in 40D to 46.44 microseconds per evaluation in 320D;
- peak process RSS rose from 78.0 MB to 95.1 MB;
- the empirical log2–log2 overhead slope was 0.384;
- the archive remained at its fixed capacity of 80 nodes once filled;
- median graph-edge count remained approximately dimension-invariant;
- serialized result size grew approximately linearly with dimension.

At 320D, BasinGraph optimizer overhead was lower than CMA-ES and L-SHADE 1.0.1, higher than L-SRTDE and multi-start L-BFGS-B, and used substantially less peak RSS than CMA-ES, L-SHADE 1.0.1 and L-SRTDE.

## Interpretation

Track D supports the claim that BasinGraph preserves a fixed-capacity operational state and sparse observed-transition graph through 40–320 dimensions, with modest optimizer-side overhead and competitive large-scale target-runtime performance under a `200d` budget.

It does not support claims that BasinGraph is the fastest wall-clock optimizer, uniformly outperforms multi-start L-BFGS-B, is best in every dimension or function group, or that empirical scaling slopes prove asymptotic complexity.

## Next stage

Proceed to Track C scientific-model and observed-data applications. The frozen Track D suite and instances must not be reused for tuning or candidate revision.