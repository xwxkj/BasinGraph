# Decision: one structural rc2 revision is required

## Scope

This decision uses only the frozen COCO development partition, instances 1-3.
No prospective COCO or CUTEst holdout result was accessed.

## Evidence

The official cocopp development output shows a consistent pattern:

- BasinGraph is competitive at early and intermediate budgets;
- its aggregate ECDF plateaus relative to CMA-ES and BIPOP-CMA-ES near the
  1000d budget;
- the late-budget deficit is strongest on high-conditioning/unimodal
  functions and at finer targets;
- hardest-target successes are 40/72, 22/72 and 15/72 in 2D, 5D and 10D.

The serialized mechanism diagnostics show:

- all 216 development runs saturate the 80-node archive;
- low-dimensional graphs are much denser than higher-dimensional graphs;
- center-local contraction is activated in almost every run;
- center-local budget fraction has a weak positive association with worse
  final-value rank.

## Semantic defect in rc1

In rc1, raw initial anchors, far-basin probes and budget-completion probes can
be committed directly as `BasinNode` objects. Consequently, archive saturation
does not necessarily mean that 80 distinct attraction basins were discovered.
This is inconsistent with the intended manuscript semantics of nodes as
refined basin representatives.

The rc1 anisotropy statistic is based on box-width ratios. BBOB uses isotropic
box widths, so this statistic cannot diagnose objective-landscape
conditioning.

## Decision

Implement exactly one structural release candidate, `v2.0.0-rc2`, before
opening any prospective holdout. The revision is mechanism-driven rather than
a post-hoc parameter search. Its design and acceptance gate are frozen in this
directory before code is changed.
