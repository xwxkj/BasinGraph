# Track A confirmatory decision

Status: **audited and frozen**  
Run ID: `b21_track_a_confirmatory_20260808T052204Z`  
Decision: **HOLD — preserve the frozen candidate, narrow the mechanism claims, and proceed to Track B.**

## Integrity

The 1,920-run matrix completed without failed runs. Internal and COCO observer
evaluation counts were exact, phase counts summed to `nfe`, archive capacity
and graph referential integrity passed, all best-so-far histories were
monotone, and official cocopp post-processing completed.

The returned archives were independently verified:

- result ZIP SHA-256: `ff3f42c1a17d8b786c9e0d5e16cef30f093a2953c30d4eca89bd6147d03b732c`;
- observer archive SHA-256: `02214e8fa02b635fc054ad09af598b8d3a4fa38662257c74f5b1342873b069a5`;
- raw-results SHA-256: `18e37b6277a950051c54f875fb251819327e3e9abce5ef572525b12a50b137a7`.

## Primary target-runtime finding

Across the standard 51-target grid, Full attained 37.63% of run-target pairs
by 300 evaluations per dimension and 45.37% by 1,000 evaluations per
dimension. The corresponding best ablations attained 38.35% (`SingleBracket`)
and 45.69% (`NoGeometryController`). Except for `NoCenterLocal`, aggregate
primary differences were small.

Official cocopp overall ordering at the right edge of the ECDF plots was:

- 5D: `NoArchiveFallback > NoGraphGuidance > NoFinalPolish > SingleBracket > NoGeometryController > Full > NoFarBasin > NoCenterLocal`;
- 20D: `NoGraphGuidance > NoArchiveFallback > NoFinalPolish > SingleBracket > NoFarBasin > Full > NoGeometryController > NoCenterLocal`.

These orders are descriptive curve positions, not stand-alone significance
tests. They show that Full was not the aggregate target-runtime leader in
either dimension.

## Secondary final-value finding

The final-value Friedman test was significant (`p=0.00383`). After Holm
correction, `SingleBracket`, `NoFarBasin` and `NoArchiveFallback` had better
final values than Full. Final-value evidence remains secondary to the official
target-runtime analysis.

## Mechanism interpretation

- Center-local contraction supplied the clearest early- and intermediate-budget
  benefit, especially on G1–G3, but its effect was not uniformly positive on
  multimodal problems.
- Graph guidance changed every trajectory, helped 5D weak-global-structure
  problems and hurt several 20D conditioned groups. It is operational, but a
  uniform aggregate benefit is not supported.
- Single-bracket refinement was modestly better than the retained
  multi-bracket configuration in the registered checkpoint summary.
- Far-basin exploration, archive fallback and final polishing redistributed
  performance across strata rather than improving it uniformly.
- Disabling the geometry controller changed 39 of 240 trajectories and was
  neutral in most registered strata.

## Why the decision is HOLD

This is not a GO result because Full was not the official aggregate ECDF leader
and several secondary ablations were significantly better. It is not a NO-GO
result because no ablation produced a large, uniform advantage across both
dimensions, most function groups and all budget stages. The dominant pattern is
mechanism interaction with dimension, problem geometry and budget.

The frozen `2.0.0-rc1` candidate must not be changed using instances 16–20.
Changing it now would convert Track A into development data and require an
entirely new confirmatory cycle. The scientifically valid next step is the
pre-planned, disjoint modern-baseline Track B, while retaining every favorable,
null and unfavorable Track A result.

## Manuscript boundary

The manuscript may state that BasinGraph mechanisms alter trajectories and
redistribute target-runtime performance across geometry and budget, with the
clearest contribution arising from center-local contraction. It must not state
that every archive, graph, bracket, exploration, fallback, polishing or
controller component improves aggregate performance uniformly.
