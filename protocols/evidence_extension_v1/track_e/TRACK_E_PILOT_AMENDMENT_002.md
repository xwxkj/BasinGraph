# Track E pilot Amendment 002: phase-boundary state probe

Status: **registered after engineering pilot 2 and before engineering pilot 3**  
Confirmatory Track E evaluations at amendment: **0**

## Trigger

Pilot 2 captured an exact `40d` checkpoint inside a run planned for `140d`.
All snapshots occurred during the coordinate-sweep phase, before completed
coordinate candidates had been inserted into the graph; every prefix graph
therefore had zero edges. In addition, the common continuation kernel retained
the incumbent-based stall-polish branch. That branch produced the same local
solution from the shared incumbent and again dominated all five arms. Pilot 2
is retained as a second negative engineering result.

## Registered correction

Pilot 3 captures the first objective-call boundary **after the coordinate-sweep
phase has completed**. At this boundary, completed coordinate candidates have
already updated the archive and observed-transition graph. The planned parent
run uses a `250d` budget only to ensure that a later phase is available; the
captured prefix length is whatever the immutable phase logic actually consumed.

Pilot 3 then applies a common, fixed `100d` **state-probe continuation kernel**
to each state view. The kernel is the sampling component of the immutable
BasinGraph budget-completion policy: graph-guided archive sampling, unguided
archive sampling and global sampling. The incumbent-based stall-polish branch
is deliberately omitted because it is identical across matched-trace arms and
would test the shared local solver rather than retained-state information. No
arm receives a local solver.

Every evaluated continuation point is hashed. Thus the analysis distinguishes
(a) different evaluated trajectories from (b) different best-so-far traces.

## Unchanged items

The six task families, dimensions, instances, five state views, continuation
budget, target ratios, paired continuation seeds, primary `FullState` versus
`TraceOnly` contrast, statistical procedures and success gate remain
unchanged. Pilot 3 is engineering-only. A positive pilot still requires a
separately frozen, disjoint confirmatory experiment before manuscript use.

No file under `basingraph_v2/` is modified. Pilot 1 and Pilot 2 remain preserved
and are not pooled with Pilot 3.
