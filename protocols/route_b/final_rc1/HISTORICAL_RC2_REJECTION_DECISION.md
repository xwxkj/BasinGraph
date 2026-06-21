# BasinGraph v2.0.0-rc2 rejection decision

## Decision

BasinGraph v2.0.0-rc2 is rejected under the predeclared development
acceptance gate.

The prospective COCO/BBOB and CUTEst holdouts remain unopened.

## Development comparison

- rc1 run:
  v2rc1_coco_development_20260620T053200Z
- rc2 run:
  v2rc2_coco_development_20260621T035558Z
- paired development problems: 216
- rc2 better than rc1: 64
- rc2 worse than rc1: 90
- ties: 62

## Passed mechanism and integrity gates

- all runs completed;
- exact function-evaluation accounting;
- all archive nodes certified;
- raw probes in archive: 0;
- graph referential integrity verified;
- graph degree caps verified;
- landscape anisotropy nonconstant;
- principal-direction phase exercised;
- archive saturation fraction: 0.1481481481;
- maximum group-dimension mean edges per node: 0.6920430206;
- center-local activation fraction: 0.1481481481.

## Failed performance gates

- fixed-budget mean rank:
  3.5324074074, exceeding the noninferiority limit of 3.49;
- hardest-target successes:
  69, below the improvement target of 82;
- fixed-budget improvement target:
  not passed;
- known improvement gates passed:
  0 of at least 2 required.

## Consequence

rc2 must not be used for the prospective holdout or as the final manuscript
algorithm.

No rc3 algorithmic revision is authorized under the current protocol.

The next candidate is the rc1 implementation, subject to a new exact
paper-code semantic contract that describes its archive nodes and transition
graph without claiming rc2-style certified-basin semantics.
