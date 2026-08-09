# Track C frozen analysis plan

## Primary evidence

Primary evidence is normalized target-runtime across the seven registered gap
targets. Results are reported overall, by C1 family, by C2 dataset, by decision
dimension and by data source. Unsuccessful attempts consume the full prescribed
budget.

Only BasinGraph versus each comparator is confirmatory. The registered tests are
Friedman omnibus testing, paired Wilcoxon signed-rank tests, Holm correction,
paired rank-biserial effects and bootstrap 95% confidence intervals.

## Secondary evidence

Final normalized gap, final rank, task-specific physical metrics, wall time and
failures are secondary. Raw objective values are never pooled across families.
Task-specific references are descriptive and are not included in the
seven-algorithm aggregate rank.

## Decision boundary

Track C supports a broad scientific-usefulness claim only if BasinGraph remains
in a leading statistical tier on multiple independent scientific families or
shows reproducible finite-budget advantages on physically meaningful metrics,
and if the NIST observed-data anchor does not contradict that claim.


## Implementation note 003

Only checkpoints available under a task budget are analyzed, and the family-specific final budget is explicitly identified. Registered task-specific references comprise spectral Wirtinger flow, alternating ridge least squares and NIST far/near-start least squares. They are descriptive and excluded from the general-purpose aggregate ranking. Paired primary target-fraction inference is frozen before confirmatory evaluation.
