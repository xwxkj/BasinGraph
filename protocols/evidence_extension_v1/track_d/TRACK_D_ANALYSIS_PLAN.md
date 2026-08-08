# Track D frozen analysis plan

## Primary performance evidence

The primary performance evidence is official COCO/cocopp target-runtime
analysis on `bbob-largescale`. Registered checkpoints are:

```text
1, 3, 10, 30, 100, 200 evaluations per dimension
```

Results are reported overall, by dimension and by the five frozen BBOB function
groups.

Only BasinGraph versus each of the four comparators is confirmatory. Comparator-
to-comparator contrasts are descriptive.

For paired performance endpoints:

- Friedman omnibus test across algorithms;
- paired Wilcoxon signed-rank tests for BasinGraph versus each comparator;
- Holm correction across the four registered tests;
- paired rank-biserial effect size;
- bootstrap 95% confidence intervals where appropriate.

Unsuccessful target attempts consume the full prescribed budget in ERT.

## Primary overhead evidence

The sequential overhead probe is analyzed separately from the parallel
performance matrix. For each algorithm and dimension, report:

- median total wall time;
- median observed objective-call time;
- median optimizer-side time;
- median optimizer-side microseconds per evaluation;
- maximum process peak resident memory;
- median estimated serialized result size.

Empirical scaling exponents are obtained by ordinary least squares on
`log2(metric)` versus `log2(dimension)` for positive finite medians. These slopes
are descriptive scaling summaries, not asymptotic-complexity proofs.

For BasinGraph additionally report median and maximum archive nodes and graph
edges, phase fractions, serialized result size, and their dimensional trends.

## Secondary evidence

- final objective-value rank;
- final-target success count;
- parallel confirmatory wall-clock time;
- implementation failures and messages;
- detailed function-group interactions.

Final-value rank and parallel wall time cannot replace official target-runtime
or sequential overhead evidence.

## Frozen strata

- G1: functions 1–5, separable;
- G2: functions 6–9, low or moderate conditioning;
- G3: functions 10–14, highly conditioned unimodal;
- G4: functions 15–19, multimodal with adequate global structure;
- G5: functions 20–24, multimodal with weak global structure;
- dimensions: 40, 80, 160 and 320.

Any post-result grouping is labelled exploratory.

## Interpretation

A controlled-scalability claim requires complete runs, exact evaluation
accounting, no BasinGraph integrity failure and an overhead/memory profile that
remains bounded and scientifically interpretable. Unfavorable and null results
are retained. Track D does not alter the frozen candidate.
