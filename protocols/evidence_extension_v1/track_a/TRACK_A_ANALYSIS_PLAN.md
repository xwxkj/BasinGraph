# Track A frozen analysis plan

## Primary evidence

The primary evidence is the official COCO/cocopp target-runtime analysis,
including empirical cumulative distributions and expected running times across
the standard target grid.

Predeclared evaluations-per-dimension checkpoints are:

```text
1, 3, 10, 30, 100, 300, 1000
```

Results are reported overall, by dimension and by the five frozen BBOB function
groups.

## Primary comparisons

Only `Full` versus each of the seven ablations is confirmatory. All other
pairwise contrasts are exploratory.

For paired endpoints:

- Friedman omnibus test across variants;
- paired Wilcoxon signed-rank test for Full versus each ablation;
- Holm correction across the seven Full-versus-ablation tests;
- paired rank-biserial effect size;
- bootstrap 95% confidence intervals where the endpoint supports resampling.

Unsuccessful target attempts consume the full prescribed budget in ERT.

## Secondary evidence

- final objective-value rank;
- final-target success count;
- fixed-budget checkpoint ranks;
- phase-allocation fractions;
- archive-node and graph-edge counts;
- wall-clock time and optimizer overhead.

Final-value rank is descriptive and cannot replace official target-runtime
evidence.

## Frozen strata

- G1: functions 1–5, separable;
- G2: functions 6–9, low or moderate conditioning;
- G3: functions 10–14, highly conditioned unimodal;
- G4: functions 15–19, multimodal with adequate global structure;
- G5: functions 20–24, multimodal with weak global structure;
- dimensions: 5D and 20D.

Any additional grouping created after results are observed is explicitly
labelled exploratory.

## Interpretation

A module is not claimed to be uniformly beneficial merely because Full has a
lower aggregate rank. Evidence is interpreted jointly across target runtime,
dimension, function group, budget checkpoint and effect size. Unfavorable or
null results are retained.
