# Track B frozen analysis plan

## Primary evidence

The primary evidence is official COCO/cocopp target-runtime analysis over the
standard target grid, including empirical cumulative distributions and expected
running times. Predeclared checkpoints are:

```text
1, 3, 10, 30, 100, 300, 1000 evaluations per dimension
```

Results are reported overall, by dimension and by the five frozen BBOB function
groups.

## Primary comparisons

Only BasinGraph versus each of the seven baselines is confirmatory. Baseline-to-
baseline contrasts are descriptive unless explicitly labelled exploratory.

For paired endpoints:

- Friedman omnibus test across algorithms;
- paired Wilcoxon signed-rank tests for BasinGraph versus each baseline;
- Holm correction across the seven registered tests;
- paired rank-biserial effect size;
- bootstrap 95% confidence intervals where appropriate.

Unsuccessful target attempts consume the full prescribed budget in ERT.

## Secondary evidence

- final objective-value rank;
- final-target success count;
- fixed-budget checkpoint ranks;
- wall-clock time;
- implementation failures and messages;
- BasinGraph phase, archive and graph diagnostics.

Final-value rank and wall time are secondary and cannot replace official
target-runtime evidence.

## Frozen strata

- G1: functions 1–5, separable;
- G2: functions 6–9, low or moderate conditioning;
- G3: functions 10–14, highly conditioned unimodal;
- G4: functions 15–19, multimodal with adequate global structure;
- G5: functions 20–24, multimodal with weak global structure;
- dimensions: 5D and 20D.

Any post-result grouping is labelled exploratory.

## Interpretation

Track B supports a competitiveness claim only if BasinGraph remains in the
leading statistical tier without being uniformly dominated across dimensions,
function groups and budget stages. Unfavorable and null results are retained.
The experiment does not alter the frozen candidate.
