# Prospective CUTEst holdout analysis plan

Run ID: `v2rc1_cutest_holdout_20260621T175759Z`

This plan was frozen after integrity validation and before performance values
were inspected.

## Primary analysis

For each problem, define `f_ref` as the minimum final objective observed across
all seven algorithms and 30 paired seeds. Define

`s_p = max(|f(x0)-f_ref|, 1e-12*(1+|f_ref|))`.

Targets are `f_ref + tau*s_p` for

`tau in {1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6}`.

The first best-so-far history entry reaching each target defines runtime.
Unsuccessful runs are charged the full prescribed budget. ERT is the sum of
successful runtimes and failed-run budgets divided by the number of successes.
The primary ECDF uses function evaluations divided by dimension.

## Secondary analysis

- final normalized gaps;
- task-normalized final ranks;
- pairwise BasinGraph better/worse/tied counts;
- dimension-group descriptive summaries.

## Confirmatory statistics

Use 24 problems as blocks. Apply a Friedman test to per-problem median final
normalized gaps. Compare BasinGraph with each baseline using paired,
two-sided Wilcoxon signed-rank tests, Holm correction and paired rank-biserial
effect sizes.

No algorithm or parameter change is permitted after this holdout.
