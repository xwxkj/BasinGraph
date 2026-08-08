# Prospective CUTEst holdout results

## Evidence integrity

The review archive passed its package manifest and frozen evidence-lock checks.
The analysis used the predeclared target and statistical definitions. The
prospective holdout contains 24 CUTEst instances, 30 paired seeds, seven
algorithms, 720 problem–seed jobs and 5,040 algorithm records, with zero exact
instance overlap with the 50-instance development/comparability set.

## Manuscript-ready Results text

On the prospective CUTEst holdout, BasinGraph provided the strongest overall
target-runtime profile. Across 4,320 problem–seed–target triples per
algorithm—24 problems, 30 seeds and six relative targets—BasinGraph reached
56.5% of targets, compared with
44.1%
for multi-start L-BFGS-B,
33.9%
for CMA-ES and
32.2%
for BIPOP-CMA-ES. BasinGraph attained the lowest mean ERT rank
(1.944) and solved at least one seed in
124 of 144 problem–target blocks,
including all 30 seeds in
69 blocks. Its ECDF fraction
was 0.364
at 10 evaluations per dimension,
0.517
at 50 evaluations per dimension and
0.565
at 100 evaluations per dimension.

The advantage persisted as targets tightened. BasinGraph success rates were
80.7%,
70.7%,
51.0%,
48.9%,
44.3% and
43.5%
for relative targets from 10⁻¹ to 10⁻⁶, respectively. Descriptively, BasinGraph
also had the best mean ERT rank in each size stratum: small
(2.045), medium
(1.417) and large
(2.375).

As a secondary fixed-budget analysis, BasinGraph achieved the lowest mean
task-normalized final-value rank (1.753), was
best or tied in 509 of 720
problem–seed runs, and had a median normalized final gap of
0.000593. Its mean ranks were
1.985,
1.262 and
1.900
for the small, medium and large strata.

The predeclared Friedman test rejected equal performance across the seven
algorithms (χ²₆=107.821,
P=5.83e-21). Holm-adjusted paired Wilcoxon tests on
per-problem median normalized gaps showed significant BasinGraph advantages
over Differential Evolution, Latin hypercube sampling and random search.
Differences from CMA-ES, BIPOP-CMA-ES and multi-start L-BFGS-B did not remain
significant after correction, despite positive paired rank-biserial effect
sizes.

## Confirmatory pairwise tests

- versus Differential Evolution: adjusted P=7.153e-07, rank-biserial=1.000; significant.
- versus Latin Hypercube Sampling: adjusted P=7.153e-07, rank-biserial=1.000; significant.
- versus Random Search: adjusted P=7.153e-07, rank-biserial=1.000; significant.
- versus CMA-ES: adjusted P=0.08078, rank-biserial=0.500; not significant after Holm correction.
- versus Multi-start L-BFGS-B: adjusted P=0.08078, rank-biserial=0.533; not significant after Holm correction.
- versus BIPOP-CMA-ES: adjusted P=0.08078, rank-biserial=0.513; not significant after Holm correction.

## Interpretation limits

- The primary evidence is the predeclared target-runtime ECDF/ERT analysis.
- The pooled best-observed reference is symmetric across all algorithms but is
  not a claim of a known global optimum.
- Size-stratified findings are descriptive because the strata contain only 11,
  7 and 6 problems.
- The active archive reached its capacity of 80 in every BasinGraph run; it is
  an operational search memory, not a complete basin enumeration.
- The 24-problem holdout must be reported separately from the 50-problem
  development/comparability suite.
