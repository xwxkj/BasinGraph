# B21 Track C scientific applications and observed-data protocol

Status: **frozen before confirmatory evaluation**  
Confirmatory objective evaluations before freeze: **0**

## Scope

Track C evaluates the immutable BasinGraph candidate on nine deterministic
scientific-model families and six NIST StRD observed-data nonlinear-regression
problems. It does not revise the candidate or reuse any Track A–D evidence for
tuning.

## C1 scientific-model matrix

- families: elliptic PDE inverse estimation; Lorenz-63 calibration; noiseless
  and noisy real phase retrieval; low-rank and large low-rank matrix
  factorization; viscous Burgers control; Allen–Cahn energy minimization; sparse
  nonlinear inverse least squares;
- development instances: 1–2;
- confirmatory instances: 11–13;
- paired algorithm seeds: 0–9;
- algorithms: seven;
- confirmatory runs: 1,890.

The exact dimension and budget table is frozen in
`TRACK_C_TASK_SPECIFICATIONS.json`.

## C2 NIST observed-data matrix

The registered datasets are Chwirut1, Roszman1, ENSO, Eckerle4, Bennett5 and
BoxBOD. Official ASCII files are vendored unchanged from NIST/ITL and identified
by SHA-256. Each dataset uses ten paired algorithm seeds, seven algorithms and a
budget of `1,000d`, for 420 runs.

## Algorithms

1. BasinGraph;
2. CMA-ES through pycma;
3. BIPOP-CMA-ES through pycma;
4. corrected L-SHADE 1.0.1 source-aligned transparent Python port;
5. jSO source-aligned transparent Python port;
6. L-SRTDE transparent source-guided Python port with registered deviations;
7. multi-start L-BFGS-B through SciPy.

The transparent ports are not byte-identical official executables. DIRECT-L is
excluded before evaluation because the heterogeneous tasks include moderate-
and high-dimensional parameterizations for which its rectangle representation
is not an appropriate primary comparator.

## Normalization and primary endpoints

For each task, fixed reference and baseline objective values are computed from
the registered task construction before algorithm execution. The normalized gap
is

```text
g = max(0, (f - f_ref) / max(f_base - f_ref, epsilon)).
```

Target ratios are `1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001`. Primary evidence is
the fraction of task–seed–target triples reached and evaluations to target,
reported overall, by family, by decision dimension and by synthetic versus
observed-data source. Fixed-budget checkpoints are `1d`, `3d`, `10d`, `30d`,
`100d`, `300d` when available, and the registered final budget.

Only BasinGraph versus each comparator is confirmatory. Inference uses Friedman
tests, paired Wilcoxon signed-rank tests, Holm correction, paired rank-biserial
effects and bootstrap confidence intervals.

## Secondary endpoints

- final normalized gap and final rank;
- parameter or certified-parameter error;
- phase-invariant signal error;
- matrix reconstruction error;
- PDE state-data misfit;
- Lorenz parameter and trajectory error;
- Burgers tracking and control-energy terms;
- Allen–Cahn energy and interface-location error;
- NIST certified-RSS ratio;
- wall-clock time and implementation failures.

Task-specific references are descriptive anchors and are not pooled into the
general-purpose mean rank.

## Integrity

- Every algorithm uses the same task data, bounds, seed, budget and objective.
- Every objective call is counted by a single internal ledger.
- Smoke uses development tasks only.
- Confirmatory C1 instances 11–13 and all C2 outputs cannot be used to revise
  BasinGraph.
- Favorable, null and unfavorable results are retained.
- Post-result groupings are labelled exploratory.
- The Apple Silicon Mac is the primary machine; cross-machine numerical and
  timing results are not pooled.
