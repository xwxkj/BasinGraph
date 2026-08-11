# Track E2: Operational-State Predictive Information Test

Status: **development protocol frozen before any Track E2 objective evaluation**  
Track E2 development objective evaluations at freeze: **0**  
Track E2 confirmatory objective evaluations at freeze: **0**

## 1. Question

Track E2 asks whether the bounded operational state retained by BasinGraph
contains out-of-sample information about future finite-budget progress beyond
that contained in the scalar best-so-far trace.

The experiment does **not** test whether a particular continuation policy can
convert that information into better optimization performance. The earlier
matched-trace engineering pilots did not establish such a performance benefit.
Track E2 instead tests incremental predictive information under a fixed,
transparent statistical model.

## 2. Immutable optimizer identity

- implementation: `2.0.0-rc1`;
- selected candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`;
- frozen options hash:
  `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- active archive capacity: 80.

No file under `basingraph_v2/` may be modified for Track E2.

## 3. Deterministic task families and partitions

The task families are:

1. rotated Rastrigin;
2. rotated Ackley;
3. rotated Griewank;
4. Gallagher-style Gaussian-basin mixture;
5. asymmetric double funnel;
6. Lunacek-style bi-Rastrigin.

Dimensions are 10 and 20. Each task has known optimum value zero.

- development instances: 1--8;
- reserved confirmatory instances: 101--108.

The transformation seed is a deterministic function of family, dimension and
instance. Confirmatory instances must not be evaluated unless the development
gate in Section 9 is met and a second machine-readable lock records the
selected ridge penalties before confirmatory access.

## 4. Parent runs and snapshots

Each parent run has budget `250d`. Operational snapshots are captured at the
first archive or graph update at or after `80d`, `140d` and `200d`. The actual
snapshot evaluation count is stored.

The capture layer monkey-patches only the constructors and update hooks used by
the frozen optimizer. It does not change candidate generation, phase budgets,
objective values, acceptance rules, archive capacity or graph scoring.

The future-progress response at a snapshot is

```text
y = log10(1 + f_snapshot) - log10(1 + f_final),
```

where the task optimum is zero and `f_final` is the final best value of the same
frozen parent run. The response is non-negative up to numerical tolerance.

## 5. Predictors

### Trace-only predictors

- dimension and normalized snapshot budget;
- current `log10(1 + fbest)`;
- total logarithmic improvement from the first evaluation;
- number of incumbent improvements;
- current stall length;
- recent improvement over the last 10% and 25% of the trace;
- normalized area under the best-so-far curve;
- recent trace slope.

### Operational-state predictors

The trace predictors are augmented with:

- archive size;
- archive objective-spread and quantile summaries relative to the incumbent;
- normalized spatial dispersion and farthest representative distance;
- novelty and visit-count summaries;
- graph edge count and directed density;
- accessibility and edge-improvement summaries;
- proportions of edge source modes;
- phase-resolved evaluation fractions.

All features are computed from the captured state only. No future values enter
the predictors.

## 6. Models

The primary comparison uses linear ridge regression for transparency.
Predictors are standardized using training data only. Ridge penalties are
selected independently for the trace-only and trace-plus-state models from the
fixed grid

```text
1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100.
```

Development selection uses leave-one-instance-out grouped cross-validation.
The primary metric is pooled out-of-fold mean squared error (MSE) for `y`.

A negative-control model uses the same trace predictors and state predictors
permuted within family, dimension and snapshot checkpoint.

Secondary outputs include mean absolute error, pooled R-squared, per-family
MSE, and the binary log-loss for the fixed event `y >= 0.10` when both outcome
classes are present.

## 7. Uncertainty

The development comparison is bootstrapped over instances, preserving all
families, dimensions and checkpoints within a resampled instance. Ten thousand
deterministic bootstrap replicates are used. The reported interval is the
percentile 95% interval for

```text
MSE_trace_only - MSE_trace_plus_state.
```

## 8. Integrity

Every run records:

- implementation version and options hash;
- parent seed, task identity and budget;
- exact snapshot evaluation counts;
- trace and state feature vectors;
- final response;
- archive and graph sizes;
- graph--archive referential-integrity status after snapshot sanitation;
- file hashes and a result manifest.

A snapshot copy is sanitized by removing graph edges whose endpoints are not in
the copied active archive. The number of removed transient edges is reported.
This sanitation is part of observation only and does not alter the running
optimizer.

## 9. Development gate

Track E2 confirmatory evaluation may be opened only if all conditions hold:

1. the trace-plus-state relative MSE improvement is at least 5%;
2. the 95% bootstrap interval for `MSE_trace - MSE_state` has lower bound above
   zero;
3. the real-state improvement exceeds the shuffled-state improvement;
4. at least five of the six task families show non-negative MSE improvement;
5. all integrity checks pass.

If the gate fails, the confirmatory instances remain untouched and the result
is reported as a development-stage negative or inconclusive finding.

## 10. Reserved confirmatory rule

If the gate passes, the selected ridge penalties and feature ordering are
committed in `TRACK_E2_CONFIRMATORY_LOCK.json` before instances 101--108 are
accessed. Confirmatory success requires:

- at least 3% relative MSE improvement;
- a positive instance-bootstrap 95% lower bound;
- real-state improvement greater than shuffled-state improvement.

No feature, penalty grid, response, checkpoint or task partition may be changed
after confirmatory access.
