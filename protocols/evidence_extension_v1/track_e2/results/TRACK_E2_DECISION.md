# Track E2 predictive-information decision

Status: **audited on independent registered partitions**  
Immutable optimizer: BasinGraph `2.0.0-rc1`  
Options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## Decision

**Predictive-information claim supported for the registered binary future-progress event, with alignment specificity confirmed. The original continuous-MSE development gate was not met and remains closed. No continuation-performance claim is supported.**

## Development continuous endpoint

Track E2 development used 96 parent runs and 288 operational snapshots across
six deterministic families, dimensions 10 and 20, and instances 1--8.

The trace-plus-state ridge model reduced pooled nested-CV MSE from
`0.0017338054` to `0.0016827088`, a relative reduction of `2.95%`. The
instance-bootstrap 95% interval for `MSE_trace - MSE_state` was positive, and
five of six families had non-negative MSE changes. The pre-registered gate
required at least a 5% relative reduction. It was therefore **not passed**, and
no continuous-endpoint confirmatory cycle was opened.

## Track E2-B independent event confirmation

The binary event `future_log_improvement >= 0.10` had been specified before
development execution. Models, feature order, penalties and standardization
were frozen before instances 101--124 were accessed.

Across 288 parent runs and 864 snapshots:

- event positives: 17;
- trace-only log loss: `0.06664379`;
- trace-plus-state log loss: `0.06118155`;
- relative reduction: `8.20%`;
- bootstrap 95% interval for `loss_trace - loss_state`:
  `[0.0004013, 0.0105914]`.

All registered Track E2-B conditions passed. A single cyclic-shuffle control
also improved on trace-only and was close to the real-state model, so
alignment-specific attribution required a second untouched partition.

## Track E2-C alignment-specificity confirmation

Seven cyclic-shuffle control models were trained on development data and frozen
before instances 201--224 were accessed. Their probabilities formed a
permutation-ensemble negative control.

Across another 288 parent runs and 864 snapshots:

- event positives: 36;
- trace-only log loss: `0.11429563`;
- correctly aligned state log loss: `0.09634053`;
- permutation-ensemble log loss: `0.10624575`;
- aligned-state reduction relative to trace: `15.71%`;
- aligned-state reduction relative to the permutation ensemble: `9.32%`;
- bootstrap interval for `loss_trace - loss_aligned`:
  `[0.0098991, 0.0261951]`;
- bootstrap interval for `loss_permutation - loss_aligned`:
  `[0.0010778, 0.0192542]`.

Every pre-registered Track E2-C condition passed. The result supports the
statement that alignment between a trace and its own operational state carries
out-of-sample information about the registered future-progress event beyond
both the scalar trace and generic state-feature distributions.

## Integrity and limitations

All runs used the immutable optimizer and exact options hash. Snapshot copies
were sanitized by removing transient graph endpoints not present in the copied
active archive; the running optimizer was not changed. The number of such
transient endpoints is reported. This does not alter the established integrity
of returned final states, but it prevents a claim of unsanitized arbitrary-time
checkpoint integrity.

The result supports:

> The bounded operational state retains alignment-specific predictive
> information about a registered future finite-budget progress event beyond
> that contained in the best-so-far trace.

It does not support:

- a claim that retaining state improves continuation performance;
- a claim that the state is a sufficient statistic for optimal continuation;
- a claim that every task family benefits equally;
- arbitrary mid-run exact replay;
- reconstructed landscape or basin topology.
