# Track E2-C: alignment-specificity test for operational-state prediction

Status: **frozen after Track E2-B and before any Track E2-C objective evaluation**  
Track E2-C objective evaluations at freeze: **0**

## 1. Motivation

Track E2-B confirmed that the locked trace-plus-state model improved event
log loss over the trace-only model on independent instances. Its real-state
improvement was only slightly larger than the single cyclic-shuffle control,
and a post-result instance bootstrap did not separate those two models. Track
E2-C therefore asks a narrower question on a second untouched partition:

> Is predictive performance specific to the alignment between each trace and
> its own operational state, rather than to generic state-feature distributions
> or added model capacity?

Track E2-C does not revise the Track E2-B endpoint or reuse its instances.

## 2. Partition and parent runs

- families: the six frozen Track E2 families;
- dimensions: 10 and 20;
- instances: 201--224;
- parent budget: `250d`;
- snapshot checkpoints: `80d`, `140d`, `200d`;
- expected parent runs: 288;
- expected records: 864.

## 3. Locked models

The real-state logistic-ridge model is identical to the Track E2-B locked
model. Seven negative-control models are trained on the complete development
set after cyclically shifting state features by 1, 2, ..., 7 instance positions
within each family, dimension and checkpoint stratum. Each control model uses
its own development-selected penalty from the original fixed grid. Model
means, scales, coefficients, feature order and penalties are committed before
Track E2-C access.

On confirmatory data, each control model receives the corresponding cyclically
shifted state features. Their predicted probabilities are averaged to form a
permutation-ensemble negative control.

## 4. Endpoints

The event remains

```text
future_log_improvement >= 0.10.
```

The primary specificity endpoint is

```text
log_loss_permutation_ensemble - log_loss_real_state.
```

The supporting trace endpoint is

```text
log_loss_trace_only - log_loss_real_state.
```

Ten thousand deterministic bootstrap replicates resample instances and retain
all families, dimensions and checkpoints belonging to each sampled instance.

## 5. Success rule

Alignment-specific predictive information is supported only if:

1. real state improves log loss over the permutation ensemble by at least 2%;
2. the 95% instance-bootstrap lower bound for
   `loss_permutation - loss_real` is above zero;
3. real state improves over trace-only by at least 5%;
4. the 95% lower bound for `loss_trace - loss_real` is above zero;
5. both event classes are present;
6. all identity, accounting and snapshot-integrity checks pass.

If this gate fails, the correct conclusion is that incremental predictive value
was confirmed relative to the trace in Track E2-B, but state-to-trace alignment
specificity was not established. No continuation-performance claim follows in
either case.
