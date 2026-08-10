# Track E2-B: independent confirmation of event-prediction information

Status: **frozen after Track E2 development and before any Track E2-B confirmatory objective evaluation**  
Confirmatory objective evaluations at freeze: **0**

## 1. Origin and separation from Track E2 continuous endpoint

The original Track E2 development gate for continuous future log-improvement
was not met: the trace-plus-state model reduced pooled nested-CV MSE by 2.95%,
below the pre-registered 5% gate, although its instance-bootstrap interval was
positive and five of six families showed non-negative improvement. No
confirmatory cycle is opened for that continuous endpoint.

Track E2-B is a distinct confirmatory study of a secondary endpoint that was
specified before Track E2 development execution: whether future logarithmic
improvement is at least 0.10. The development log loss was 0.12625 for the
trace-only model and 0.11335 for the trace-plus-state model. The event threshold,
feature definitions and model class are unchanged. This protocol freezes an
independent test rather than lowering the original continuous-endpoint gate.

## 2. Confirmatory partition

- task families: the six frozen Track E2 families;
- dimensions: 10 and 20;
- confirmatory instances: 101--124;
- parent budget: `250d`;
- snapshots: first graph update at or after `80d`, `140d` and `200d`;
- expected parent runs: 288;
- expected snapshot records: 864.

No confirmatory instance was used to choose features, penalties or model
parameters.

## 3. Locked models

Three logistic-ridge models are fit once on all Track E2 development records:

1. trace-only;
2. trace plus operational state;
3. trace plus cyclically shuffled operational state.

The penalty is 0.01 for all three models. Standardization means, scales,
intercepts and coefficients are frozen in
`TRACK_E2B_CONFIRMATORY_LOCK.json`. Confirmatory data are never used to refit or
recalibrate the models.

## 4. Primary endpoint

The primary endpoint is the paired difference in binary log loss:

```text
log_loss_trace_only - log_loss_trace_plus_state.
```

The event label is

```text
future_log_improvement >= 0.10.
```

The negative control replaces each confirmatory record's state features by a
cyclic permutation across instances within the same family, dimension and
snapshot checkpoint, while retaining the locked shuffled-state model.

## 5. Uncertainty and success rule

Ten thousand deterministic bootstrap replicates resample confirmatory
instances while preserving all families, dimensions and checkpoints belonging
to each sampled instance.

Track E2-B succeeds only if:

1. trace-plus-state relative log-loss improvement is at least 5%;
2. the 95% instance-bootstrap lower bound for
   `log_loss_trace - log_loss_state` is above zero;
3. the real-state log-loss improvement is larger than the shuffled-state
   improvement;
4. both event classes occur in the confirmatory data;
5. all optimizer identity, accounting and snapshot-integrity checks pass.

All favorable, null and unfavorable results are retained. A successful result
supports incremental predictive information, not continuation-performance
improvement or optimal-state sufficiency.
