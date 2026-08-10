# Track E matched-trace engineering-pilot decision

Status: **completed; pilot gate not met**  
Decision: **do not open a confirmatory state-utility cycle and do not claim that retained operational state improves continuation performance over a best-so-far trace.**

## Frozen identity

All pilots used the immutable BasinGraph implementation `2.0.0-rc1`, options
hash
`031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`,
and unchanged files under `basingraph_v2/`.

## Pilot 1

Pilot 1 used a completed `30d` run as the common prefix and a `100d`
continuation. Across 48 matched blocks and five state views, `FullState` and
`TraceOnly` had zero mean target-fraction difference, Wilcoxon `p = 1`, a
bootstrap interval `[0, 0]`, and 48 ties.

Diagnosis: a completed short run compresses all frozen phases, including final
polishing and completion, into the prefix. The shared incumbent-based stall
polish then dominated every continuation arm. The null result is retained.

## Pilot 2

Pilot 2 captured an exact `40d` checkpoint inside a parent run planned for
`140d`. The primary result was again zero difference, Wilcoxon `p = 1`,
bootstrap interval `[0, 0]`, and 48 ties.

Diagnosis: every snapshot occurred inside the coordinate-sweep phase before
completed coordinate candidates had updated the graph; all prefix graphs had
zero edges. The common stall-polish branch again produced identical
best-so-far histories. The null result is retained.

## Pilot 3

Pilot 3 captured the first objective-call boundary after coordinate-sweep
completion and used a common `100d` state-probe sampling kernel without a local
solver. Snapshot graphs were restricted to endpoints present in the active
archive; the number of transient edges removed at serialization was recorded.

The five arms evaluated different point sequences in all 48 blocks: 46 blocks
had five distinct sequence hashes and two blocks had four. Prefix archives
contained 56–80 nodes and sanitized graphs contained 8–57 edges. Nevertheless,
no arm improved its shared prefix incumbent in any block. All final target
fractions were zero, and `FullState - TraceOnly` was again exactly zero with
Wilcoxon `p = 1`, bootstrap interval `[0, 0]`, and 48 ties.

Thus the test established that different retained states changed evaluated
trajectories, but it did not establish continuation-value improvement under the
registered budget and tasks.

## Additional implementation finding

The first Pilot 3 execution exposed a transient mid-run graph edge whose target
had just been evicted from the capacity-limited archive. Final BasinGraph
results already clean graph endpoints before return and all prior final-state
integrity audits passed. Arbitrary mid-run serialization, however, requires an
explicit referential cleanup. This finding prevents an unqualified exact-replay
or arbitrary-snapshot claim for the current implementation.

## Claim boundary

The manuscript may claim that the operational object:

- preserves information absent from a scalar best-so-far trace;
- changes subsequent sampled trajectories when supplied to a state-conditioned
  policy;
- enables bounded inspection, accounting and integrity checks at returned
  states.

The manuscript must not claim, on the basis of Track E, that:

- Full operational state improves continuation target attainment over
  `TraceOnly`;
- graph attachment provides a causal performance gain;
- arbitrary mid-run states are directly serializable or exactly replayable
  without cleanup;
- the null engineering pilots are confirmatory evidence.

## Next action

Do not proceed to the proposed confirmatory matched-trace experiment or to
follow-on steps whose premise is a positive continuation-utility result. Retain
the current auditability-focused manuscript positioning. Theory may still
formalize best-so-far trace non-identifiability and bounded returned-state
storage, but not continuation sufficiency or exact replay for the present
implementation.
