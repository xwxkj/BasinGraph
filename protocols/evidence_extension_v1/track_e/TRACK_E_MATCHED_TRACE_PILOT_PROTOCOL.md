# Track E matched-trace operational-state utility pilot

Status: **registered engineering pilot before evaluation**  
Track E objective evaluations at registration: **0**

## Question

Does the bounded BasinGraph operational state contain continuation-relevant
information that is absent from an incumbent and its best-so-far trace?

## Design

For every deterministic task block, the immutable BasinGraph `2.0.0-rc1`
candidate is first run once for a prefix budget of `30d`. The resulting
best-so-far trace, incumbent, archive and observed-transition graph are then
frozen. Every continuation arm therefore starts from exactly the same prefix
objective history and incumbent.

A common continuation kernel, derived from the frozen BasinGraph
`budget_completion` policy, receives one of five state views:

1. `FullState`: incumbent, full active archive and correctly attached graph;
2. `ArchiveOnly`: incumbent and full archive, with graph removed;
3. `EdgeShuffled`: full archive and a topology-preserving permutation of graph
   node labels relative to archive identities;
4. `TraceOnly`: incumbent and prefix best-so-far trace, represented to the
   continuation kernel by the incumbent alone;
5. `ColdRestart`: incumbent retained for accounting, but no operational nodes
   or graph are supplied to the sampling policy.

Every arm receives the same continuation budget (`100d`) and the same paired
continuation seed. State corruption uses a separate deterministic control seed
and therefore does not consume the continuation random stream.

## Pilot task matrix

The engineering pilot uses six deterministic, non-negative synthetic families
with known optimum zero:

- rotated Rastrigin;
- rotated Ackley;
- rotated Griewank;
- Gallagher-style basin mixture;
- double-funnel landscape;
- Lunacek-style bi-Rastrigin landscape.

Dimensions are 10 and 20. Four independently generated instances are used per
family, giving 48 matched prefix blocks and 240 continuation runs.

## Endpoints

For each block, continuation targets are fixed fractions of the prefix gap:

```text
0.3, 0.1, 0.03, 0.01, 0.003
```

The primary endpoint is the final fraction of these targets reached within the
continuation budget. Secondary endpoints are:

- target fraction at `1d`, `3d`, `10d`, `30d` and `100d`;
- log10 gap reduction from prefix to continuation end;
- evaluations to each target;
- final gap;
- continuation archive and graph sizes.

The primary confirmatory contrast within this pilot is `FullState` versus
`TraceOnly`. Paired Wilcoxon inference, paired rank-biserial effect and a
10,000-resample paired bootstrap confidence interval are reported.

## Pilot success gate

The pilot is classified as positive only when all conditions hold:

1. mean paired final target-fraction difference
   `FullState - TraceOnly >= 0.03`;
2. two-sided paired Wilcoxon `p < 0.05`;
3. the 95% paired bootstrap interval has a positive lower bound;
4. the mean difference is positive in at least four of six task families;
5. all integrity checks, exact evaluation counts and matched-prefix hashes pass.

A positive pilot permits—but does not itself establish—a manuscript claim. It
triggers a separately frozen confirmatory experiment on disjoint official or
scientific tasks. A null or negative pilot prevents any claim that the retained
state improves continuation performance; the paper must then remain an
`auditability` contribution only.

## Integrity boundary

No file under `basingraph_v2/` is modified. This pilot is engineering-only and
is not pooled with Tracks A–D/C. Favorable, null and unfavorable results are
retained.
