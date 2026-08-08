# B21 Track A confirmatory mechanism-ablation protocol

Status: **frozen before confirmatory evaluation**  
Protocol date: 2026-08-07  
Step 0 merge commit: `0457d98f3f7041bee491f3ea9998db5e8c656dba`  
Result-bearing candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`

## Purpose

Track A tests whether the frozen BasinGraph mechanisms alter target-runtime
performance on a new, disjoint COCO/BBOB evidence extension. The experiment is
not a retrospective addition to the original holdout. The original
COCO/BBOB instances 4–15 remain immutable and are not rerun, replaced or pooled
with this extension.

## Confirmatory matrix

- suite: official noiseless `bbob`;
- functions: 1–24;
- dimensions: 5 and 20;
- actual instances: 16–20;
- budget: `1,000d` objective evaluations per run;
- variants: eight;
- total problems per variant: 240;
- total runs: 1,920.

The random seed is paired across variants:

```text
seed = 20260808 + 100000*function + 1000*dimension + instance
```

## Frozen variants

1. Full
2. NoGraphGuidance
3. SingleBracket
4. NoFarBasin
5. NoGeometryController
6. NoArchiveFallback
7. NoFinalPolish
8. NoCenterLocal

No variant-specific retuning is allowed. Disabled phase allocations pass to
`budget_completion` exactly as implemented by the frozen candidate.

## Immutable implementation

- internal implementation version: `2.0.0-rc1`;
- package: `basingraph_v2/`;
- entry point: `minimize_basingraph_v2`;
- Full options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- archive capacity: 80;
- selected-candidate tag: `route-b-v2.0.0-rc1-selected-final-candidate`;
- selected-candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`.

No file under `basingraph_v2/` may be changed during Track A.

## Development-only smoke

The infrastructure smoke uses functions 1, 6, 10, 15 and 20, dimensions 5 and
20, actual instance 1, and a budget of `100d`. Smoke outputs are engineering
records only and cannot be pooled with confirmatory evidence.

## Execution and recovery

The experiment is split into 16 independent variant-dimension shards. Each
shard owns a separate COCO observer folder. A completed shard is immutable.
If a shard is interrupted, its partial result directory and observer data are
retained as an incomplete attempt; the entire shard is rerun under the same
frozen configuration in a new attempt folder. No completed problem is silently
overwritten.

## Recorded evidence

Every run records exact internal and observer evaluation counts, phase
allocations, final value, official final-target status, fixed-budget
checkpoints, archive size, graph size, diagnostics, event counts and integrity
hashes. Full traces are retained for the predeclared subset
`f={1,8,15,20,24}, instance=16` in both dimensions. Official COCO observer
logs are retained for all runs.

## Non-negotiable rules

- no access to instances 16–20 before protocol and source identity are frozen;
- no tuning after confirmatory evaluation begins;
- no deletion of failures or unfavorable results;
- no pooling with original instances 4–15;
- no change to the candidate, seed formula, budget, variants or endpoints;
- any amendment must be timestamped, justified and made before further
  confirmatory evaluations.
