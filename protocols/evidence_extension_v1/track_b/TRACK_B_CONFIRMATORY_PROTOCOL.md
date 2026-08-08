# B21 Track B confirmatory modern-baseline protocol

Status: **frozen and amended before confirmatory evaluation**  
Amendment 001: jSO initial population corrected to `round(25 ln(D) sqrt(D))`; confirmatory objective evaluations before the amendment: **0**.  
Protocol date: 2026-08-08  
Track A merge commit: `c955bdba08a6a50316248b6c1dd9ff61f1a4840b`  
Result-bearing candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`

## Purpose

Track B tests whether the immutable BasinGraph candidate remains competitive
against modern and established strong bound-constrained optimizers on a new,
disjoint official COCO/BBOB partition. It is not a retrospective enlargement
of the original holdout or Track A.

## Confirmatory matrix

- suite: official noiseless `bbob`;
- functions: 1–24;
- dimensions: 5 and 20;
- actual instances: 21–30;
- budget: `1,000d` objective evaluations per run;
- algorithms: eight;
- problems per algorithm: 480;
- total runs: 3,840.

The paired seed is

```text
seed = 20260808 + 100000*function + 1000*dimension + instance
```

DIRECT-L is deterministic for fixed objective and bounds; the paired seed is
still recorded but is not consumed by that implementation.

## Frozen algorithms

1. BasinGraph;
2. CMA-ES;
3. BIPOP-CMA-ES;
4. corrected L-SHADE 1.0.1 port;
5. jSO port;
6. L-SRTDE port;
7. DIRECT-L;
8. multi-start L-BFGS-B.

No algorithm-specific tuning is allowed after protocol freeze. All objective
calls pass through a strict external counter and must equal the prescribed
budget. Bounds, objective wrappers, problem IDs and seeds are shared.

## Implementation provenance

- BasinGraph: immutable `basingraph_v2` implementation and frozen options hash;
- CMA-ES and BIPOP-CMA-ES: public pycma interfaces;
- L-SHADE 1.0.1: transparent Python port based on the corrected author release;
- jSO: transparent Python port of the published CEC-2017 algorithm, with initial population `round(25 ln(D) sqrt(D))`;
- L-SRTDE: transparent Python port of the public CEC-2024 C++ core, generalized
  only to arbitrary finite per-coordinate bounds and an external objective;
- DIRECT-L: `scipy.optimize.direct(..., locally_biased=True)`;
- multi-start L-BFGS-B: repeated SciPy bounded minimizations until budget
  exhaustion.

The three ports are frozen research implementations, not claimed to be
byte-identical official executables. Full parameter values and known deviations
are recorded in the provenance and parameter tables.

## Immutable BasinGraph identity

- internal implementation version: `2.0.0-rc1`;
- package: `basingraph_v2/`;
- entry point: `minimize_basingraph_v2`;
- Full options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- selected-candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`.

No file under `basingraph_v2/` may change during Track B.

## Development-only smoke

The infrastructure smoke uses functions 1, 6, 10, 15 and 20; dimensions 5
and 20; actual instance 1; and budget `100d`. Smoke is an engineering gate and
cannot enter confirmatory statistics.

## Execution and recovery

The experiment is split into 16 algorithm-dimension shards. Each shard owns a
separate COCO observer folder. A completed shard is immutable. If interrupted,
the partial attempt and observer data are retained and the complete shard is
rerun under the same frozen configuration in a new attempt folder.

## Recorded evidence

Every run records internal and observer evaluation counts, final value,
final-target status, fixed-budget checkpoints, wall time, implementation
identity and integrity hashes. BasinGraph additionally records phase counts,
archive size and graph integrity. Full traces are retained for
`f={1,8,15,20,24}, instance=21` in both dimensions. Official COCO observer logs
are retained for every run.

## Non-negotiable rules

- no objective evaluation on instances 21–30 before protocol and source freeze;
- no tuning or code replacement after confirmatory access begins;
- no deletion of failures or unfavorable results;
- no pooling with instances 4–15 or 16–20;
- no change to candidate, baseline parameters, seed formula, budget or endpoint;
- any amendment must be timestamped before further confirmatory evaluations.
