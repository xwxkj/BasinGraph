# B21 Track D large-scale scalability and overhead protocol

Status: **frozen before Track D confirmatory evaluation**  
Protocol date: 2026-08-08  
Track B merge commit: `e567b085b5ec6621a57fc3f7bd66d758ea10ddeb`  
Track B decision commit: `6e92e70e27e433d5ad951540218cc7b6340f36b3`  
Result-bearing BasinGraph candidate: `adbc0ecdf1153044188f0508321c47001ad9bdb0`

## Purpose

Track D tests whether the immutable BasinGraph candidate remains operationally
tractable as dimension increases and whether its bounded archive and observed-
transition graph create controlled rather than unbounded optimizer-side cost.
The track separates two questions:

1. performance scaling on the official `bbob-largescale` suite; and
2. optimizer-side timing and memory scaling under a sequential registered
   overhead probe.

No result from Track D may be used to retune BasinGraph or any comparator.

## Confirmatory performance matrix

- suite: official noiseless `bbob-largescale`;
- functions: 1–24;
- dimensions: 40, 80, 160 and 320;
- actual instances: 16–18;
- budget: `200d` objective evaluations per run;
- algorithms: five;
- problems per algorithm: 288;
- total confirmatory runs: 1,440.

The paired seed is

```text
seed = 20260808 + 100000*function + 1000*dimension + instance
```

The large-scale suite is separate from the standard `bbob` suite used in the
original holdout and Tracks A and B. Instance 1 is reserved for Track D
engineering and overhead work; instances 16–18 are frozen for the Track D
confirmatory matrix.

## Registered overhead probe

The primary overhead probe is run before the performance matrix and uses:

- suite: `bbob-largescale`;
- functions: 1, 10 and 20;
- dimensions: 40, 80, 160 and 320;
- actual instance: 1;
- budget: `50d`;
- algorithms: the same five algorithms;
- workers: exactly one;
- total runs: 60.

For every run, the wrapper records total wall time and cumulative time inside
the observed objective call. Optimizer-side time is defined before evaluation
as

```text
optimizer-side time = total wall time - observed objective-call time.
```

This subtraction includes Python control flow, candidate generation, state
updates, numerical linear algebra and result construction inside the optimizer,
but excludes the timed COCO objective and observer call. Timing from the
parallel confirmatory matrix is secondary and descriptive only.

## Frozen algorithms

1. BasinGraph, immutable `basingraph_v2` Full candidate;
2. CMA-ES through pycma;
3. corrected L-SHADE 1.0.1 source-aligned transparent Python port;
4. L-SRTDE transparent source-guided Python port with registered deviations;
5. multi-start L-BFGS-B through SciPy.

The comparator implementations are imported unchanged from the frozen Track B
source. No Track D-specific algorithm parameter is introduced.

## Recorded evidence

Every run records:

- internal, timed-objective and COCO observer evaluation counts;
- best value and registered fixed-budget checkpoints;
- total, objective-call and optimizer-side wall time;
- optimizer-side microseconds per evaluation;
- process peak resident memory before result serialization;
- serialized result-core, history and estimated full JSON sizes;
- implementation identity, message and algorithm metadata;
- for BasinGraph, phase counts, archive-node count, graph-edge count and graph
  referential integrity.

Every algorithm–dimension combination owns an independent process and observer
folder. Completed shards are immutable. Interrupted attempts are retained and
the complete shard is rerun with the same frozen configuration.

## Primary endpoints

The primary scalability endpoints are:

- official cocopp target-runtime/ECDF evidence on `bbob-largescale`;
- registered fixed-budget ranks at 1, 3, 10, 30, 100 and 200 evaluations per
  dimension;
- sequential optimizer-side microseconds per evaluation;
- sequential process peak resident memory;
- empirical log2–log2 scaling slopes across dimensions;
- BasinGraph archive-node and graph-edge growth.

Final objective-value rank and parallel wall-clock time are secondary.

## Interpretation boundary

Track D supports a controlled-scalability claim only if BasinGraph completes
all registered dimensions without integrity failure and its archive, graph,
memory and optimizer-side cost remain bounded and interpretable. A favorable
result does not establish universal large-scale superiority. A failure or sharp
cost escalation is retained and reported.
