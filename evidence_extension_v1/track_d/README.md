# B21 Track D: large-scale scalability and overhead

Track D evaluates the immutable BasinGraph candidate on the official
`bbob-largescale` suite and measures optimizer-side overhead separately from
observed objective-call time.

## Registered performance matrix

- functions: 1–24;
- dimensions: 40, 80, 160 and 320;
- actual instances: 16–18;
- budget: `200d`;
- algorithms: BasinGraph, CMA-ES, corrected L-SHADE 1.0.1 port, L-SRTDE
  transparent port and multi-start L-BFGS-B;
- total confirmatory runs: 1,440.

The performance matrix is independent of the earlier standard-BBOB partitions
because it uses the separate `bbob-largescale` suite.

## Registered overhead probe

The primary overhead probe uses development instance 1, functions 1, 10 and
20, all four dimensions, budget `50d`, and exactly one worker. It records:

- total wall time;
- cumulative time inside the observed objective call;
- optimizer-side time obtained by subtraction;
- optimizer-side microseconds per evaluation;
- peak resident memory per algorithm–dimension shard;
- serialized result size;
- BasinGraph archive, graph and phase scaling.

The parallel confirmatory run also records timing, but those values are
descriptive only because shards may execute concurrently.

## Mac execution

First run:

```bash
bash START_B21_TRACK_D_SMOKE.command
```

After the returned smoke package is audited, run:

```bash
bash START_B21_TRACK_D_CONFIRMATORY.command
```

The second command executes the sequential overhead probe and the 1,440-run
large-scale matrix. Completed packages are placed in
`~/Downloads/B21_RESULTS_READY/` and Finder opens the exact folder.
