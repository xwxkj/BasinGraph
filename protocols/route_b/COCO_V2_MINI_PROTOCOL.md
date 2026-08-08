# Route B Step B3: COCO v2 mini protocol

This protocol is a smoke-level official COCO/BBOB benchmark for BasinGraph v2.0.0.

It is not a final manuscript experiment.

## Suite

- COCO suite: bbob noiseless
- dimensions: 2
- functions: 1-3
- instances: 1-3
- budget: 1000 * dimension

## Algorithms

- BasinGraph_v2
- CMA_ES
- BIPOP_CMA_ES

## Required BasinGraph_v2 outputs

Each BasinGraph_v2 run must save a JSON file containing:

- xbest
- fbest
- nfe
- history
- archive
- graph_edges
- diagnostics
- event_log
- message

## Acceptance criteria

The mini run passes only if:

1. 27 raw result rows are produced;
2. each algorithm has 9 rows;
3. official COCO .info and .dat files are produced;
4. each BasinGraph_v2 row exhausts the 1000d budget;
5. each BasinGraph_v2 row has nonempty archive;
6. BasinGraph_v2 produces at least one transition edge across the mini suite;
7. every BasinGraph_v2 JSON contains archive, graph_edges, diagnostics and event_log.

## Manuscript rule

These mini results are engineering validation only and must not be reported as final
paper results.
