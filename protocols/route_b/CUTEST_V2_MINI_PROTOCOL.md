# Route B Step B4: CUTEst v2 mini protocol

This is a mini validation for full BasinGraph v2.0.0.

It is not a final manuscript experiment.

## Source list

The mini benchmark uses the pre-registered CUTEst v2 list and selects:

- global_protocol_order 1
- global_protocol_order 21
- global_protocol_order 41

These represent small, medium and large dimension groups.

## Algorithms

- BasinGraph_v2
- CMA_ES
- BIPOP_CMA_ES
- Multi_start_LBFGSB

## Seeds

- 20260619
- 20260620

## Budget

budget = min(3000, max(300, 50 * dimension))

## Acceptance criteria

The mini run passes only if:

1. 24 rows are produced;
2. all runs complete;
3. each algorithm has 6 rows;
4. each dimension group has 8 rows;
5. each BasinGraph_v2 run exhausts its assigned budget;
6. each BasinGraph_v2 run has nonempty archive;
7. BasinGraph_v2 produces at least one transition edge across the mini benchmark;
8. each BasinGraph_v2 JSON contains archive, graph_edges, diagnostics and event_log.

## Manuscript rule

These mini results are engineering validation only and must not be reported as final manuscript results.
