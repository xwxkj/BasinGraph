
# BasinGraph v2.0.0 consistency contract

## Non-negotiable rule

No manuscript claim, figure, table, COCO result, CUTEst result, AppliedMath result
or ablation result may be reported as BasinGraph v2.0.0 unless it was generated
by the `basingraph_v2` implementation on this branch or by a later tagged v2
successor.

## Required public objects

- BasinNode
- BasinArchive
- TransitionEdge
- BasinTransitionGraph
- GeometryDiagnostics
- BasinGraphResult

## Required optimizer outputs

- xbest
- fbest
- nfe
- history
- archive
- graph_edges
- diagnostics
- event_log
- message

## Required result regeneration

The following must be rerun after v2.0.0 implementation freeze:

1. official COCO/BBOB;
2. pre-registered CUTEst;
3. AppliedMath suite;
4. ablation suite;
5. MATLAB consistency tests, if MATLAB is retained as an independent validation.

Earlier v1.0.0 results and DOI records remain historical/development evidence only.
