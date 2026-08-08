# BasinGraph final consistency contract

## Non-negotiable rule

No manuscript claim, figure or table may be attributed to the selected
BasinGraph candidate unless it is supported by the byte-frozen
`basingraph_v2` implementation, options hash `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`, and the frozen
prospective COCO/BBOB or CUTEst evidence.

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

## Final evidence boundary

Primary external evidence consists only of the prospective COCO/BBOB and
CUTEst holdouts. Development records, rejected candidate records and legacy
archives remain historical/audit material and must not be presented as final
performance evidence.
