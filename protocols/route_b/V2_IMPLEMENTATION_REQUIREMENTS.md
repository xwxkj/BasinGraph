# BasinGraph v2.0.0 implementation requirements

## Required algorithmic objects

The optimizer must explicitly maintain:

- BasinNode
- BasinArchive
- TransitionEdge
- BasinTransitionGraph
- GeometryDiagnostics
- BasinGraphResult

## Required outputs

Each run must return:

- xbest
- fbest
- nfe
- history
- archive
- graph
- diagnostics
- event_log

## Required modules

- anchors and initial design
- basin-node creation and merge rule
- local contraction
- multi-bracket coordinate-Brent sweep
- far-basin exploration
- transition-edge update
- archive fallback
- graph-aware budget completion
- final polishing
- strict function-evaluation accounting

## Required tests

- budget accounting test
- basin-node merge test
- transition-edge test
- graph output serialization test
- COCO smoke test
- CUTEst smoke test
