# Track E pilot Amendment 001: exact mid-run snapshot

Status: **registered after engineering pilot 1 and before engineering pilot 2**  
Confirmatory Track E evaluations at amendment: **0**

## Trigger

Pilot 1 used a completed `30d` BasinGraph run as the common prefix. All five
continuation arms then produced identical best-so-far histories. Inspection
showed that this prefix was not equivalent to a `30d` checkpoint of a longer
run: all seven BasinGraph phases had been compressed into the shorter total
budget, including final polishing and completion. The common incumbent was
therefore already strongly locally polished, and the continuation was dominated
by the same incumbent-based stall polish in every arm.

Pilot 1 remains an immutable negative engineering result. It is not discarded,
pooled or reinterpreted as confirmatory evidence.

## Registered correction

Pilot 2 captures an exact operational snapshot at `40d` within a run whose
phase limits were computed for a total budget of `140d`. The capture occurs at
the next objective-call boundary after evaluation `40d`, when all updates from
the preceding evaluation are complete. The runner then prevents further
objective calls. Each continuation arm receives the same remaining budget of
`100d` and the same paired continuation seed.

The snapshot hook is implemented only in `evidence_extension_v1/track_e/` by
temporarily substituting capturing subclasses for the ledger, archive and graph
constructors used by the immutable optimizer. No file under `basingraph_v2/` is
modified, and the frozen options hash is unchanged.

## Unchanged items

The six task families, dimensions, instances, continuation arms, target ratios,
primary contrast, statistical procedures and success gate are unchanged.
Pilot 2 remains engineering-only. Any positive result must still be followed by
a separately frozen, disjoint confirmatory experiment.
