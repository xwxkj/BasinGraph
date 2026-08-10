# Track E pilot Amendment 003: snapshot referential sanitization

Status: **registered after the failed Pilot 3 execution and before rerun**  
Confirmatory Track E evaluations at amendment: **0**

## Trigger

The first Pilot 3 execution completed 35 of 48 engineering blocks before the
job failed. Thirteen blocks raised an integer `KeyError` while constructing the
`EdgeShuffled` control. Inspection identified graph edges whose endpoint was no
longer present in the active archive at the captured mid-run boundary.

The transient condition arises when a newly proposed archive node is immediately
evicted by the capacity rule: the archive-removal callback is applied before the
optimizer records the transition to that newly returned node. Final BasinGraph
results perform a graph cleanup before returning and therefore passed all prior
final-state integrity checks, but an arbitrary mid-run snapshot can expose the
transient edge.

No aggregate Pilot 3 result was produced. The partial execution is retained as
an engineering failure record and is not pooled.

## Registered correction

At snapshot serialization only, Pilot 3 now retains graph edges whose source
and target identifiers are both present in the captured active archive. The
number of removed transient edges is recorded for every block. This is the same
referential condition enforced on returned BasinGraph results; it does not
change archive contents, objective histories, optimizer decisions already made,
or any file under `basingraph_v2/`.

The task matrix, continuation arms, seeds, budgets, endpoints, statistical
procedures and success gate remain unchanged.
