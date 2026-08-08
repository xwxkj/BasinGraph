# Track D development-only smoke protocol

Status: **engineering gate only; excluded from scientific evidence**

The Track D smoke uses:

- suite: `bbob-largescale`;
- functions: 1, 10 and 20;
- dimensions: 40 and 80;
- actual instance: 1;
- budget: `50d`;
- five frozen algorithms;
- total runs: 30.

The smoke validates suite enumeration, source identity, exact internal/timed-
objective/observer accounting, result serialization, memory measurement,
BasinGraph phase/archive/graph integrity, shard recovery, official cocopp and
Mac packaging. It cannot enter confirmatory statistics.

The confirmatory command is locked until a local smoke gate exists for the exact
source commit and source-identity digest.
