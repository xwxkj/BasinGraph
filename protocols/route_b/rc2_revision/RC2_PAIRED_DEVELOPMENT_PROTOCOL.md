# rc1 versus rc2 paired COCO development protocol

## Scope

- official noiseless BBOB;
- functions 1-24;
- dimensions 2, 5 and 10;
- development instances 1-3 only;
- budget 1000d;
- exact problem-specific seeds reused from rc1.

## rc2 identity

- implementation: `2.0.0-rc2`;
- options hash:
  `15fe9bcbf8e87aabe4767f811524c00f67b74e3ebfa31fa81cdf6f461cbfeb08`;
- source code must match tag `route-b-v2.0.0-rc2-codefreeze`;
- machine spec must match tag `route-b-v2.0.0-rc2-machinefreeze`.

## Baselines

The unchanged six baseline records and COCO observer logs are reused from the
validated rc1 development run. Reuse is valid only because:

1. the problem set and problem-specific seeds are identical;
2. baseline source provenance is frozen;
3. the rc1 validation established exact internal/observer FE agreement;
4. rc2 is the only changed algorithm.

## Decision

Record-level integrity, mechanism, fixed-budget rank and hardest-target gates
are evaluated automatically. Official cocopp ECDF gates remain pending until
the combined rc1/rc2 output is reviewed.

## Holdout rule

Do not instantiate, inspect or summarize COCO instances 4-15 or the
prospective CUTEst holdout.
