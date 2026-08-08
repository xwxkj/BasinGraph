# Prospective COCO/BBOB holdout execution

## Candidate identity

- implementation: `2.0.0-rc1`;
- options hash:
  `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- selected-candidate tag:
  `route-b-v2.0.0-rc1-selected-final-candidate`;
- runner-freeze tag:
  `route-b-v2.0.0-rc1-holdout-runner-freeze-v2`.

## Holdout partition

- suite: official noiseless BBOB;
- functions: 1-24;
- dimensions: 2, 3, 5, 10 and 20;
- instances: 4-15;
- budget: 1000d;
- seven algorithms.

## Pairing

Each problem uses the same deterministic seed for all algorithms:

`20260620 + 100000*f + 1000*d + instance`.

## Expected size

- 1,440 unique holdout problems;
- 10,080 algorithm-problem rows;
- 1,440 BasinGraph detailed records;
- 168 `.info` files;
- 840 `.dat` files.

## Interpretation order

1. complete all seven algorithms;
2. validate integrity;
3. complete official cocopp processing;
4. only then inspect or summarize performance;
5. report holdout separately from development;
6. any pooled 1-15 result must be explicitly secondary.

No algorithm modification is permitted after opening this holdout.

## COCO instance-construction semantics

Actual BBOB instance identifiers 4-15 are supplied through the suite-instance argument:

`cocoex.Suite("bbob", "instances: 4-15", suite_options)`.

The `instance_indices` suite option is not used because it filters ordinal positions in an instantiated suite rather than actual instance numbers. A zero-evaluation preflight verifies 1,440 IDs with instances exactly 4-15 before an observer is attached.
