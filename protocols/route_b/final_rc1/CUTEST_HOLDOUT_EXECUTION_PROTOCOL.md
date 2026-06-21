# Prospective CUTEst holdout execution

## Candidate identity

- selected implementation: `2.0.0-rc1`;
- options hash:
  `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- code-freeze tag:
  `route-b-v2.0.0-rc1-codefreeze`;
- final-candidate tag:
  `route-b-v2.0.0-rc1-selected-final-candidate`;
- CUTEst runner-freeze tag:
  `route-b-v2.0.0-rc1-cutest-holdout-runner-freeze`.

## Frozen problem set

`protocols/route_b/formal_v2_protocol/CUTEST_V2_PROSPECTIVE_HOLDOUT_24.csv`

- 24 instances;
- 11 small, 7 medium and 6 large;
- exact instance overlap with the 50-problem development set: zero;
- selected without optimizer-performance information.

## Algorithms and paired seeds

Seven algorithms:

- BasinGraph 2.0.0-rc1;
- CMA-ES;
- BIPOP-CMA-ES;
- Differential Evolution;
- Multi-start L-BFGS-B;
- Latin Hypercube Sampling;
- Random Search.

Thirty paired seeds are used. For holdout order `h` and zero-based seed index
`s`:

`seed = 20260621 + 100000*h + s`.

The same seed is supplied to all seven algorithms in a problem-seed job.

## Budget

For dimension `n`:

`budget = min(20000, max(1000, 50*n))`.

Early algorithm termination is retained in the raw record. Later ERT analysis
charges unsuccessful runs the full prescribed budget.

## Atomic execution

One atomic job consists of one problem and one paired seed, with all seven
algorithms run sequentially after a single PyCUTEst import. Each job writes one
gzip-compressed JSON file containing:

- seven raw rows;
- improvement-only histories for all algorithms;
- the full serialized BasinGraph result;
- code, protocol, seed and runner identities.

## Run batches

- seed indices 0-9;
- seed indices 10-19;
- seed indices 20-29.

Partial performance results must not be summarized between batches.

## Expected final size

- problem-seed jobs: `24 * 30 = 720`;
- result rows: `24 * 30 * 7 = 5040`;
- BasinGraph detail records: 720.

No algorithm modification is permitted after opening this holdout.
