# BasinGraph v2.0.0-rc1 ablation mini protocol

## Status

Engineering validation only. These results must not be used as final manuscript
evidence.

## Frozen implementation

- implementation version: `2.0.0-rc1`
- full options hash:
  `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## Tasks

The suite contains seven deterministic diagnostic tasks fixed before running:

1. shifted sphere, 5 variables;
2. Rosenbrock, 5 variables;
3. shifted Rastrigin, 5 variables;
4. shifted Ackley, 5 variables;
5. rotated ellipsoid, 10 variables;
6. far-basin double well, 5 variables;
7. boundary quartic, 8 variables.

All seven objectives have known global value zero.

## Variants

- `Full`
- `NoGraphGuidance`
- `SingleBracket`
- `NoFarBasin`
- `NoGeometryController`
- `NoArchiveFallback`
- `NoFinalPolish`

Each variant differs from `Full` by exactly one frozen Boolean option.

## Seeds

Five paired seeds:

- 20260619
- 20260620
- 20260621
- 20260622
- 20260623

## Acceptance criteria

1. 245 completed runs;
2. implementation version and full options hash match the frozen contract;
3. all runs exhaust the assigned FE budget;
4. phase evaluation counts sum exactly to total nfe;
5. every archive is nonempty;
6. every graph edge references active archive nodes;
7. disabled phases have zero evaluations;
8. the graph-guidance ablation still constructs graph edges;
9. each ablation changes the search trajectory in at least one paired run;
10. no performance claim is made from this mini suite.
