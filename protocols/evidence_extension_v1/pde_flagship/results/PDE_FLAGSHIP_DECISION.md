# Elliptic-PDE non-identifiability flagship decision

Status: **confirmed on a registered independent partition**  
Immutable optimizer: BasinGraph `2.0.0-rc1`  
Options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## Decision

**The flagship supports the claim that a finite operational archive can expose
multiple baseline-compatible coefficient explanations and select an additional
measurement that sharply reduces ambiguity for that finite explanation set.**

The result does not establish complete inverse-solution enumeration, global
identifiability or universal superiority of BasinGraph as an inverse solver.

## Transparent development history

Development cycle v1 used six tasks with dimensions 8--12. Only two tasks
retained at least three separated explanations, and its source-only
interventions improved the separation score by less than the registered 25%
gate. The v1 gate failed and its confirmatory instances were not accessed.

Development cycle v2 used a deliberately underdetermined five-sensor regime,
dimensions 12--20, broader source and sensor interventions, ten seeds per task
and budget `400d`. All eight tasks retained 16 separated explanations and all
eight selected interventions had D-optimal scores at least 25% above the median
candidate score. Its fixed baseline likelihood temperature was not invariant to
the objective scale, causing baseline weights to collapse before a new
measurement was introduced. The original v2 gate therefore failed.

Analysis amendment v2.1 introduced no objective evaluations and changed only
the development-stage baseline weighting to use the already registered
eligibility tolerance as its scale. The task generator, optimizer runs,
explanation selection, intervention set, D-optimal score, observations and
integrity checks remained unchanged. The v2.1 development gate passed all
conditions before confirmatory instances were accessed.

## Independent v2.1 confirmation

The locked confirmatory study used instances 201--208, ten BasinGraph seeds per
task and 80 optimization runs. All runs exhausted the registered budget and
passed implementation identity, phase accounting, archive-capacity and returned
graph-integrity checks.

Across eight tasks:

- seven retained at least four separated explanations;
- six had selected D-optimal score at least 25% above the median intervention;
- all eight reduced normalized effective candidate count by at least 20%;
- all eight reduced weighted parameter dispersion by at least 20%;
- all eight selected interventions were no worse than the median intervention
  on both ambiguity measures.

Median confirmatory outcomes were:

- retained explanations: `16`;
- selected-score gain over median intervention: `43.07%`;
- normalized effective-count reduction: `93.67%`;
- weighted parameter-dispersion reduction: `99.999%`.

The selected intervention was determined solely from candidate predictions and
the registered noise scale. It did not access the hidden truth or intervention
observation. In six of eight tasks the selected intervention was the registered
Gaussian source perturbation centred at 0.50 with width 0.12 and the uniform
12-sensor layout; two tasks selected closely related registered interventions.

## Scientific interpretation

A single incumbent state fit can conceal multiple coefficient vectors that are
widely separated in parameter space. The operational archive provides a finite,
inspectable collection of such explanations. Their predicted responses under
candidate interventions define a measurement-design criterion. On the
independent partition, the criterion selected measurements that concentrated
likelihood weight and reduced parameter dispersion within the retained set.

This supports:

> A bounded operational archive can expose alternative baseline-compatible PDE
> coefficient explanations hidden by a single incumbent and can guide an
> informative additional measurement for that finite candidate set.

It does not support:

- complete enumeration of all inverse solutions;
- proof that the selected measurement makes the continuous inverse problem
  globally identifiable;
- a claim that archived representatives are mathematical attraction basins;
- a claim that low objective value uniquely identifies the true coefficient;
- replacement of specialized Bayesian or experimental-design methods.
