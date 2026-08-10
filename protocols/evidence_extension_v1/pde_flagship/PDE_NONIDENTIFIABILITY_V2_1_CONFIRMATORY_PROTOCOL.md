# Elliptic-PDE non-identifiability flagship v2.1 confirmatory protocol

Status: **frozen after the v2.1 development gate and before confirmatory access**  
Confirmatory objective evaluations at freeze: **0**

## 1. Frozen design

- confirmatory instances: 201--208;
- dimensions: 12, 16 and 20, assigned cyclically;
- ten BasinGraph seeds per task;
- budget: `400d` per seed;
- five baseline sensors;
- explanation eligibility, farthest-first rule, 0.08 separation and maximum 16
  explanations exactly as in development cycle v2;
- complete sinusoidal/Gaussian source and five-layout intervention set;
- D-optimal selection score and lexical tie breaking;
- scale-normalized baseline weighting from amendment v2.1;
- fixed observation and intervention-noise models.

The immutable optimizer is BasinGraph `2.0.0-rc1` with options hash
`031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`.

## 2. Primary scientific endpoints

For each task, the selected intervention is evaluated on the finite operational
explanation set. Primary outcomes are:

- normalized effective candidate-count reduction relative to baseline;
- weighted parameter-dispersion reduction relative to baseline;
- D-optimal score gain relative to the median candidate intervention.

The hidden truth and generated intervention observation are unavailable to the
design selector. Maximum-weight candidate parameter error is secondary.

## 3. Success rule

The confirmatory flagship succeeds only if all conditions hold:

1. at least six of eight tasks retain at least four separated explanations;
2. at least six tasks have selected score at least 25% above the median design;
3. at least six tasks reduce normalized effective candidate count by at least
   20%;
4. at least six tasks reduce weighted parameter dispersion by at least 20%;
5. selected-design ambiguity is no worse than the median design on at least six
   tasks;
6. design selection never accesses truth or intervention observations;
7. all optimizer identity, evaluation-accounting, archive-capacity and returned
   graph-integrity checks pass.

A successful result supports the claim that a bounded operational archive can
expose multiple baseline-compatible explanations and guide an informative
additional measurement for that finite set. It does not establish complete
inverse-solution enumeration, global identifiability or universal superiority
of BasinGraph as an inverse solver.
