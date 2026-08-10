# Elliptic-PDE flagship analysis amendment v2.1

Status: **registered after v2 development optimization and before any reserved confirmatory access**  
Additional objective evaluations introduced by this amendment: **0**  
Reserved confirmatory objective evaluations at registration: **0**

## 1. Development finding requiring amendment

Development cycle v2 successfully produced 16 separated operational
explanations on all eight tasks and selected interventions whose D-optimal
scores were 38--58% above the median candidate score. The ambiguity gate could
not be interpreted, however, because the fixed baseline likelihood temperature
`tau = 5` was expressed in raw objective units while eligible objective ranges
varied by orders of magnitude. Baseline weights consequently collapsed to one
candidate before any new measurement was applied.

This is a scale-definition failure in a development-stage secondary metric, not
an adverse measurement-design result. Cycle v2 and its failed gate remain
preserved. Amendment v2.1 reanalyses the same frozen development runs without
new objective calls and uses the already registered eligibility tolerance to
normalize baseline candidate weights.

## 2. Unchanged elements

The following remain unchanged:

- development instances 21--28 and reserved confirmatory instances 201--208;
- immutable BasinGraph implementation, seeds, budgets and returned archives;
- explanation eligibility and farthest-first selection;
- maximum 16 explanations and 0.08 normalized separation;
- complete fixed intervention set;
- D-optimal design score and lexical tie breaking;
- intervention-noise scale and generated intervention observations;
- all integrity checks.

## 3. Scale-normalized baseline weights

For each task, let `delta_f` be the frozen eligibility tolerance recorded by the
v2 development analysis. Baseline candidate weights are redefined as

```text
w_i proportional to exp[-0.5 * (f_i - f_best) / max(delta_f, 1e-12)].
```

Every eligible candidate therefore receives a likelihood exponent between
zero and approximately -0.5 before the intervention. This implements the
scientific premise that the retained candidates are alternative explanations
at the baseline-resolution scale.

Intervention residual likelihoods and all posterior ambiguity metrics are
otherwise unchanged.

## 4. V2.1 development gate

The v2.1 analysis gate is the original v2 gate applied to the scale-normalized
weights:

1. at least six of eight tasks retain at least four separated explanations;
2. at least six have selected D-optimal score at least 25% above the median;
3. at least six reduce normalized effective candidate count by at least 20%
   relative to the scale-normalized baseline;
4. at least six reduce weighted parameter dispersion by at least 20%;
5. selected-design ambiguity is no worse than the median candidate design on at
   least six tasks;
6. truth is not accessed by design selection;
7. integrity checks pass.

If the gate passes, the unchanged task generator, intervention set, selection
rule and v2.1 weighting formula are frozen before instances 201--208 are
accessed. Confirmatory claims remain limited to the finite operational
explanation set.
