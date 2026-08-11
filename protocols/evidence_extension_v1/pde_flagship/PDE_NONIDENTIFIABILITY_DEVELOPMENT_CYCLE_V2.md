# Elliptic-PDE non-identifiability flagship: development cycle v2

Status: **new development cycle frozen before v2 objective evaluation**  
V2 development objective evaluations at freeze: **0**  
V2 reserved confirmatory objective evaluations at freeze: **0**

## 1. Why a second development cycle is scientifically justified

Development cycle v1 retained three or more separated explanations on only two
of six tasks. Its baseline layouts were therefore too informative for a study
whose explicit subject is non-identifiability. The sinusoidal source-only
design family improved the separation score by at most about 14%, and the
registered likelihood scale made posterior weights numerically collapse.
Cycle v1 remains an immutable negative engineering record; none of its
instances is reused.

Cycle v2 changes the scientific observation regime rather than relaxing the v1
gate after seeing outcomes. It deliberately studies an underdetermined inverse
problem with fewer baseline sensors, higher coefficient dimension, a broader
fixed intervention family and an observation-noise model used consistently in
design and evaluation.

## 2. Partitions and immutable optimizer

- development instances: 21--28;
- reserved confirmatory instances: 201--208;
- dimensions: 12, 16 and 20, assigned cyclically;
- BasinGraph seeds per task: 10;
- budget: `400d` per seed;
- implementation: BasinGraph `2.0.0-rc1`;
- options hash:
  `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`.

No file under `basingraph_v2/` is changed.

## 3. Baseline inverse problem

The coefficient field uses the same Fourier log-conductivity
parameterization and conservative finite-volume solver as cycle v1. Baseline
observations use five sensors and one source. Observation noise is Gaussian
with standard deviation 1% of the standard deviation of the noiseless baseline
sensor vector, bounded below by `1e-6`.

The objective is mean squared sensor residual divided by the registered noise
variance, plus `1e-8` times the mean squared coefficient vector. This creates a
well-defined likelihood-scaled misfit while leaving the inverse problem
underdetermined.

## 4. Operational explanation set

All returned archive representatives and incumbents from ten paired runs are
pooled. Duplicates within Euclidean distance `1e-8` are removed. A candidate is
eligible when its objective is no more than

```text
f_best + max(5, 0.15 * (f_zero - f_best)).
```

The additive value five corresponds to one normalized residual unit per
baseline sensor and is fixed before evaluation. A deterministic farthest-first
rule retains at most 16 explanations with normalized parameter distance at
least 0.08 of the box diameter.

The set is operational and finite. It is not a complete inverse-solution set.

## 5. Fixed candidate interventions

Candidate interventions combine four families:

1. sinusoidal source perturbations, frequencies 1--8, amplitudes 0.25 and 0.50;
2. localized Gaussian source perturbations, centres 0.15--0.85 and widths 0.06
   or 0.12;
3. twelve-sensor approximately uniform layouts with offsets 0 and 0.5;
4. eight-sensor left-, centre- and right-weighted layouts.

Every source perturbation is paired with every registered sensor layout. The
candidate set and lexical tie-breaking rule are fixed.

## 6. Design score and ambiguity metrics

For each design, candidate prediction vectors are centred and divided by the
registered intervention-noise standard deviation, equal to 2% of the baseline
truth-state standard deviation and bounded below by `1e-6`. The design score is

```text
log det(I + covariance(predictions) / sigma_design^2).
```

This D-optimal score uses candidates only; it cannot access the hidden truth or
new observations.

Baseline candidate weights use the baseline objective through

```text
w_i proportional to exp(-0.5 * (f_i - f_best) / tau),
```

with fixed temperature `tau = 5`. After the selected intervention, the baseline
log weights are augmented by the intervention residual likelihood using the
same registered intervention-noise scale.

Ambiguity is measured by:

- normalized Shannon effective candidate count, `exp(H(w)) / K`;
- weighted parameter dispersion divided by the squared box diameter;
- maximum-weight candidate parameter error, reported only as a secondary
  truth-dependent evaluation metric.

The selected design is compared with the baseline ambiguity and with the
median result over all fixed candidate interventions.

## 7. V2 development gate

A v2 confirmatory cycle may be opened only if all conditions hold:

1. at least six of eight tasks retain at least four separated explanations;
2. at least six tasks have selected design score at least 25% above the median
   candidate-design score;
3. at least six tasks reduce normalized effective candidate count by at least
   20% relative to baseline;
4. at least six tasks reduce weighted parameter dispersion by at least 20%
   relative to baseline;
5. selected-design median ambiguity reduction is no worse than the median
   candidate design on at least six tasks;
6. design selection never accesses the truth or intervention observations;
7. all optimizer identity, accounting, archive-capacity and returned graph
   checks pass.

If the gate fails, instances 201--208 remain untouched. A successful result
supports measurement-design utility for the finite operational explanation
set, not complete identifiability of the PDE model.
