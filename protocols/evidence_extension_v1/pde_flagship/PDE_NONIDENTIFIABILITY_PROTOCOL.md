# Elliptic-PDE non-identifiability flagship protocol

Status: **development protocol frozen before any flagship objective evaluation**  
Development objective evaluations at freeze: **0**  
Reserved confirmatory objective evaluations at freeze: **0**

## 1. Scientific question

A single incumbent can fit observed PDE states accurately while concealing
multiple substantially different coefficient fields that explain the same
measurements. The flagship experiment asks whether BasinGraph's bounded
operational archive can expose such alternative explanations and whether those
alternatives can be used, without knowing the truth, to select an additional
measurement configuration that reduces parameter ambiguity.

## 2. Model

The one-dimensional elliptic equation is

\[
-\frac{\mathrm d}{\mathrm dx}\left(k(x;\boldsymbol\theta)
\frac{\mathrm du}{\mathrm dx}\right)=s(x),\qquad
u(0)=u(1)=0,
\]

with

\[
\log k(x;\boldsymbol\theta)=
\sum_{j=1}^{d}\theta_j\frac{\sin(j\pi x)}{j}.
\]

A conservative finite-volume discretization and Thomas tridiagonal solve are
used. Baseline observations are noisy state values at a sparse fixed sensor
layout under a baseline source. The optimization objective is normalized
state-data misfit plus a fixed weak quadratic regularizer.

## 3. Partitions

- development instances: 1--6;
- reserved confirmatory instances: 101--106;
- dimensions: 8, 10 and 12, assigned cyclically;
- BasinGraph seeds per instance: 8;
- budget: `300d` per seed.

The frozen BasinGraph implementation and options hash are identical to the
manuscript evidence. No existing Track C task or instance is reused.

## 4. Alternative-explanation set

For each task, all returned active-archive representatives and incumbents from
the eight paired runs are pooled. Duplicate parameter vectors are removed.
Candidates are eligible when

```text
(f - f_best) / max(f_zero - f_best, epsilon) <= 0.10,
```

where `f_zero` is the objective at the zero coefficient vector. A deterministic
farthest-first procedure retains at most 12 explanations, each separated from
previously retained explanations by at least 0.12 of the parameter-box
diameter.

This is an operational candidate set, not a claim that all mathematical
solutions or attraction basins have been enumerated.

## 5. Candidate measurement designs

Candidate designs are formed without using the true parameter vector. They
combine:

- sinusoidal source perturbation frequencies 1--8;
- amplitudes 0.25 and 0.50;
- two twelve-sensor offsets.

For each design, every retained explanation predicts a state observation
vector. The design score is the 25th percentile of pairwise prediction
distances divided by the registered observation-noise scale. The selected
design maximizes this score, with deterministic lexical tie breaking.

## 6. Ambiguity reduction

After design selection, synthetic observations under the hidden truth are
created with a deterministic independent noise seed. Candidate likelihood
weights are

```text
w_i proportional to exp(-0.5 * mean_squared_residual_i / sigma^2).
```

Ambiguity is quantified by:

- effective candidate count, `1 / sum(w_i^2)`;
- weighted parameter dispersion normalized by the box diameter;
- error of the maximum-weight candidate.

The selected design is compared with the median result over the complete fixed
candidate-design set. The median-design comparison is not chosen after seeing
the selected-design outcome.

## 7. Development gate

A confirmatory cycle may be opened only if:

1. at least four of six tasks retain at least three separated explanations;
2. at least four tasks show a selected-design separation score at least 25%
   above the median candidate-design score;
3. at least four tasks show weighted parameter dispersion at least 20% below
   the median candidate-design dispersion;
4. the design-selection code never accesses the truth or new observations;
5. all optimizer identity, accounting, archive-capacity and returned graph
   integrity checks pass.

If the gate fails, the reserved confirmatory instances remain untouched.

## 8. Confirmatory boundary

If the gate passes, all thresholds, design lists, source/sensor definitions and
analysis code are committed in a machine-readable lock before instances
101--106 are accessed. Confirmatory claims will be limited to operationally
exposed alternative explanations and measurement-design utility. No claim of
complete inverse-solution enumeration will be made.
