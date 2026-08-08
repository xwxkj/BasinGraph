# B21 Track C master protocol draft

Status: **drafted after audited Track D and before any Track C objective evaluation**  
Track C confirmatory objective evaluations at drafting: **0**

## 1. Purpose

Track C tests whether the immutable BasinGraph candidate provides useful
finite-budget behavior on scientific-model calibration, inverse problems,
field optimization and observed-data nonlinear regression. The experiment does
not alter `basingraph_v2`, the frozen options hash or any earlier Track A–D
evidence.

Track C is divided into:

- **C1 scientific-model applications**: deterministic, reproducibly generated
  model instances with known data-generation mechanisms and frozen physical or
  mathematical metrics;
- **C2 observed-data anchor**: six NIST Statistical Reference Datasets for
  nonlinear regression, retaining the official observations, models, starting
  values and certified residual sums of squares.

## 2. General-purpose algorithms

The registered general-purpose comparison set is:

1. BasinGraph;
2. CMA-ES through pycma;
3. BIPOP-CMA-ES through pycma;
4. corrected L-SHADE 1.0.1 source-aligned transparent Python port;
5. jSO source-aligned transparent Python port;
6. L-SRTDE transparent source-guided Python port with registered deviations;
7. multi-start L-BFGS-B through SciPy.

DIRECT-L is not included because the heterogeneous scientific tasks include
moderate- and high-dimensional parameterizations for which its rectangle
representation is not an appropriate primary comparator. This exclusion is
registered before Track C evaluation.

Task-specific references are reported separately and are never pooled into the
general-purpose mean rank:

- Wirtinger-flow reference for phase retrieval;
- alternating-minimization reference for matrix factorization;
- SciPy nonlinear least-squares references from the official NIST far and near
  starting values.

## 3. C1 scientific-model families

The frozen family list is:

1. one-dimensional elliptic PDE inverse estimation;
2. Lorenz-63 parameter and initial-state calibration;
3. noiseless real phase retrieval;
4. noisy real phase retrieval;
5. low-rank matrix factorization;
6. large low-rank matrix factorization;
7. one-dimensional viscous Burgers control;
8. Allen–Cahn energy minimization with fixed boundary values;
9. sparse nonlinear inverse least squares.

Each family has:

- development instances 1–2;
- confirmatory instances 11–13;
- ten paired algorithm seeds per confirmatory instance;
- the same task data, bounds, objective, seed mapping and evaluation budget for
  every general-purpose algorithm.

The registered C1 general-purpose matrix is therefore:

```text
9 families × 3 instances × 10 paired seeds × 7 algorithms = 1,890 runs
```

### Registered dimensions and budgets

| Family | Confirmatory dimensions | Budget |
|---|---|---:|
| elliptic PDE inverse | 6, 8, 10 | `300d` |
| Lorenz-63 calibration | 6, 6, 6 | `300d` |
| phase retrieval | 16, 24, 32 | `200d` |
| noisy phase retrieval | 16, 24, 32 | `200d` |
| matrix factorization | 36, 60, 96 | `150d` |
| large matrix factorization | 120, 200, 320 | `75d` |
| Burgers control | 8, 12, 16 | `200d` |
| Allen–Cahn energy | 32, 64, 96 | `100d` |
| sparse nonlinear inverse | 20, 40, 80 | `150d` |

Dimensions denote the number of decision variables after the frozen
parameterization, not the numerical grid size or observation count.

## 4. C2 NIST observed-data anchor

The registered observed-data datasets are:

- Chwirut1;
- Roszman1;
- ENSO;
- Eckerle4;
- Bennett5;
- BoxBOD.

All are NIST StRD nonlinear-regression datasets with observed data and
best-available certified solutions. Official ASCII data files are vendored
unchanged with source URL, retrieval date and SHA-256.

Each dataset is evaluated with ten paired algorithm seeds and all seven
general-purpose algorithms:

```text
6 datasets × 10 paired seeds × 7 algorithms = 420 runs
```

The objective is the residual sum of squares divided by the certified residual
sum of squares. The physical parameter vector is generated from a bounded
standardized coordinate. The affine scale is frozen from the official far,
near and certified parameter vectors solely to ensure that all three official
reference points lie inside the registered search domain. This use of the
certified vector is disclosed and is not an optimizer-specific tuning step.

The registered NIST budget is `1,000d`.

## 5. Primary endpoints

Raw objective values are not pooled across scientific families. For every task,
a fixed reference value `f_ref` and fixed baseline value `f_base` are computed
before algorithm execution. The primary normalized gap is

```text
g = max(0, (f - f_ref) / max(f_base - f_ref, epsilon)).
```

The registered target ratios are:

```text
1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001
```

Primary evidence consists of:

- fraction of task–seed–target triples reached;
- evaluations to each target;
- fixed-budget checkpoints at `1d`, `3d`, `10d`, `30d`, `100d`, `300d` when
  available, and the family-specific final budget;
- results overall, by family, by decision dimension and by synthetic versus
  observed-data source.

Only BasinGraph versus each general-purpose comparator is confirmatory.
Registered inference uses Friedman tests, paired Wilcoxon signed-rank tests,
Holm correction, paired rank-biserial effects and bootstrap confidence
intervals.

## 6. Secondary endpoints

Secondary endpoints include:

- final normalized gap and final objective rank;
- parameter error where ground truth or certified values exist;
- phase-invariant signal error;
- matrix reconstruction error;
- PDE state-data misfit;
- Lorenz parameter and trajectory error;
- Burgers tracking and control-energy terms;
- Allen–Cahn energy and interface-location error;
- NIST certified-RSS ratio and certified-parameter error;
- wall-clock time and implementation failures.

Task-specific reference solvers are descriptive anchors, not members of the
general-purpose aggregate ranking.

## 7. Integrity and decision rules

- Development tasks may be used only to debug wrappers and freeze numerical
  settings.
- Confirmatory instances and NIST results cannot be used to revise BasinGraph.
- Every objective call is counted by a single internal ledger.
- Favorable, null and unfavorable outcomes are retained.
- Any post-result family grouping is labelled exploratory.
- Mac outputs are written to
  `~/Downloads/B21_RESULTS_READY/trackC_<mode>_<timestamp>/`, Finder opens the
  exact folder and the upload ZIP is highlighted automatically.

Track C supports a scientific-usefulness claim only if BasinGraph remains in a
leading statistical tier on multiple independent scientific families or shows
a reproducible finite-budget advantage on practically meaningful task metrics.
A result confined to internally generated tasks without support from the NIST
observed-data anchor is insufficient for a broad application claim.

## 8. Source records

NIST StRD general record: `https://doi.org/10.18434/T43G6C`.

Official dataset pages and ASCII files will be recorded in
`TRACK_C_NIST_PROVENANCE.csv` before the first Track C smoke run.
