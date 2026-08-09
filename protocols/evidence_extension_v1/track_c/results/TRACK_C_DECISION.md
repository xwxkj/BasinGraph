# Track C confirmatory decision

Status: **audited and frozen**  
Run ID: `b21_track_c_confirmatory_20260809T005314Z`  
Frozen source commit: `49c802225a87ebe3ce33b87ed79bfabd03a7d9cd`  
Decision: **GO to integrated manuscript revision and evidence release, with a finite-budget general-purpose scientific-optimization claim and explicit task-dependent limitations.**

## Integrity

The registered evidence completed without failed runs:

- C1 scientific-model matrix: 1,890/1,890 runs;
- C2 NIST observed-data matrix: 420/420 runs;
- seven frozen general-purpose algorithms;
- 330 paired task-seed blocks;
- 105 task-algorithm shards, all completed on attempt 1;
- 18 registered descriptive task-specific reference runs;
- exact internal evaluation accounting;
- monotone best-so-far histories;
- BasinGraph phase accounting, archive capacity and graph referential-integrity checks passed;
- confirmatory preflight reported zero objective evaluations before access.

Independently verified archive identities:

- result ZIP SHA-256: `d1f1805bc7b7d4364a97562369725afbacf771835cf1896ae706c34ed3a859c4`;
- details archive SHA-256: `e5b580375b5d1476d600c542e68018f07a6bf30e54d86fa9abc4d3beb0373c9f`;
- raw-results SHA-256: `fbdf20878465edd81b2469b418b3e786db5b8dcfc7f64f9541a337f874f4b92e`;
- target-runtime records SHA-256: `a9434037c13bb11b64d635cfd5134fc6d144c74d2d730f9a596b11e12146c630`.

The run used BasinGraph implementation `2.0.0-rc1` and the frozen options hash
`031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`.

## Primary finite-budget result

Across all 330 task-seed blocks and seven registered target ratios, BasinGraph
attained the following pooled target fractions at common checkpoints:

| evaluations per dimension | BasinGraph | descriptive rank |
|---:|---:|---:|
| 1 | 0.1342 | 1 |
| 3 | 0.1610 | 2 |
| 10 | 0.2359 | 2 |
| 30 | 0.4658 | 2 |
| 100 | 0.8338 | 2 |

At each task's registered final budget, pooled target fractions were:

1. multi-start L-BFGS-B: 0.8766;
2. BasinGraph: 0.8009;
3. CMA-ES: 0.7468;
4. BIPOP-CMA-ES: 0.7364;
5. jSO: 0.5792;
6. L-SRTDE: 0.5095;
7. L-SHADE 1.0.1: 0.4190.

At the final registered budgets, paired target-fraction tests showed that
BasinGraph was significantly better than CMA-ES, BIPOP-CMA-ES, L-SHADE 1.0.1,
jSO and L-SRTDE after Holm correction, and significantly worse than multi-start
L-BFGS-B. BasinGraph also had the highest pooled target fraction at `1d`; the
advantage over multi-start L-BFGS-B at that checkpoint was small but registered
as statistically significant.

## C1 scientific-model result

On the nine scientific-model families, BasinGraph's final registered-budget
target fraction was 0.7566, second to multi-start L-BFGS-B at 0.8492. It was
significantly better than every other registered general-purpose comparator and
significantly worse than multi-start L-BFGS-B.

BasinGraph was best or tied for best in final target fraction on six families:
Allen–Cahn energy, Burgers control, elliptic PDE inverse estimation, Lorenz-63
calibration, noisy phase retrieval and sparse nonlinear inverse least squares.
It was not the leading method on noiseless phase retrieval, matrix factorization
or large matrix factorization.

Practically meaningful secondary metrics showed especially strong results for:

- noisy phase retrieval: median phase-invariant relative error 0.719, lower than
  every registered general-purpose comparator;
- sparse nonlinear inverse estimation: median parameter relative error 0.00590,
  close to multi-start L-BFGS-B at 0.00561 and far below the remaining methods;
- Allen–Cahn energy: median energy 0.9387, essentially matching multi-start
  L-BFGS-B at 0.9386;
- elliptic PDE, Lorenz-63 and Burgers tasks: BasinGraph was in the leading tier
  on state, trajectory or control objectives.

Important limitations were also confirmed:

- low-rank matrix factorization remained weak: the median matrix relative error
  was 0.569 for the moderate matrix family, versus approximately 4.5e-6 for
  multi-start L-BFGS-B;
- on noiseless phase retrieval, low measurement error did not translate into
  uniformly accurate phase-invariant signal recovery;
- Allen–Cahn energy minimization did not imply recovery of the registered
  reference interface location, because multiple low-energy states are
  possible;
- elliptic state fit was substantially more accurate than coefficient recovery,
  consistent with inverse-problem non-identifiability.

## C2 NIST observed-data result

All seven methods reached all seven registered target ratios by the final
`1,000d` budget on the six NIST StRD datasets, so the final primary endpoint was
saturated and did not distinguish methods.

BasinGraph had the highest NIST pooled target fraction at `1d` (0.1667), was
second at `3d`–`100d`, and reached all targets by `300d`. It substantially
outperformed most population-based baselines at intermediate checkpoints, but
multi-start L-BFGS-B was generally faster after the earliest checkpoint.

Final certified-RSS ratios were near one for most methods. BasinGraph's final
parameter recovery was mixed: it was accurate on BoxBOD, Chwirut1, ENSO and
Eckerle4, but had larger scaled parameter errors on Roszman1 and Bennett5 than
several comparators. Because the certified parameter vector was used only to
construct a common bounded coordinate system, these experiments are a
reference-data calibration anchor, not a blind real-world deployment test.

## Secondary aggregate endpoint

The final normalized-gap Friedman test was significant (`p = 2.50e-217`).
Descriptive mean final ranks were:

1. multi-start L-BFGS-B: 2.594;
2. CMA-ES: 2.991;
3. BasinGraph: 3.076;
4. BIPOP-CMA-ES: 3.088;
5. jSO: 4.591;
6. L-SRTDE: 5.192;
7. L-SHADE 1.0.1: 6.468.

BasinGraph was significantly better than the three differential-evolution
comparators and marginally better than CMA-ES and BIPOP-CMA-ES after Holm
correction, but significantly worse than multi-start L-BFGS-B. This is a
secondary endpoint and does not replace the target-runtime analysis.

## Task-specific references

The 18 spectral Wirtinger-flow, alternating-ridge-least-squares and NIST
least-squares reference results are descriptive anchors only. They are excluded
from the seven-algorithm aggregate ranking. Their mixed outcomes reinforce that
task-specific structure can outperform a general-purpose optimizer on some
problems and that BasinGraph must not be presented as a replacement for
specialized scientific solvers.

## Interpretation and manuscript boundary

Track C supports the following bounded claim:

> BasinGraph is a competitive general-purpose scientific optimizer under tight
> and intermediate evaluation budgets. Across heterogeneous model-calibration,
> inverse, control, energy and observed-data tasks, it provides the earliest
> pooled target progress and the second-highest final registered-budget target
> fraction, while retaining exact evaluation accounting and bounded operational
> search state.

Allowed supporting claims:

- performance depends on task geometry, dimension and budget stage;
- the strongest application evidence arises from noisy phase retrieval, sparse
  nonlinear inversion, energy minimization, dynamical calibration and control;
- BasinGraph is competitive with CMA-ES and BIPOP-CMA-ES and substantially
  stronger than the registered DE ports on the pooled finite-budget endpoint;
- multi-start L-BFGS-B remains the strongest aggregate comparator.

Disallowed claims:

- BasinGraph is the best scientific optimizer overall;
- BasinGraph uniformly outperforms multi-start L-BFGS-B, CMA-ES or BIPOP-CMA-ES;
- BasinGraph is a state-of-the-art matrix-factorization solver;
- the NIST experiments are blind real-world deployments;
- low objective value always implies accurate physical-parameter or latent-state
  recovery;
- the transparent L-SHADE, jSO or L-SRTDE ports are byte-identical official
  executables.

## Next stage

Freeze and release the Track A–D/C evidence package, update the software/data
DOIs, generate Source Data and Supplementary tables/figures, and rewrite the
manuscript around bounded operational state, auditable transitions and
finite-budget scientific optimization. No Track C confirmatory task or NIST
result may be reused for tuning or candidate revision.
