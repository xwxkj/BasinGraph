# Track B confirmatory decision

Status: **audited and frozen**  
Run ID: `b21_track_b_confirmatory_20260808T183416Z`  
Frozen source commit: `e405cb5d34e7a26242d2c9b77a03f63f1b19c54a`  
Decision: **conditional GO — proceed to scalability and scientific-application evidence, while rejecting any broad final-budget superiority claim.**

## Integrity

The registered 3,840-run matrix completed without failures:

- official noiseless COCO/BBOB functions 1–24;
- dimensions 5 and 20;
- actual instances 21–30;
- eight frozen algorithms;
- budget `1,000d` per run;
- exact internal and observer evaluation accounting;
- all 16 algorithm–dimension shards completed on first attempt;
- official cocopp post-processing completed;
- BasinGraph phase, archive and graph integrity checks passed.

Independently verified archive identities:

- result ZIP SHA-256: `a0535d301bb2a13a68717580ba0f28d4b49f5c0f1b98e8058f08380cb0465010`;
- observer archive SHA-256: `0fa002d95cb98cf57159386dae5dce17167c1a8f58d174460812822d9bb22cff`;
- raw-results SHA-256: `1357bf1b2d0bd4cba7cbbc3522441c55a640a7a7ca7da6fe41b7ade442b63561`.

## Primary target-runtime result

Across the standard 51-target grid, BasinGraph attained the following pooled
fractions of function–instance–target pairs:

| evaluations per dimension | fraction | descriptive rank |
|---:|---:|---:|
| 1 | 0.0421 | 1 |
| 3 | 0.0472 | 2 |
| 10 | 0.1002 | 2 |
| 30 | 0.1941 | 2 |
| 100 | 0.3033 | 2 |
| 300 | 0.3890 | 2 |
| 1,000 | 0.4666 | 4 |

BasinGraph therefore showed a strong early-to-intermediate finite-budget
profile, but it was overtaken at the final budget by BIPOP-CMA-ES (0.5928),
CMA-ES (0.5893) and multi-start L-BFGS-B (0.4835).

At `1,000d`, paired per-problem target fractions showed that BasinGraph was
significantly worse than CMA-ES, BIPOP-CMA-ES and multi-start L-BFGS-B after
Holm correction, statistically indistinguishable from L-SHADE 1.0.1, jSO and
L-SRTDE, and significantly better than DIRECT-L.

The result is dimension-dependent. At `1,000d`, BasinGraph ranked seventh in
5D and fourth in 20D. It remained second in 20D through `300d`, whereas in 5D
CMA-ES and BIPOP-CMA-ES overtook it earlier.

## Secondary final-value result

The final-value Friedman test was significant (`p = 2.45e-145`). Mean final
ranks were led by BIPOP-CMA-ES (2.809) and CMA-ES (2.954); BasinGraph ranked
seventh with mean rank 5.222. BasinGraph final values were significantly worse
than CMA-ES, BIPOP-CMA-ES, L-SHADE 1.0.1, jSO, L-SRTDE and multi-start
L-BFGS-B after Holm correction, and were not significantly different from
DIRECT-L. This endpoint is secondary and does not replace the official
target-runtime evidence.

## Interpretation

Track B supports the claim that BasinGraph is competitive under tight and
intermediate evaluation budgets and that its performance profile differs from
population-based optimizers that improve more strongly late in the budget.
It does not support a claim of aggregate final-budget or final-value
state-of-the-art performance.

The candidate is not uniformly dominated across dimensions, function groups
and budget stages, so the pre-registered competitiveness condition is met in a
qualified sense. The correct next step is to retain the frozen candidate and
proceed to scalability/overhead and scientific-application evidence.

## Manuscript boundary

Allowed claim:

> BasinGraph provides strong early-to-intermediate target-runtime progress and
> remains competitive with modern optimizers under finite evaluation budgets,
> while its relative performance depends on dimension, landscape class and
> budget stage.

Disallowed claims:

- BasinGraph is the best overall optimizer;
- BasinGraph uniformly outperforms CMA-ES or BIPOP-CMA-ES;
- BasinGraph leads at the final `1,000d` budget;
- BasinGraph achieves state-of-the-art final objective values across BBOB;
- the transparent L-SHADE, jSO or L-SRTDE ports are byte-identical official
  executables.

## Next stage

Proceed to Track D scalability and optimizer-overhead analysis, followed by
Track C scientific-model and observed-data applications. The frozen Track B
instances 21–30 must never be reused for tuning or candidate revision.
