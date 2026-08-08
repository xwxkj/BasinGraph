# Prospective COCO/BBOB holdout results

## Evidence status

The uploaded review package passed all supplied SHA-256 checks. The 631
official `cocopp` output files matched the frozen manifest. The successful
holdout comprises 10,080 algorithm--problem records over 1,440 distinct
problems, with functions 1--24, dimensions 2, 3, 5, 10 and 20, actual BBOB
instances 4--15, seven algorithms and a budget of 1000 evaluations per
dimension.

## Manuscript-ready Results text

On the prospective COCO/BBOB holdout, BasinGraph displayed a budget-dependent
anytime profile. At 100 function evaluations per dimension, the official
`cocopp` ECDF fractions for BasinGraph were
0.521, 0.448, 0.348, 0.293, 0.256
in 2, 3, 5, 10 and 20 dimensions, respectively; at this intermediate budget,
BasinGraph exceeded both CMA-ES and BIPOP-CMA-ES in every tested dimension,
although multi-start L-BFGS-B was slightly higher in several dimensions. At
the full 1000-evaluation-per-dimension budget, the corresponding BasinGraph
fractions were
0.712, 0.611, 0.538, 0.475, 0.381.
BasinGraph therefore ranked third by aggregate target attainment in 5, 10 and
20 dimensions and fourth in 2 and 3 dimensions, behind the late-budget
performance of CMA-ES and BIPOP-CMA-ES.

As a secondary fixed-budget sanity check, BasinGraph achieved a
task-normalized mean final-value rank of 3.359,
placing third overall, with 450
best-or-tied records and 437
strict final-target successes among 1,440 holdout problems. It was better than
Differential Evolution on 685 paired problems, worse on 602 and tied on 153,
and it outperformed multi-start L-BFGS-B on 818 paired problems. Its clearest
specialization occurred on weakly structured multimodal functions (f20--f24),
where it obtained 60 strict
final-target successes out of 300, the highest count
among the tested algorithms, and a mean final-value rank of
2.863.

The prospective results were consistent with development behavior rather than
showing a post-selection collapse. BasinGraph's mean final-value rank changed
from 3.391 on development instances 1--3 to
3.359 on holdout instances 4--15. Across the common
dimensions 2, 5 and 10, the holdout-minus-development change in aggregate ECDF
was -0.001 at 100 evaluations per dimension and
-0.014 at 1000 evaluations per dimension.

## Required interpretation limits

- Do not claim overall superiority to CMA-ES or BIPOP-CMA-ES.
- Treat final-value ranks as secondary to official target-runtime ECDF/ERT.
- Describe the method as especially competitive at early-to-intermediate
  budgets and on weakly structured multimodal functions.
- State that the active archive reached its fixed capacity of 80 in all
  holdout runs; it is an operational memory, not a complete basin enumeration.
- Use the frozen terminology “operational basin-state node” and “observed
  search transition”.
