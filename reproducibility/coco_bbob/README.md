# BasinGraph final official COCO/BBOB core validation

## Run identity

- RUN_ID: `step12c_corrected_coco_d2351020_f1_24_i1_15_20260618_172900`
- Status: frozen final core COCO/BBOB validation
- Suite: official COCO/BBOB noiseless
- Functions: 1-24
- Dimensions: 2, 3, 5, 10 and 20
- Instances: 1-15
- Budget: 1000d function evaluations
- Algorithms: seven

## Integrity

- Result rows: 12600
- Unique COCO problems: 1800
- `.info` files: 168
- `.dat` files: 840
- Formal COCO labels: BIPOP_CMA_ES, BasinGraph, CMA_ES, DE, LHS, MS_LBFGSB, Random
- cocopp status: ALL done

## Corrected baseline status

- CMA-ES uses pycma 4.4.4.
- BIPOP-CMA-ES uses `cma.fmin2` with `restarts=9` and `bipop=True`.
- Official COCO short labels are stored in the `.info` logs.
- BasinGraph and BIPOP-CMA-ES use at least 99% of the prescribed budget in every run.

## Contents

- `tables/`: final CSV and budget-usage summary.
- `logs/`: run, validation, COCO-label and cocopp logs.
- `protocols/`: environment, dependency lock, provenance and run metadata.
- `source_snapshot/`: source files used for this run.
- `compressed_raw_data/`: compressed COCO observer logs and cocopp outputs.

This archive supersedes the earlier development Step 9 COCO run.
