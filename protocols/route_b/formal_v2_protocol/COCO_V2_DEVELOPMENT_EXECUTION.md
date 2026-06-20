# COCO v2 development execution

## Frozen partition

- suite: official noiseless BBOB;
- functions: 1-24;
- dimensions: 2, 5 and 10;
- instances: 1-3 only;
- budget: 1000d;
- seven algorithms;
- paired deterministic seed per problem across algorithms.

## Holdout exclusion

The runner hard-codes instances 1-3 and raises an exception if any problem
outside the development partition is encountered. Instances 4-15 must not be
run or inspected before the final v2.0.0 code tag.

## Expected size

- unique problems: 216;
- result rows: 1512;
- BasinGraph detail records: 216;
- COCO `.info` files: 168;
- COCO `.dat` files: 504.

## Interpretation

The result is development evidence only. It may be used to decide whether the
rc1 algorithm needs modification. It may not be reported as prospective
holdout or final manuscript evidence.
