# COCO v2 development diagnostic policy

## Permitted inputs

- COCO instances 1-3 only;
- frozen v2.0.0-rc1 detail records;
- development cocopp output;
- development-only ablation evidence.

## Prohibited inputs

- COCO instances 4-15;
- prospective CUTEst holdout results;
- any unpublished holdout summary.

## Revision threshold

A change to the algorithm requires all of the following:

1. a mechanism-level diagnosis from serialized archive/graph/phase data;
2. agreement with development cocopp target-runtime evidence;
3. improvement under a predeclared development-only comparison;
4. no use of prospective holdout information;
5. a new options hash and updated algorithm specification;
6. rerunning all development evidence after the change.

Final-value ranks alone are insufficient to justify revision.
