# Route B freeze: full BasinGraph v2.0.0

This branch implements the full BasinGraph algorithm and supersedes the earlier
development implementation used in the initial v1.0.0 validation package.

## Frozen legacy status

The following materials are retained only as legacy/development evidence:

- BasinGraph v1.0.0 software release
- original official COCO/BBOB validation
- original pre-registered CUTEst validation
- original Zenodo reproducibility dataset
- independent MATLAB v3.3 validation package

These materials must not be presented as final evidence for the full
BasinGraph v2.0.0 algorithm.

## Route B target

The goal of this branch is to implement a complete BasinGraph optimizer with:

1. explicit basin-node archive;
2. merge rule for basin identity;
3. directed transition graph;
4. edge cost, improvement, barrier and accessibility metadata;
5. multi-bracket coordinate-Brent basin sweep;
6. geometry diagnostics including anisotropy, boundary signal and ruggedness;
7. graph-aware archive fallback;
8. budget-completion probing;
9. full logging of archive and graph outputs;
10. new official COCO/BBOB, CUTEst, AppliedMath and ablation results.

## Version target

The new version will be released as:

BasinGraph v2.0.0
