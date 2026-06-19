# CUTEst protocol amendment: v1 to v1.1

## Timing

This amendment was made after technical compilation and validity
screening, but before any optimizer-performance experiment or
performance-based problem selection.

## Reason

Under the predefined strict eligibility criteria, only 18 problems
in the 2–20-variable stratum were technically valid. The original
target of 20 therefore could not be met.

## Revised allocation

- Small, 2–20 variables: 18 problems
- Medium, 21–100 variables: 22 problems
- Large, 101–500 variables: 10 problems
- Total: 50 problems

## Unchanged criteria

The following requirements remain unchanged:

- regular CUTEst classification;
- fixed dimensionality;
- bound constraints only;
- no general constraints;
- continuous variables only;
- finite lower and upper bounds;
- positive box widths;
- feasible CUTEst initial point;
- finite objective at the initial point;
- deterministic SHA-256 ordering;
- no use of optimizer performance during selection.

The failed v1 log is retained as part of the audit trail.
