# BasinGraph CUTEst pre-registration protocol v2

## Timing and audit status

This protocol was frozen after technical inventory and before any
optimizer-performance experiment on the final 50-problem set.

The failed v1 and v1.1 quota attempts are retained as part of the audit trail.
The technical inventory v2 identified 84 valid instances: 41 small,
27 medium and 16 large. No optimizer result was used in the inventory.

## Final benchmark size

- Small, 2-20 variables: 20 instances
- Medium, 21-100 variables: 20 instances
- Large, 101-500 variables: 10 instances
- Total: 50 instances

## Eligibility criteria

- regular CUTEst classification;
- bound constraints only;
- no general equality or inequality constraints;
- continuous variables only;
- fixed or formally parameterized dimension between 2 and 500;
- finite lower and upper bounds;
- strictly positive box widths;
- feasible CUTEst initial point;
- finite objective value at the initial point;
- objective not none, constant or purely linear.

## Deterministic selection

Selection seed:
`BasinGraph-CUTEst-v2-final-20260619`

Each technically valid instance is assigned SHA-256 of:
`BasinGraph-CUTEst-v2-final-20260619|<instance_id>`

Selection is performed in scarcity-first order:
large, medium, then small.

Within each stratum, instances are ordered by SHA-256 and CUTEst base
problem names not previously used are preferred. A second instance from
a previously used base problem is allowed only if needed to fill a stratum.
No optimizer performance is used.

## Source balance and family diversity

- Selected fixed instances: 21
- Selected scalable instances: 29
- Unique CUTEst base names: 48
- Additional instances from repeated base names: 2

## Frozen artifacts

- `cutest_pre_registered_problem_list_v2.csv`
- `cutest_valid_not_selected_v2.csv`
- `cutest_selection_audit_v2.csv`
- `CUTEST_PRE_REGISTRATION_SUMMARY_v2.json`
- `CUTEST_PRE_REGISTRATION_MANIFEST_v2.csv`

The technical inventory artifacts from Step 13C-v2A remain part of the
complete audit trail.
