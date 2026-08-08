# Track B Amendment 001: jSO initial population

Status: **registered before confirmatory evaluation**  
Date: 2026-08-08  
Confirmatory objective evaluations before amendment: **0**

## Trigger

A post-smoke provenance review found that the frozen v1 Python port used an
initial jSO population of `25D`. The published jSO specification states

```text
N_init = 25 ln(D) sqrt(D)
```

The v1 smoke used development instance 1 only and did not access Track B
instances 21--30.

## Registered correction

The Track B jSO port now uses

```text
N_init = max(4, round(25 ln(D) sqrt(D)))
```

which gives 90 individuals in 5D and 335 in 20D. No other algorithm, seed,
budget, instance, comparator or endpoint setting is changed.

## Consequence

The v1 smoke remains an infrastructure record but is superseded for baseline
fidelity. A new development-only smoke is required before confirmatory access.
All v1 records are retained and excluded from confirmatory statistics.
