# Track B Amendment 003: L-SRTDE provenance clarification

Status: **registered before confirmatory evaluation**  
Date: 2026-08-08  
Confirmatory objective evaluations before amendment: **0**

## Trigger

The final provenance review found that the result metadata label
`python_port_official_CEC2024_core` could be read as implying an official
executable.  Track B uses a transparent Python port, not the GPL C++ binary.

## Clarification

The implementation label is now
`transparent_python_port_public_CEC2024_core`.  The registered deviations are
NumPy random-number generation, arbitrary finite per-coordinate bounds and an
external objective, success-delta measurement before cyclic-front replacement,
stable elite reduction, and replacement-index normalization after population
reduction.

No numerical operation, parameter, seed, budget, problem partition or endpoint
was changed in this amendment.  The source-identity commit changes, so a final
development-only smoke remains mandatory before confirmatory access.
