# Track B Amendment 002: source-alignment corrections

Status: **registered before confirmatory evaluation**  
Date: 2026-08-08  
Confirmatory objective evaluations before amendment: **0**

## Trigger

A second, line-by-line provenance audit compared the transparent Python ports
with the corrected L-SHADE release and public jSO CEC-2017 C++ source.  The
review identified source-level rules that were simplified in the first Track B
port: strict-success memory admission, the negative CR-memory sentinel,
source-style external-archive replacement, C++ count rounding, and the jSO
fixed sampling slot, half-old/half-new memory update and early p-best target
restriction.

## Registered correction

The L-SHADE and jSO ports now implement those rules explicitly.  The objective
wrapper, arbitrary finite bounds, NumPy random-number generator, seeds, budget,
problem partition, endpoints and all other algorithms remain unchanged.  The
ports remain transparent research ports and are not described as byte-identical
official executables.

## Consequence

The original local smoke and the Amendment-001 CI smoke remain retained as
infrastructure records but are superseded as baseline-fidelity gates.  A new
development-only smoke on the Amendment-002 source is mandatory before any
Track B confirmatory access.
