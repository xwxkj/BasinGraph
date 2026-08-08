# Track C development-only smoke protocol

Smoke uses paired seed 0 and development instance 1 on:

- elliptic PDE inverse estimation;
- noiseless phase retrieval;
- Allen–Cahn energy minimization;
- NIST BoxBOD;
- NIST ENSO.

Seven algorithms produce 35 runs. Smoke verifies task construction, NIST data
integrity, all optimizer wrappers, exact budgets, histories, metrics, result
packaging and the local confirmatory gate. Smoke is not confirmatory evidence.
