# Track C implementation note 001: elliptic interface assembly

Status: **registered before any Track C objective evaluation**  
Date: 2026-08-08  
Track C confirmatory objective evaluations before correction: **0**

The development preflight identified a shape inconsistency in the one-dimensional elliptic finite-volume interface assembly. The `n+1` harmonic-interface sequence is now split into its left and right `n`-entry views (`k_half[:-1]` and `k_half[1:]`). This is the standard conservative tridiagonal assembly for `n` interior unknowns. No development, confirmatory or observed-data result was produced before this correction. The correction is covered by a permanent construction test and is part of the frozen Track C source identity.
