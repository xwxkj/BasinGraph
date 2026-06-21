# rc2 implementation order

1. Add `ProbeRecord`, node certification metadata and result serialization.
2. Refactor archive so only certified nodes can be inserted.
3. Implement adaptive merge threshold and quality-diversity-accessibility
   eviction.
4. Implement curvature anisotropy and retain domain anisotropy separately.
5. Generalize the line search to arbitrary directions.
6. Implement principal-direction generation and refinement.
7. Convert far-basin and budget completion to probe-refine-commit batches.
8. Add graph degree pruning.
9. Update controller and frozen phase allocations.
10. Add unit tests for every semantic invariant.
11. Run smoke tests only.
12. Run paired rc1-versus-rc2 development evaluation.
13. Apply the frozen acceptance gate.
14. Only after acceptance, create the final v2.0.0 code tag and open holdout.
