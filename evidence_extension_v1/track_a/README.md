# B21 Track A: confirmatory mechanism ablation

Track A is an evidence extension. It does not replace or modify the original
COCO/BBOB instances 4–15 or the prospective CUTEst holdout.

## Mac sequence

1. Run the development-only smoke:

```bash
bash START_B21_TRACK_A_SMOKE.command
```

2. Return the automatically opened result files for audit.

3. Only after the smoke gate passes, run the confirmatory matrix:

```bash
bash START_B21_TRACK_A_CONFIRMATORY.command
```

Both commands place all upload files in:

```text
~/Downloads/B21_RESULTS_READY/trackA_<mode>_<timestamp>/
```

Finder opens automatically and highlights the main ZIP.

The confirmatory design is frozen at 1,920 runs:

```text
24 functions × 2 dimensions × 5 new instances × 8 variants
```

Actual COCO/BBOB instances are 16–20. They are disjoint from development
instances 1–3 and the original prospective holdout instances 4–15.
