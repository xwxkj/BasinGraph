# B21 Track B: modern strong-baseline comparison

Track B is a new evidence extension. It does not alter BasinGraph, the original
COCO/BBOB instances 4–15, the Track A instances 16–20, or the prospective
CUTEst holdout.

## Mac sequence

Run the development-only smoke first:

```bash
bash START_B21_TRACK_B_SMOKE.command
```

After the returned smoke package is audited, run the confirmatory matrix:

```bash
bash START_B21_TRACK_B_CONFIRMATORY.command
```

Both commands place upload-ready files in:

```text
~/Downloads/B21_RESULTS_READY/trackB_<mode>_<timestamp>/
```

Finder opens automatically and highlights the main ZIP.

## Confirmatory matrix

```text
24 functions × 2 dimensions × 10 new instances × 8 algorithms
= 3,840 runs
```

Actual COCO/BBOB instances are 21–30. They are disjoint from development
instances 1–3, the original holdout instances 4–15 and Track A instances
16–20.

## Algorithm identity

BasinGraph is the immutable result-bearing implementation. CMA-ES and
BIPOP-CMA-ES use pycma; DIRECT-L and multi-start L-BFGS-B use SciPy. L-SHADE
1.0.1, jSO and L-SRTDE are transparent frozen Python ports, with their source
basis and deviations recorded in the protocol. They are not described as
byte-identical official executables.
