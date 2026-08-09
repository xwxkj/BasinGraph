# B21 Track C

Track C evaluates the immutable BasinGraph candidate on nine scientific-model
families and six NIST StRD observed-data nonlinear-regression datasets.

The registered general-purpose confirmatory matrix contains 2,310 runs. Run the
development-only smoke first:

```bash
bash START_B21_TRACK_C_SMOKE.command
```

After the returned smoke package is audited, run:

```bash
bash START_B21_TRACK_C_CONFIRMATORY.command
```

Results are placed in `~/Downloads/B21_RESULTS_READY/`, and Finder opens the
exact delivery folder automatically.
