# B21 NCS evidence extension

This directory contains independent evidence-extension runners. It does not
modify the result-bearing BasinGraph implementation or the original prospective
COCO/BBOB and CUTEst records.

## First Mac batch

Quick smoke:

```bash
chmod +x START_B21_SMOKE.command
./START_B21_SMOKE.command
```

Engineering ablation pilot:

```bash
chmod +x START_B21_ABLATION_PILOT.command
./START_B21_ABLATION_PILOT.command
```

The launchers create `.venv-b21`, install the repository in editable mode,
generate and verify the repository manifest, run the Step 0 identity audit,
execute the selected B21 mode, seal the result directory and produce a ZIP plus
SHA-256 file for return.

Available runner modes:

- `smoke`: 32 runs, including a 25D controller-activation probe;
- `pilot`: 112 runs;
- `full-mini`: 280 runs.

Direct invocation:

```bash
bash evidence_extension_v1/run_b21_mac.sh full-mini
```

The smoke and pilot outputs are engineering evidence only. They must not be
pooled with later confirmatory experiments.
