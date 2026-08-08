# B21 NCS evidence extension

This directory contains independent evidence-extension runners. It does not
modify the result-bearing BasinGraph implementation or the original prospective
COCO/BBOB and CUTEst records.

## Mac runners

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

Direct full-mini invocation:

```bash
bash evidence_extension_v1/run_b21_mac.sh full-mini
```

The launchers create `.venv-b21`, install the repository in editable mode,
generate and verify the repository manifest, run the Step 0 identity audit,
execute the selected B21 mode, seal the result directory and produce a ZIP plus
SHA-256 file for return.

## Automatic result delivery

Every successful Mac run copies all user-facing outputs into one fixed root:

```text
~/Downloads/B21_RESULTS_READY/<mode>_<UTC timestamp>/
```

The directory contains the upload ZIP, its adjacent `.zip.sha256` file,
`validation_report.json`, `variant_summary.csv`,
`paired_differences_vs_full.csv`, `run_metadata.json`, the result manifest and
the Step 0 identity audit. At completion, Finder opens that exact directory and
highlights the ZIP automatically. No manual file search is required.

Set `B21_DELIVERY_DIR` only when a different delivery root is deliberately
required.

Available runner modes:

- `smoke`: 32 runs, including a 25D controller-activation probe;
- `pilot`: 112 runs;
- `full-mini`: 280 runs.

The smoke, pilot and full-mini outputs are engineering evidence only. They must
not be pooled with later confirmatory experiments.
