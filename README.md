# BasinGraph

[![DOI](https://zenodo.org/badge/1274531480.svg)](https://doi.org/10.5281/zenodo.20765883)

BasinGraph is a geometry-controlled basin-graph optimization framework for
nonconvex mathematical models.

## Repository contents

- `basingraph/`: proposed optimizer.
- `baselines/`: standardized benchmark implementations and wrappers.
- `experiments/`: COCO/BBOB and benchmark runners.
- `scripts/`: reproducible experiment and analysis entry points.
- `protocols/`: frozen benchmark protocols and manifests.
- `reproducibility/`: lightweight Source Data, summary tables, and metadata.
- `environments/`: dependency locks and CUTEst environment information.

Large COCO observer logs, CUTEst run-level histories, and full raw archives are
not committed to GitHub. They are deposited on Zenodo at https://doi.org/10.5281/zenodo.20765047.

## Main validation

- Official noiseless COCO/BBOB: 24 functions, dimensions 2, 3, 5, 10 and 20,
  15 instances, and a budget of 1,000d evaluations.
- Pre-registered CUTEst: 50 instances, seven algorithms, and 30 paired seeds.
- Independent MATLAB full-parallel validation and applied-mathematics tests.

## Installation

Create the COCO environment from the archived dependency lock or install:

```bash
python -m pip install numpy scipy pandas matplotlib cma coco-experiment cocopp
```

CUTEst requires SIFDecode, CUTEst, MASTSIF and PyCUTEst. See
`protocols/` and `environments/` for the frozen toolchain information.

## Reproduction

Experiment entry points are documented in `scripts/`. The final public release
will contain a DOI-backed Zenodo archive with raw logs and convergence histories.

## Citation

See `CITATION.cff`.

## License

BasinGraph is released under the BSD-3-Clause license. See `LICENSE`.

## Software archive

- Version 1.0.0 DOI: https://doi.org/10.5281/zenodo.20765884
- All-versions DOI: https://doi.org/10.5281/zenodo.20765883
- Reproducibility Dataset DOI: https://doi.org/10.5281/zenodo.20765047

## Data and reproducibility archive

The official COCO/BBOB logs, pre-registered CUTEst results, MATLAB
validation, Source Data and frozen protocols are openly available on
Zenodo:

- Version-specific Dataset DOI: https://doi.org/10.5281/zenodo.20765047
- All-versions Dataset DOI: https://doi.org/10.5281/zenodo.20765046

The version-specific DOI identifies the exact files supporting the
reported study.
