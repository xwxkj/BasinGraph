# BasinGraph [![DOI](https://zenodo.org/badge/1274531480.svg)](https://doi.org/10.5281/zenodo.20765883)

BasinGraph is a geometry-controlled optimization framework that maintains a
fixed-capacity archive of operational basin-state representatives and a
directed graph of observed search transitions.

## Selected result-bearing implementation

- Implementation: `BasinGraph 2.0.0-rc1`
- Selected-candidate tag: `route-b-v2.0.0-rc1-selected-final-candidate`
- Options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

Software release `v2.0.1` is a documentation-only cleanup. It does not change
the selected algorithm, benchmark code, frozen options, raw results or
numerical conclusions.

## Prospective validation

- Official noiseless COCO/BBOB holdout: 24 functions; dimensions 2, 3, 5, 10
  and 20; actual instances 4-15; seven algorithms; budget 1,000d; 10,080
  records.
- Prospective CUTEst holdout: 24 performance-independently selected instances;
  30 paired seeds; seven algorithms; 5,040 records.

COCO instances 1-3 and the 50-instance CUTEst development/comparability set
are retained separately and are not pooled with the prospective holdouts.

## Repository contents

- `basingraph_v2/`: selected optimizer implementation.
- `baselines/`: frozen comparator implementations and wrappers.
- `experiments_v2/`: COCO/BBOB and CUTEst runners.
- `analysis_v2/`: validation and analysis scripts.
- `scripts_v2/`: reproducibility and release workflows.
- `protocols/`: frozen protocols, machine contracts and manifests.
- `results_v2/final_analysis/`: lightweight final analysis tables and text.
- `environments/`: dependency and CUTEst toolchain records.

Large observer logs, atomic histories and raw archives are deposited on Zenodo.

## Installation

```bash
python -m pip install numpy scipy pandas matplotlib cma coco-experiment cocopp
```

CUTEst additionally requires SIFDecode, CUTEst, MASTSIF and PyCUTEst. See
`protocols/` and `environments/` for the frozen toolchain records.

## Reproducibility data

- Dataset version 2.0.1 DOI: https://doi.org/10.5281/zenodo.20791231
- Dataset all-versions DOI: https://doi.org/10.5281/zenodo.20765046

The v2.0.1 dataset contains the prospective COCO/BBOB and CUTEst evidence,
Source Data, protocols, manuscript files and integrity records. Obsolete
legacy implementation materials are intentionally excluded.

## Software archive

- Software all-versions DOI: https://doi.org/10.5281/zenodo.20765883
- Version 2.0.0 DOI: https://doi.org/10.5281/zenodo.20789002

The version-specific DOI for software release v2.0.1 is added to the current
branch after Zenodo completes the GitHub release archive.

## Citation

See `CITATION.cff`.

## License

BasinGraph is released under the BSD-3-Clause license. See `LICENSE`.
