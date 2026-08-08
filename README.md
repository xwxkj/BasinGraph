# BasinGraph

[![Result-bearing software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20789002.svg)](https://doi.org/10.5281/zenodo.20789002)
[![Reproducibility dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20788903.svg)](https://doi.org/10.5281/zenodo.20788903)

BasinGraph is a geometry-controlled basin-graph optimizer for nonconvex
mathematical models. It maintains a fixed-capacity archive of operational
search-state representatives and a directed graph of observed transitions,
while counting every objective call in a single evaluation ledger.

## Result-bearing implementation

The implementation that produced the reported prospective COCO/BBOB and
CUTEst results is immutable and is identified by all of the following:

- algorithm name in the manuscript: `BasinGraph`;
- internal implementation version: `2.0.0-rc1`;
- selected-candidate tag: `route-b-v2.0.0-rc1-selected-final-candidate`;
- selected-candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`;
- source package: `basingraph_v2/`;
- result-bearing entry point: `basingraph_v2.optimizer.minimize_basingraph_v2`;
- frozen options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`;
- active-archive capacity: `80`.

The public compatibility API forwards to that exact implementation:

```python
from basingraph import BasinGraphOptions, minimize_basingraph

result = minimize_basingraph(
    objective,
    lower_bounds,
    upper_bounds,
    max_evals=10_000,
    seed=0,
    options=BasinGraphOptions(),
)
```

The historical simplified optimizer formerly exposed from
`basingraph/optimizer.py` is not the result-bearing manuscript implementation.
Its history remains available through earlier repository commits and tags.

## Prospective evidence already reported

- Official noiseless COCO/BBOB holdout: functions 1–24; dimensions 2, 3, 5,
  10 and 20; actual instances 4–15; seven algorithms; budget `1,000d`;
  10,080 run-level records.
- Prospective CUTEst holdout: 24 performance-independently selected problems;
  30 paired seeds; seven algorithms; 5,040 run-level records.

COCO/BBOB instances 1–3 and the 50-problem CUTEst development/comparability
set are retained separately and are not pooled with the prospective holdouts.

## Repository layout

- `basingraph_v2/`: frozen result-bearing optimizer implementation.
- `basingraph/`: public compatibility import that forwards to
  `basingraph_v2/`.
- `baselines/`: frozen comparator implementations and wrappers.
- `experiments_v2/`: prospective COCO/BBOB and CUTEst runners.
- `analysis_v2/`: validation and analysis scripts.
- `scripts_v2/`: reproducibility, integrity and release workflows.
- `protocols/`: frozen protocols, machine contracts and manifests.
- `results_v2/final_analysis/`: lightweight final analysis outputs.
- `evidence_extension_v1/`: newly registered NCS evidence-extension runners;
  these do not alter the frozen candidate or the original holdouts.

## Installation

Python 3.11 or later is recommended.

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

COCO/BBOB additionally requires `cma`, `coco-experiment` and `cocopp`.
CUTEst additionally requires SIFDecode, CUTEst, MASTSIF and PyCUTEst; the
frozen toolchain is documented under `protocols/` and `environments/`.

## Identity audit

Run the repository identity audit before any new experiment:

```bash
python scripts_v2/audit_step0_identity.py
```

A deterministic repository manifest can be generated with:

```bash
python scripts_v2/generate_repository_manifest.py
```

## Public archives used by the manuscript

| Resource | Public version-specific DOI |
|---|---|
| Result-bearing software archive v2.0.0 | `10.5281/zenodo.20789002` |
| Reproducibility dataset v2.0.0 | `10.5281/zenodo.20788903` |

The GitHub release `v3.0.0` is a documentation and archive-composition release.
It does not change the result-bearing optimizer, frozen options, benchmark
runs, raw results, statistical analyses or numerical conclusions. Reserved or
unverified DOI values in historical release-preparation files are retained only
as provenance and are not used as the authoritative manuscript citations.

## Licence

BasinGraph source code is released under the BSD-3-Clause licence. Benchmark
libraries and externally developed comparator implementations retain their
original licences.
