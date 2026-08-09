# BasinGraph

[![Software DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21857175.svg)](https://doi.org/10.5281/zenodo.21857175)
[![Clean evidence DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21863259.svg)](https://doi.org/10.5281/zenodo.21863259)

BasinGraph is a geometry-controlled optimizer for finite-budget nonconvex and
scientific optimization. It maintains a fixed-capacity archive of operational
search-state representatives and a directed graph of observed transitions,
while counting every objective call in a single evaluation ledger.

## Result-bearing implementation

The implementation used throughout the manuscript evidence is immutable and is
identified by all of the following:

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

## Evidence base

### Original prospective holdouts

- Official noiseless COCO/BBOB holdout: functions 1–24; dimensions 2, 3, 5,
  10 and 20; actual instances 4–15; seven algorithms; budget `1,000d`;
  10,080 run-level records.
- Prospective CUTEst holdout: 24 performance-independently selected problems;
  30 paired seeds; seven algorithms; 5,040 run-level records.

COCO/BBOB instances 1–3 and the 50-problem CUTEst development/comparability
set are retained separately and are not pooled with the prospective holdouts.

### Registered evidence extensions

The v3.1.0 evidence release preserves the frozen candidate and adds disjoint,
registered evaluations:

- Track A: 1,920-run mechanism ablation;
- Track B: 3,840-run modern-baseline comparison;
- Track D: 1,440-run `bbob-largescale` comparison in 40–320 dimensions;
- Track D overhead: 60 sequential optimizer-overhead runs;
- Track C: 2,310-run scientific-model and NIST observed-data matrix;
- Track C references: 18 separately reported task-specific reference runs.

This is 9,510 confirmatory general-purpose runs, 60 sequential overhead runs
and 18 descriptive task-specific references. Favorable, null and unfavorable
outcomes are retained. The evidence supports a bounded finite-budget
competitiveness claim; it does not support a claim that BasinGraph is the best
overall optimizer or that every mechanism improves aggregate performance.

## Repository layout

- `basingraph_v2/`: frozen result-bearing optimizer implementation.
- `basingraph/`: public compatibility import that forwards to
  `basingraph_v2/`.
- `baselines/`: frozen comparator implementations and wrappers.
- `experiments_v2/`: prospective COCO/BBOB and CUTEst runners.
- `analysis_v2/`: validation and analysis scripts.
- `scripts_v2/`: reproducibility, integrity and release workflows.
- `protocols/`: frozen protocols, machine contracts, decisions and manifests.
- `results_v2/final_analysis/`: lightweight final analysis outputs.
- `evidence_extension_v1/`: registered Track A–D/C runners and protocols;
  these do not alter the frozen candidate or original holdouts.

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

## Public archives

| Resource | Version | DOI |
|---|---:|---|
| Current software archive | 3.1.0 | `10.5281/zenodo.21857175` |
| Submission-facing clean evidence record | 3.1.0-clean | `10.5281/zenodo.21863259` |
| Complete timestamped priority/provenance record | 3.1.0 | `10.5281/zenodo.20791231` |
| Original result-bearing software archive | 2.0.0 | `10.5281/zenodo.20789002` |
| Original prospective reproducibility dataset | 2.0.0 | `10.5281/zenodo.20788903` |

The clean evidence record excludes author manuscripts, redline drafts, internal
submission packages and deposition-management files. It retains the complete
scientific evidence, Source Data, Supplementary Information, figures, logs,
analyses, provenance reports and integrity manifests. The complete v3.1.0
record remains unchanged as the earliest full public priority and provenance
record.

The v3.1.0 software archive is identical to release-source commit
`2084438e679f61660de497f5464fbe5636373163`. The result-bearing optimizer itself
remains candidate commit `adbc0ecdf1153044188f0508321c47001ad9bdb0` with the
frozen options hash shown above.

## Licence

BasinGraph source code is released under the BSD-3-Clause licence. Research
outputs and documentation in the evidence records are released under CC BY
4.0. Benchmark libraries and externally developed comparator implementations
retain their original licences.
