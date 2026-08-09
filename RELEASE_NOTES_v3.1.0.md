# BasinGraph v3.1.0

## Release status

Published 8 August 2026.

- Software DOI: https://doi.org/10.5281/zenodo.21857175
- Submission-facing clean evidence DOI: https://doi.org/10.5281/zenodo.21863259
- Complete timestamped priority/provenance DOI: https://doi.org/10.5281/zenodo.20791231
- Release-source commit: `2084438e679f61660de497f5464fbe5636373163`
- Result-bearing candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`
- Result-bearing implementation: `2.0.0-rc1`
- Frozen options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## Scope

Version 3.1.0 is an evidence-integration, documentation and archive release. It
does not modify the result-bearing BasinGraph candidate, its frozen options or
any completed confirmatory outcome.

The release integrates:

- the original prospective COCO/BBOB and CUTEst holdouts;
- Track A: 1,920-run mechanism ablation;
- Track B: 3,840-run modern-baseline comparison;
- Track D: 1,440-run 40-320-dimensional `bbob-largescale` comparison;
- Track D overhead: 60 sequential optimizer-overhead runs;
- Track C: 2,310-run scientific-model and NIST observed-data matrix;
- Track C: 18 separately reported task-specific references;
- the 21-sheet Source Data workbook;
- Supplementary Information and publication figures;
- frozen protocols, source identities, validation reports and SHA-256 manifests;
- explicit evidence-to-claim boundaries.

The extension therefore contains 9,510 general-purpose confirmatory runs, 60
sequential overhead runs and 18 descriptive task-specific references.

## Clean public evidence edition

The preferred submission-facing reproducibility record is the independent
clean public evidence edition at https://doi.org/10.5281/zenodo.21863259. It
excludes author manuscripts, redline drafts, internal submission packages,
internal claim-audit documents and deposition-management files while retaining
all scientific evidence, Source Data, Supplementary Information, figures,
logs, analyses, provenance reports and integrity records.

The complete record at https://doi.org/10.5281/zenodo.20791231 remains public
and unchanged as the earliest full timestamped priority and provenance record.
No code, data, protocol, run, statistic, figure or numerical conclusion differs
between the clean and complete records.

## Main evidence boundary

The evidence supports the following bounded conclusion:

> BasinGraph is a competitive general-purpose optimizer under tight and
> intermediate evaluation budgets, with a fixed-capacity operational archive,
> auditable observed-transition graph and exact evaluation accounting.

The release does not support claims that BasinGraph is the best overall
optimizer, uniformly outperforms CMA-ES, BIPOP-CMA-ES or multi-start
L-BFGS-B, or that every archive, graph, bracket, exploration, fallback,
polishing or controller mechanism improves aggregate performance.

## Licences

- BasinGraph source code: BSD-3-Clause.
- Integrated research outputs and documentation: CC BY 4.0.
- External benchmark libraries, datasets and comparator implementations retain
  their original licences.
