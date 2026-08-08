# B21 Step 0 repository-identity contract

Status: **frozen for the NCS evidence-extension branch**  
Date: 2026-08-07

## Purpose

This contract separates the immutable result-bearing BasinGraph algorithm from
later documentation, packaging and independent evidence-extension work. It
must be checked before any B21 run is accepted.

## Authoritative algorithm identity

| Field | Frozen value |
|---|---|
| Manuscript-facing method name | `BasinGraph` |
| Internal implementation version | `2.0.0-rc1` |
| Selected-candidate tag | `route-b-v2.0.0-rc1-selected-final-candidate` |
| Selected-candidate commit | `adbc0ecdf1153044188f0508321c47001ad9bdb0` |
| Result-bearing package | `basingraph_v2/` |
| Result-bearing function | `basingraph_v2.optimizer.minimize_basingraph_v2` |
| Frozen options hash | `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69` |
| Archive capacity | `80` |
| Coarse coordinate samples | `11` |
| Retained refinement intervals | at most `3` |
| Phase fractions | `0.10, 0.15, 0.30, 0.15, 0.10, 0.05, remainder` |

The concise public import `basingraph.minimize_basingraph` must be the same
Python function object as `basingraph_v2.optimizer.minimize_basingraph_v2`.
No adapter may change the objective-call ledger, options, phase budgets,
archive, graph or result fields.

## Public archive identity

The authoritative, publicly resolvable records used by the manuscript are:

- result-bearing software archive v2.0.0:
  `10.5281/zenodo.20789002`;
- reproducibility dataset v2.0.0:
  `10.5281/zenodo.20788903`.

Historical release-preparation files may contain reserved or superseded DOI
values. Such values are provenance only and must not be used in the current
README, citation file, Code availability or Data availability statements.

## Original prospective evidence

The following records are immutable and are not rerun, replaced or pooled with
B21:

- official noiseless COCO/BBOB prospective holdout, actual instances 4–15;
- prospective 24-problem CUTEst holdout with 30 paired seeds;
- associated raw logs, Source Data, analysis outputs and integrity records.

## B21 evidence-extension boundary

B21 runners are stored under `evidence_extension_v1/`. They may:

- execute independent smoke tests, mechanism ablations, modern baselines,
  scientific applications and scalability experiments;
- add wrappers, protocol files, validation scripts and new result directories;
- use new, disjoint benchmark instances defined by a frozen B21 protocol.

They may not:

- modify `basingraph_v2/`;
- alter `BasinGraphOptions()` defaults or the frozen options hash;
- overwrite the original prospective result directories;
- represent development or smoke outputs as confirmatory evidence;
- change claims after results are inspected without starting a new registered
  development cycle.

## Manifest policy

The root `MANIFEST_SHA256.csv` is generated from tracked files by
`scripts_v2/generate_repository_manifest.py`. The generator excludes the
manifest itself to avoid self-reference. Every B21 result directory also
contains its own SHA-256 manifest.

## Step 0 acceptance checks

Step 0 passes only when all of the following hold:

1. public import and result-bearing import are identical;
2. implementation version and frozen options hash match this contract;
3. archive capacity and phase fractions match this contract;
4. README, citation and availability files use the authoritative public DOI
   records above;
5. a deterministic smoke run exhausts its prescribed budget, has exact phase
   accounting and preserves graph–archive referential integrity;
6. the repository manifest verifies without missing or mismatched files.
