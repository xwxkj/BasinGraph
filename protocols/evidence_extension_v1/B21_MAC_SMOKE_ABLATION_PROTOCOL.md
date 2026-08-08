# B21 Mac smoke and ablation-pilot protocol

Status: **engineering validation only; not confirmatory manuscript evidence**  
Frozen date: 2026-08-07

## Purpose

This first B21 batch verifies that the Step 0 repository repair is executable on
macOS and that every registered mechanism switch changes real search behaviour
without violating budget accounting or graph/archive integrity. It does not
replace the later confirmatory COCO/BBOB ablation on disjoint instances 16–20.

## Immutable optimizer

All runs use the result-bearing BasinGraph source identified by
`STEP0_IDENTITY_CONTRACT.json`. No file under `basingraph_v2/` may be edited.

## Variants

1. `Full`
2. `NoGraphGuidance`
3. `SingleBracket`
4. `NoFarBasin`
5. `NoGeometryController`
6. `NoArchiveFallback`
7. `NoFinalPolish`
8. `NoCenterLocal`

Unused phase allocations pass to `budget_completion`. No ablation-specific
retuning is allowed.

`NoCenterLocal` is implemented only through frozen options
(`center_local_max_dim=-1` and `local_mode_min_score=inf`); the optimizer source
is unchanged.

## Diagnostic tasks

The pilot uses seven deterministic engineering tasks chosen before execution:

- shifted sphere, 5D;
- Rosenbrock, 5D;
- shifted Rastrigin, 5D;
- shifted Ackley, 5D;
- rotated ellipsoid, 10D;
- far-basin double well, 5D;
- boundary quartic, 8D.

The `smoke` mode uses the shifted sphere, shifted Rastrigin and far-basin
double-well subset. These tasks are not part of the planned confirmatory
BBOB/CUTEst extension.

## Modes

| Mode | Tasks | Variants | Seeds | Runs | Purpose |
|---|---:|---:|---:|---:|---|
| `smoke` | 3 | 8 | 1 | 24 | installation and mechanism-path check |
| `pilot` | 7 | 8 | 2 | 112 | first Mac engineering ablation |
| `full-mini` | 7 | 8 | 5 | 280 | optional expanded engineering check |

## Required integrity conditions

- implementation version equals `2.0.0-rc1`;
- Full options hash equals the frozen hash;
- every completed run uses exactly its prescribed evaluation budget;
- phase counts sum exactly to `nfe`;
- archive size remains in `[1, 80]`;
- every graph edge references two active archive nodes;
- disabled phases have zero evaluations where applicable;
- every ablation changes at least one paired trajectory relative to Full;
- raw records, specifications, metadata and validation outputs are sealed by a
  directory-level SHA-256 manifest.

## Interpretation rule

Rank and gap summaries from this pilot are diagnostic. They may reveal coding,
budget or mechanism problems, but they must not be cited as confirmatory
performance evidence. If an ablation outperforms Full, the output is retained
and investigated; it is not deleted or silently tuned away.
