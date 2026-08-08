# BasinGraph v2 formal experiment protocol

## Frozen algorithm

- implementation: `2.0.0-rc1`
- full options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`
- branch: `route-b/full-basingraph-v2`
- protocol-freeze commit: `438b9d302ba85e220bbd225b6b80cc71ec4f5090`
- rc1 ablation-freeze tag commit: `438b9d302ba85e220bbd225b6b80cc71ec4f5090`

## Non-negotiable inspection rule

No result from a prospective holdout partition may be inspected before the
final BasinGraph v2.0.0 code tag is created. Engineering changes and parameter
choices may use development partitions only.

## Official COCO/BBOB

### Development partition

- functions 1-24;
- dimensions 2, 5 and 10;
- instances 1-3;
- budget 1000d;
- seven algorithms.

### Prospective holdout

- functions 1-24;
- dimensions 2, 3, 5, 10 and 20;
- instances 4-15;
- budget 1000d;
- seven algorithms.

### Final report

After the final code tag, run instances 1-15 and report development (1-3) and
prospective holdout (4-15) separately before any pooled summary.

## CUTEst

### Development/comparability suite

The existing frozen 50-instance list is retained for development and
comparability with the historical implementation. Because prior results on
these problems have already been observed, it is not treated as a prospective
v2 holdout.

### Prospective holdout

Twenty-four instances are selected from the valid-but-not-selected technical
inventory with the deterministic seed `BasinGraph-v2-holdout-20260619`:

- 11 small;
- 7 medium;
- 6 large.

No optimizer result is used in selection. The holdout list must remain
uninspected until the final v2.0.0 code tag.

### Budget and seeds

- 30 paired seeds;
- budget `min(20000, max(1000, 50*n))`;
- seven algorithms.

## Ablation

Ablation uses development data only. It must not use the prospective COCO or
CUTEst holdouts. The seven frozen variants are:

- Full
- NoGraphGuidance
- SingleBracket
- NoFarBasin
- NoGeometryController
- NoArchiveFallback
- NoFinalPolish

## Result identity requirements

Every BasinGraph v2 formal result must contain:

- implementation version;
- options hash;
- exact Git commit;
- phase evaluation counts;
- explicit archive nodes;
- graph edges with referential integrity;
- diagnostics;
- event log;
- seed and budget;
- protocol manifest hash.

## Manuscript rule

Only results generated after the final v2.0.0 code tag and under this protocol
may replace the historical v1.0.0 evidence in the manuscript.
