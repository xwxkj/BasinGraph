# Track E2-C and PDE v2.1 Apple Silicon replication audit

Status: **passed**  
Returned ZIP SHA-256: `d19c454ea9bbeeb11b82462b083d211475d19b1ac9c642f6c869028288be7aa8`  
Frozen source commit: `480b296a0939270351bdd40105079a67780cc9cb`  
Optimizer blob: `4433d2b92075dcd858529f7f13342631847def4c`  
Implementation: `2.0.0-rc1`  
Options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## Archive integrity

- uploaded checksum matched independent SHA-256 recomputation;
- ZIP CRC check passed;
- unsafe paths: 0;
- symbolic links: 0;
- Track E2-C result manifest: 3/3 entries valid;
- PDE v2.1 result manifest: 4/4 entries valid;
- independent audit checks: 25/25 passed.

## Track E2-C replication

The returned Apple Silicon package reproduced the registered decision
`TRACK_E2C_ALIGNMENT_SPECIFICITY_SUPPORTED`.

- parent runs: 288;
- snapshots: 864;
- future-progress events: 36 positive and 828 negative;
- trace-only log loss: `0.11450343`;
- correctly aligned state log loss: `0.09702068`;
- permutation-ensemble log loss: `0.10664198`;
- relative reduction versus trace: `15.27%`;
- relative reduction versus permutation ensemble: `9.02%`;
- trace-minus-state bootstrap 95% interval:
  `[0.0097436, 0.0255222]`;
- permutation-minus-state bootstrap 95% interval:
  `[0.0008842, 0.0190597]`.

All six registered decision conditions remained true. Relative to Ubuntu CI,
the log-loss values changed slightly, but both bootstrap lower bounds remained
strictly positive and the confirmatory decision was unchanged.

Matrix and integrity checks also passed:

- six task families;
- dimensions 10 and 20;
- instances 201--224;
- three snapshots for each of 288 parent tasks;
- parent phase counts equalled parent budgets;
- best-so-far histories were monotone;
- snapshot graph--archive referential integrity passed;
- archive capacity never exceeded 80.

## PDE flagship v2.1 replication

The returned Apple Silicon package reproduced the registered decision
`PDE_FLAGSHIP_V2_1_CONFIRMATORY_SUCCESS`.

- runs: 80;
- tasks: 8;
- at least four separated explanations: 8/8 tasks;
- D-optimal score gain at least 25%: 8/8 tasks;
- effective candidate-count reduction at least 20%: 8/8 tasks;
- weighted parameter-dispersion reduction at least 20%: 8/8 tasks;
- selected design no worse than the median design: 8/8 tasks;
- median retained explanations: 16;
- median D-optimal score gain: `42.41%`;
- median effective-count reduction: `93.67%`;
- median weighted-dispersion reduction: `99.9860%`.

The Apple Silicon replication was at least as strong as Ubuntu CI on every
registered task-count gate. Seven of eight selected designs matched exactly;
the remaining task selected another member of the same preregistered
intervention family. The aggregate decision was unchanged.

All 80 runs exhausted their registered budgets, phase-count sums equalled
objective-call counts, returned graph endpoints were valid, archive capacity
never exceeded 80, and each task evaluated all 130 registered candidate
designs.

## Decision

The cross-platform replication gate is passed. PR #8 may be merged and the
following bounded claims may be incorporated into the manuscript:

1. correctly aligned operational state contains out-of-sample predictive
   information about the registered future-progress event beyond both the
   best-so-far trace and permutation-control state distributions;
2. a finite operational archive can expose multiple baseline-compatible PDE
   coefficient explanations and guide an informative additional measurement
   for that finite explanation set.

The following limitations remain mandatory:

- no demonstrated continuation-performance improvement;
- no sufficient-statistic or arbitrary-time exact-replay claim;
- no complete inverse-solution enumeration;
- no global PDE identifiability claim.
