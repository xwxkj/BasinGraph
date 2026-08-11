# Bounded operational state: definitions, propositions and claim boundary

Status: **formalized against the immutable BasinGraph 2.0.0-rc1 source**  
Selected candidate commit: `adbc0ecdf1153044188f0508321c47001ad9bdb0`  
Frozen options hash: `031b9c3df716889e48e2db753c73ec960b96a0239173ce791b4ed1ee63ed0f69`

## 1. Search histories and best-so-far traces

Let a finite-budget search history through evaluation \(t\) be

\[
\mathcal H_t = ((\mathbf{x}_1,f_1,\phi_1),\ldots,(\mathbf{x}_t,f_t,\phi_t)),
\]

where \(\mathbf{x}_i\in\mathbb R^d\) is the evaluated point, \(f_i\) is the
returned objective value and \(\phi_i\) is the phase responsible for the call.
The scalar best-so-far trace is

\[
b_i = \min_{1\le j\le i} f_j,\qquad i=1,\ldots,t.
\]

The trace records incumbent progress. It does not record the visited points,
non-incumbent values, phases, retained representatives or observed
transitions.

### Theorem 1 | Non-identifiability of search history from the best-so-far trace

For every dimension \(d\ge1\) and every \(t\ge2\), the map
\(\mathcal H_t\mapsto(b_1,\ldots,b_t)\) is not injective. In particular, there
exist distinct histories with identical best-so-far traces, different visited
regions and different operational states.

**Proof.** Choose three distinct points \(\mathbf{x}_\star,\mathbf{a},\mathbf{b}\)
in the search domain. Let the first evaluation in both histories be
\((\mathbf{x}_\star,0,\phi_1)\). Let the second evaluation be
\((\mathbf{a},1,\phi_2)\) in the first history and
\((\mathbf{b},1,\phi_2)\) in the second. Complete both histories with any
sequence of non-negative objective values while keeping the evaluated points
different. Both best-so-far traces are identically zero, but the visited sets
and any archive or graph constructed from them can differ. Therefore the trace
does not identify the history or the operational state. \(\square\)

### Corollary 1 | A scalar trace is not sufficient for state-dependent allocation

An allocation rule that can condition on retained locations or observed
transitions may select different future evaluations for two histories with the
same best-so-far trace. This is an information statement; it does not imply
that the state-dependent rule must obtain a better objective value.

## 2. Effective active state, raw graph container and audit ledger

Let \(A_t\) be the active archive. Let \(\widetilde G_t\) denote the raw internal
graph container, and define its active projection

\[
G_t^{\mathrm{act}}=
\{(u,v)\in\widetilde G_t:u\in A_t,\ v\in A_t\}.
\]

The effective active operational state is

\[
\mathcal S_t^{\mathrm{act}}=
(\mathbf{x}^{\mathrm{best}}_t,A_t,G_t^{\mathrm{act}},D_t,P_t),
\]

where \(D_t\) contains fixed-size geometry diagnostics and \(P_t\) contains
fixed-size option and phase information. Track E2 computes its state features
from this sanitized active projection.

The append-only audit ledger \(L_t\) stores one best-so-far record for every
successfully completed objective call and phase-resolved counts. The event log
is also append-only. Audit records are not part of the bounded active state.

### Theorem 2 | Storage bound for the effective active state

Let the archive capacity be \(C\). The effective active operational state
requires

\[
\mathcal O(Cd+C^2)
\]

scalar storage. For the frozen value \(C=80\), the effective active-state
storage is \(\mathcal O(d)\) as decision dimension grows.

**Proof.** Each archive node stores one \(d\)-dimensional centre and a fixed
number of scalars. The archive evicts one current worst node whenever insertion
would make its size exceed \(C\), so \(|A_t|\le C\) and archive storage is
\(\mathcal O(Cd)\).

The active projected graph contains at most one edge for each ordered pair of
active identifiers because graph edges are keyed by `(source_id, target_id)`;
self-loops are rejected. Hence

\[
|G_t^{\mathrm{act}}|\le C(C-1),
\]

and projected-graph storage is \(\mathcal O(C^2)\). The incumbent contributes
\(\mathcal O(d)\); diagnostics, options and phase counters are fixed-size.
Summing the terms gives \(\mathcal O(Cd+C^2)\). \(\square\)

### Limitation 1 | The raw transient graph container is not proven bounded by \(C^2\)

Track E snapshots exposed an operation ordering in which a newly created node
can be evicted by the archive-capacity rule and subsequently receive an edge in
the same higher-level step. Such an edge has a target outside the active
archive. Final cleanup removes it, and Track E2 removes it from copied
snapshots, but the raw internal container can retain inert stale-target edges
between cleanup boundaries. The immutable implementation therefore does not
provide a strict \(\mathcal O(C^2)\) worst-case bound for the unsanitized raw
graph container during an arbitrary run.

Stale-target edges do not enter the active projection and are not queried as
incoming edges of an active target. The rigorous bound applies to the
decision-relevant sanitized projection and to the returned graph, not to the
raw transient container.

### Limitation 2 | The full audit record is not fixed-capacity

The best-so-far history, objective-call ledger and event log grow with the
number of evaluations \(T\), requiring \(\mathcal O(T)\) records. The precise
claim is:

> BasinGraph has a fixed-capacity active archive, a bounded sanitized active
> graph projection and an append-only audit ledger.

It is not correct to describe the entire runtime container or complete
serialized result as fixed-capacity.

## 3. Returned-state integrity

### Proposition 1 | Archive-capacity invariant

Every returned BasinGraph result satisfies \(|A_T|\le C\).

**Justification.** `BasinArchive.add_or_merge` appends at most one node and,
whenever the resulting length exceeds `max_nodes`, immediately removes one
current worst node. No other operation inserts nodes into the active archive.

### Proposition 2 | Returned graph--archive referential integrity

Every edge in the returned graph has both endpoints in the returned active
archive.

**Justification.** Archive eviction requests removal of edges incident to the
evicted identifiers. In addition, immediately before constructing the result,
the optimizer computes active identifiers and removes every graph endpoint not
in that set. Hence the returned object equals its active projection and has
referential integrity.

### Limitation 3 | Arbitrary mid-run snapshots require sanitation

An arbitrary mid-run snapshot must project graph endpoints onto the copied
active archive. No claim of unsanitized arbitrary-time integrity or exact
arbitrary-time replay is made.

## 4. Exact objective-call accounting

### Proposition 3 | One successful objective call, one ledger increment

For every successfully completed call to `EvaluationLedger.evaluate`:

1. the objective is evaluated exactly once;
2. the global counter `nfe` increases by one;
3. exactly one current-phase counter increases by one;
4. exactly one best-so-far history record is appended.

Budget and phase-limit exceptions occur before the objective call and before
any counter increment.

### Corollary 2 | Accounting identities

At every returned result,

\[
|L_T|=T=\sum_{\phi} n_{\phi},
\]

where \(|L_T|\) is the number of best-so-far records and \(n_\phi\) is the
number of objective calls assigned to phase \(\phi\).

### Proposition 4 | Monotone incumbent trace

The returned best-so-far sequence is non-increasing.

**Proof.** The ledger changes `fbest` only when a finite new value is strictly
smaller. Every appended history value is the current `fbest`. \(\square\)

## 5. Predictive information established by Track E2

Theorem 1 establishes only that state can contain information absent from the
trace. Track E2 tests whether some of that information predicts a registered
future-progress event out of sample. The successful Track E2-B and E2-C
partitions support alignment-specific predictive information for the event
`future_log_improvement >= 0.10`. They do not convert Theorem 1 into a claim of
continuation-performance improvement or optimal-state sufficiency.

## 6. What is and is not established

Established:

- best-so-far traces do not identify search histories;
- the archive and sanitized active graph have an explicit storage bound;
- returned graph endpoints are active representatives;
- each successful objective call is assigned to one ledger and one phase;
- returned best-so-far histories are monotone;
- the registered sanitized state features carry out-of-sample predictive
  information beyond the trace for one future-progress event.

Not established:

- topology reconstruction of mathematical attraction basins;
- a strict fixed-capacity bound for the raw transient graph container;
- sufficiency of the active state for optimal continuation;
- continuation-performance improvement from retaining state;
- exact arbitrary-time resume or replay;
- asymptotic time complexity from empirical scaling slopes.

Exact continuation would additionally require a formally defined safe
checkpoint boundary, complete random-generator state, all loop-local control
variables and a continuation function proven to depend only on the serialized
checkpoint.
