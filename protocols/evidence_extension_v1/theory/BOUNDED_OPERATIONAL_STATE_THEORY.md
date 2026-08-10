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

### Corollary 1 | A scalar trace is not a sufficient representation for state-dependent allocation

Any allocation rule that can condition on retained locations or observed
transitions may select different future evaluation points for two histories
with the same best-so-far trace. The corollary concerns information content;
it does not imply that the state-dependent rule must obtain a better objective
value.

## 2. Active operational state and audit ledger

At an evaluation boundary, define the active operational state

\[
\mathcal S_t=(\mathbf{x}^{\mathrm{best}}_t,A_t,G_t,D_t,P_t),
\]

where:

- \(\mathbf{x}^{\mathrm{best}}_t\in\mathbb R^d\) is the incumbent;
- \(A_t\) is the active archive of operational representatives;
- \(G_t\) is the directed graph of observed transitions between active
  representatives;
- \(D_t\) contains the fixed-size geometry diagnostics;
- \(P_t\) contains fixed-size phase and option information.

The append-only audit ledger is denoted \(L_t\). It stores one best-so-far
record for every successfully completed objective call and phase-resolved call
counts. The event log is also append-only. These audit records are deliberately
separated from the bounded active state.

### Theorem 2 | Storage bound for the active operational state

Let the archive capacity be \(C\). The BasinGraph active operational state
requires

\[
\mathcal O(Cd+C^2)
\]

scalar storage. For the frozen value \(C=80\), the active-state storage is
\(\mathcal O(d)\) as the decision dimension grows.

**Proof.** Each archive node stores one \(d\)-dimensional centre and a fixed
number of scalar attributes. The archive implementation evicts one current
worst node whenever insertion would make its size exceed \(C\), so
\(|A_t|\le C\). Archive storage is therefore \(\mathcal O(Cd)\).

The transition graph stores at most one edge for each ordered pair of node
identifiers because edges are held in a dictionary keyed by
`(source_id, target_id)`. Self-loops are rejected. A returned graph whose
endpoints all belong to the active archive therefore has at most
\(C(C-1)\) edges. Every edge stores a fixed number of scalars, giving
\(\mathcal O(C^2)\) graph storage. The incumbent contributes \(\mathcal O(d)\),
and diagnostics, options and phase counters contribute fixed-size terms.
Summing the terms gives \(\mathcal O(Cd+C^2)\). \(\square\)

### Limitation 1 | The full audit record is not fixed-capacity

The best-so-far history, objective-call ledger and event log grow with the
number of evaluations \(T\), requiring \(\mathcal O(T)\) records. The precise
claim is therefore:

> BasinGraph maintains a bounded active decision state coupled to an
> append-only audit ledger.

It is not correct to describe the complete serialized result, including the
ledger and event log, as fixed-capacity.

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
the optimizer computes the active node identifiers and removes every graph
endpoint not in that set. Hence the returned object has referential integrity.

### Limitation 2 | Arbitrary mid-run snapshots require sanitation

Track E engineering probes exposed a transient ordering in which a just-created
node can be evicted by the capacity rule and subsequently receive an edge in
the same higher-level operation. The final cleanup removes such endpoints
before return, so all prior returned-state audits remain valid. An arbitrary
mid-run snapshot, however, must explicitly sanitize graph endpoints against the
copied active archive. No claim of unsanitized arbitrary-time integrity is made.

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

where \(|L_T|\) is the number of best-so-far history records and \(n_\phi\) is
the number of objective calls assigned to phase \(\phi\).

### Proposition 4 | Monotone incumbent trace

The returned best-so-far sequence is non-increasing.

**Proof.** The ledger changes `fbest` only when a finite new value is strictly
smaller. Every appended history value is the current `fbest`. \(\square\)

## 5. What is and is not established

The definitions and results establish that:

- the best-so-far trace cannot reconstruct the search history;
- the active archive and returned graph have a dimension-explicit storage
  bound;
- returned graph endpoints are valid active representatives;
- every successful objective call is assigned to one ledger and one phase;
- the returned incumbent trace is monotone.

They do not establish that:

- the operational graph reconstructs the topology of mathematical attraction
  basins;
- the active state is a sufficient statistic for optimal continuation;
- retaining the state necessarily improves objective performance;
- arbitrary mid-run states can be resumed exactly;
- the empirical scaling measurements prove an asymptotic time-complexity law.

Exact continuation would additionally require a formally defined safe
checkpoint boundary, the complete random-generator state, all loop-local
control variables and a continuation implementation whose transition function
is proven to depend only on the serialized checkpoint.
