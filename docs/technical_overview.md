# SensEI: Technical Overview

This document is a from-the-code technical reference for the SensEI pipeline,
what is actually implemented, with the exact mathematical formulations used.

---

## 1. System boundary

```
Ensense core     vendored, pinned (src/sensei/config/ensense_pin.txt),
                 read-only. A DETECTOR: given a tree ensemble and a
                 protected feature, finds (x1, x2) that agree outside the
                 flip set but disagree in prediction.

SensEI           everything in src/sensei/. A REPAIRER built around
                 Ensense core: oracle (Tier A), validity encoding, convex
                 repair, evaluation.
```

Two oracle tiers exist side by side:

| Tier | Implementation | Role |
|---|---|---|
| A | `sensei.oracle.sensei` -- our own MILP | primary search + repair driver |
| B | `sensei.oracle.ensense.adapter` -- vendored Ensense core | independent certifier, held-out re-check |

---

## 2. Notation

| Symbol | Meaning | Code |
|---|---|---|
| $D$ | training dataset | `Dataset` |
| $S$ | flip set: features allowed to differ between $x_1, x_2$ | `flip_set: tuple[str, ...]` |
| $P, M, R$ | protected / monotone / free feature groups | `spec.protected`, `spec.monotone` |
| $T$ | number of trees in the ensemble | `len(structure.trees_leaves)` |
| $n$ | a global leaf index, $n \in \{0, \dots, N-1\}$, $N$ = total leaf count over all trees | `LeafMap.leaf_index`, flat |
| $v \in \mathbb{R}^N$ | vector of all leaf values across all trees; $v_0$ = original (pre-repair) | `leaf_map.v0`, `CegsalLoop.v` |
| $b$ | base score (global bias, constant, never repaired) | `leaf_map.base_score` |
| $\phi(x) \in \{0,1\}^N$ | leaf-indicator vector: $\phi(x)_n = 1$ iff $x$ reaches leaf $n$ | `LeafMap.phi_for` |
| $E_v(x)$ | ensemble margin | $b + \phi(x)^\top v$ |
| $\ell_1, \ell_2$ | leaf-indicator vectors for $x_1, x_2$ (two separate vectors) | `Pair.ell1`, `Pair.ell2` |
| $\mathrm{gap}$ | **signed**: $E_v(x_1) - E_v(x_2)$ | `Pair.gap` |
| $\varepsilon$ | sensitivity budget (margin space) | `eps` |
| $\theta$ | plausibility threshold | `theta` |
| $\mathrm{plaus}(x)$ | plausibility score, product of frozen marginals -- **not a density** | `FrozenBins.log_plaus` |
| $d$ | a cut's sparse difference vector, $d = \ell_1 - \ell_2$ | `Cut` |
| $\mu, \kappa$ | repair QP proximal / slack-price weights | `settings.repair.mu`, `.kap` |

$\varepsilon$ and $\theta$ are never abbreviated to the same symbol and never
share meaning: $\varepsilon$ bounds a margin gap, $\theta$ floors a
plausibility score.

---

## 3. The ensemble margin and leaf indexing

XGBoost's own `trees_to_dataframe()` is walked once, per tree, to assign
every leaf a single flat index $n$ (`model/leaves.py::LeafMap._index_leaves`).
This map is built once from the frozen structure and never rebuilt during a
repair run -- **the central invariant of the whole project**:

> Tree structure is frozen after initial training, forever. A cut refers to
> specific leaves; retrain, and it silently means something else.

For a row $x$, and tree $t \in \{1, \dots, T\}$, let $\mathrm{leaf}(t, x)$ be
the global index of the leaf $x$ reaches in tree $t$. Then:

$$
E_v(x) \;=\; b \;+\; \sum_{t=1}^{T} v_{\mathrm{leaf}(t, x)} \;=\; b + \phi(x)^\top v
$$

$\phi(x) \in \{0,1\}^N$ has exactly $T$ ones (one per tree) and $N-T$ zeros;
$\phi$ over a dataset is stored as a sparse matrix
$\Phi \in \{0,1\}^{|D| \times N}$ (`LeafMap.phi_for`), one row per training
example.

**$b$ is a constant.** For a `binary:logistic` model (the only objective
currently verified), XGBoost's own base score $p_0$ is stored as a
probability; SensEI converts it to margin space via the logit link:

$$
b = \log\!\left(\frac{p_0}{1 - p_0}\right)
$$

$b$ cancels in every gap ($E_v(x_1) - E_v(x_2)$ never contains $b$) but is
required whenever an actual prediction or accuracy number is computed:
$\hat y(x) = \mathbb{1}[E_v(x) > 0]$.

**Margin, not probability.** All sensitivity reasoning (gaps, cuts,
no-goods, the repair objective) happens on $E_v(x)$ directly. The sigmoid
$\sigma(E_v(x))$ is applied only at reporting time, never inside the search
or the repair QP.

---

## 4. Tier A: the MILP oracle

`sensei/oracle/sensei/{encoding,nogoods,solve,warm_start}.py`. One Gurobi model
per `worst_valid_pair` call, built fresh (no persistent model across calls).

### 4.1 Decision variables

For every feature $f$ in the dataset's columns, two point-variables (Gurobi
continuous, $[0,1]$ since data is min-max scaled):

$$
x_1^f, \quad x_2^f \quad \text{for } f \in \text{flip\_set}, \qquad
x_2^f := x_1^f \quad \text{for } f \notin \text{flip\_set}
$$

i.e. $x_2$ only gets its own free variable for features in $S$; every other
feature is *the same variable object* for both points, which is what "agree
outside the flip set" means at the encoding level -- not a constraint that
can be violated, a structural identity.

### 4.2 Threshold order variables and the split-epsilon trick

For every distinct split threshold $t$ that appears for feature $f$
anywhere in the ensemble, one binary order variable:

$$
u_{f,t} \in \{0, 1\}, \qquad
u_{f,t} = 1 \iff x_f \le t - \delta, \qquad
u_{f,t} = 0 \iff x_f \ge t
$$

implemented as two Gurobi indicator constraints
(`TreeEncoder.link_order_variables`) plus a monotonicity chain over sorted
thresholds $t_1 < t_2 < \dots$ for the same feature:

$$
u_{f,t_1} \le u_{f,t_2} \le \dots
$$

$\delta = 10^{-4}$ (`_SPLIT_EPS`) is a fixed small constant, needed because
Gurobi only has non-strict inequalities but XGBoost's split test is strict
($x < t$ goes left). This structurally excludes the half-open interval
$[t - \delta, t)$ from the searchable domain for every threshold $t$. This is
provably harmless for the objectives this encoding is used for (margins and
plausibility-bin membership are both piecewise-constant between thresholds,
so nothing achievable in the excluded sliver is lost) **provided** $\delta$
is smaller than the gap between any two thresholds for the same feature.
`TreeEncoder._assert_thresholds_separated` checks this at encoding time and
raises loudly rather than silently risking two order variables' linking
constraints overlapping (which would corrupt leaf routing or bin membership
outright, not just narrow the domain).

### 4.3 Leaf indicators

For every global leaf $n$, with root-to-leaf path
$(f_1, t_1, \mathrm{dir}_1), \dots, (f_k, t_k, \mathrm{dir}_k)$
($\mathrm{dir}_i \in \{\text{yes: } x \le t_i,\ \text{no: } x \ge t_i\}$),
define the literal set

$$
L_n = \{\, u_{f_i, t_i} \text{ if } \mathrm{dir}_i = \text{yes}, \ \ (1 - u_{f_i, t_i}) \text{ if } \mathrm{dir}_i = \text{no} \,\}
$$

and the indicator $\ell_n \in \{0,1\}$ via the standard AND-linearization:

$$
\ell_n \le \lambda \quad \forall \lambda \in L_n, \qquad
\ell_n \ge \Big(\sum_{\lambda \in L_n} \lambda\Big) - (|L_n| - 1)
$$

plus, per tree $t$, exactly one active leaf:

$$
\sum_{n \in \mathrm{leaves}(t)} \ell_n = 1
$$

Two full copies of this encoding are built per model -- $\ell_1$ (order
variables `a_*`, linked to $x_1$) and $\ell_2$ (order variables `b_*`: reused
from $\ell_1$'s for every feature outside $S$, fresh for features inside
$S$, linked to $x_2$).

**Compact representation.** `Pair.ell1`/`ell2` are stored not as the full
$N$-length one-hot vector but compactly as the $T$ active global leaf
indices (one per tree) that are set. Set-difference on these compact arrays
(`CutBuilder.make_cut`, `NoGoodBuilder.make_nogood`) is mathematically
identical to differencing the full one-hot vectors -- leaves active in both
$\ell_1$ and $\ell_2$ cancel either way.

### 4.4 Objective: the signed gap

$$
E_{v}(x_1) = b + \sum_{n} v_n\, \ell_{1,n}, \qquad
E_{v}(x_2) = b + \sum_{n} v_n\, \ell_{2,n}, \qquad
\mathrm{gap} = E_v(x_1) - E_v(x_2)
$$

$b$ is omitted from the actual Gurobi objective (it cancels identically), so
the objective Gurobi sees is $\sum_n v_n(\ell_{1,n} - \ell_{2,n})$, maximized:

$$
\max_{x_1, x_2, \ell_1, \ell_2} \;\; \mathrm{gap} \;=\; \sum_n v_n\,(\ell_{1,n} - \ell_{2,n})
$$

**Two modes**:

- **`feasibility`** (used inside the loop): adds the constraint
  $\mathrm{gap} \ge \varepsilon + 10^{-9}$ before solving, then still
  *maximizes*, but stops at the first feasible incumbent Gurobi is content
  with (`MIPGap` from settings). Returns the first pair with a genuine
  violation, or infeasible.
- **`optimality`** (reporting, final certification): no `eps` constraint at
  all; `MIPGap = 0`; solves to true optimality, giving the actual worst gap.

**Direction handling** (`direction` argument):

- `protected`: $S \subseteq P$, and the search space is symmetric under
  swapping $x_1 \leftrightarrow x_2$ (both range over the same domain). One
  search per flip set suffices; maximizing the signed gap covers both
  directions. `_extract_result` asserts $\mathrm{gap} \ge -10^{-6}$ on return
  -- a negative value means this symmetry assumption broke.
- `monotone_wrong`: not symmetric. One extra linear constraint pins the
  order of $x_1, x_2$ on the monotone feature $f^\star$, so that a positive
  gap is *exactly* a wrong-direction violation:

$$
\text{increasing:} \quad x_1^{f^\star} \le x_2^{f^\star}
\qquad\qquad
\text{decreasing:} \quad x_2^{f^\star} \le x_1^{f^\star}
$$

  The objective (maximize $E(x_1) - E(x_2)$) never changes between the two
  cases; only which point is pinned to "should score no higher" flips.

### 4.5 No-goods: joint pattern, not union

Given a returned pair with active-leaf sets $N_1^\star, N_2^\star$
(each of size $T$, one leaf per tree), the no-good forbids exactly that
*joint* assignment, not a ban on ever reusing any of those leaves again:

$$
\sum_{n \in N_1^\star} \ell_{1,n} \;+\; \sum_{n \in N_2^\star} \ell_{2,n} \;\le\; 2T - 1
$$

This says "at least one of the $2T$ leaf choices differs from last time."
`NoGoodBuilder.make_nogood` builds this in $O(T)$ from a `Pair`.

**Optional, conditionally-valid strengthening**: a cut on the difference
pattern $d = \ell_1 - \ell_2$ bounds the gap for *every* pair sharing that
$d$; once that cut's slack is exactly $0$, every such pair is provably
non-violating and can be forbidden with the stronger, sparser difference
no-good. This must never fire when a cut's slack is nonzero -- that would
hide real remaining violations.

### 4.6 Warm starts

`WarmStartApplier.apply_warm_start` sets Gurobi `.Start` values on
$x_1, x_2, \ell_1, \ell_2$ from a previous `Pair`. **Not currently used** by
`CegsalLoop._run_stage`: the only candidate (the most recently found pair)
already has its own no-good in the same `nogoods` list passed to the next
call, so seeding a `.Start` with it is guaranteed infeasible under a
constraint the same solve already enforces. Left as reachable
infrastructure, not wired into the loop's own calls.

---

## 5. Validity constraints (Q1)

Emitted directly into the same Gurobi model as the tree encoding
(`validity/encode_milp.py::MilpValidityEncoder`), for *both* $x_1$ and $x_2$
independently.

**Ranges.** $\mathrm{lo}_f \le x_f \le \mathrm{hi}_f$ for declared
`spec.ranges`.

**One-hot groups.** For a declared group of columns $C$:
$\sum_{c \in C} x_c = 1$, with each $x_c \in \{0,1\}$.

**Linear domain rules.** Each `DomainRule` is
$\sum_f a_f x_f \;\{\le, \ge, =\}\; r$ for declared coefficients $a_f$ and
constant $r$.

**Functional dependencies.** A declared dependency
$(\mathrm{if\_col}, \mathrm{if\_val}, \mathrm{then\_col}, \mathrm{then\_val})$
(e.g. `education = 0.667 -> education-num = 1.0`, the Adult
`education`/`education-num` redundancy) is encoded via an exact-integer
indicator: an auxiliary integer $a = x_{\mathrm{if\_col}} \cdot
\mathrm{span}$ is constrained to a target integer level, and a binary $b$
is $1$ iff $a$ equals that target, via two more indicator constraints
($b = 1 \Rightarrow a \le \mathrm{target}{-}1$ is FALSE... concretely
$b=1 \Rightarrow$ neither "below" nor "above" holds, i.e. $a = \mathrm{target}$
exactly); then $b = 1 \Rightarrow x_{\mathrm{then\_col}} = \mathrm{then\_val}$.

**Integrality.** For every feature in `spec.integer_features`, with known
bounds $[\mathrm{lo}, \mathrm{hi}]$ (from the fitted scaler) and
$\mathrm{span} = \mathrm{hi} - \mathrm{lo}$:

$$
x_f \cdot \mathrm{span} = a, \qquad a \in \mathbb{Z}, \;\; 0 \le a \le \mathrm{round}(\mathrm{span})
$$

i.e. $x_f$ is forced onto one of $\mathrm{round}(\mathrm{span}) + 1$ evenly
spaced points in $[0,1]$ -- exactly the raw integer levels the min-max
scaler mapped from. **Gated purely on `spec.integer_features`**: any encoded
column omitted from that list has no integrality constraint at all and can
take any real value in $[0,1]$.

---

## 6. Plausibility (Q2)

$$
\mathrm{plaus}(x) = \prod_{f} p_f(x_f), \qquad
\log \mathrm{plaus}(x) = \sum_f \log p_f(x_f)
$$

**This is a product of per-feature marginals under an independence
assumption -- a plausibility SCORE, never a density estimate of the joint
$p(x)$.** It will score an individually-common-but-jointly-absurd
combination highly. Never named `density`/`logpdf`/`p_x` anywhere in the
code; always `plaus`/`log_plaus`.

**Frozen bins** (`data/bins.py::FrozenBins`), computed once from $D_{\text{train}}$,
never from model split thresholds:

- Numeric feature: $n_{\text{bins}} + 1$ quantile edges from
  $D_{\text{train}}$ (default $n_{\text{bins}} = 10$).
- Categorical feature (declared via `categorical_levels`, sourced from
  `data/builder.py`'s own `encoding_map.json`): **one bin per level**, not
  quantile bins. For an $L$-level ordinal-encoded categorical, level $i$
  is assumed scaled to $i / (L-1) \in [0,1]$ (true only if the scaler's
  fitted min/max for that column are exactly $0$ and $L-1$ -- see
  remediation item 17 for a case where this breaks), and bin edges are the
  midpoints between consecutive levels' scaled positions:

$$
\text{edge}_0 = -\tfrac{1}{2(L-1)}, \quad
\text{edge}_i = \frac{1}{2}\left(\frac{i-1}{L-1} + \frac{i}{L-1}\right)\ \text{for } 1 \le i \le L-1, \quad
\text{edge}_L = 1 + \tfrac{1}{2(L-1)}
$$

**Laplace smoothing.** For bin $i$ of feature $f$ with count $c_i$ over
$N_f$ total training rows and $n_{\text{bins}}(f)$ bins:

$$
p_f(\text{bin } i) = \frac{c_i + 1}{N_f + n_{\text{bins}}(f)}
$$

(zero counts would make $\log p_f$ undefined, for reasons unrelated to
sensitivity).

**The threshold constraint** (Tier A, `emit_plausibility`), using the same
order-variable machinery as tree thresholds (bin edges are just another
set of thresholds per feature), with one binary "active bin" indicator per
bin (same AND-linearization pattern as leaf indicators):

$$
\sum_f \sum_i \log p_f(i) \cdot \mathbb{1}[\text{bin}_f(x_f) = i] \;\ge\; \log \theta
$$

$\theta \le 0$ is treated as "no floor at all" (skips the constraint
entirely on the Tier A side, returns `True` immediately on the Tier B
postfilter side) since $\log \mathrm{plaus}(x)$ is always finite and the
constraint would be vacuously true anyway -- `math.log(0.0)` is undefined
and was a live crash before this special case was added.

**Certificates are relative to $\theta$.** Raising $\theta$ shrinks the
plausible region and makes UNSAT easier to reach trivially (declare
everything implausible); every certified result must therefore be reported
as a curve over $\theta$ (`eval/sweeps.py::theta_sweep`), never a single
number.

---

## 7. Cuts: sparse difference vectors, never two-leaf

For a returned pair with $\ell_1, \ell_2$:

$$
d = \ell_1 - \ell_2 \in \{-1, 0, 1\}^N
$$

Only leaves where the two points diverge are nonzero (shared leaves cancel
exactly). `CutBuilder.make_cut` computes this directly as a set
symmetric-difference on the compact per-tree leaf-id arrays: $+1$ where only
$\ell_1$ has the leaf, $-1$ where only $\ell_2$ has it.

**The correct and only cut form** (never the two-leaf special case
$|v_a - v_b| \le \varepsilon$, which is only correct when a pair diverges in
exactly one tree -- with $T$ large, a pair typically diverges across many
trees and the constraint must couple every leaf in $d$):

$$
-\varepsilon - s \;\le\; d^\top v \;\le\; \varepsilon + s, \qquad s \ge 0
$$

$s$ is a per-cut slack variable, always present: once $\sim$30-50 cuts accumulate,
some leaf is typically pulled in two directions by different cuts and a
hard-constrained repair would go infeasible. $\sum_i s_i > 0$ at termination is a
finding to report (leaf-only repair ran out of expressive power for this
configuration), never something to hide or drop a cut to avoid.

---

## 8. The repair QP

`repair/qp.py::RepairQP`. Solves for a delta $d = v - v_0$ (not $v$
directly), then reconstructs $v = v_0 + d$ at the end
(`_extract_solution`).

### 8.1 Squared-error variant (`solve_qp_fast`, dev speed only)

$$
\min_{d} \;\; d^\top \big(\Phi^\top \Phi\big)\, d \;+\; \mu \lVert d \rVert_2^2 \;+\; \kappa \sum_i s_i
\qquad \text{s.t. cuts on } v_0 + d
$$

$\Phi \in \{0,1\}^{|D_{\text{train}}| \times N}$ is the training leaf-indicator
matrix; $d^\top(\Phi^\top\Phi)d = \sum_{x \in D_{\text{train}}} (\phi(x)^\top d)^2$
is the sum of squared margin changes -- a proxy for log-loss, not the
reported metric.

### 8.2 Log-loss variant (`solve_qp`, reported runs)

$$
\min_{d} \;\; \sum_{x \in D_{\text{train}}} \log\!\big(1 + e^{-y_x^{\pm}\, E_{v_0+d}(x)}\big) \;+\; \mu \lVert d \rVert_2^2 \;+\; \kappa \sum_i s_i
$$

where $y_x^{\pm} = +1$ if the true label is positive, $-1$ otherwise, and
$E_{v_0+d}(x) = b + \phi(x)^\top v_0 + \phi(x)^\top d$. The log-loss term is
convex but nonlinear in $d$; Gurobi handles it via a per-row piecewise-linear
(PWL) approximation of $z \mapsto \log(1+e^{-z})$ (softplus of $-z$),
evaluated at **41 evenly spaced breakpoints over $z \in [-20, 20]$**
(`n_pwl_breakpoints`, `margin_bound`), one `addGenConstrPWL` call per
training row:

$$
z_r = y_r^{\pm}\big(b + \phi(x_r)^\top v_0 + \phi(x_r)^\top d\big), \qquad
\text{loss}_r \approx \mathrm{PWL}\big(z_r;\, \{z_j, \log(1+e^{-z_j})\}_{j=1}^{41}\big)
$$

**Known performance risk** (remediation item 5): one PWL constraint plus one
`gp.quicksum` per training row means this scales linearly in
$|D_{\text{train}}|$ per QP solve, once per CEGSAL iteration -- unverified
at real dataset scale (a 50-tree Adult run with an actual cut did not finish
in a live test after tens of minutes).

### 8.3 Cut constraints inside the QP

Each `Cut` (sparse `idx`, `coef`, `eps`) is added as, in terms of $d$:

$$
\mathrm{const} = \sum_{i} \mathrm{coef}_i \cdot v_0[\mathrm{idx}_i], \qquad
\Big(\sum_i \mathrm{coef}_i \cdot d[\mathrm{idx}_i]\Big) + \mathrm{const} \le \varepsilon + s, \qquad
\ge -\varepsilon - s
$$

algebraically identical to $-\varepsilon - s \le d_{\text{cut}}^\top v \le
\varepsilon + s$, just expressed in the delta variable the solver actually optimizes over.

---

## 9. The CEGSAL loop

`repair/loop.py::CegsalLoop`. Per stage (one flip_set/direction pair):

```
search (Tier A, mode="feasibility", eps) -> pair or None
  pair found  -> cut = make_cut(pair, eps); nogood = make_nogood(pair, T)
                 repair: v <- solve_qp(v0, cuts, mu, kap)
                 record snapshot; check accuracy floor / stall; loop
  none found  -> confirm via mode="optimality" (no eps):
                   raises OracleTimeout         -> Inconclusive("TIMEOUT")
                   returns None                 -> Inconclusive("EMPTY_DOMAIN")
                   gap > eps (shouldn't happen) -> Inconclusive("INCONSISTENT_OPTIMALITY")
                   else                         -> ConditionalCertificate
```

**Why the confirmation search exists.** `mode="feasibility"` returning
`None` conflates two different things: "a valid pair exists but none has
gap $> \varepsilon$" (real UNSAT) and "the encoded validity+plausibility
domain has no valid pair at all" (a misconfiguration, e.g. $\theta$ too high
for this flip set -- never a certificate). A fresh `mode="optimality"`
search (no `eps`) tells them apart: `None` there specifically means the
domain is empty.

**Exit types**, never merged:

$$
\begin{aligned}
\texttt{ConditionalCertificate} &: \text{solver\_status} = \text{"UNSAT"}, \text{ records } \varepsilon, \theta, S, \text{ spec\_hash, oracle@sha, time\_limit} \\
\texttt{ParetoBest} &: \text{accuracy floor, stall (}\Delta_{\text{worst}} < \text{stall\_delta}\text{), or iteration cap hit} \\
\texttt{Inconclusive} &: \text{TIMEOUT} \mid \text{ORACLE\_SATURATED} \mid \text{EMPTY\_DOMAIN} \mid \text{INCONSISTENT\_OPTIMALITY}
\end{aligned}
$$

**Pareto checkpoint.** Among snapshots meeting the accuracy floor
$a_{\min}$, keeps the one with the smallest $|\mathrm{gap}|$ seen so far
(`ParetoCheckpoint.record`/`.restore`). Meaningful only *within* one
flip_set/direction's own iterations -- comparing `worst_gap` values from
different flip sets is comparing different quantities (this was the source
of the schedule-level bug fixed in remediation item 13).

**Stall condition:** stop if $|\,\mathrm{gap}_{t} - \mathrm{gap}_{t-1}\,| <
\Delta_{\text{stall}}$, where $\mathrm{gap}_t$ is the triggering violation's
$|\mathrm{gap}|$ at iteration $t$ (the cheap, per-`found`-pairs value, not a
fresh optimality search -- that would double the per-iteration solver cost).

---

## 10. Tier B (Ensense core) and the postfilter

`oracle/ensense/adapter.py::EnsenseOracle` calls the vendored solver, then
runs every raw result through `validity/postfilter.py::Postfilter
.find_valid_pair` before returning it -- Ensense's own search does not know
about SensEI's declared validity/plausibility rules at all.

**Rejection budget**, since Tier B exposes no no-good mechanism (a
deterministic solver call can return the exact same invalid pair
indefinitely otherwise):

$$
\text{MAX\_REJECTIONS\_PER\_ITER} = 20
$$

After 20 consecutive rejections, raises `OracleSaturated` -- **never** a
certificate, logged and reported as its own distinct status, never
collapsed into UNSAT or into "no violation found."

**Degenerate pairs.** A returned pair failing basic sanity ($x_1 = x_2$,
differs outside the declared flip set, or zero margin gap) raises
`OracleDegenerate`, counted as a rejection against the same budget --
distinct from Tier B genuinely searching and finding nothing.

**Timeout reliability.** The vendored `pb` search method (the default for
held-out verification) has its own `signal.alarm`-based timeout enforcement
commented out in the vendored source; the vendored `milp` method hardcodes
its own Gurobi time limit to one hour regardless of what is requested. The
`timeout` value SensEI passes to Tier B is therefore not reliably honored
by either method (see remediation item 16) -- `ensense/` is never edited to
fix this; the caller-side handling of Tier B's `None` result is where this
has to be addressed.

---

## 11. Metrics (`eval/metrics.py`, frozen -- `Metrics.VERSION`)

**Accuracy:**

$$
\mathrm{acc}(v) = \frac{1}{|D_{\text{eval}}|} \sum_{x \in D_{\text{eval}}} \mathbb{1}\big[\,\mathbb{1}[E_v(x) > 0] = y_x\,\big]
$$

**Sensitivity rate**, over a fixed random sample of $\min(1000, |D|)$ rows,
each protected feature flipped to its own next sorted level independently:

$$
\mathrm{sens}(v) = \frac{1}{|R|}\sum_{x \in R} \mathbb{1}\Big[\exists\, f \in P : \hat y_v(x) \ne \hat y_v(\mathrm{flip}_f(x))\Big]
$$

**Worst valid gap**: the true $\mathrm{gap}$ from Tier A in `mode="optimality"`
at the current $\theta$, for one flip_set/direction. `None`/empty-domain is
reported as `NaN`, never `0.0` -- a `0.0` would misrepresent "we don't know" as
"certified insensitive."

**Slack mass:** $\sum_i s_i$ at repair termination.

**Oracle status counts:** tallies of UNSAT / TIMEOUT / SATURATED /
EMPTY\_DOMAIN across a run -- kept as four distinct counters, never
collapsed into a boolean.

---

## 12. Certificates: language is fixed

```python
ConditionalCertificate(tier, eps, theta, flip_sets, directions, spec_hash,
                        bins_hash, validity_rules, plaus_model, oracle,
                        solver_status, time_limit_s)
```

Rendered as a sentence, never a verdict:

> No pair $(x_1, x_2)$ was found that satisfies the encoded type and domain
> rules, scores $\mathrm{plaus} \ge \theta$ under a
> product-of-frozen-marginals plausibility model, differs only on $S$, and
> produces a margin gap exceeding $\varepsilon$ on the repaired model --
> under oracle `<name@sha>` within a `<t>` s limit.

Never "certified fair," "proved insensitive," or "the model is safe."
`solver_status` for a certificate must be exactly `"UNSAT"` -- never
`SATURATED`/`TIMEOUT`. Tier A's UNSAT means absence within the encoded
domain; Tier B's UNSAT (when trustworthy) means absence under Ensense's own
independent search, which does not enforce SensEI's declared validity rules
inside its own search at all -- a structurally weaker claim.

---

## 13. Baseline 4's labeling policies

For counterexample-retraining, since Ensense pairs are solver outputs with
no ground-truth label:

$$
\textbf{P1 (majority-protected anchoring):}\quad
\hat y_{\text{pair}} = \mathbb{1}\big[E_{v}(x \mid P \leftarrow \mathrm{mode}(P \text{ in } D_{\text{train}})) > 0\big]
$$
applied to both $x_1, x_2$ identically.

$$
\textbf{P2 (score-averaging):}\quad
\hat y_{\text{pair}} = \mathrm{round}\!\left(\sigma\!\left(\frac{E_v(x_1) + E_v(x_2)}{2}\right)\right)
$$

Both are always run; the policy name is required (no default) in every
baseline-4 results JSON, since the labeling rule is a declared modeling
choice, not a neutral one.

---

## 14. Known gaps in this formulation

This document describes the code as it stands, including its current
limitations -- it is not a claim that every formula above is airtight. See
the project's remediation-tracking notes for the live list of confirmed
issues (categorical integrality gaps outside Adult, the Tier-B timeout
reliability issue, the rare-categorical-level bin-boundary interaction,
etc.) rather than duplicating that list here, since it changes independently
of the underlying math.
