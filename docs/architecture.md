# SensEI: Sensitivity Repair for Decision Tree Ensembles

**Architecture document**

---

## 1. The problem

Ensense finds problems but cannot fix them. Given a trained tree ensemble, it finds two inputs that differ only in gender (or race) but get opposite predictions. It reports them. That is all.

We want to repair the model so those pairs stop existing.

```
Ensense:  detector
SensEI:   repairer
```

---

## 2. The idea

Do not retrain the model. **Freeze the tree structure and change only the numbers in the leaves.**

Why this works: once the trees are frozen, we already know exactly which leaves each input lands in. The score becomes a simple sum of leaf values. The condition "these two inputs must score alike" becomes a plain linear inequality. And a linear inequality is something a solver can handle in seconds.

So the loop is:

```
train model
      │
      ▼
find a bad pair (slow, exact)
      │
      ├── none found ─► stop, we have a (conditional) certificate
      │
      ▼
turn the pair into one linear rule
      │
      ▼
pick new leaf values obeying all rules so far (fast, cheap)
      │
      ▼
    repeat
```

Each round adds one rule. After maybe 40 rounds we have 40 rules instead of infinitely many, and they are exactly the 40 that mattered.

---

## 3. Two hard facts about the setup

**We cannot edit Ensense.** It is vendored, pinned to a commit, and read-only. Everything we build lives beside it.

**Tree structure never changes after training.** A rule says "leaf 41 and leaf 57 must stay close." Retrain, and leaf 41 is a different leaf. Every rule silently becomes wrong.

Names we use consistently:

| Name | Meaning |
|---|---|
| Ensense core | the upstream tool, untouched |
| SensEI | our layer |
| SensEI+Ensense | the whole pipeline |

---

## 4. Symbols

| Symbol | Meaning |
|---|---|
| `D` | training data |
| `P` | protected features. Flipping these must not change the answer |
| `M` | monotone features. May change the answer, but only one way |
| `R` | everything else. No rule |
| `S` | flip set: which features are allowed to differ |
| `v` | all leaf values. `v0` = original |
| `E_v(x)` | model score for `x` (sum of leaf values reached) |
| `gap` | `E_v(x1) − E_v(x2)`. Always signed |
| `eps` | how big a gap we tolerate |
| `theta` | how realistic a point must be to count |
| `phi` | leaf indicator matrix |

`eps` and `theta` are different things. Never share a symbol.

`P`, `M`, `R` are written by hand, per dataset, before anything runs. No data can tell you gender should not matter but income should. That is a human judgement and it is an **input** to the project.

---

## 5. Which counterexamples we care about

Ensense produces things like `education-num = 13.500002`. Nobody has 13.5 years of school. That is not an unlikely person, it is an impossible one.

Three separate questions:

| Question | Example failure | Answered by |
|---|---|---|
| Is it well-formed? | age = 250 | type rules — cheap, exact |
| Is it realistic? | age 25, PhD, 40 years experience | statistics — approximate |
| Should the flip change the answer? | gender flips a loan | a human, in advance |

Only pairs that pass all three are worth repairing. Everything else is noise.

---

## 6. Where the search rules live

We cannot put rules inside Ensense. Two ways around it.

### Tier B — start here

Let Ensense search anywhere, then check the pair ourselves and reject junk.

Plus the trick that actually helps: **make bad values impossible to write down.** Drop the `education` column, keep only `education-num`, and "Doctorate with 13.5 years" cannot exist. Integer-code integral features and 13.5 is outside the space.

Weakness: Ensense does not know we rejected anything, so it can return the same junk forever. Hence a rejection budget. After 20 rejects we stop and record `ORACLE_SATURATED`, which means *we gave up*, not *nothing is left*.

### Tier A — target

Write our own search MILP. Standard, well-documented work, roughly 1–2 weeks. Because it is ours, the rules go inside the search: junk is never proposed, and we can say "never show me that pair again."

The real reason to do it: **the repairer stops grading its own work.** We repair with our solver, then hand the finished model to untouched Ensense for the verdict.

| | Tier B | Tier A |
|---|---|---|
| Rules live | outside the search | inside the search |
| "Not that one again" | impossible | yes |
| Can spin forever | yes | no |
| Certifier | same tool that searched | independent |

Before either: read Ensense's actual inputs and write down what it already accepts. If it takes a feature spec with types and bounds, some of Tier B is already done.

---

## 7. Validity rules

All linear, all cheap.

**Type rules.** Integers stay integers. One-hot columns sum to 1. Ranges: `0 <= age <= 120`. Immutable features cannot be flipped.

**Functional dependencies.** `education = PhD` implies `education-num = 16`.

**Domain rules.** `experience <= age − 18`.

**Plausibility.**

```
plaus(x) = product over features of p_j(x_j)
```

Require `plaus(x1) >= theta` and `plaus(x2) >= theta` as a **hard rule**, not a preference. Ensense treats it as a preference, which means it returns the least-bad option even when every option is nonsense.

Two things about `plaus`:

- **It is not a density.** It is a product of per-feature marginals under an independence assumption. It will happily score "age 25, PhD, 40 years experience" as fine. Say this in the paper. Never name a variable `density`.
- **The bins are frozen.** Computed once from `D_train` by quantile, never from the model's split thresholds. Otherwise "realistic" is redefined every iteration, and a person who was rare in round 1 is common in round 20 without changing.

---

## 8. The repair

**The rule from one pair.** Both inputs pass through many leaves. Shared leaves cancel. What is left is the source of the gap.

```
d = ell1 − ell2                 (sparse; zero where both agree)

−eps − s  <=  d @ v  <=  eps + s ,   s >= 0
```

The common mistake is writing `|v_a − v_b| <= eps`. That is only true when the pair diverges in exactly one tree. With 200 trees it diverges in many, and the rule couples all of them. The two-leaf form looks fine on toy models and is wrong on real ones.

**The solve.**

```
minimize over v, s:

    LogLoss(D_train, v)          stay accurate
  + mu  * ||v − v0||^2           stay near the original model
  + kap * sum of s               price of rules we cannot meet

subject to every rule collected so far
```

Convex. Seconds.

**Rules are always soft.** Around 30–50 rules in, some leaf gets pulled two ways and a hard-constrained problem returns "no solution." The slack `s` turns a crash into a measurement: `sum(s) > 0` means leaf-only repair has run out of room. That is a **result to report**, not a failure to hide.

**Never drop a rule to make the solver feasible.**

**Known limit.** If a bad pair and a legitimate customer pass through the same leaves, no leaf value can separate them — the model literally cannot tell them apart. Measure how often this happens. It is a finding either way.

---

## 9. No-goods

If the search returns gaps of 10.0, then 9.9, then 9.8, that is the same spot three times.

After each pair, forbid **that exact pair of leaf assignments**:

```
sum of ell1 over its leaves  +  sum of ell2 over its leaves  <=  2T − 1
```

At least one of the `2T` leaf choices must differ.

Do **not** forbid the union of leaves. That reads as "never use any of these leaves again," which is a much stronger and wrong constraint, and it will make the search infeasible for reasons unrelated to sensitivity.

Tier B cannot do this at all. That is exactly why Tier B needs a rejection budget.

---

## 10. Keep the two rule sets apart

```
SEARCH side                 ║   REPAIR side
tree encoding               ║   sensitivity rules (d @ v)
validity + plausibility     ║   slack variables
no-goods                    ║   proximal term
warm starts                 ║   log-loss

    a no-good must NEVER enter the repair solve
    a sensitivity rule must NEVER enter the search
```

Enforce this with separate types and asserts, not comments. A no-good quietly added to the repair produces plausible-looking numbers rather than a crash, which is the worst kind of bug.

---

## 11. The loop

**"Worst" is defined.** `gap` is always signed, `E_v(x1) − E_v(x2)`. The search maximizes the signed gap, never the absolute value.

- Protected features: the search space is symmetric, so one search per flip set covers both directions.
- Monotone features: not symmetric. `x1` must be pinned as the lower side, so a positive gap is exactly a wrong-direction violation.

**Two modes.**

- `feasibility` — used in the loop. `eps` goes into the solver: find any valid pair with `gap > eps`. Fast. Returns a pair, or `None`.
- `optimality` — used for reporting. Ignores `eps`, finds the true worst gap. Lets us sweep `eps` afterwards without re-solving.

**Warm starts.** Structure is frozen, so the previous counterexample is still a legal point in the new problem. Feed it back. This is free and much stronger than warm-starting a loop that retrains.

**Flip set order.** Single features in `P`, then pairs from `P`, then features in `M` wrong-direction only. No-goods reset between stages.

**Stopping.**

| Condition | Result |
|---|---|
| search returns `None` | conditional certificate |
| accuracy below floor | Pareto best |
| sensitivity barely moving | Pareto best |
| 50 iterations | Pareto best |
| timeout or saturated | **inconclusive** |

**Safety net.** Record `(accuracy, sensitivity)` every round and keep the best non-dominated model. Fixing A may create B, and fixing B may bring back A. The loop is allowed to fail — log it and return the checkpoint.

---

## 12. What the certificate actually says

Not "the model is fair." Not "proved insensitive."

> No pair was found that is type-valid, scores `plaus >= theta` under a product-of-marginals model, differs only on `S`, and exceeds `eps` — under oracle `<name@sha>` within `<t>` seconds.

**The certificate is relative to `theta`.** Raise `theta`, the realistic region shrinks, and "nothing found" gets trivially easy. You can always win by declaring everything implausible.

**So every certified result is reported as a curve over `theta`, never a single number.** A reviewer finds this in one line if we hide it.

**Three outcomes, never merged.**

| Outcome | Meaning |
|---|---|
| Certificate | search came back empty inside the encoded domain |
| Pareto best | loop stopped early; here is the best model seen |
| Inconclusive | timeout, saturated, or empty domain |

A timeout reported as a certificate is the worst bug this project can produce.

---

## 13. Evaluation

**The experiment that decides everything.** After repair, run a **fresh Ensense search**: new seed, new starting points, new flip sets.

- Comes back empty → the fix generalized. This is the win.
- Finds new problems → we patched cases, not the property. **Report it.** A negative result here is publishable; a hidden one is not.

**Why we expect it to generalize.** A rule on `d @ v` constrains *every input pair whose leaf paths differ by `d`* — a whole region of input space, not one point. Adversarial training patches points; we patch regions. Make this measurable: for each fresh counterexample, check whether its `d` matches one already covered.

**Metrics, frozen after Stage 0.**

| Metric | Definition |
|---|---|
| Sensitivity rate | fraction of 1,000 held-out rows whose prediction flips when `P` is flipped |
| Worst valid gap | max signed gap in `optimality` mode |
| Accuracy | test accuracy on `D_eval` |
| L2 distance to data | Ensense's own metric, for comparability |
| Runtime | wall clock for the whole loop |
| Slack mass | `sum(s)` at the end |

**Data discipline.** `D_train` (80%) builds everything. `D_eval` (20%) is untouched until final reporting.

**Datasets.**

| Dataset | `P` | `M` |
|---|---|---|
| Adult | gender, race | education-num (up), hours (up) |
| German credit | age, gender | amount (down), duration (down) |
| Pima | — | glucose (up) |
| Churn | as available | tenure (up) |

---

## 14. Baselines

1. Plain XGBoost
2. XGBoost with protected features deleted
3. XGBoost with `monotone_constraints`
4. **Counterexample retraining** — add Ensense's pairs to the training set with corrected labels, retrain

Baseline 4 is the one to beat. If we do not beat it, there is no paper. Build it in Stage 0.

**The labeling rule must be declared**, because there is no neutral choice. Ensense's pairs are synthetic, so there is often no true label at all, and giving both members the same label injects a fairness assumption by construction.

- **P1 (default):** set protected features to their most common value in `D_train`, predict with `M0`, give that label to both members.
- **P2 (check):** both get the label from the average of the two scores.

Run both. If baseline 4 swings between them, it is an artifact of labeling and we say so.

Only baselines 1–4 give no proof. That is the claim.

---

## 15. Stages

**Stage 0 — Setup.** Pin Ensense. Document what inputs it accepts. Reproduce its results. Write `P`/`M`/`R` by hand. Freeze bins and metrics. Build all four baselines.

**Stage 1 — Tier B validity.** Preprocessing plus post-filter with a budget.
*Deliverable:* their counterexample vs ours, side by side. They produce `education-num = 13.500002`; our encoding cannot.

**Stage 2 — Tier A search plus leaf repair.** Own MILP, verified to agree with Ensense on the unrepaired model. Soft rules, joint no-goods, warm starts, Pareto checkpoint.
*Deliverable:* a certified model, its accuracy cost, the Pareto curve.

**Stage 3 — Certify and stress-test.** Fresh Ensense re-check. `theta` sweep. `eps` sweep. Slack report.

---

## 16. Out of scope

| Idea | Why not |
|---|---|
| Editing Ensense | Read-only, pinned |
| Validity as appended trees | Blocks bad `x1` but *helps* bad `x2`. Filters one side only. Unsound |
| Causal validity | Contradicts `x1 = x2 outside S`. Needs a new definition of sensitivity |
| Structure as solver variables | Optimal-tree learning caps near depth 3–4 on one tree; we need 200+ |
| Soft differentiable trees | Guarantees do not survive hardening |
| Chow–Liu joint density | Weeks of work for a small gain over frozen marginals |
| Learned validity model | Adds approximation error to the thing we are trying to prove |
| Penalising splits on protected features | Reduces to leaf repair; also breaks accuracy for no gain |
| Pruning | Irreversible, unpredictable cost |

---

## 17. Related work

| Area | Works | Why |
|---|---|---|
| Individual fairness | Dwork et al. | Similar people, similar treatment |
| Counterfactual fairness | Kusner et al. | The causal route we deliberately skip |
| Fairness verification for trees | Calzavara et al. | Already cited by Ensense |
| In-processing fairness | FairXGBoost | Empirical baseline |
| Robust tree training | Chen et al. 2019a | Ensense names this as the analogy |
| Optimal decision trees | Bertsimas & Dunn; Verwer & Zhang | Cite when explaining scope |
| Model repair | PRDNN and similar | Leaf repair belongs to this family |

---

## 18. Risks

| Risk | Likely | What we do |
|---|---|---|
| Loop does not terminate | medium | Pareto checkpoint, iteration cap, report honestly |
| Repair does not generalize | high | Fresh-seed verification; report the failure |
| Leaf-only repair not expressive enough | medium | Slack mass measures it. A result either way |
| Search too slow per round | medium | Warm starts, gap tolerance, use best-so-far |
| Certificate looks stronger than it is | high | `theta` sweep required in every table |
| Timeout mistaken for success | high | Four distinct statuses, never collapsed |
