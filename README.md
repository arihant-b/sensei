# SensEI

**Sens**itivity **E**nsemble **I**ntervention

This repository contains SensEI, a tool for **sensitivity repair** of decision tree ensembles. Given a trained XGBoost model, SensEI searches for realistic pairs of inputs that differ only in a protected feature (such as sex or race) yet receive opposite predictions, and repairs the model so that no such pair is found within the encoded validity/plausibility domain. Repair adjusts leaf values only &mdash; the tree structure is never modified &mdash; which reduces the problem to a convex program (a QP with soft slack) over leaf values. When the search comes back UNSAT, SensEI reports a **conditional certificate**, not an unconditional proof: absence of a counterexample under the encoded type/domain/plausibility rules at the current plausibility threshold, independently re-checked against the vendored [Ensense](https://github.com/formal-trust-AI/ensense) core. SensEI builds on Ensense (Varshney et al., ICLR 2026) as its independent sensitivity oracle:

> _Ensense finds it, SensEI fixes it, and checks it again._

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
  - [Software](#software)
  - [Gurobi License](#gurobi-license)
  - [Ensense](#ensense)
  - [Hardware](#hardware)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Quick Start](#quick-start)
  - [Command-Line Interface](#command-line-interface)
  - [Writing Sensitivity Specification](#writing-sensitivity-specification)
  - [Tuning Main Parameters](#tuning-main-parameters)
  - [Programmatic Use](#programmatic-use)
  - [Debugging](#debugging)
- [Reproducing Experimental Results](#reproducing-experimental-results)
- [Contribution](#contribution)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Repository Structure

`sensei` (the importable package) lives at `src/sensei/` (the standard "src layout" -- `pyproject.toml` discovers it there, and `pip install -e .` points a plain path entry at `src/`, so `import sensei` works and static analyzers/IDEs resolve it too). `experiments/` is deliberately a separate, top-level directory: plain runnable scripts, not part of the installed package.

```
dataset/                              # built datasets (csv files, scaler, etc.)

docs/
├── architecture.md                   # system overview: the idea, the loop, the guardrails
└── technical_overview.md             # from-the-code reference: exact math, per module

ensense/                              # Ensense's own code (read-only, not ours)

experiments/                          # scripts you run from the terminal
├── run_baseline_sweep.py             # run baselines on every dataset
├── run_baselines.py                  # run the four baselines
├── run_certify.py                    # repair one feature and check the result
└── run_hyperparam_sweep.py           # run run_certify.py across a hyperparameter grid

logs/                                 # one timestamped log file per run (gitignored)

results/                              # one JSON file per run

src/
└── sensei/                           # the sensei package
    ├── config/                       # settings
    │   ├── defaults.yaml             # default hyperparameters
    │   └── ensense_pin.txt           # exact Ensense version we use
    ├── data/                         # building and loading datasets
    │   ├── bins.py                   # groups feature values into bins
    │   ├── builder.py                # turns a raw CSV into a dataset
    │   └── loader.py                 # loads a built dataset
    ├── eval/                         # measuring and checking models
    │   ├── heldout_verify.py         # re-checks a repair with Ensense
    │   ├── metrics.py                # accuracy and sensitivity scores
    │   ├── plots.py                  # saves the per-run diagnostic plots
    │   ├── region_overlap.py         # checks if a fix already covers a case
    │   ├── results_writer.py         # writes one results JSON per run
    │   └── sweeps.py                 # repeats a check across many settings
    ├── model/                        # training and reading the model
    │   ├── baselines.py              # trains the four baseline comparison models
    │   ├── leaves.py                 # reads and edits leaf values
    │   └── train.py                  # trains an XGBoost model
    ├── oracle/                       # finds unfair pairs of inputs
    │   ├── types.py                  # shared data types (Pair, NoGood, TreeStructure, ...)
    │   ├── ensense/                  # wraps Ensense core
    │   │   └── adapter.py            # asks Ensense to find a pair
    │   └── sensei/                   # our own oracle
    │       ├── encoding.py           # writes the model as MILP constraints
    │       ├── nogoods.py            # blocks a solution we already saw
    │       ├── solve.py              # runs our oracle
    │       └── warm_start.py         # reuses the last solution to solve faster
    ├── repair/                       # fixing unfair pairs
    │   ├── cuts.py                   # turns a pair into a constraint
    │   ├── loop.py                   # the main search-fix-repeat loop
    │   ├── pareto.py                 # keeps the best model seen so far
    │   └── qp.py                     # solves the fix as an optimization problem
    ├── spec/                         # per-dataset settings
    │   ├── adult.yaml                # settings for the adult dataset
    │   ├── churn.yaml                # settings for the churn dataset
    │   ├── german_credit.yaml        # settings for the german credit dataset
    │   └── pimadiabetes.yaml         # settings for the diabetes dataset
    ├── validity/                     # checking if an input is realistic
    │   ├── domain_rules.py           # checks simple rules between features
    │   ├── encode_milp.py            # adds these rules to our oracle
    │   ├── fd_rules.py               # checks features that must match
    │   ├── plausibility.py           # scores how realistic an input is
    │   ├── postfilter.py             # rejects bad inputs from Ensense
    │   └── type_rules.py             # checks types, ranges, and categories
    ├── pipeline.py                   # runs search, repair, and checks together
    ├── args.py                       # reads command-line options
    ├── logging_setup.py              # configures console + logs/ file logging
    └── cli.py                        # the `sensei` command

tests/
├── test_bins_frozen.py               # checks bins don't change by accident
├── test_domain_rules.py              # checks domain rule logic
├── test_ensense_unmodified.py        # checks Ensense's code wasn't edited
├── test_leaf_roundtrip.py            # checks leaf values save and load correctly
├── test_milp_agrees_with_ensense.py  # checks our oracle agrees with Ensense
└── test_postfilter.py                # checks bad inputs get rejected
```

**Key files:**

- `repair/qp.py` and `repair/loop.py` do the actual repair.
- `oracle/ensense/adapter.py` talks to Ensense; `oracle/sensei/` is our own oracle.
- `pipeline.py` runs the whole repair-and-check process, used by both the CLI and the experiment scripts.

---

## Prerequisites

### Software

| Requirement          | Version              | Notes                                                                    |
| -------------------- | -------------------- | ------------------------------------------------------------------------ |
| Python               | ≥ 3.11               | core programming language                                                |
| XGBoost              | 3.1.1 (pinned)       | matches the vendored `ensense/`'s own requirements                       |
| Gurobi               | ≥ 13.0               | **license required** (see below)                                         |
| z3-solver            | ≥ 4.15               | installed to match `ensense/`'s own pin; not called by sensei's own code |
| NumPy, SciPy, pandas | see `pyproject.toml` | sparse matrix operations, data handling                                  |
| scikit-learn         | ≥ 1.7                | splits, scaling, baseline classifiers                                    |

### Gurobi License

Gurobi is commercial. Free options:

- **Academic named-user license** &mdash; free for students and faculty at recognised institutions, full-size models. Request at [gurobi.com/academia](https://www.gurobi.com/academia/)
- **Limited pip license** &mdash; installed automatically with `pip install gurobipy`, capped at $2000$ variables. Sufficient for the tests in `tests/` but **not** for large ensembles.

An $800$-tree ensemble has roughly $200,000$ leaves. Plan for the academic license.

### Ensense

SensEI vendors [Ensense](https://github.com/formal-trust-AI/ensense) into `ensense/` at the project root, pinned to the commit recorded in `src/sensei/config/ensense_pin.txt`, and treats it as **read-only** (a pre-commit hook, `.pre-commit-config.yaml`, rejects any diff under it). `ensense/` is gitignored, so each checkout needs to vendor it itself:

```bash
git clone https://github.com/formal-trust-AI/ensense.git ensense
git -C ensense checkout $(cat src/sensei/config/ensense_pin.txt)
```

See `docs/architecture.md` for what was found by reading the vendored source, not assumed.

### Hardware

Development runs on a laptop. Reproducing results is CPU-bound on MILP solves.

---

## Installation

```bash
git clone https://github.com/arihant-b/sensei.git
cd sensei

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

This also installs the `sensei` console command (see [Command-Line Interface](#command-line-interface)) into your virtual environment.

Vendor Ensense as described above, then activate the Gurobi license:

```bash
grbgetkey <your-license-key>
python -c "import gurobipy; gurobipy.Model()"   # should print no error
```

Verify:

```bash
pytest tests/ -q
python experiments/run_baselines.py --dataset adult
sensei --help
```

The test suite covers leaf round-tripping, cut/no-good construction, sensei-oracle vs. Ensense SAT/UNSAT agreement, plausibility bin freezing, domain-rule checks, the Ensense postfilter's rejection budget, and that `ensense/` matches its pinned SHA. `run_baselines` trains and evaluates all four baselines on the `adult` dataset and writes a results JSON. `sensei --help` should print the CLI's usage without error.

---

## Usage Guide

### Quick Start

Once installed (see [Installation](#installation)), the fastest way to run SensEI is the `sensei` command:

```bash
sensei --dataset adult --feature sex
```

Equivalently, without installing the console command, the same pipeline is runnable as a script:

```bash
python experiments/run_certify.py --dataset adult --feature sex
```

Either way, this trains M0, runs CEGSAL to repair sensitivity to `sex`, then independently re-checks the repair with a fresh call into Ensense core, runs a required `theta` sweep and a companion `eps` sweep, and writes one results JSON. One real run (`results/certify_adult_sex_*.json`, 30 trees/depth 4) landed on a `ParetoBest` snapshot after 12 rounds (`iteration_cap`): worst gap dropped to `0.2559`, accuracy `0.8707`, sensitivity rate `0.025`, 60 cuts, zero slack. Held-out verification against Ensense core still found a fresh violation there (gap `0.1217`, region-overlap rate `0.0`) &mdash; an honest negative result, reported as such rather than hidden (see [Held-out Verification](#held-out-verification)).

### Command-Line Interface

`sensei` (installed by `pip install -e .`, see [Installation](#installation)) runs the same search -> repair -> verify -> sweep pipeline as `experiments/run_certify.py`, but is driven entirely by CLI flags instead of `src/sensei/config/defaults.yaml` -- `src/sensei/args.py` defines one optional flag per `defaults.yaml` field, at that file's own default value, so an unconfigured invocation behaves identically to the YAML-driven experiment script. Run `sensei --help` for the full, current list; the required and most commonly-overridden flags are:

| Flag                                                                  | Required? | Default                   | Meaning                                                                    |
| --------------------------------------------------------------------- | --------- | ------------------------- | -------------------------------------------------------------------------- |
| `--dataset`                                                           | **yes**   | &mdash;                   | dataset name, matching `dataset/<name>/` and `src/sensei/spec/<name>.yaml` |
| `--feature`                                                           | **yes**   | &mdash;                   | feature to repair sensitivity to                                           |
| `--direction`                                                         | no        | `protected`               | `protected` or `monotone_wrong`                                      |
| `--eps`                                                               | no        | `0.10`                    | sensitivity budget, margin space                                           |
| `--theta`                                                             | no        | `1e-9`                    | plausibility threshold                                                     |
| `--gap`                                                               | no        | `0.30`                    | Ensense's confident-flip margin, probability space (`output_gap=(gap, 1-gap)`); avoid `0.5`, a zero-width band Ensense's own solver mishandles |
| `--mu`, `--kap`                                                       | no        | `0.1`, `100.0`            | proximal weight, slack price                                        |
| `--n-estimators`, `--max-depth`                                       | no        | `200`, `5`                | M0's size                                                                  |
| `--max-iters`, `--a-min`, `--stall-delta`, `--cuts-per-round`         | no        | `50`, `0.82`, `1e-3`, `5` | CEGSAL stop conditions                                              |
| `--oracle-time-limit-s`, `--oracle-mip-gap`                           | no        | `300.0`, `0.05`           | per-call Gurobi limits                                                     |
| `--oracle-type`                                                       | no        | `sensei`                  | `sensei` (own MILP) or `ensense` (Ensense core + postfilter, weaker) |
| `--oracle-method`                                                     | no        | `milp`                    | which Ensense core solver family to call (`pb` or `milp`) whenever Ensense core is used |
| `--n-quantile-bins`                                                   | no        | `10`                      | plausibility bins                                                   |
| `--seed`                                                              | no        | `42`                      | single global seed: data split, model training, oracle search sampling, baseline retraining |
| `--results-dir`                                                       | no        | `results/`                | where the output JSON is written, relative to your current directory       |

```bash
sensei --dataset adult --feature sex --eps 0.2 --max-iters 20 --n-estimators 50 --max-depth 5
```

### Writing Sensitivity Specification

Everything downstream depends on this file. It encodes Q3 and must be written by someone who understands the domain.

```yaml
# src/sensei/spec/adult.yaml
dataset: adult

protected: [sex, race]
monotone:
  education-num: increasing # more education must not lower the score
  hours-per-week: increasing
immutable: [native-country]

# --- Q1 type validity ---
# Every ordinal-encoded categorical (workclass, education, marital-status,
# race, sex, ...) belongs here too, not just the genuinely-numeric features --
# data/builder.py encodes each one as a raw ordinal index, so its raw value
# is always an integer. Omitting a categorical lets the oracle propose a
# fractional category code that doesn't correspond to any real category.
integer_features:
  [age, education-num, hours-per-week, capital-gain, capital-loss,
   workclass, education, marital-status, occupation, relationship,
   race, sex, native-country]
one_hot_groups: {}
ranges:
  age: [0.0, 1.0]
  hours-per-week: [0.0, 0.9999999999999999]
  # ... one entry per feature, taken verbatim from dataset/adult/details.csv
functional_deps: []
```

`ranges`/`functional_deps` values are in the same min-max-scaled `[0, 1]` space `data/builder.py` writes to `dataset/<name>/train.csv`, not raw units -- the same `details.csv` Ensense core's own search uses for its bounds (`oracle/ensense/adapter.py`'s `_details_csv_for`), so the spec and Ensense's own domain are drawn from one source of truth.

### Tuning Main Parameters

| Parameter        | Meaning                                            | Effect of increasing                                                                                                                           |
| ---------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `eps`            | Sensitivity budget (margin-space cut tightness)    | Weaker guarantee, less accuracy cost, faster convergence                                                                                       |
| `theta`          | Plausibility threshold                             | Fewer counterexamples accepted; too high shrinks the plausible region until UNSAT is trivial &mdash; always report as a sweep, never one value |
| `mu`             | Proximal weight on `\|\|v - v0\|\|^2`              | Safer on unseen inputs, more resistance to repair                                                                                              |
| `kap`            | Slack price on cut violations                      | Cuts bind harder (slack toward 0), at more accuracy cost                                                                                       |
| `gap`            | Ensense's confident-flip margin, probability space | Only confident flips count in ensense-oracle calls; avoid `0.5` (zero-width band, mishandled by Ensense's own solver)                          |
| `oracle.method`  | Which Ensense solver family (`pb` or `milp`)       | Different solver, different runtime/results whenever Ensense core is used                                                                      |
| `cuts_per_round` | Cuts batched per oracle call before repairing      | Fewer MILP calls, possibly redundant cuts (no effect for the ensense oracle, which has no no-goods to diversify with -- one search per round)  |

### Programmatic Use

```python
import dataclasses

from sensei.config import load_defaults
from sensei.data.loader import Dataset
from sensei.spec import load_spec
from sensei.model.train import Trainer
from sensei.repair.loop import CegsalLoop, ConditionalCertificate

settings = load_defaults()  # every hyperparameter comes from config/defaults.yaml
settings = dataclasses.replace(
    settings,
    model=dataclasses.replace(settings.model, n_estimators=30, max_depth=4),
    loop=dataclasses.replace(settings.loop, max_iters=15),
)

spec = load_spec("adult")
ds = Dataset("adult", eval_holdout=settings.dataset.eval_holdout, seed=settings.seed).load()
booster = Trainer.train_baseline(
    ds.X_train, ds.y_train, settings.model.n_estimators, settings.model.max_depth, settings.seed
)

loop = CegsalLoop(
    booster, ds.X_train, ds.y_train, ds.X_test, ds.y_test,
    ds.columns, ds.feature_bounds, spec, settings,
)
result = loop.run(flip_set=("sex",), direction="protected")

if isinstance(result, ConditionalCertificate):
    print(result.statement())
else:
    print(type(result).__name__, result.snapshot.worst_gap, result.snapshot.accuracy)
```

`CegsalLoop` takes one `Settings` object (fixed for the loop's lifetime) instead of separate `eps`/`theta`/`mu`/... arguments -- override just the fields you need via `dataclasses.replace` before constructing it, as above. This is the same `Settings` object `sensei.args.parse_args()`/`load_defaults()` produce, so CLI runs and programmatic use share one config shape.

### Interpreting Output

| Result                                | Meaning                                                                                                                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ConditionalCertificate`              | Oracle returned UNSAT: no valid, plausible pair with gap > `eps` was found in the encoded domain. Conditional on Q1+Q2 and the current `theta` &mdash; not an unconditional proof. |
| `ParetoBest`, reason `accuracy_floor` | Repair cost more accuracy than `a_min` allows. Loosen `eps`/`kap` or inspect for leaf collisions.                                                                                  |
| `ParetoBest`, reason `stalled`        | Worst gap stopped improving between rounds. Best accuracy-meeting snapshot restored.                                                                                               |
| `ParetoBest`, reason `iteration_cap`  | `max_iters` rounds ran out before UNSAT, a floor, or a stall.                                                                                                                      |
| `Inconclusive`                        | The oracle timed out or its rejection budget saturated &mdash; never reported as UNSAT.                                                                                            |

### Debugging

There's no dedicated debug flag. Every entry point (the experiment scripts and `sensei.cli`) calls `sensei.logging_setup.setup_logging(name)`, which writes a full, timestamped log to `logs/<name>_<timestamp>.log` in addition to the console -- read that first. Increase verbosity by passing a different `level` to `setup_logging`, or use the pieces directly (see [Programmatic Use](#programmatic-use)) and inspect `result.snapshot.cuts` yourself &mdash; each `Cut` (`repair/cuts.py`) carries its own sparse leaf indices and coefficients. Every run's full parameters and final snapshot also land in `results/` regardless of verbosity.

---

## Reproducing Experimental Results

### Datasets

| Dataset       | Spec (`src/sensei/spec/`) | Dataset artifacts (`dataset/`) | Protected `P`              | Monotone `M`                            |
| ------------- | -------------------------- | ------------------------------ | -------------------------- | --------------------------------------- |
| Adult         | reviewed                   | built                          | `sex`, `race`              | `education-num` up, `hours-per-week` up |
| German credit | drafted, UNREVIEWED        | built                          | `PersonalStatusSex`, `Age` | `CreditAmount` down, `Duration` down    |
| Pima diabetes | drafted, UNREVIEWED        | built                          | &mdash;                    | `Glucose` up                            |
| Churn         | drafted, UNREVIEWED        | built                          | `gender`, `SeniorCitizen`  | `tenure` up                             |

All four have built dataset artifacts; only `adult`'s spec has been reviewed. `dataset/` also holds a few datasets built for baseline comparisons only (`breast_cancer`, `diabetes`, `ijcnn`, `iris`, `winequality_red`) -- these have no `spec/*.yaml`, so `sensei --dataset <name> ...` (certify) doesn't work on them, only `run_baselines.py`/`run_baseline_sweep.py`. Run `src/sensei/data/builder.py` against a raw CSV to add a new one.

### Running Experiments

```bash
# all four baselines, one dataset
python experiments/run_baselines.py --dataset adult

# ...or across every dataset that currently has a built train.csv
python experiments/run_baseline_sweep.py

# Ensense-oracle validity and the sensei oracle + repair QP are library code,
# not separate CLI entry points -- exercised by
# tests/test_milp_agrees_with_ensense.py and by run_certify.py itself.

# repair one flip set via CEGSAL, held-out verify, region-overlap,
# theta sweep (required) + eps sweep, sensei vs. ensense oracle comparison
python experiments/run_certify.py --dataset adult --feature sex \
    --n-estimators 30 --max-depth 4 --max-iters 15 --oracle-time-limit-s 30
```

Each run writes one JSON to `results/`. The installed `sensei` command (see [Command-Line Interface](#command-line-interface)) runs the same single-feature certify pipeline too, sourced from CLI flags instead of `defaults.yaml`.

### Held-out Verification

**Never skipped.** `run_certify.py` automatically re-checks the repaired model with a fresh call into the vendored Ensense core (via `EnsenseOracle`) &mdash; a genuinely different implementation than the `SenseiOracle` MILP that did the repairing, not merely a different seed on the same solver (Ensense core's own seed is a hardcoded module constant, not exposed to callers &mdash; see `docs/architecture.md`).

- fresh UNSAT &rarr; the repair generalized
- fresh SAT &rarr; the repair patched specific cases, not the property &mdash; reported as `heldout_verify.generalized: false` in the results JSON, not hidden

The one real end-to-end run so far found the latter: worst gap fell to `0.256` over 12 rounds, but Ensense core still found a fresh violation (gap `0.122`) with a `0.0` region-overlap rate against the accumulated cuts &mdash; an honest negative result, not a certificate.

### Metrics

Frozen by design (`eval/metrics.py::Metrics`, `VERSION`):

| Metric           | Definition                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy         | test accuracy on `D_eval`                                                                                                                   |
| Sensitivity rate | fraction of held-out rows whose prediction flips when a protected feature is flipped                                                        |
| Worst valid gap  | max signed gap from the sensei oracle (default) in `mode="optimality"`, at the current `theta`; `NaN` (logged) on `EMPTY_DOMAIN`, never a silent zero |
| Slack mass       | `sum(s_i)` at repair termination                                                                                                            |

Changing a definition invalidates every earlier result and requires bumping `Metrics.VERSION`.

### Baselines

| Baseline                          | Handles label-flip constraint?             | Data-aware?                              | Certificate?    |
| --------------------------------- | ------------------------------------------ | ---------------------------------------- | --------------- |
| Plain XGBoost                     | no                                         | no                                       | no              |
| Drop protected feature            | no &mdash; proxies survive                 | no                                       | no              |
| XGBoost `monotone_constraints`    | no                                         | no &mdash; enforced in empty regions too | no              |
| Reweighing                        | partially                                  | no                                       | no              |
| Counterexample retraining (P1/P2) | partially, and only where a pair was found | yes                                      | no              |
| **SensEI**                        | **yes**                                    | **yes**                                  | **conditional** |

Baselines 2 and 3 are built-in tools that partially address the problem and will be raised by any reviewer. Including them strengthens the case.

### Reproducibility

Every experiment run writes one JSON to `results/<kind>_<dataset>[_<feature>]_<timestamp>.json`, never overwritten: git SHA, the vendored Ensense pin SHA, dataset, spec hash, every hyperparameter (`eps`/`theta`/`mu`/`kap` for a certify run, `metrics_version` for a baselines run), the seed, and the full outcome. There's a single global seed, fixed in `src/sensei/config/defaults.yaml`'s `seed:` field (or via the `sensei` CLI's `--seed` flag); there's no built-in multi-seed runner yet, so reporting across seeds means invoking the experiment script (or CLI) once per seed.

---

## Contribution

Contributions are welcome, particularly on:

- Additional dataset specifications
- Alternative repair options
- Characterising when leaf repair is insufficient

### Workflow

```bash
git checkout -b feature/<name>
pytest tests/ -q
ruff check . && ruff format .
mypy src/
```

Open a PR against `main` with a description of what changed and why.

### Guidelines

- New solver encodings need a correctness test against a small real trained model with a known SAT/UNSAT answer (see `tests/test_milp_agrees_with_ensense.py`)
- Preserve the module dependency order; no cycles
- Negative results are welcome. "Leaf repair provably cannot converge on models with property X" is a valuable contribution, and this repository documents its own limitations.

---

## Citation

If you use SensEI, please cite this repository and the underlying Ensense work:

```bibtex
@software{sensei,
  title  = {SensEI},
  author = {arihant-b},
  year   = {2026},
  url    = {https://github.com/arihant-b/sensei}
}

@inproceedings{varshney2026dataaware,
  title     = {Data-Aware and Scalable Sensitivity Analysis for
               Decision Tree Ensembles},
  author    = {Varshney, Namrita and Gupta, Ashutosh and Ahmad, Arhaan and
               Tayal, Tanay V. and Akshay, S.},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

Related work worth citing:

- Ahmad et al. (2025), _Sensitivity verification for additive decision tree ensembles_ &mdash; ICLR
- Kantchelian et al. (2016), _Evasion and hardening of tree ensemble classifiers_ &mdash; ICML

---

## License

See the [LICENSE](LICENSE) file for full statutory terms and legal disclosures.

Gurobi is licensed separately and is not distributed with this repository. Datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml) under their respective terms.

---

## Contact

**Authors:**

- Arihant Bedagkar
- Ashutosh Gupta

**Issues and bugs:** For bugs, solver integration errors, or feature requests, please open an issue [here](https://github.com/arihant-b/sensei/issues).

**Discussion:** For questions about the formulation, certificates, or extending the encodings, open a discussion [here](https://github.com/arihant-b/sensei/discussions).
