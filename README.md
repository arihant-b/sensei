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
dataset/                              # built dataset artifacts (currently: adult) -- see data/builder.py

docs/
└── ensense_interface.md              # inventory of Ensense core's actual input surface

ensense/                              # vendored Ensense core, pinned to a commit SHA, read-only, gitignored

experiments/                          # plain scripts, OUTSIDE the installed package
├── run_all_features.py               # CertifyPipeline across every P/M feature a spec declares
├── run_baseline_sweep.py             # run_stage0_baselines across every dataset with a built train.csv
├── run_stage0_baselines.py           # all four baselines, one dataset
└── run_stage3_certify.py             # CertifyPipeline for one feature, sourced from defaults.yaml

results/                              # one JSON per experiment or CLI run, append-only, never overwritten

src/
└── sensei/                           # the package (standard src-layout, import sensei still works, see above)
    ├── config/                       # defaults.yaml (eps/theta/mu/kap/seeds/...), ensense_pin.txt
    │   ├── defaults.yaml             #
    │   └── ensense_pin.txt           #
    ├── data/                         # builder.py (raw CSV -> dataset/<name>/), loader.py, bins.py, preprocess.py
    │   ├── bins.py                   #
    │   ├── builder.py                #
    │   ├── loader.py                 #
    │   └── preprocess.py             #
    ├── eval/                         # metrics.py, baselines.py, heldout_verify.py, region_overlap.py, sweeps.py
    │   ├── baselines.py              #
    │   ├── heldout_verify.py         #
    │   ├── metrics.py                #
    │   ├── region_overlap.py         #
    │   └── sweeps.py                 #
    ├── model/                        # train.py, leaves.py (leaf indexing + phi(x)), freeze.py
    │   ├── freeze.py                 #
    │   ├── leaves.py                 #
    │   └── train.py                  #
    ├── oracle/
    │   ├── base.py                   #
    │   ├── ensense_adapter.py        # Tier B: TierBOracle, one method per Ensense solver family (pb, milp)
    │   ├── types.py                  #
    │   └── milp/                     # Tier A: sensei's own MILP oracle (encoding, no-goods, warm starts, solve)
    │       ├── encoding.py           #
    │       ├── nogoods.py            #
    │       ├── solve.py              #
    │       └── warm_start.py         #
    ├── repair/                       # cuts.py, qp.py (soft-slack repair QP), loop.py (CEGSAL), pareto.py
    │   ├── cuts.py                   #
    │   ├── loop.py                   #
    │   ├── pareto.py                 #
    │   └── qp.py                     #
    ├── spec/                         # per-dataset protected/monotone/immutable groups + preprocessing decisions
    │   ├── adult.yaml                #
    │   ├── churn.yaml                #
    │   ├── german_credit.yaml        #
    │   └── pimadiabetes.yaml         #
    ├── validity/                     # type/FD/domain/plausibility rules, MILP constraint export, postfilter
    │   ├── domain_rules.py           #
    │   ├── encode_milp.py            #
    │   ├── fd_rules.py               #
    │   ├── plausibility.py           #
    │   ├── postfilter.py             #
    │   └── type_rules.py             #
    ├── pipeline.py                   # CertifyPipeline: the search->repair->verify->sweep pipeline, shared by
    │                                   experiments/run_stage3_certify.py and the `sensei` CLI
    ├── args.py                       # sensei CLI's argument parsing -- used INSTEAD OF defaults.yaml for that
    │                                   entry point (see Command-Line Interface below)
    └── cli.py                        # `sensei` console command's entry point (args.py -> pipeline.py -> JSON)

tests/
├── test_bins_frozen.py               #
├── test_domain_rules.py              #
├── test_ensense_unmodified.py        #
├── test_leaf_roundtrip.py            #
├── test_milp_agrees_with_ensense.py  #
├── test_postfilter.py                #
└── test_stage1_appendix_example3.py  #
```

**Key files:**

- `repair/qp.py` and `repair/loop.py` hold the core contribution: the soft-slack repair QP and the CEGSAL search-cut-repair loop.
- `oracle/ensense_adapter.py` is the only integration point with the vendored Ensense core (Tier B); `oracle/milp/` is sensei's own independent oracle (Tier A), used both to repair and, via a fresh Ensense call, to be checked by something other than itself.
- `pipeline.py` is the single implementation both `experiments/run_stage3_certify.py` and the `sensei` command run -- they only differ in where they get their settings from.

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

See `docs/ensense_interface.md` for what Ensense core's own CLI/API actually exposes (seeds, time limits, UNSAT-vs-timeout ambiguity, etc.) &mdash; found by reading the vendored source, not assumed.

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
python experiments/run_stage0_baselines.py --dataset adult
sensei --help
```

The test suite covers leaf round-tripping, cut/no-good construction, Tier A vs. Ensense SAT/UNSAT agreement, and the Stage 1 Appendix Example 3 deliverable. `run_stage0_baselines` trains and evaluates all four baselines on the `adult` dataset and writes a results JSON. `sensei --help` should print the CLI's usage without error.

---

## Usage Guide

### Quick Start

Once installed (see [Installation](#installation)), the fastest way to run SensEI is the `sensei` command:

```bash
sensei --dataset adult --feature sex
```

Equivalently, without installing the console command, the same pipeline is runnable as a script:

```bash
python experiments/run_stage3_certify.py --dataset adult --feature sex
```

Either way, this trains M0, runs CEGSAL to repair sensitivity to `sex`, then independently re-checks the repair with a fresh call into Ensense core, runs a required `theta` sweep and a companion `eps` sweep, and writes one results JSON. One real run (`results/stage3_certify_adult_sex_*.json`, 30 trees/depth 4) landed on a `ParetoBest` snapshot after 12 rounds (`iteration_cap`): worst gap dropped to `0.2559`, accuracy `0.8707`, sensitivity rate `0.025`, 60 cuts, zero slack. Held-out verification against Ensense core still found a fresh violation there (gap `0.1217`, region-overlap rate `0.0`) &mdash; an honest negative result, reported as such rather than hidden (see [Held-out Verification](#held-out-verification)).

### Command-Line Interface

`sensei` (installed by `pip install -e .`, see [Installation](#installation)) runs the same search -> repair -> verify -> sweep pipeline as `experiments/run_stage3_certify.py`, but is driven entirely by CLI flags instead of `src/sensei/config/defaults.yaml` -- `src/sensei/args.py` defines one optional flag per `defaults.yaml` field, at that file's own default value, so an unconfigured invocation behaves identically to the YAML-driven experiment script. Run `sensei --help` for the full, current list; the required and most commonly-overridden flags are:

| Flag                                                                  | Required? | Default                   | Meaning                                                                    |
| --------------------------------------------------------------------- | --------- | ------------------------- | -------------------------------------------------------------------------- |
| `--dataset`                                                           | **yes**   | &mdash;                   | dataset name, matching `dataset/<name>/` and `src/sensei/spec/<name>.yaml` |
| `--feature`                                                           | **yes**   | &mdash;                   | feature to repair sensitivity to                                           |
| `--direction`                                                         | no        | `protected`               | `protected` or `monotone_wrong`                                      |
| `--eps`                                                               | no        | `0.10`                    | sensitivity budget, margin space                                           |
| `--theta`                                                             | no        | `1e-6`                    | plausibility threshold                                                     |
| `--mu`, `--kap`                                                       | no        | `0.01`, `100.0`           | proximal weight, slack price                                        |
| `--n-estimators`, `--max-depth`                                       | no        | `200`, `5`                | M0's size                                                                  |
| `--max-iters`, `--a-min`, `--stall-delta`, `--cuts-per-round`         | no        | `50`, `0.82`, `1e-3`, `5` | CEGSAL stop conditions                                              |
| `--oracle-time-limit-s`, `--oracle-mip-gap`                           | no        | `300.0`, `0.05`           | per-call Gurobi limits                                                     |
| `--n-quantile-bins`                                                   | no        | `10`                      | plausibility bins                                                   |
| `--data-split-seed`, `--model-train-seed`, `--baseline4-retrain-seed` | no        | `42` each                 | pass seeds explicitly, never a global default                         |
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
integer_features:
  [age, education-num, hours-per-week, capital-gain, capital-loss]
one_hot_groups: {}
ranges:
  age: [0.0, 1.0]
  hours-per-week: [0.0, 1.0]
functional_deps:
  - [education, 0.6666666666666666, education-num, 1.0]

# --- Tier B1 preprocessing decisions ---
preprocessing:
  drop_fd_redundant_columns: []
  integer_code_features: []
```

`ranges`/`functional_deps` values are in the same min-max-scaled `[0, 1]` space `data/builder.py` writes to `dataset/<name>/train.csv`, not raw units.

### Tuning Main Parameters

| Parameter        | Meaning                                            | Effect of increasing                                                                                                                           |
| ---------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `eps`            | Sensitivity budget (margin-space cut tightness)    | Weaker guarantee, less accuracy cost, faster convergence                                                                                       |
| `theta`          | Plausibility threshold                             | Fewer counterexamples accepted; too high shrinks the plausible region until UNSAT is trivial &mdash; always report as a sweep, never one value |
| `mu`             | Proximal weight on `\|\|v - v0\|\|^2`              | Safer on unseen inputs, more resistance to repair                                                                                              |
| `kap`            | Slack price on cut violations                      | Cuts bind harder (slack toward 0), at more accuracy cost                                                                                       |
| `gap`            | Ensense's confident-flip margin, probability space | Only confident flips count in Tier B calls                                                                                                     |
| `cuts_per_round` | Cuts batched per oracle call before repairing      | Fewer MILP calls, possibly redundant cuts                                                                                                      |

### Programmatic Use

```python
from sensei.data.loader import Dataset
from sensei.spec import load_spec
from sensei.model.train import Trainer
from sensei.repair.loop import CegsalLoop, ConditionalCertificate

spec = load_spec("adult")
ds = Dataset("adult", eval_holdout=0.2, seed=42).load()
booster = Trainer.train_baseline(ds.X_train, ds.y_train, n_estimators=30, max_depth=4, seed=42)

result = CegsalLoop(
    booster, ds.X_train, ds.y_train, ds.X_test, ds.y_test,
    ds.columns, ds.feature_bounds, spec, seed=42,
).run(
    flip_set=("sex",), direction="protected",
    eps=0.10, theta=1e-6, mu=0.01, kap=100.0,
    max_iters=15, a_min=0.82, stall_delta=1e-3, cuts_per_round=5,
    oracle_time_limit_s=30.0, oracle_mip_gap=0.05,
)

if isinstance(result, ConditionalCertificate):
    print(result.statement())
else:
    print(type(result).__name__, result.snapshot.worst_gap, result.snapshot.accuracy)
```

### Interpreting Output

| Result                                | Meaning                                                                                                                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ConditionalCertificate`              | Oracle returned UNSAT: no valid, plausible pair with gap > `eps` was found in the encoded domain. Conditional on Q1+Q2 and the current `theta` &mdash; not an unconditional proof. |
| `ParetoBest`, reason `accuracy_floor` | Repair cost more accuracy than `a_min` allows. Loosen `eps`/`kap` or inspect for leaf collisions.                                                                                  |
| `ParetoBest`, reason `stalled`        | Worst gap stopped improving between rounds. Best accuracy-meeting snapshot restored.                                                                                               |
| `ParetoBest`, reason `iteration_cap`  | `max_iters` rounds ran out before UNSAT, a floor, or a stall.                                                                                                                      |
| `Inconclusive`                        | The oracle timed out or its rejection budget saturated &mdash; never reported as UNSAT.                                                                                            |

### Debugging

There's no dedicated debug flag. Increase verbosity by editing the `logging.basicConfig` call in an experiment script's or `sensei.cli`'s `main()`, or use the pieces directly (see [Programmatic Use](#programmatic-use)) and inspect `result.snapshot.cuts` yourself &mdash; each `Cut` (`repair/cuts.py`) carries its own sparse leaf indices and coefficients. Every run's full parameters and final snapshot land in `results/` regardless of verbosity &mdash; start there before adding logging.

---

## Reproducing Experimental Results

### Datasets

| Dataset       | Spec (`src/sensei/spec/`)         | Dataset artifacts (`dataset/`) | Protected `P`              | Monotone `M`                            |
| ------------- | --------------------------------- | ------------------------------ | -------------------------- | --------------------------------------- |
| Adult         | reviewed                          | built                          | `sex`, `race`              | `education-num` up, `hours-per-week` up |
| German credit | drafted, preprocessing UNREVIEWED | not yet built                  | `PersonalStatusSex`, `Age` | `CreditAmount` down, `Duration` down    |
| Pima diabetes | drafted, preprocessing UNREVIEWED | not yet built                  | &mdash;                    | `Glucose` up                            |
| Churn         | drafted, preprocessing UNREVIEWED | not yet built                  | `gender`, `SeniorCitizen`  | `tenure` up                             |

Only `adult` currently has built dataset artifacts. The others need `src/sensei/data/builder.py` run against their raw CSVs, and their preprocessing decisions confirmed, before any stage will work on them.

### Running Stages

```bash
# Stage 0 -- all four baselines, one dataset
python experiments/run_stage0_baselines.py --dataset adult

# ...or across every dataset that currently has a built train.csv
python experiments/run_baseline_sweep.py

# Stage 1 (Tier B validity) and Stage 2 (Tier A oracle + repair QP) are library
# code, not separate CLI stages -- exercised by tests/test_stage1_appendix_example3.py,
# tests/test_milp_agrees_with_ensense.py, and by Stage 3 itself.

# Stage 3 -- repair one flip set via CEGSAL, held-out verify, region-overlap,
# theta sweep (required) + eps sweep, Tier A vs Tier B comparison
python experiments/run_stage3_certify.py --dataset adult --feature sex \
    --n-estimators 30 --max-depth 4 --max-iters 15 --oracle-time-limit-s 30

# ...or across every protected/monotone feature a dataset's spec declares
python experiments/run_all_features.py --dataset adult
```

Each run writes one JSON to `results/`. The installed `sensei` command (see [Command-Line Interface](#command-line-interface)) runs the single-feature Stage 3 pipeline too, sourced from CLI flags instead of `defaults.yaml`.

### Held-out Verification

**Never skipped.** `run_stage3_certify.py` automatically re-checks the repaired model with a fresh call into the vendored Ensense core (Tier B) &mdash; a genuinely different implementation than the Tier A MILP that did the repairing, not merely a different seed on the same solver (Ensense core's own seed is a hardcoded module constant, not exposed to callers &mdash; see `docs/ensense_interface.md`).

- fresh UNSAT &rarr; the repair generalized
- fresh SAT &rarr; the repair patched specific cases, not the property &mdash; reported as `heldout_verify.generalized: false` in the results JSON, not hidden

The one real end-to-end run so far found the latter: worst gap fell to `0.256` over 12 rounds, but Ensense core still found a fresh violation (gap `0.122`) with a `0.0` region-overlap rate against the accumulated cuts &mdash; an honest negative result, not a certificate.

### Metrics

Frozen after Stage 0 (`eval/metrics.py::Metrics`, `VERSION`):

| Metric           | Definition                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Accuracy         | test accuracy on `D_eval`                                                                                                                   |
| Sensitivity rate | fraction of held-out rows whose prediction flips when a protected feature is flipped                                                        |
| Worst valid gap  | max signed gap from the Tier A oracle in `mode="optimality"`, at the current `theta`; `NaN` (logged) on `EMPTY_DOMAIN`, never a silent zero |
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

Every experiment run writes one JSON to `results/<stage>_<dataset>[_<feature>]_<timestamp>.json`, never overwritten: git SHA, the vendored Ensense pin SHA, dataset, spec hash, every hyperparameter (`eps`/`theta`/`mu`/`kap` for Stage 3, `metrics_version` for Stage 0), seeds, and the full outcome. Seeds are fixed in `src/sensei/config/defaults.yaml`'s `seeds:` block (or via the `sensei` CLI's `--data-split-seed`/`--model-train-seed`/`--baseline4-retrain-seed` flags); there's no built-in multi-seed runner yet, so reporting across seeds means invoking the experiment script (or CLI) once per seed.

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
