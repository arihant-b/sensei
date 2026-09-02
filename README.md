# SensEI

**Sens**itivity **E**nsemble **I**ntervention

This repository contains SensEI, a tool for **certified sensitivity repair** of decision tree ensembles. Given a trained XGBoost model, SensEI searches for realistic pairs of inputs that differ only in a protected feature (such as sex or race) yet receive different predictions, and then repairs the model so that such pairs no longer exist. Repair is performed by adjusting leaf values only &mdash; the tree structure is never modified &mdash; which reduces the problem to a convex quadratic program that solves in seconds even for ensembles of 800 trees. Because the counterexample search is complete, SensEI terminates with a **proof** that no valid undesirable counterexample remains, rather than the empirical "we tested and found none" that fairness heuristics offer. SensEI builds directly on [Ensense](https://github.com/formal-trust-AI/ensense) (Varshney et al., ICLR 2026), which it uses as a sensitivity oracle:
> *Ensense finds it, SensEI fixes it, and proves it.*

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

```
src/
├── config.py        # all hyperparameters in one dataclass
├── data.py          # loading, splits, frozen quantile bins, density
├── loop.py          # the CEGSAL loop
├── main.py          # entry point
├── metrics.py       # evaluation + Pareto checkpointing
├── model.py         # XGBoost wrapper + leaf indicator matrix phi(x)
├── oracle.py        # Ensense / MILP counterexample search (the only UNSAT source)
├── repair.py        # convex QP over leaf values + cut store
├── sampler.py       # cheap solver-free violation screen
├── spec.py          # sensitivity specification: protected/monotone/free
└── validity.py      # Q1 type rules, Q2 plausibility, MILP constraint export

experiments/
├── stage0_baselines.py
├── stage1_validity.py
├── stage2_repair.py
├── stage3_training.py
└── plots/

tests/
├── test_leaf_indexing.py
├── test_cut_construction.py
├── test_qp_convergence.py
└── fixtures/synthetic.py

configs/
├── adult.yaml
├── german_credit.yaml
└── pima.yaml

runs/                # artifacts, checkpoints, logs (gitignored)
```

**Module dependency order:** `config` → `spec` → `data` → `model` → `validity` → {`sampler`, `oracle`} → `repair` → `loop` → `main`.

**Key files:**
- `repair.py` holds the core contribution.
- `oracle.py` is the only integration point with [Ensense](https://github.com/formal-trust-AI/ensense).

---

## Prerequisites

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | core programming language |
| XGBoost | 1.7.1 | decision tree ensemble |
| Gurobi | ≥ 10.0 | **license required** (see below) |
| z3-solver | ≥ 4.12 | SMT for disjunctive domain rules |
| NumPy, SciPy | recent | sparse matrix operations |
| scikit-learn | ≥ 1.3 | splits, metrics, baseline density |

### Gurobi License

Gurobi is commercial. Free options:

- **Academic named-user license** &mdash; free for students and faculty at recognised institutions, full-size models. Request at [gurobi.com/academia](https://www.gurobi.com/academia/)
- **Limited pip license** &mdash; installed automatically with `pip install gurobipy`, capped at $2000$ variables. Sufficient for the toy fixtures in `tests/` but **not** for real benchmarks.

An $800$-tree ensemble has roughly $200,000$ leaves. Plan for the academic license.

### Ensense

SensEI uses [Ensense](https://github.com/formal-trust-AI/ensense) as its sensitivity oracle. Clone it alongside this repository:

```bash
git clone https://github.com/formal-trust-AI/ensense.git
```

### Hardware

Development runs on a laptop. Full benchmark reproduction is CPU-bound on MILP solves.

---

## Installation

```bash
git clone https://github.com/arihant-b/sensei.git
cd sensei

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

Activate the Gurobi license:

```bash
grbgetkey <your-license-key>
python -c "import gurobipy; gurobipy.Model()"   # should print no error
```

Point SensEI at Ensense:

```bash
export ENSENSE_PATH=/path/to/ensense
```

Verify:

```bash
pytest tests/ -q
python -m sensei.main --config configs/synthetic.yaml --smoke
```

The smoke test runs the full loop on a $3$-tree synthetic model with a known planted violation. It should terminate with a certificate in under $10$ seconds.

---

## Usage Guide

### Quick Start

```bash
python -m sensei.main --config configs/adult.yaml
```

Output:

```
baseline accuracy: 0.8641
--- repairing sensitivity to 'sex' ---
iter  0 | gap 3.4021 | acc 0.8641 | cuts  5
iter  1 | gap 2.1044 | acc 0.8638 | cuts 10
iter  2 | gap 0.8510 | acc 0.8629 | cuts 15
iter  3 | gap 0.0612 | acc 0.8624 | cuts 20
UNSAT at iteration 4 -- certified
worst gap 0.0612 | accuracy 0.8624 | cost 0.0017
certified: True
```

### Writing Sensitivity Specification

Everything downstream depends on this file. It encodes Q3 and must be written by someone who understands the domain.

```yaml
# configs/adult.yaml
dataset: adult

spec:
  protected: [sex, race]
  monotone:
    education-num: up        # more education must not lower the score
    hours-per-week: up
  immutable: [native-country]

  integer_features: [age, education-num, hours-per-week,
                     capital-gain, capital-loss]
  ranges:
    age: [17, 90]
    hours-per-week: [1, 99]
  functional_deps:
    - [education, Doctorate, education-num, 16]
    - [education, Bachelors, education-num, 13]
```

### Tuning Main Parameters

| Parameter | Meaning | Effect of increasing |
|---|---|---|
| `epsilon` | Sensitivity budget (cut tightness) | Weaker guarantee, less accuracy cost, faster convergence |
| `theta` | Plausibility threshold | Fewer counterexamples accepted; too high yields vacuous certificates |
| `mu` | Weight on leaf drift | Safer on unseen inputs, more behaviour change on seen data |
| `gap` | Ensense's $g$, confidence margin | Only confident flips count |
| `cuts_per_round` | Cuts batched per oracle call | Fewer MILP calls, possibly redundant cuts |

### Programmatic Use

```python
from sensei import Config, Ensemble, SensEILoop, load_spec, Dataset

cfg  = Config(epsilon=0.1, theta=1e-6, mu=0.01)
spec = load_spec("configs/adult.yaml")

data  = Dataset(cfg); data.load()
model = Ensemble(cfg).fit(data.X_train, data.y_train)

loop = SensEILoop(model, data, oracle, sampler, cfg)
best = loop.run(flip_set=["sex"])

print(best.worst_gap, best.accuracy, loop.certified)
```

### Interpreting Output

| Result | Meaning |
|---|---|
| `certified: True` | Oracle returned UNSAT. No valid undesirable counterexample above $\varepsilon$ exists. |
| `certified: False`, stalled | Loop oscillated. Pareto checkpoint restored. Report the gap, not a certificate. |
| `certified: False`, accuracy floor | Repair cost too much. Loosen `epsilon` or inspect for leaf collisions. |
| UNSAT on **round 0** | Suspicious. Usually means `theta` is too high, so no counterexample is plausible enough to count. |

### Debugging

```bash
python -m sensei.main --config configs/adult.yaml --log-level DEBUG --dump-cuts
```

`--dump-cuts` writes every cut with its leaf indices and coefficients to `runs/<id>/cuts.jsonl`. Useful when a cut appears to have no effect &mdash; usually a leaf-indexing mismatch between `model.phi()` and the oracle's encoding.

---

## Reproducing Experimental Results

### Benchmarks

Models match Ensense's Table $1$ so runtimes are directly comparable.

| Dataset | Protected $\mathcal{P}$ | Monotone $\mathcal{M}$ | Trees | Depth |
|---|---|---|---|---|
| Adult | `sex`, `race` | `education-num` $\uparrow$, `hours` $\uparrow$ | $200$/$300$/$500$ | $5$, $6$ |
| German credit | `age`, `sex` | `amount` $\downarrow$, `duration` $\downarrow$ | $500$/$800$ | $5$, $6$ |
| Pima diabetes | &mdash; | `glucose` $\uparrow$ | $200$/$300$/$500$ | $5$, $6$ |
| Churn | &mdash; | `tenure` $\uparrow$ | $200$/$300$/$500$ | $5$, $6$ |

Pima and Churn have no natural protected attribute; they exercise the monotone path.

### Running Stages

```bash
# Stage 0 -- baselines
python -m experiments.stage0_baselines --all

# Stage 1 -- validity ablation
python -m experiments.stage1_validity --dataset adult

# Stage 2 -- repair (the main result)
python -m experiments.stage2_repair --all --seeds 42,43,44

# Stage 3 -- train-vs-repair comparison
python -m experiments.stage3_training --dataset adult

# Figures
python -m experiments.plots.pareto --input runs/
```

### Held-out Verification

**Do not skip this.** Adversarial training is notorious for patching exactly the counterexamples it saw while leaving the property broken elsewhere.

```bash
python -m experiments.stage2_repair --verify-generalization --seed 43
```

Runs a **fresh** oracle with a different seed and no accumulated no-goods against the repaired model.

- UNSAT $\to$ the repair generalised
- New violations found easily $\to$ the repair didn't work

### Metrics

| Metric | Definition |
|---|---|
| Sensitivity rate | fraction of $1,000$ held-out rows whose label flips when $\mathcal{P}$ is flipped |
| Worst valid gap | $\max_{\mathcal{V}} V$ from the oracle |
| Accuracy cost | baseline accuracy − repaired accuracy |
| $\ell_2$ distance to data | Ensense's own counterexample quality metric |
| Certified | did the oracle return UNSAT |
| Total runtime | wall-clock for the entire loop, not per instance |

### Baselines

| Baseline | Handles label-flip constraint? | Data-aware? | Certificate? |
|---|---|---|---|
| Plain XGBoost | no | no | no |
| Drop protected feature | no &mdash; proxies survive | no | no |
| XGBoost `monotone_constraints` | no | no &mdash; enforced in empty regions too | no |
| Reweighing | partially | no | no |
| **SensEI** | **yes** | **yes** | **yes** |

Baselines $2$ and $3$ are built-in tools that partially address the problem and will be raised by any reviewer. Including them strengthens the case.

### Expected Figures

1. **Pareto front** &mdash; accuracy vs sensitivity across $\lambda$ / $\varepsilon$. The headline result; a single number hides the trade-off.
2. **Convergence traces** &mdash; worst gap per iteration, per benchmark
3. **Validity comparison** &mdash; Ensense vs SensEI counterexamples, showing malformed values eliminated
4. **Held-out density** &mdash; mean log-likelihood under an independent model fitted on `X_eval`
5. **Leaf collision rate** &mdash; how often repair is blocked by shared leaves
6. **Ablation** &mdash; sampling screen, warm starts, and cut batching on MILP call count

### Reproducibility

All runs write to `runs/<timestamp>-<git-sha>/` containing the resolved config, cut log, per-iteration snapshots, final leaf values, and environment capture. Seeds are fixed in config; report across at least three.

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
mypy sensei/
```

Open a PR against `main` with a description of what changed and why.

### Guidelines

- New solver encodings need a correctness test on a synthetic fixture with a known answer
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

- Ahmad et al. (2025), *Sensitivity verification for additive decision tree ensembles* &mdash; ICLR
- Kantchelian et al. (2016), *Evasion and hardening of tree ensemble classifiers* &mdash; ICML

---

## License

See the [LICENSE](LICENSE) file for full statutory terms and legal disclosures..

Gurobi is licensed separately and is not distributed with this repository. Datasets are from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml) under their respective terms.

---

## Contact

**Authors:**
- Arihant Bedagkar
- Ashutosh Gupta

**Issues and bugs:** For bugs, solver integration errors, or feature requests, please open an issue [here](https://github.com/arihant-b/sensei/issues).

**Discussion:** For questions about the formulation, certificates, or extending the encodings, open a discussion [here](https://github.com/arihant-b/sensei/discussions).
