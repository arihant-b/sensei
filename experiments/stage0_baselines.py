"""
Stage 0 -- baselines.

Compares plain XGBoost, dropping the protected feature(s), XGBoost's
native monotone_constraints, a simple reweighing baseline, and SensEI
itself on accuracy + sensitivity_rate. Per README's baselines table,
none of the first four handle the label-flip constraint, are
data-aware, or produce a certificate -- this measures exactly that gap,
it does not try to close it for them.

    python -m experiments.stage0_baselines --all
    python -m experiments.stage0_baselines --dataset adult
"""

import argparse
import logging
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402

from config import Config  # noqa: E402
from data import Dataset  # noqa: E402
from experiments._common import (  # noqa: E402
    ALL_DATASETS,
    build_config,
    resolve_datasets,
)
from loop import SensEILoop  # noqa: E402
from metrics import accuracy, sensitivity_rate  # noqa: E402
from model import Ensemble  # noqa: E402
from oracle import SensitivityOracle  # noqa: E402
from runlog import RunWriter  # noqa: E402
from sampler import SamplingScreen  # noqa: E402
from spec import SPECS, Direction  # noqa: E402
from validity import ValidityChecker  # noqa: E402

log = logging.getLogger("experiments.stage0")


class _ClassifierAdapter:
    """model.predict(X, v=None)-compatible wrapper for a plain xgboost classifier."""

    def __init__(self, clf: xgb.XGBClassifier):
        self.clf = clf

    def predict(self, X, v=None) -> np.ndarray:
        return self.clf.predict(X)


def _fit_plain(cfg: Config, X, y) -> Ensemble:
    return Ensemble(cfg).fit(X, y)


def _fit_dropped(cfg: Config, X, y, protected: list) -> tuple[xgb.XGBClassifier, list]:
    keep = [c for c in X.columns if c not in protected]
    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.seed)
    clf.fit(X[keep], y)
    return clf, keep


def _fit_monotone(cfg: Config, X, y, monotone: dict) -> xgb.XGBClassifier:
    """XGBoost's OWN monotone_constraints -- covers monotone features,
    not protected ones."""
    direction = {Direction.UP: 1, Direction.DOWN: -1}
    constraints = tuple(
        direction[monotone[c]] if c in monotone else 0 for c in X.columns)
    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.seed,
        monotone_constraints=constraints)
    clf.fit(X, y)
    return clf


def _fit_reweighed(cfg: Config, X, y, protected: list) -> xgb.XGBClassifier:
    """
    Kamiran & Calders reweighing: instance weight = P(group)*P(label) /
    P(group, label), so each (protected-group, label) combination
    contributes equally regardless of its base rate in the data.
    """
    clf = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=cfg.seed)

    if not protected:
        clf.fit(X, y)
        return clf

    group = X[protected[0]].to_numpy()
    labels = np.asarray(y)
    weights = np.ones(len(y))

    for g in np.unique(group):
        for label in np.unique(labels):
            mask = (group == g) & (labels == label)
            if mask.sum() == 0:
                continue
            p_group = (group == g).mean()
            p_label = (labels == label).mean()
            p_joint = mask.mean()
            weights[mask] = (p_group * p_label) / p_joint

    clf.fit(X, y, sample_weight=weights)
    return clf


def _proxy_leakage(predictions: np.ndarray, protected_values: np.ndarray) -> float:
    """
    |mean prediction difference between protected groups| -- a simple,
    honest stand-in for 'does the sensitive signal still leak through
    correlated features' when the protected column itself isn't in the
    model, so sensitivity_rate's flip-based measurement doesn't apply.
    """
    groups = np.unique(protected_values)
    if len(groups) < 2:
        return float("nan")
    means = [predictions[protected_values == g].mean() for g in groups[:2]]
    return abs(float(means[0] - means[1]))


def run_dataset(
    dataset: str, seed: int,
    n_estimators: int | None = None, max_depth: int | None = None,
) -> dict:
    cfg = build_config(dataset, seed, n_estimators, max_depth)
    spec = SPECS[dataset]

    data = Dataset(cfg)
    data.load()
    assert data.X_train is not None and data.y_train is not None
    assert data.X_test is not None and data.y_test is not None
    bins = data.build_bins()
    validity = ValidityChecker(spec, bins, cfg, data.feature_bounds)
    sampler = SamplingScreen(spec, validity, cfg)

    results: dict = {"dataset": dataset, "seed": seed, "protected": spec.protected}

    plain = _fit_plain(cfg, data.X_train, data.y_train)
    plain_sens = (
        sensitivity_rate(plain, data.X_test, spec, sampler) if spec.protected else None)
    results["plain_xgboost"] = {
        "accuracy": accuracy(plain, data.X_test, data.y_test),
        "sensitivity_rate": plain_sens,
        "certified": False,
    }

    if spec.protected:
        dropped_clf, keep_cols = _fit_dropped(
            cfg, data.X_train, data.y_train, spec.protected)
        preds = dropped_clf.predict(data.X_test[keep_cols])
        results["drop_protected"] = {
            "accuracy": float((preds == data.y_test).mean()),
            "proxy_leakage": _proxy_leakage(
                preds.astype(float), data.X_test[spec.protected[0]].to_numpy()),
            "certified": False,
        }

    if spec.monotone:
        mono_clf = _fit_monotone(cfg, data.X_train, data.y_train, spec.monotone)
        mono_model = _ClassifierAdapter(mono_clf)
        results["monotone_constraints"] = {
            "accuracy": accuracy(mono_model, data.X_test, data.y_test),
            "sensitivity_rate": (
                sensitivity_rate(mono_model, data.X_test, spec, sampler)
                if spec.protected else None),
            "certified": False,
        }

    if spec.protected:
        rw_clf = _fit_reweighed(cfg, data.X_train, data.y_train, spec.protected)
        rw_model = _ClassifierAdapter(rw_clf)
        results["reweighing"] = {
            "accuracy": accuracy(rw_model, data.X_test, data.y_test),
            "sensitivity_rate": sensitivity_rate(rw_model, data.X_test, spec, sampler),
            "certified": False,
        }

    if spec.protected:
        model = Ensemble(cfg).fit(data.X_train, data.y_train)
        baseline_acc = accuracy(model, data.X_test, data.y_test)
        oracle = SensitivityOracle(validity, spec, cfg)
        loop = SensEILoop(model, data, oracle, sampler, cfg)

        best = loop.run(flip_set=[spec.protected[0]])
        results["sensei"] = {
            "accuracy": best.accuracy if best else baseline_acc,
            "sensitivity_rate": best.sensitivity_rate if best else None,
            "certified": loop.certified,
        }

    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Stage 0: baseline comparison")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", type=str, choices=ALL_DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=None,
                    help="override Config.n_estimators (smaller = faster)")
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = resolve_datasets(args)

    for dataset in datasets:
        log.info("=== stage0 baselines: %s ===", dataset)
        results = run_dataset(dataset, args.seed, args.n_estimators, args.max_depth)
        for name, metrics in results.items():
            if isinstance(metrics, dict):
                log.info("  %-20s %s", name, metrics)

        writer = RunWriter(tag=f"stage0-{dataset}")
        writer.write_summary(results)
        writer.write_environment()
        log.info("  -> %s", writer.dir)


if __name__ == "__main__":
    main()
