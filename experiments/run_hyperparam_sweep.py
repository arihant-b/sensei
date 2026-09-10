"""
One-at-a-time (OFAT) hyperparameter sweep, restricted to the adult dataset's
"sex" feature (direction="protected") -- this script is deliberately scoped
to that single dataset/feature pair, not a general-purpose sweep runner.

Varies n_estimators, max_depth, theta, eps, gap, and mu, each across its own
small default grid, ONE PARAMETER AT A TIME (every other setting held at a
shared baseline) -- under BOTH oracle_type="sensei" and oracle_type="ensense"
for every grid point. This is NOT a full cross-product grid search: 6 grids
of ~3 values each combined exhaustively would be 3**6 * 2 = 1458 runs, and
every run here is a full `CertifyPipeline.run()` call (train a fresh M0, run
the CEGSAL loop, held-out verify against Ensense core, a sensei-oracle
self-check, AND its own theta/eps sweep -- see sensei/pipeline.py).
One-at-a-time keeps the run count linear in the number of grid points (6
params * 3 values * 2 oracle types = 36 runs by default) instead of
exponential, and matches how sensei.eval.sweeps.Sweeper already reports
theta/eps sensitivity: hold everything else fixed, vary one thing, read off
the effect.

Every (parameter, value, oracle_type) point writes its own results JSON via
CertifyPipeline.run() -- same as any other sensei run -- plus one combined
summary JSON (results/hyperparam_sweep_adult_sex_summary_<ts>.json) indexing
every point this script produced, including which ones errored.

oracle_type="ensense" never needs direction="monotone_wrong" handling here:
"sex" is a protected feature, and direction is fixed to "protected"
throughout.

Usage:
    python -m experiments.run_hyperparam_sweep
    python -m experiments.run_hyperparam_sweep --oracle-types sensei
    python -m experiments.run_hyperparam_sweep --n-estimators-grid 10 20 \
        --theta-grid 1e-6
"""

import argparse
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sensei.config import Settings, load_defaults
from sensei.eval.results_writer import ResultsWriter
from sensei.logging_setup import setup_logging
from sensei.pipeline import CertifyPipeline

log: logging.Logger = logging.getLogger("sensei.experiments.run_hyperparam_sweep")

_DATASET = "adult"
_FEATURE = "sex"
_DIRECTION = "protected"

_ORACLE_TYPES: tuple[str, ...] = ("sensei", "ensense")

# Deliberately small -- each point is a full CertifyPipeline run (see module
# docstring). Override any one grid via its matching --*-grid CLI flag.
_DEFAULT_GRIDS: dict[str, list[Any]] = {
    "n_estimators": [10, 30, 50, 80, 100, 150, 200],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "theta": [1e-9, 1e-6, 1e-3, 0.01, 0.1],
    "eps": [0.05, 0.1, 0.2],
    "gap": [0.3, 0.5, 0.7],
    "mu": [0.001, 0.01, 0.1],
}

# One updater per sweepable parameter -- dispatch table instead of an
# if/elif chain, and the single place that knows which Settings sub-config
# each parameter lives under.
_APPLIERS: dict[str, Callable[[Settings, Any], Settings]] = {
    "n_estimators": lambda s, v: dataclasses.replace(
        s, model=dataclasses.replace(s.model, n_estimators=v)
    ),
    "max_depth": lambda s, v: dataclasses.replace(
        s, model=dataclasses.replace(s.model, max_depth=v)
    ),
    "theta": lambda s, v: dataclasses.replace(
        s, sensitivity=dataclasses.replace(s.sensitivity, theta=v)
    ),
    "eps": lambda s, v: dataclasses.replace(
        s, sensitivity=dataclasses.replace(s.sensitivity, eps=v)
    ),
    "gap": lambda s, v: dataclasses.replace(
        s, sensitivity=dataclasses.replace(s.sensitivity, gap=v)
    ),
    "mu": lambda s, v: dataclasses.replace(
        s, repair=dataclasses.replace(s.repair, mu=v)
    ),
}


def _baseline_settings(
    n_estimators: int, max_depth: int, max_iters: int, oracle_time_limit_s: float
) -> Settings:
    """
    The shared starting point every OFAT sweep point is built from. Small
    n_estimators/max_depth/max_iters/oracle_time_limit_s by default --
    matching experiments/run_certify.py's own CLI defaults, NOT
    config/defaults.yaml's reported-run values -- so a 36-run sweep finishes
    in a reasonable time. Every field this function doesn't touch (theta,
    eps, gap, mu, oracle.type) stays at config/defaults.yaml's own default
    until `_apply` overrides it for one sweep point.
    """

    settings: Settings = load_defaults()
    return dataclasses.replace(
        settings,
        model=dataclasses.replace(
            settings.model, n_estimators=n_estimators, max_depth=max_depth
        ),
        loop=dataclasses.replace(settings.loop, max_iters=max_iters),
        oracle=dataclasses.replace(settings.oracle, time_limit_s=oracle_time_limit_s),
    )


def _apply(settings: Settings, param: str, value: Any, oracle_type: str) -> Settings:
    """Apply ONE swept (param, value) pair plus oracle_type onto `settings`."""

    if param not in _APPLIERS:
        raise ValueError(
            f"unknown sweep parameter {param!r} -- one of {sorted(_APPLIERS)}"
        )

    settings = _APPLIERS[param](settings, value)
    return dataclasses.replace(
        settings, oracle=dataclasses.replace(settings.oracle, type=oracle_type)
    )


def run(
    grids: dict[str, list[Any]] | None = None,
    oracle_types: tuple[str, ...] = _ORACLE_TYPES,
    n_estimators: int = 30,
    max_depth: int = 4,
    max_iters: int = 15,
    oracle_time_limit_s: float = 30.0,
) -> dict[str, Any]:
    """
    Run the OFAT sweep over `grids` (default `_DEFAULT_GRIDS`) x
    `oracle_types` for adult/sex/protected. Writes one results JSON per
    point plus a combined summary JSON; returns the summary dict.

    A single point raising (Gurobi/license issues, an unexpected exception
    -- NOT the oracle's own TIMEOUT/ORACLE_SATURATED/EMPTY_DOMAIN statuses,
    which CegsalLoop already turns into an Inconclusive result rather than
    raising) is logged and recorded as an "ERROR" point rather than aborting
    the rest of the sweep, matching how a single failed diagnostic plot
    doesn't take down an otherwise-successful CertifyPipeline run elsewhere
    in this codebase.
    """

    grids = grids if grids is not None else _DEFAULT_GRIDS
    baseline: Settings = _baseline_settings(
        n_estimators, max_depth, max_iters, oracle_time_limit_s
    )

    runs: list[dict[str, Any]] = []

    for param, values in grids.items():
        for value in values:
            for oracle_type in oracle_types:
                log.info(
                    "=== sweep: %s=%s, oracle_type=%s ===", param, value, oracle_type
                )
                settings: Settings = _apply(baseline, param, value, oracle_type)

                out_path: Path | None = None
                status = "ERROR"
                error: str | None = None
                try:
                    payload: dict = CertifyPipeline().run(
                        _DATASET, _FEATURE, _DIRECTION, settings
                    )
                    out_path = _write(payload, param, value, oracle_type)
                    status = payload.get("cegsal_exit", "unknown")
                except Exception:
                    log.exception(
                        "sweep point %s=%s, oracle_type=%s failed -- continuing "
                        "with the rest of the sweep",
                        param,
                        value,
                        oracle_type,
                    )
                    error = "see log for the full traceback"

                runs.append(
                    {
                        "param": param,
                        "value": value,
                        "oracle_type": oracle_type,
                        "cegsal_exit": status,
                        "results_path": str(out_path) if out_path else None,
                        "error": error,
                    }
                )

    summary: dict[str, Any] = {
        "dataset": _DATASET,
        "feature": _FEATURE,
        "direction": _DIRECTION,
        "grids": grids,
        "oracle_types": list(oracle_types),
        "baseline": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "max_iters": max_iters,
            "oracle_time_limit_s": oracle_time_limit_s,
        },
        "runs": runs,
    }
    _write_summary(summary)
    return summary


def _safe_value(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def _write(payload: dict, param: str, value: Any, oracle_type: str) -> Path:
    name: str = (
        f"hyperparam_sweep_{_DATASET}_{_FEATURE}_{param}_{_safe_value(value)}_"
        f"{oracle_type}"
    )
    return ResultsWriter.write(payload, name)


def _write_summary(summary: dict) -> Path:
    name: str = f"hyperparam_sweep_{_DATASET}_{_FEATURE}_summary"
    return ResultsWriter.write(summary, name)


def main() -> None:
    setup_logging("run_hyperparam_sweep")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-estimators-grid", type=int, nargs="+", default=None, metavar="N"
    )
    parser.add_argument(
        "--max-depth-grid", type=int, nargs="+", default=None, metavar="D"
    )
    parser.add_argument(
        "--theta-grid", type=float, nargs="+", default=None, metavar="THETA"
    )
    parser.add_argument(
        "--eps-grid", type=float, nargs="+", default=None, metavar="EPS"
    )
    parser.add_argument(
        "--gap-grid", type=float, nargs="+", default=None, metavar="GAP"
    )
    parser.add_argument("--mu-grid", type=float, nargs="+", default=None, metavar="MU")
    parser.add_argument(
        "--oracle-types",
        choices=["sensei", "ensense"],
        nargs="+",
        default=list(_ORACLE_TYPES),
        help="which oracle type(s) to sweep every grid point under. "
        "Defaults to both sensei and ensense.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=30,
        help="baseline value, held fixed while a DIFFERENT parameter is swept",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="baseline value, held fixed while a DIFFERENT parameter is swept",
    )
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--oracle-time-limit-s", type=float, default=30.0)
    args: argparse.Namespace = parser.parse_args()

    grids: dict[str, list[Any]] = dict(_DEFAULT_GRIDS)
    if args.n_estimators_grid is not None:
        grids["n_estimators"] = args.n_estimators_grid
    if args.max_depth_grid is not None:
        grids["max_depth"] = args.max_depth_grid
    if args.theta_grid is not None:
        grids["theta"] = args.theta_grid
    if args.eps_grid is not None:
        grids["eps"] = args.eps_grid
    if args.gap_grid is not None:
        grids["gap"] = args.gap_grid
    if args.mu_grid is not None:
        grids["mu"] = args.mu_grid

    run(
        grids=grids,
        oracle_types=tuple(args.oracle_types),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
        oracle_time_limit_s=args.oracle_time_limit_s,
    )


if __name__ == "__main__":
    main()
