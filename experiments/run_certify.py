import argparse
import dataclasses
import logging

from sensei.config import Settings, load_defaults
from sensei.eval.results_writer import ResultsWriter
from sensei.logging_setup import setup_logging
from sensei.pipeline import CertifyPipeline

log: logging.Logger = logging.getLogger("sensei.experiments.run_certify")


def run(
    dataset: str,
    feature: str,
    n_estimators: int,
    max_depth: int,
    max_iters: int,
    oracle_time_limit_s: float,
    theta: float,
    oracle_type: str | None = None,
) -> dict:
    settings: Settings = load_defaults()
    oracle_type = oracle_type if oracle_type is not None else settings.oracle.type
    settings = dataclasses.replace(
        settings,
        model=dataclasses.replace(
            settings.model, n_estimators=n_estimators, max_depth=max_depth
        ),
        loop=dataclasses.replace(settings.loop, max_iters=max_iters),
        oracle=dataclasses.replace(
            settings.oracle, time_limit_s=oracle_time_limit_s, type=oracle_type
        ),
        sensitivity=dataclasses.replace(settings.sensitivity, theta=theta),
    )

    payload: dict = CertifyPipeline().run(dataset, feature, "protected", settings)
    ResultsWriter.write(payload, f"certify_{dataset}_{feature}")
    return payload


def main() -> None:
    setup_logging("run_certify")
    defaults: Settings = load_defaults()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--n-estimators", type=int, default=defaults.model.n_estimators)
    parser.add_argument("--max-depth", type=int, default=defaults.model.max_depth)
    parser.add_argument("--max-iters", type=int, default=defaults.loop.max_iters)
    parser.add_argument("--theta", type=float, default=defaults.sensitivity.theta)
    parser.add_argument(
        "--oracle-time-limit-s", type=float, default=defaults.oracle.time_limit_s
    )
    parser.add_argument(
        "--oracle-type",
        choices=["sensei", "ensense"],
        default=defaults.oracle.type,
        help="which oracle drives the CEGSAL loop's own search: 'sensei' (own "
        "MILP, preferred) or 'ensense' (Ensense core + postfilter, weaker)",
    )
    args: argparse.Namespace = parser.parse_args()

    run(
        args.dataset,
        args.feature,
        args.n_estimators,
        args.max_depth,
        args.max_iters,
        args.oracle_time_limit_s,
        args.theta,
        args.oracle_type,
    )


if __name__ == "__main__":
    main()
