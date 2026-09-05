import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from sensei.config import Settings, load_defaults
from sensei.pipeline import CertifyPipeline
from sensei.spec import Spec, load_spec

log: logging.Logger = logging.getLogger("sensei.experiments.run_all_features")

_RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"


def run(
    dataset: str,
    n_estimators: int,
    max_depth: int,
    max_iters: int,
    oracle_time_limit_s: float,
) -> list[dict]:
    settings: Settings = load_defaults()
    settings = dataclasses.replace(
        settings,
        model=dataclasses.replace(
            settings.model, n_estimators=n_estimators, max_depth=max_depth
        ),
        loop=dataclasses.replace(settings.loop, max_iters=max_iters),
        oracle=dataclasses.replace(settings.oracle, time_limit_s=oracle_time_limit_s),
    )
    spec: Spec = load_spec(dataset)

    schedule: list[tuple[str, str]] = [(f, "protected") for f in spec.protected] + [
        (f, "monotone_wrong") for f in spec.monotone
    ]

    if not schedule:
        log.warning(
            "%s: spec declares no protected or monotone features -- nothing to run",
            dataset,
        )
        return []

    payloads: list[dict] = []

    for feature, direction in schedule:
        log.info("=== %s: %s (%s) ===", dataset, feature, direction)
        payload = CertifyPipeline().run(dataset, feature, direction, settings)
        _write(payload, dataset, feature)
        payloads.append(payload)

    return payloads


def _write(payload: dict, dataset: str, feature: str) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path: Path = (
        _RESULTS_DIR / f"stage3_certify_{dataset}_{feature}_{int(time.time())}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    log.info("wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-estimators", type=int, default=30)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--oracle-time-limit-s", type=float, default=30.0)
    args: argparse.Namespace = parser.parse_args()

    run(
        args.dataset,
        args.n_estimators,
        args.max_depth,
        args.max_iters,
        args.oracle_time_limit_s,
    )


if __name__ == "__main__":
    main()
