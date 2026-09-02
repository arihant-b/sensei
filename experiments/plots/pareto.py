"""
Pareto front: accuracy vs worst sensitivity gap across runs. The
headline result -- a single number hides the trade-off (README).

Reads every runs/*/summary.json under --input and plots one point per
(dataset, seed, feature) repair result it can find (stage2_repair's
format; other stages' summaries are skipped since they don't carry a
worst_gap). Needs the optional 'plots' extra: pip install -e ".[plots]"

    python -m experiments.plots.pareto --input runs/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

log = logging.getLogger("experiments.plots.pareto")


def collect_points(input_dir: Path) -> list[dict]:
    """Walk runs/*/summary.json, pulling (dataset, seed, feature, accuracy, gap)."""
    points = []

    for summary_path in sorted(input_dir.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        per_feature = summary.get("per_feature")
        if not isinstance(per_feature, dict):
            continue

        for feature, entry in per_feature.items():
            if not isinstance(entry, dict) or entry.get("worst_gap") is None:
                continue
            points.append({
                "run": summary_path.parent.name,
                "dataset": summary.get("dataset"),
                "seed": summary.get("seed"),
                "feature": feature,
                "accuracy": entry.get("accuracy"),
                "worst_gap": entry.get("worst_gap"),
                "certified": entry.get("certified", False),
            })

    return points


def plot(points: list[dict], output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required for plots. Install with: "
            'pip install -e ".[plots]"'
        ) from e

    fig, ax = plt.subplots(figsize=(6, 4.5))
    certified = [p for p in points if p["certified"]]
    uncertified = [p for p in points if not p["certified"]]

    for group, label, marker in [(uncertified, "not certified", "x"),
                                   (certified, "certified", "o")]:
        if not group:
            continue
        ax.scatter([p["accuracy"] for p in group], [p["worst_gap"] for p in group],
                   marker=marker, label=label)

    ax.set_xlabel("accuracy")
    ax.set_ylabel("worst sensitivity gap")
    ax.set_title("Accuracy vs sensitivity trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    log.info("wrote %s", output)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Plot the accuracy/sensitivity Pareto front")
    p.add_argument("--input", type=str, default="runs/",
                    help="directory of run subdirectories (each with summary.json)")
    p.add_argument("--output", type=str, default=None,
                    help="output PNG path (default: <input>/pareto.png)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    input_dir = Path(args.input)
    output = Path(args.output) if args.output else input_dir / "pareto.png"

    points = collect_points(input_dir)
    log.info("found %d repair result(s) with a gap under %s", len(points), input_dir)

    if not points:
        log.warning("nothing to plot -- run experiments.stage2_repair first")
        return

    plot(points, output)


if __name__ == "__main__":
    main()
