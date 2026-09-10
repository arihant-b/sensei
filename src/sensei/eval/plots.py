import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

log: logging.Logger = logging.getLogger("sensei.eval.plots")

_PLOTS_ROOT: Path = Path(__file__).resolve().parents[3] / "results" / "plots"

SnapshotLike = Any


def _field(obj: SnapshotLike, name: str) -> Any:
    """
    Read `name` off a snapshot-like object, whether it's a dataclass
    instance (`Snapshot`) or a plain dict (e.g. loaded from JSON).

    Args:
        obj (SnapshotLike): A `Snapshot` instance or a dict.
        name (str): The field/key to read.

    Returns:
        Any: The field's value.
    """

    if isinstance(obj, dict):
        return obj.get(name)

    return getattr(obj, name)


def _sweep_field(point: Any, key_candidates: tuple[str, ...]) -> Any:
    """
    Read one of `key_candidates` off a sweep point, whether it's a
    `sensei.eval.sweeps.SweepPoint` dataclass (field name always `value`) or
    a dict from a results JSON (key named `theta` or `eps` instead of
    `value`, per `pipeline.py`'s own serialization).

    Args:
        point (Any): A `SweepPoint` instance or a dict.
        key_candidates (tuple[str, ...]): Field/key names to try, in order.

    Returns:
        Any: The first matching field's value.

    Raises:
        TypeError: `point` is a class, not an instance.
        KeyError: None of `key_candidates` are present on `point`.
    """

    if isinstance(point, type):
        raise TypeError(f"expected a sweep point instance, got the class {point!r}")

    fields: dict[str, Any] = asdict(cast(Any, point)) if is_dataclass(point) else point

    for key in key_candidates:
        if key in fields:
            return fields[key]

    raise KeyError(f"none of {key_candidates} found in sweep point {point!r}")


class ResultsPlotter:
    """
    Saves every plot produced by one instance into the SAME timestamped
    directory under `results/plots/`, so a full set of diagnostics for one
    run lands together. Create one `ResultsPlotter` per run/report you want
    grouped; create a fresh one (or pass `run_dir` explicitly) to keep
    plots from different runs apart.
    """

    def __init__(
        self,
        run_dir: Path | None = None,
        label: str | None = None,
        hyperparams: dict[str, Any] | None = None,
    ) -> None:
        """
        `run_dir` reuses an existing directory instead of stamping a new
        `<timestamp>_<label>` one. `hyperparams` (e.g. `{"eps": 0.1, "theta":
        1e-6, "oracle_type": "ensense"}`) is stamped as a text caption below
        every plot this instance saves, so a PNG pulled out of
        `results/plots/` on its own still says what config produced it.

        Args:
            run_dir (Path | None): Existing directory to reuse, or None to
                stamp a fresh `<timestamp>_<label>` one.
            label (str | None): Suffix for the auto-stamped directory name.
            hyperparams (dict[str, Any] | None): Config to caption every
                plot with.
        """

        if run_dir is not None:
            self.run_dir: Path = run_dir
        else:
            stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname: str = f"{stamp}_{label}" if label else stamp
            self.run_dir = _PLOTS_ROOT / dirname

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.hyperparams: dict[str, Any] | None = hyperparams

    @classmethod
    def from_results_json(
        cls, path: str | Path, label: str | None = None
    ) -> tuple["ResultsPlotter", dict]:
        """
        Load a `results/certify*.json` (or `baselines*.json`) payload and
        construct a `ResultsPlotter` labeled after it (`label` overrides the
        auto-derived one, the file's own stem).

        Args:
            path (str | Path): Path to the results JSON file.
            label (str | None): Overrides the auto-derived label (the
                file's own stem).

        Returns:
            tuple[ResultsPlotter, dict]: The constructed plotter and the
                loaded payload, to pass to `plot_available`.
        """

        path = Path(path)
        payload: dict = json.loads(path.read_text())
        hyperparams: dict[str, Any] = {
            key: payload[key]
            for key in ("eps", "theta", "oracle_type")
            if key in payload
        }

        return (
            cls(label=label or path.stem, hyperparams=hyperparams or None),
            payload,
        )

    def plot_available(self, payload: dict) -> list[Path]:
        """
        Plot whatever a results JSON payload actually contains -- a results
        JSON does not persist the full per-iteration history (only the
        final snapshot and the sweeps), so this only produces the sweep /
        baseline-comparison plots when their data is present, never the
        per-iteration or Pareto plots (use `training_dashboard`/
        `pareto_curve` directly with in-memory `Snapshot` history for those).

        Args:
            payload (dict): A loaded results JSON payload.

        Returns:
            list[Path]: Every plot actually produced, empty if none applied.
        """

        produced: list[Path] = []

        if payload.get("theta_sweep"):
            produced.append(self.theta_sweep(payload["theta_sweep"]))

        if payload.get("eps_sweep"):
            produced.append(self.eps_sweep(payload["eps_sweep"]))

        if payload.get("baselines"):
            produced.append(self.baseline_comparison(payload["baselines"]))

        if not produced:
            log.warning(
                "plot_available: payload had none of theta_sweep/eps_sweep/baselines "
                "-- nothing to plot from this file alone"
            )

        return produced

    def accuracy_vs_iteration(
        self, snapshots: list[SnapshotLike], name: str = "accuracy_vs_iteration"
    ) -> Path:
        """
        Line plot of accuracy against CEGSAL iteration.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, acc = self._iter_series(snapshots, "accuracy")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(iter, acc, marker="o", color="tab:blue")
        ax.set_xlabel("iteration")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs. iteration")
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def sensitivity_vs_iteration(
        self, snapshots: list[SnapshotLike], name: str = "sensitivity_vs_iteration"
    ) -> Path:
        """
        Line plot of the sensitivity rate against CEGSAL iteration.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, sens = self._iter_series(snapshots, "sensitivity_rate")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(iter, sens, marker="o", color="tab:red")
        ax.set_xlabel("iteration")
        ax.set_ylabel("sensitivity rate")
        ax.set_title("Sensitivity rate vs. iteration")
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def worst_gap_vs_iteration(
        self, snapshots: list[SnapshotLike], name: str = "worst_gap_vs_iteration"
    ) -> Path:
        """
        Line plot of worst valid gap vs. iteration, with a zero reference line.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, gap = self._iter_series(snapshots, "worst_gap")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(iter, gap, marker="o", color="tab:purple")
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("iteration")
        ax.set_ylabel("worst valid gap")
        ax.set_title("Worst valid gap vs. iteration")
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def slack_mass_vs_iteration(
        self, snapshots: list[SnapshotLike], name: str = "slack_mass_vs_iteration"
    ) -> Path:
        """
        Bar chart of slack mass (sum of cut slacks at repair termination)
        against CEGSAL iteration. Nonzero slack means leaf-only repair ran
        out of expressive power for some cuts at that iteration -- a
        finding to see, not something this plot should hide.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, slack = self._iter_series(snapshots, "slack_mass")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(iter, slack, color="tab:orange")
        ax.set_xlabel("iteration")
        ax.set_ylabel("slack mass (sum of s_i)")
        ax.set_title("Slack mass vs. iteration")
        ax.grid(True, alpha=0.3, axis="y")
        return self._save(fig, name)

    def cuts_vs_iteration(
        self, snapshots: list[SnapshotLike], name: str = "cuts_vs_iteration"
    ) -> Path:
        """
        Step plot of the accumulated cut count against CEGSAL iteration.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, n_cuts = self._iter_series(snapshots, "n_cuts")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.step(iter, n_cuts, where="post", color="tab:green", marker="o")
        ax.set_xlabel("iteration")
        ax.set_ylabel("accumulated cuts")
        ax.set_title("Cut count vs. iteration")
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def pareto_curve(
        self,
        snapshots: list[SnapshotLike],
        x: str = "sensitivity_rate",
        y: str = "accuracy",
        name: str = "pareto_curve",
    ) -> Path:
        """
        Scatter plot of `y` against `x` (default accuracy vs. sensitivity
        rate), with iteration numbers annotated and colored.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            x (str): Field to plot on the x-axis.
            y (str): Field to plot on the y-axis.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        xs: list[float] = [_field(s, x) for s in snapshots]
        ys: list[float] = [_field(s, y) for s in snapshots]
        iters: list[int] = [_field(s, "iteration") for s in snapshots]

        fig, ax = plt.subplots(figsize=(6.5, 6))
        sc: PathCollection = ax.scatter(xs, ys, c=iters, cmap="viridis", s=60, zorder=3)

        for xi, yi, iter in zip(xs, ys, iters, strict=True):
            ax.annotate(
                str(iter),
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.7,
            )

        cbar: Colorbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("iteration")
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel(y.replace("_", " "))
        ax.set_title(f"Pareto curve: {y.replace('_', ' ')} vs. {x.replace('_', ' ')}")
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def training_dashboard(
        self, snapshots: list[SnapshotLike], name: str = "training_dashboard"
    ) -> Path:
        """
        A 2x3 grid summarizing CEGSAL's training history: accuracy,
        sensitivity rate, worst gap, slack mass, cut count (all vs.
        iteration), plus the accuracy/sensitivity Pareto scatter.

        Args:
            snapshots (list[SnapshotLike]): Per-iteration training history.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        iter, acc = self._iter_series(snapshots, "accuracy")
        _, sens = self._iter_series(snapshots, "sensitivity_rate")
        _, gap = self._iter_series(snapshots, "worst_gap")
        _, slack = self._iter_series(snapshots, "slack_mass")
        _, n_cuts = self._iter_series(snapshots, "n_cuts")

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        axes[0, 0].plot(iter, acc, marker="o", color="tab:blue")
        axes[0, 0].set_title("accuracy")

        axes[0, 1].plot(iter, sens, marker="o", color="tab:red")
        axes[0, 1].set_title("sensitivity rate")

        axes[0, 2].plot(iter, gap, marker="o", color="tab:purple")
        axes[0, 2].axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        axes[0, 2].set_title("worst valid gap")

        axes[1, 0].bar(iter, slack, color="tab:orange")
        axes[1, 0].set_title("slack mass")

        axes[1, 1].step(iter, n_cuts, where="post", color="tab:green", marker="o")
        axes[1, 1].set_title("cumulative cuts")

        sc = axes[1, 2].scatter(sens, acc, c=iter, cmap="viridis", s=50)
        axes[1, 2].set_title("Pareto: accuracy vs. sensitivity")
        axes[1, 2].set_xlabel("sensitivity rate")
        axes[1, 2].set_ylabel("accuracy")
        fig.colorbar(sc, ax=axes[1, 2], label="iteration")

        for ax in axes.flat[:5]:
            ax.set_xlabel("iteration")
            ax.grid(True, alpha=0.3)

        fig.suptitle("CEGSAL training dashboard", fontsize=14)
        fig.tight_layout()
        return self._save(fig, name)

    def theta_sweep(self, points: list[Any], name: str = "theta_sweep") -> Path:
        """
        Worst gap against theta, certified (UNSAT) points marked distinctly
        from violated ones.

        Args:
            points (list[Any]): `SweepPoint`s or results-JSON dicts.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        return self._sweep_plot(
            points, key=("theta", "value"), xlabel="theta", name=name
        )

    def eps_sweep(self, points: list[Any], name: str = "eps_sweep") -> Path:
        """
        Same as `theta_sweep`, swept over eps instead.

        Args:
            points (list[Any]): `SweepPoint`s or results-JSON dicts.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        return self._sweep_plot(points, key=("eps", "value"), xlabel="eps", name=name)

    def _sweep_plot(
        self, points: list[Any], key: tuple[str, ...], xlabel: str, name: str
    ) -> Path:
        values: list[float] = [_sweep_field(p, key) for p in points]
        certified: list[bool] = [_sweep_field(p, ("certified",)) for p in points]
        raw_gaps: list[float | None] = [_sweep_field(p, ("gap",)) for p in points]
        plot_gaps: list[float] = [0.0 if g is None else g for g in raw_gaps]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(values, plot_gaps, color="gray", linewidth=1.0, zorder=1)

        cert_x: list[float] = [v for v, c in zip(values, certified, strict=True) if c]
        cert_y: list[float] = [
            g for g, c in zip(plot_gaps, certified, strict=True) if c
        ]
        sat_x: list[float] = [
            v for v, c in zip(values, certified, strict=True) if not c
        ]
        sat_y: list[float] = [
            g for g, c in zip(plot_gaps, certified, strict=True) if not c
        ]

        ax.scatter(
            cert_x,
            cert_y,
            marker="o",
            s=70,
            color="tab:green",
            label="certified (UNSAT) -- no violation",
            zorder=3,
        )
        ax.scatter(
            sat_x,
            sat_y,
            marker="x",
            s=70,
            color="tab:red",
            label="violation found",
            zorder=3,
        )

        if xlabel == "theta":
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("worst gap (0 = certified, no violation)")
        ax.set_title(f"{xlabel} sweep")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return self._save(fig, name)

    def oracle_status_counts(
        self, counts: dict[str, int], name: str = "oracle_status_counts"
    ) -> Path:
        """
        Bar chart of `counts` (status label -> tally, e.g. `{"UNSAT": 5,
        "TIMEOUT": 2}`), colored per `_status_color`.

        Args:
            counts (dict[str, int]): Status label -> tally.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        labels: list[str] = list(counts.keys())
        values: list[int] = list(counts.values())
        colors: list[str] = [self._status_color(label) for label in labels]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("count")
        ax.set_title("Oracle status counts")
        ax.grid(True, alpha=0.3, axis="y")
        fig.autofmt_xdate(rotation=20)
        return self._save(fig, name)

    def baseline_comparison(
        self, baselines: dict[str, dict[str, Any]], name: str = "baseline_comparison"
    ) -> Path:
        """
        Grouped bar chart of accuracy and sensitivity rate across `baselines`.

        Args:
            baselines (dict[str, dict[str, Any]]): Baseline name -> its
                `{"accuracy": ..., "sensitivity_rate": ...}` result.
            name (str): Output filename stem (without extension).

        Returns:
            Path: Where the plot was saved.
        """

        names: list[str] = list(baselines.keys())
        acc: list[float] = [baselines[n]["accuracy"] for n in names]
        sens: list[float] = [baselines[n]["sensitivity_rate"] for n in names]

        x = np.arange(len(names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 5))
        ax.bar(x - width / 2, acc, width, label="accuracy", color="tab:blue")
        ax.bar(x + width / 2, sens, width, label="sensitivity rate", color="tab:red")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylabel("value")
        ax.set_title("Baseline comparison")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return self._save(fig, name)

    def _iter_series(
        self, snapshots: list[SnapshotLike], field: str
    ) -> tuple[list[int], list[float]]:
        iter: list[int] = [_field(s, "iteration") for s in snapshots]
        values: list[float] = [_field(s, field) for s in snapshots]
        return iter, values

    def _hyperparams_text(self) -> str | None:
        if not self.hyperparams:
            return None
        return "  |  ".join(f"{k}={v}" for k, v in self.hyperparams.items())

    def _save(self, fig: Figure, name: str) -> Path:
        text: str | None = self._hyperparams_text()
        if text is not None:
            fig.text(
                0.5,
                -0.02,
                text,
                ha="center",
                va="top",
                fontsize=8,
                color="dimgray",
                transform=fig.transFigure,
            )

        path: Path = self.run_dir / f"{name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", path)
        return path

    def _status_color(self, label: str) -> str:
        return {
            "UNSAT": "tab:green",
            "TIMEOUT": "tab:orange",
            "ORACLE_SATURATED": "tab:red",
            "EMPTY_DOMAIN": "tab:gray",
            "INCONSISTENT_OPTIMALITY": "tab:pink",
        }.get(label, "tab:blue")
