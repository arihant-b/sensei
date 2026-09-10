import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

from sensei.config import Settings
from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.types import (
    NoGood,
    OracleDegenerate,
    OracleTimeout,
    Pair,
    TreeStructure,
)
from sensei.spec import Spec
from sensei.validity.postfilter import MAX_REJECTIONS_PER_ITER, Postfilter

log: logging.Logger = logging.getLogger("sensei.oracle.ensense.adapter")

_ENSENSE_SRC: Path = Path(__file__).resolve().parents[4] / "ensense" / "src"
_ENSENSE_CLI: Path = _ENSENSE_SRC / "sensitive.py"
_DATASET_ROOT: Path = Path(__file__).resolve().parents[4] / "dataset"

_ANSI_ESCAPE: re.Pattern[str] = re.compile(r"\x1b\[[0-9;]*m")


def _details_csv_for(dataset: str) -> str | None:
    """
    Path to `dataset/<dataset>/details.csv`, if it has been built, else None.
    Ensense core uses this file to derive each feature's search range
    (`ensense/src/ensemble.py`'s `Ensemble.load`); without it, it falls back
    to `(min(tree_split_thresholds) - 1, max(tree_split_thresholds) + 1)` per
    feature -- a range with no relation to the feature's real declared
    domain, which can hand back "counterexamples" with feature values
    outside `[0, 1]` (e.g. a negative categorical code). Resolved here, once,
    so every `EnsenseOracle` call site gets the real bounds automatically
    instead of each caller having to remember to look this file up and pass
    it through.

    Args:
        dataset (str): Dataset name, matching `dataset/<dataset>/`.

    Returns:
        str | None: The resolved path, or None if the file doesn't exist.
    """

    path: Path = _DATASET_ROOT / dataset / "details.csv"
    return str(path) if path.exists() else None


def _parse_bracketed_floats(line: str) -> list[float]:
    """
    `"Sensitive sample 1:[ 0.5 , 1 , 0.25 ]"` -> `[0.5, 1.0, 0.25]`.

    Args:
        line (str): One line containing a single `[ ... ]` bracketed,
            comma-separated list of numbers.

    Returns:
        list[float]: The parsed values, in order.
    """

    inner: str = line.split("[", 1)[1].rsplit("]", 1)[0]
    return [float(v.strip()) for v in inner.split(",") if v.strip()]


def _parse_ensense_stdout(
    stdout: str,
) -> tuple[list[float], list[float]] | None:
    """
    Parse Ensense CLI's own `Sensitive sample 1:[ ... ]` / `Sensitive sample
    2:[ ... ]` lines into `(x1, x2)`. Both solver families print this same
    format (`ensense/src/utils.py::print_array`), just via different wrapper
    calls -- "milp" prints it bare, "pb" prints it through
    `print_array_verbose`, which prepends `"# "` -- and "milp" additionally
    wraps values that differ between x1/x2 in ANSI color codes, stripped
    here. Neither line appears at all when Ensense found no violation
    (its own "Insensitive" message differs in wording and prefix between
    the two solvers, so absence of the "Sensitive sample" lines -- not
    presence of some specific "insensitive" string -- is the UNSAT signal).

    Args:
        stdout (str): Captured stdout from the Ensense CLI subprocess.

    Returns:
        tuple[list[float], list[float]] | None: `(x1, x2)` if both lines
            were found, else None (no violation).
    """

    x1: list[float] | None = None
    x2: list[float] | None = None

    for raw_line in _ANSI_ESCAPE.sub("", stdout).splitlines():
        line: str = raw_line.strip().lstrip("#").strip()

        if line.startswith("Sensitive sample 1:"):
            x1 = _parse_bracketed_floats(line)
        elif line.startswith("Sensitive sample 2:"):
            x2 = _parse_bracketed_floats(line)

    if x1 is None or x2 is None:
        return None
    return x1, x2


class EnsenseOracle:
    """
    A wrapper around the Ensense core that provides a simple interface for calling the
    different solver families.
    """

    def worst_valid_pair(
        self,
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        flip_set: tuple[str, ...],
        spec: Spec,
        bins: FrozenBins,
        feature_bounds: dict[str, tuple[float, float]],
        settings: Settings,
        max_rejections: int = MAX_REJECTIONS_PER_ITER,
    ) -> Pair | None:
        """
        Call Ensense core's `settings.oracle.method` solver ("pb" or "milp")
        to find a counterexample, then run every candidate it returns
        through `Postfilter.find_valid_pair` (Q1/Q2 checks Ensense itself
        knows nothing about) until one passes or `max_rejections` are
        exhausted. `settings.sensitivity.gap` becomes Ensense's OWN search
        criterion (a confident-flip band `(gap, 1-gap)` in probability
        space) -- a different quantity from our `eps`, which this method
        never sees at all (Ensense's own search has no `eps` concept; only
        `worst_valid_pair_loop` below gates on `eps`).

        `spec.dataset`'s own `details.csv` (via `_details_csv_for`) is always
        used when it exists, never left for the caller to look up and pass
        in -- without it, Ensense derives each feature's search range from
        its own tree splits instead of our declared bounds, which can hand
        back a "counterexample" with a feature value outside its real domain
        (see `_details_csv_for`'s docstring).

        Args:
            booster (xgb.Booster): The (possibly repaired) model to search.
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            spec (Spec): Supplies `spec.dataset` and every Q1/Q2 rule the
                postfilter checks.
            bins (FrozenBins): Frozen plausibility bins for the Q2 check.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature, for the postfilter.
            settings (Settings): Supplies `oracle.method`, `sensitivity.gap`/
                `.theta`, `oracle.time_limit_s`.
            max_rejections (int): Consecutive invalid pairs allowed before
                giving up.

        Returns:
            Pair | None: A postfilter-validated pair, or None if Ensense's
                own search found nothing.

        Raises:
            ValueError: `settings.oracle.method` isn't "pb" or "milp"
                (already validated by `OracleConfig.__post_init__`, so this
                should be unreachable in practice).
            OracleSaturated: `max_rejections` consecutive pairs all failed
                the postfilter.
        """

        method: str = settings.oracle.method

        if method not in ("pb", "milp"):
            raise ValueError(
                f"unknown Ensense solver method {method!r} -- 'pb' and 'milp' are "
                f"wired up here; Ensense core also exposes naive_smt, rounding, "
                f"roundingsoplex, veritas, monitor (ensense/src/options.py's "
                f"--solver choices), not yet adapted"
            )

        details_csv: str | None = _details_csv_for(spec.dataset)
        output_gap: tuple[float, float] = (
            settings.sensitivity.gap,
            1.0 - settings.sensitivity.gap,
        )
        timeout: int = int(settings.oracle.time_limit_s)

        def _pair_source() -> Pair | None:
            return self._solve_via(
                method,
                booster,
                leaf_map,
                columns,
                flip_set,
                details_csv=details_csv,
                output_gap=output_gap,
                precision=1000.0,
                timeout=timeout,
            )

        return Postfilter.find_valid_pair(
            _pair_source,
            columns,
            spec,
            bins,
            settings.sensitivity.theta,
            feature_bounds,
            max_rejections=max_rejections,
        )

    def worst_valid_pair_loop(
        self,
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec | None,
        flip_set: tuple[str, ...],
        direction: str,
        mode: str,
        settings: Settings,
        nogoods: list[NoGood] | None = None,
        warm_start: Pair | None = None,
        enforce_validity: bool = True,
        structure: TreeStructure | None = None,
        v: NDArray[np.float64] | None = None,
        bins: FrozenBins | None = None,
    ) -> Pair | None:
        """
        `CegsalLoop`'s own search entry point into `EnsenseOracle`, sharing
        `worst_valid_pair`'s search+postfilter above rather than duplicating
        it. Takes the same kwarg shape `SenseiOracle.worst_valid_pair` does
        (so `CegsalLoop._run_stage` can call either oracle uniformly) and
        adapts it onto `self.worst_valid_pair`'s real Ensense contract, being
        honest about what Ensense can't do rather than papering over it:

        - no no-goods -- `nogoods` is accepted
          and ignored, logged once. Within one `cuts_per_round` round the
          search is unchanged, so repeat calls will typically return the same
          pair -- harmless (duplicate cuts are algebraically redundant, not
          wrong) but not diversified the way `SenseiOracle`'s no-goods
          diversify a round.
        - no warm starts, no MILP `structure`/`settings.oracle.mip_gap` --
          accepted, ignored.
        - no true feasibility/optimality distinction -- Ensense's own
          search always runs the same way; `mode="feasibility"` only changes
          whether THIS METHOD gates the returned pair on
          `gap > settings.sensitivity.eps` before handing it back.
        - no `direction="monotone_wrong"` pinning -- raises rather than
          silently mislabeling a protected-direction pair as monotone.

        `settings.seed`, `structure`, and `settings.oracle.mip_gap` are
        accepted (via `settings`) for signature compatibility and unused --
        Ensense core's search is not seeded or warm-started by this method.

        Args:
            booster (xgb.Booster): The unrepaired model (structure only;
                leaf values come from `v` if given).
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec | None): Dataset spec; must not be None (Ensense's
                postfilter always needs it).
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): Must be `"protected"`.
            mode (str): `"feasibility"` or `"optimality"`.
            settings (Settings): Supplies `sensitivity.eps`/`.gap`/`.theta`,
                `oracle.method`/`.time_limit_s`.
            nogoods (list[NoGood] | None): Accepted and ignored (logged once).
            warm_start (Pair | None): Accepted and ignored (logged once).
            enforce_validity (bool): Must be True.
            structure (TreeStructure | None): Accepted and ignored.
            v (NDArray[np.float64] | None): Leaf values to search with; if
                given, writes them into a fresh booster before searching.
            bins (FrozenBins | None): Frozen plausibility bins; must not be None.

        Returns:
            Pair | None: The found pair (re-oriented so `gap >= 0`, and
                gated on `eps` in feasibility mode), or None.

        Raises:
            ValueError: `direction` isn't `"protected"`.
        """

        assert mode in ("feasibility", "optimality")
        eps: float = settings.sensitivity.eps

        if direction != "protected":
            raise ValueError(
                f"oracle_type='ensense' does not support direction={direction!r} -- "
                f"Ensense core's search has no way to pin which side of the pair "
                f"is the lower/higher monotone side, and always returns "
                f"direction='protected' pairs (_pair_from_points below). Use "
                f"oracle_type='sensei' for direction='monotone_wrong'."
            )

        assert enforce_validity, (
            "Ensense always enforces validity via its own postfilter -- "
            "enforce_validity=False has no Ensense equivalent"
        )
        assert spec is not None and bins is not None, (
            "Ensense's postfilter requires spec/bins"
        )

        if nogoods:
            log.warning(
                "oracle_type='ensense': %d no-good(s) accumulated by the loop are "
                "being ignored -- Ensense has no mechanism to exclude a "
                "previously-found joint leaf pattern. Repeated calls within one "
                "cuts_per_round round may return the same pair; harmless "
                "(duplicate cuts), just not diversified.",
                len(nogoods),
            )

        if warm_start is not None:
            log.warning(
                "oracle_type='ensense': warm_start is ignored -- Ensense has no MIP "
                "start equivalent."
            )

        search_booster: xgb.Booster = booster
        search_leaf_map: LeafMap = leaf_map

        if v is not None:
            search_booster = leaf_map.write_leaf_values(booster, v)
            search_leaf_map = LeafMap(search_booster)

        pair: Pair | None = self.worst_valid_pair(
            search_booster,
            search_leaf_map,
            columns,
            flip_set,
            spec,
            bins,
            feature_bounds,
            settings,
        )

        if pair is None:
            return None

        if pair.gap < 0:
            pair = Pair(
                x1=pair.x2,
                x2=pair.x1,
                ell1=pair.ell2,
                ell2=pair.ell1,
                gap=-pair.gap,
                flip_set=pair.flip_set,
                direction=pair.direction,
                solver_status=pair.solver_status,
            )

        if mode == "feasibility" and pair.gap <= eps + 1e-9:
            # Ensense's own search criterion (output_gap, probability space) is
            # not the same threshold as our eps (margin space) -- a pair it
            # considers worth returning may still not clear OUR eps. Treat that
            # as "no violation this round" rather than accepting a weaker bar
            # than eps.
            log.info(
                "oracle_type='ensense': counterexample discarded -- gap=%.6f does not "
                "clear eps=%.6f",
                pair.gap,
                eps,
            )
            return None

        return pair

    def _solve_via(
        self,
        solver: str,
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        flip_set: tuple[str, ...],
        *,
        details_csv: str | None,
        output_gap: tuple[float, float],
        precision: float,
        timeout: int,
    ) -> Pair | None:
        """
        Run Ensense's `solver` ("pb"/"milp") CLI, build a `Pair` from its output.

        Args:
            solver (str): `"pb"` or `"milp"`.
            booster (xgb.Booster): Model to dump to a temp JSON file for Ensense.
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            details_csv (str | None): Path to `details.csv`, or None.
            output_gap (tuple[float, float]): Ensense's own confident-flip band.
            precision (float): Ensense's own search precision.
            timeout (int): Hard wall-clock limit, in seconds.

        Returns:
            Pair | None: The found pair, or None if Ensense found nothing.
        """

        feature_indices: list[int] = list(map(columns.index, flip_set))

        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "model.json")
            booster.save_model(model_path)

            found: tuple[list[float], list[float]] | None = (
                self._run_search_with_hard_timeout(
                    solver,
                    model_path,
                    feature_indices,
                    details_csv,
                    precision,
                    output_gap,
                    timeout,
                )
            )

            if found is None:
                return None

            x1: NDArray[np.float64] = np.asarray(found[0], dtype=float)
            x2: NDArray[np.float64] = np.asarray(found[1], dtype=float)

        return self._pair_from_points(booster, leaf_map, columns, flip_set, x1, x2)

    def _run_search_with_hard_timeout(
        self,
        solver: str,
        model_path: str,
        feature_indices: list[int],
        details_path: str | None,
        precision: float,
        output_gap: tuple[float, float],
        timeout: int,
    ) -> tuple[list[float], list[float]] | None:
        """
        Run Ensense core's own CLI (`ensense/src/sensitive.py`) as a
        subprocess -- the same command a user would type at the terminal,
        not an in-process import of its internals -- and enforce `timeout`
        ourselves via `subprocess.run`'s own `timeout=` (which kills the
        process for us on expiry): Ensense core's own `--timeout` is not
        honored by the vendored solver (pb's alarm is disabled, milp's
        Gurobi TimeLimit is hardcoded to 3600s).

        Ensense prints its result to stdout rather than a result file or
        return value; `capture_output=True` keeps that off OUR console
        entirely (never streamed, never printed) and hands it to
        `_parse_ensense_stdout` instead.

        Args:
            solver (str): `"pb"` or `"milp"`.
            model_path (str): Path to the dumped XGBoost model JSON.
            feature_indices (list[int]): Column indices of `flip_set`.
            details_path (str | None): Path to `details.csv`, or None.
            precision (float): Ensense's own search precision.
            output_gap (tuple[float, float]): Ensense's own confident-flip band.
            timeout (int): Hard wall-clock limit, in seconds.

        Returns:
            tuple[list[float], list[float]] | None: `(x1, x2)` if the search
                found a candidate pair, None if it legitimately found nothing.

        Raises:
            OracleTimeout: The process had to be killed after `timeout` seconds.
            RuntimeError: The subprocess exited with a non-zero code.
        """

        cmd: list[str] = [
            sys.executable,
            str(_ENSENSE_CLI),
            model_path,
            "--solver",
            solver,
            "--model_library",
            "xgboost",
            "--features",
            *(str(f) for f in feature_indices),
            "--output_gap",
            str(output_gap[0]),
            str(output_gap[1]),
            "--precision",
            str(precision),
            "--timeout",
            str(timeout),
        ]
        if details_path is not None:
            cmd += ["--details", details_path]

        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            raise OracleTimeout(
                f"Ensense {solver!r} CLI did not return within "
                f"time_limit_s={timeout}s -- killed. Ensense core's own "
                f"--timeout is not honored by the vendored solver (pb's "
                f"alarm is disabled, milp's Gurobi TimeLimit is hardcoded "
                f"to 3600s), so this is enforced on the sensei side instead."
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"Ensense {solver!r} CLI ({_ENSENSE_CLI}) exited with code "
                f"{result.returncode}:\n{result.stderr[-4000:]}"
            )

        return _parse_ensense_stdout(result.stdout)

    def _pair_from_points(
        self,
        booster: xgb.Booster,
        leaf_map: LeafMap,
        columns: list[str],
        flip_set: tuple[str, ...],
        x1: NDArray[np.float64],
        x2: NDArray[np.float64],
    ) -> Pair:
        """
        Build a `Pair` from Ensense's raw (x1, x2), scoring both against M0.

        Args:
            booster (xgb.Booster): Model to score x1/x2 against.
            leaf_map (LeafMap): Global leaf map for `booster`.
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            x1 (NDArray[np.float64]): Ensense's raw x1 point.
            x2 (NDArray[np.float64]): Ensense's raw x2 point.

        Returns:
            Pair: `x1`, `x2` with their leaf sets and signed margin gap.

        Raises:
            OracleDegenerate: x1/x2 don't actually form a valid
                counterexample -- identical points, points that differ
                outside the declared flip set, or a zero margin gap are all
                things Ensense has been observed to return without warning
                (region2point() can collapse a degenerate region to a single
                point). None of these are evidence of anything; a caller
                must not treat this the same as "Ensense searched and found
                no violation" (see `OracleDegenerate`'s docstring).
        """

        self._validate_points(columns, flip_set, x1, x2)

        row1 = pd.DataFrame([x1], columns=columns)
        row2 = pd.DataFrame([x2], columns=columns)

        ell1: NDArray[np.int64] = leaf_map.leaf_ids_for_row(booster, row1)
        ell2: NDArray[np.int64] = leaf_map.leaf_ids_for_row(booster, row2)

        e1 = float(
            leaf_map.margin_score(leaf_map.phi_for(booster, row1), leaf_map.v0)[0]
        )
        e2 = float(
            leaf_map.margin_score(leaf_map.phi_for(booster, row2), leaf_map.v0)[0]
        )

        if np.isclose(e1, e2):
            raise OracleDegenerate(
                f"Ensense returned x1/x2 differing on {flip_set} with zero margin "
                f"gap (e1={e1}, e2={e2}) -- not a real counterexample"
            )

        return Pair(
            x1=x1,
            x2=x2,
            ell1=ell1,
            ell2=ell2,
            gap=e1 - e2,
            flip_set=flip_set,
            direction="protected",
            solver_status="FEASIBLE",
        )

    def _validate_points(
        self,
        columns: list[str],
        flip_set: tuple[str, ...],
        x1: NDArray[np.float64],
        x2: NDArray[np.float64],
    ) -> None:
        """
        Structural sanity check on a solver-returned pair, independent of
        the model: x1 and x2 must differ, and they must differ ONLY on the
        declared flip set.

        Args:
            columns (list[str]): All feature names, in model column order.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            x1 (NDArray[np.float64]): The first point.
            x2 (NDArray[np.float64]): The second point.

        Raises:
            OracleDegenerate: x1/x2 are identical, differ outside
                `flip_set`, or don't differ on `flip_set` at all.
        """

        if np.allclose(x1, x2):
            raise OracleDegenerate(
                f"Ensense returned identical x1 and x2 for flip_set={flip_set} "
                f"-- not a counterexample"
            )

        flip_idx = [columns.index(f) for f in flip_set]
        non_flip_mask = np.ones(len(columns), dtype=bool)
        non_flip_mask[flip_idx] = False

        if not np.allclose(x1[non_flip_mask], x2[non_flip_mask]):
            raise OracleDegenerate(
                f"Ensense returned x1/x2 that differ outside the declared flip "
                f"set {flip_set} -- violates 'differ only on the flip set'"
            )

        if np.allclose(x1[flip_idx], x2[flip_idx]):
            raise OracleDegenerate(
                f"Ensense returned x1/x2 that don't actually differ on the "
                f"declared flip set {flip_set}"
            )
