import dataclasses
from dataclasses import dataclass

import xgboost as xgb

from sensei.config import Settings
from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.ensense.adapter import EnsenseOracle
from sensei.oracle.sensei.encoding import TreeEncoder
from sensei.oracle.sensei.solve import SenseiOracle
from sensei.oracle.types import (
    OracleDegenerate,
    OracleSaturated,
    OracleTimeout,
    Pair,
    TreeStructure,
)
from sensei.spec import Spec


@dataclass(frozen=True)
class SweepPoint:
    """
    A single point in a sweep, representing the result of a feasibility check at a
    specific value of theta, eps, mip_gap, or the ensense oracle's output_gap.
    """

    value: float  # the swept parameter's value
    certified: bool  # True iff the oracle found no valid pair (UNSAT)
    gap: float | None  # the found violation's gap, if any
    status: str  # "UNSAT" | "SAT" | "TIMEOUT" | "ORACLE_SATURATED"
    # ORACLE_SATURATED is EnsenseOracle only


class Sweeper:
    """
    Sweeps over theta or eps, reporting the result of a feasibility check at each value.
    """

    @staticmethod
    def theta_sweep(
        booster: xgb.Booster,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        bins: FrozenBins,
        flip_set: tuple[str, ...],
        direction: str,
        thetas: list[float],
        settings: Settings,
        v=None,
        structure: TreeStructure | None = None,
    ) -> list[SweepPoint]:
        """
        The required theta curve: re-run the sensei oracle's feasibility
        check at each `thetas` value, fixed `settings.sensitivity.eps`, on
        the same (optionally repaired via `v`) booster. Raising theta
        shrinks the plausible region, so this is never reported as a single
        number.

        Args:
            booster (xgb.Booster): The (optionally repaired) model to check.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec): Dataset spec, for validity/plausibility encoding.
            bins (FrozenBins): Frozen plausibility bins.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            thetas (list[float]): Plausibility thresholds to sweep.
            settings (Settings): Base settings; `sensitivity.theta` is
                overridden per point.
            v (NDArray[np.float64] | None): Leaf values to check with;
                `leaf_map.v0` if None.
            structure (TreeStructure | None): Pre-extracted tree structure;
                extracted fresh if None.

        Returns:
            list[SweepPoint]: One point per `thetas` value.
        """

        leaf_map = LeafMap(booster)

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        oracle = SenseiOracle()
        points: list[SweepPoint] = []

        for theta in thetas:
            point_settings: Settings = dataclasses.replace(
                settings,
                sensitivity=dataclasses.replace(settings.sensitivity, theta=theta),
            )
            try:
                pair: Pair | None = oracle.worst_valid_pair(
                    booster,
                    leaf_map,
                    columns,
                    feature_bounds,
                    spec,
                    flip_set,
                    direction,
                    mode="feasibility",
                    settings=point_settings,
                    enforce_validity=True,
                    structure=structure,
                    v=v,
                    bins=bins,
                )
            except OracleTimeout:
                points.append(
                    SweepPoint(theta, certified=False, gap=None, status="TIMEOUT")
                )
                continue

            if pair is None:
                points.append(
                    SweepPoint(theta, certified=True, gap=None, status="UNSAT")
                )
            else:
                points.append(
                    SweepPoint(theta, certified=False, gap=pair.gap, status="SAT")
                )

        return points

    @staticmethod
    def eps_sweep(
        booster: xgb.Booster,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        bins: FrozenBins,
        flip_set: tuple[str, ...],
        direction: str,
        epsilons: list[float],
        settings: Settings,
        v=None,
        structure: TreeStructure | None = None,
    ) -> list[SweepPoint]:
        """
        Same shape as `theta_sweep`, but sweeps `epsilons` at a fixed
        `settings.sensitivity.theta`.

        Args:
            booster (xgb.Booster): The (optionally repaired) model to check.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec): Dataset spec, for validity/plausibility encoding.
            bins (FrozenBins): Frozen plausibility bins.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            epsilons (list[float]): Sensitivity budgets to sweep.
            settings (Settings): Base settings; `sensitivity.eps` is
                overridden per point.
            v (NDArray[np.float64] | None): Leaf values to check with;
                `leaf_map.v0` if None.
            structure (TreeStructure | None): Pre-extracted tree structure;
                extracted fresh if None.

        Returns:
            list[SweepPoint]: One point per `epsilons` value.
        """

        leaf_map = LeafMap(booster)

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        oracle = SenseiOracle()
        points: list[SweepPoint] = []

        for eps in epsilons:
            point_settings: Settings = dataclasses.replace(
                settings,
                sensitivity=dataclasses.replace(settings.sensitivity, eps=eps),
            )
            try:
                pair = oracle.worst_valid_pair(
                    booster,
                    leaf_map,
                    columns,
                    feature_bounds,
                    spec,
                    flip_set,
                    direction,
                    mode="feasibility",
                    settings=point_settings,
                    enforce_validity=True,
                    structure=structure,
                    v=v,
                    bins=bins,
                )
            except OracleTimeout:
                points.append(
                    SweepPoint(eps, certified=False, gap=None, status="TIMEOUT")
                )
                continue

            if pair is None:
                points.append(SweepPoint(eps, certified=True, gap=None, status="UNSAT"))
            else:
                points.append(
                    SweepPoint(eps, certified=False, gap=pair.gap, status="SAT")
                )

        return points

    @staticmethod
    def mip_gap_sweep(
        booster: xgb.Booster,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        bins: FrozenBins,
        flip_set: tuple[str, ...],
        direction: str,
        mip_gaps: list[float],
        settings: Settings,
        v=None,
        structure: TreeStructure | None = None,
    ) -> list[SweepPoint]:
        """
        Sweep over the sensei oracle's solver relative-optimality tolerance
        (`mip_gap`), same shape as `theta_sweep`/`eps_sweep`: hold the
        trained (and, if `v` is given, repaired) booster fixed, re-run our
        own MILP oracle's feasibility check at each `mip_gap` (fixed
        `settings.sensitivity.eps`/`.theta`), and report UNSAT/SAT/TIMEOUT.
        Unlike theta/eps this isn't a modeling choice being certified
        against -- it's solver precision -- so a point here shows whether
        loosening the tolerance changes what the oracle reports, not a
        property of the model itself.

        Args:
            booster (xgb.Booster): The (optionally repaired) model to check.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec): Dataset spec, for validity/plausibility encoding.
            bins (FrozenBins): Frozen plausibility bins.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            direction (str): `"protected"` or `"monotone_wrong"`.
            mip_gaps (list[float]): Relative MIP gaps to sweep.
            settings (Settings): Base settings; `oracle.mip_gap` is
                overridden per point.
            v (NDArray[np.float64] | None): Leaf values to check with;
                `leaf_map.v0` if None.
            structure (TreeStructure | None): Pre-extracted tree structure;
                extracted fresh if None.

        Returns:
            list[SweepPoint]: One point per `mip_gaps` value.
        """

        leaf_map = LeafMap(booster)

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        oracle = SenseiOracle()
        points: list[SweepPoint] = []

        for mip_gap in mip_gaps:
            point_settings: Settings = dataclasses.replace(
                settings, oracle=dataclasses.replace(settings.oracle, mip_gap=mip_gap)
            )
            try:
                pair = oracle.worst_valid_pair(
                    booster,
                    leaf_map,
                    columns,
                    feature_bounds,
                    spec,
                    flip_set,
                    direction,
                    mode="feasibility",
                    settings=point_settings,
                    enforce_validity=True,
                    structure=structure,
                    v=v,
                    bins=bins,
                )
            except OracleTimeout:
                points.append(
                    SweepPoint(mip_gap, certified=False, gap=None, status="TIMEOUT")
                )
                continue

            if pair is None:
                points.append(
                    SweepPoint(mip_gap, certified=True, gap=None, status="UNSAT")
                )
            else:
                points.append(
                    SweepPoint(mip_gap, certified=False, gap=pair.gap, status="SAT")
                )

        return points

    @staticmethod
    def gap_sweep(
        booster: xgb.Booster,
        columns: list[str],
        feature_bounds: dict[str, tuple[float, float]],
        spec: Spec,
        bins: FrozenBins,
        flip_set: tuple[str, ...],
        output_gap_lowers: list[float],
        settings: Settings,
    ) -> list[SweepPoint]:
        """
        Sweep over the ensense oracle's own confident-flip margin: Ensense
        core's `output_gap` is a symmetric `(lo, 1 - lo)` band in
        PROBABILITY space (not the margin-space `eps`), and only a flip
        Ensense itself is this confident about counts as a hit. Unlike
        `theta_sweep`/`eps_sweep`/`mip_gap_sweep`, which all re-run OUR OWN
        MILP (the sensei oracle), this calls Ensense core directly
        (`EnsenseOracle.worst_valid_pair`) at each `lo` (via a per-point
        `settings.sensitivity.gap` override -- `EnsenseOracle.worst_valid_pair`
        derives its own `output_gap` from that field), on the same fixed
        booster, using `settings.sensitivity.theta`/`settings.oracle.method`/
        `.time_limit_s` fixed throughout -- the sensei oracle never sees
        `output_gap` at all (CegsalLoop's ensense entry point,
        `worst_valid_pair_loop`, doesn't expose it either).

        `OracleDegenerate`/`OracleSaturated` (the ensense oracle's postfilter
        rejection budget exhausted) is its own status, `"ORACLE_SATURATED"`
        -- this must never be folded into `"TIMEOUT"` or `"UNSAT"`, since it
        means something different (we ran out of rejection budget, not that
        the oracle proved or timed out).

        Args:
            booster (xgb.Booster): The (optionally repaired) model to check.
            columns (list[str]): All feature names, in model column order.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            spec (Spec): Dataset spec, for the Ensense postfilter.
            bins (FrozenBins): Frozen plausibility bins.
            flip_set (tuple[str, ...]): Features x1/x2 are allowed to differ on.
            output_gap_lowers (list[float]): Lower bounds `lo` to sweep;
                each point's band is `(lo, 1 - lo)`.
            settings (Settings): Base settings; `sensitivity.gap` is
                overridden per point.

        Returns:
            list[SweepPoint]: One point per `output_gap_lowers` value.
        """

        leaf_map = LeafMap(booster)
        oracle = EnsenseOracle()
        points: list[SweepPoint] = []

        for lo in output_gap_lowers:
            point_settings: Settings = dataclasses.replace(
                settings, sensitivity=dataclasses.replace(settings.sensitivity, gap=lo)
            )
            try:
                pair = oracle.worst_valid_pair(
                    booster,
                    leaf_map,
                    columns,
                    flip_set,
                    spec,
                    bins,
                    feature_bounds,
                    point_settings,
                )
            except (OracleDegenerate, OracleSaturated):
                points.append(
                    SweepPoint(lo, certified=False, gap=None, status="ORACLE_SATURATED")
                )
                continue

            if pair is None:
                points.append(SweepPoint(lo, certified=True, gap=None, status="UNSAT"))
            else:
                points.append(
                    SweepPoint(lo, certified=False, gap=pair.gap, status="SAT")
                )

        return points
