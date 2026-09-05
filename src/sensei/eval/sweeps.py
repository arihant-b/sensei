from dataclasses import dataclass

import xgboost as xgb

from sensei.data.bins import FrozenBins
from sensei.model.leaves import LeafMap
from sensei.oracle.milp.encoding import TreeEncoder, TreeStructure
from sensei.oracle.milp.solve import TierAOracle
from sensei.oracle.types import OracleTimeout
from sensei.spec import Spec


@dataclass(frozen=True)
class SweepPoint:
    """
    A single point in a sweep, representing the result of a feasibility check at a
    specific value of theta or eps.
    """

    value: float  # the theta or eps swept
    certified: bool  # True iff the oracle found no valid pair (UNSAT)
    gap: float | None  # the found violation's gap, if any
    status: str  # "UNSAT" | "SAT" | "TIMEOUT"


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
        eps: float,
        thetas: list[float],
        seed: int,
        time_limit_s: float,
        mip_gap: float,
        v=None,
        structure: TreeStructure | None = None,
    ) -> list[SweepPoint]:
        """
        Sweep over theta, reporting the result of a feasibility check at each value.

        Args:
            booster (xgb.Booster): The XGBoost booster to evaluate.
            columns (list[str]): The list of column names in the dataset.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            spec (Spec): The specification to check.
            bins (FrozenBins): The bins to use for the evaluation.
            flip_set (tuple[str, ...]): The set of features to flip.
            direction (str): The direction of the flip.
            eps (float): The epsilon value to use for the feasibility check.
            thetas (list[float]): The list of theta values to sweep over.
            seed (int): The random seed to use for the evaluation.
            time_limit_s (float): The time limit for the evaluation in seconds.
            mip_gap (float): The MIP gap to use for the evaluation.
            v (_type_, optional): The vector of values to use for the evaluation.
                                  Defaults to None.
            structure (TreeStructure | None, optional): The tree structure to use for
                                                        the evaluation. Defaults to
                                                        None.

        Returns:
            list[SweepPoint]: The list of sweep points representing the result of the
                              feasibility check at each theta value.
        """

        leaf_map = LeafMap(booster)

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        oracle = TierAOracle()
        points: list[SweepPoint] = []

        for theta in thetas:
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
                    eps=eps,
                    seed=seed,
                    time_limit_s=time_limit_s,
                    mip_gap=mip_gap,
                    enforce_validity=True,
                    structure=structure,
                    v=v,
                    bins=bins,
                    theta=theta,
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
        theta: float,
        epsilons: list[float],
        seed: int,
        time_limit_s: float,
        mip_gap: float,
        v=None,
        structure: TreeStructure | None = None,
    ) -> list[SweepPoint]:
        """
        Sweep over eps, reporting the result of a feasibility check at each value.

        Args:
            booster (xgb.Booster): The XGBoost booster to evaluate.
            columns (list[str]): The list of column names.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            spec (Spec): The specification to check.
            bins (FrozenBins): The bins to use for the evaluation.
            flip_set (tuple[str, ...]): The set of features to flip.
            direction (str): The direction of the sweep.
            theta (float): The threshold value.
            epsilons (list[float]): The list of epsilon values to sweep over.
            seed (int): The random seed to use.
            time_limit_s (float): The time limit for each evaluation in seconds.
            mip_gap (float): The MIP gap to use for the optimization.
            v (_type_, optional): The vector of values to use for the evaluation.
                                  Defaults to None.
            structure (TreeStructure | None, optional): The tree structure to use for
                                                        the evaluation. Defaults to
                                                        None.

        Returns:
            list[SweepPoint]: The list of sweep points representing the result of the
                              feasibility check at each epsilon value.
        """

        leaf_map = LeafMap(booster)

        if structure is None:
            structure = TreeEncoder.extract_tree_structure(
                booster, leaf_map.leaf_index, columns
            )

        oracle = TierAOracle()
        points: list[SweepPoint] = []

        for eps in epsilons:
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
                    eps=eps,
                    seed=seed,
                    time_limit_s=time_limit_s,
                    mip_gap=mip_gap,
                    enforce_validity=True,
                    structure=structure,
                    v=v,
                    bins=bins,
                    theta=theta,
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
