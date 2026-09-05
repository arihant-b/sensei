import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

from sensei.model.leaves import LeafMap
from sensei.oracle.types import Pair

_ENSENSE_SRC: Path = Path(__file__).resolve().parents[3] / "ensense" / "src"

if str(_ENSENSE_SRC) not in sys.path:
    sys.path.insert(0, str(_ENSENSE_SRC))


class TierBOracle:
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
        method: str = "pb",
        details_csv: str | None = None,
        output_gap: tuple[float, float] = (0.3, 0.7),
        precision: float = 1000.0,
        timeout: int = 120,
    ) -> Pair | None:
        """
        Find the worst valid pair of points using the specified solver method.

        Args:
            booster (xgb.Booster): The XGBoost booster model to analyze.
            leaf_map (LeafMap): The leaf map to use for mapping points to leaves.
            columns (list[str]): The list of column names in the dataset.
            flip_set (tuple[str, ...]): The set of features to flip.
            method (str, optional): The solver method to use. Defaults to "pb".
            details_csv (str | None, optional): The path to the details CSV file.
                                                Defaults to None.
            output_gap (tuple[float, float], optional): The range of output gaps to
                                                        consider. Defaults to (0.3,
                                                        0.7).
            precision (float, optional): The precision of the solver. Defaults to
                                         1000.0.
            timeout (int, optional): The timeout for the solver in seconds. Defaults to
                                     120.

        Raises:
            ValueError: If an unknown solver method is specified.

        Returns:
            Pair | None: The worst valid pair of points found by the solver, or None if
                         no valid pair is found.
        """

        solvers: dict[str, Callable] = {
            "pb": self._solve_via_pb,
            "milp": self._solve_via_milp,
        }

        if method not in solvers:
            raise ValueError(
                f"unknown Ensense solver method {method!r} -- one of "
                f"{sorted(solvers)} are wired up here; Ensense core also exposes "
                f"naive_smt, rounding, roundingsoplex, veritas, monitor "
                f"(ensense/src/options.py's --solver choices), not yet adapted"
            )

        return solvers[method](
            booster,
            leaf_map,
            columns,
            flip_set,
            details_csv=details_csv,
            output_gap=output_gap,
            precision=precision,
            timeout=timeout,
        )

    def _solve_via_pb(
        self,
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
        Solve the problem using the PB (Pseudo-Boolean) method.

        Args:
            booster (xgb.Booster): The XGBoost booster model to analyze.
            leaf_map (LeafMap): The map of leaves in the booster.
            columns (list[str]): The list of column names.
            flip_set (tuple[str, ...]): The set of columns to flip.
            details_csv (str | None): The path to the CSV file containing details, or
                                      None if not needed.
            output_gap (tuple[float, float]): The range of output gaps to consider.
            precision (float): The precision of the solver.
            timeout (int): The timeout for the solver in seconds.

        Returns:
            Pair | None: The worst valid pair of points found by the solver, or None if
                         no valid pair is found.
        """

        from ensemble import Ensemble  # type: ignore[import-not-found]
        from pb import search_anomaly_for_features  # type: ignore[import-not-found]

        feature_indices: list[int] = list(map(columns.index, flip_set))

        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "model.json")
            booster.save_model(model_path)

            options: object = self._build_options(
                "pb",
                model_path,
                feature_indices,
                details_csv,
                precision,
                output_gap,
                timeout,
            )
            e: Ensemble = Ensemble(options)
            e.load(print_vitals=False)
            base_val: float = e.get_base_value()

            _, _, region_pair = search_anomaly_for_features(
                e,
                feature_indices,
                precision,
                e.n_classes,
                e.model,
                e.trees,
                e.n_trees,
                e.op_range_list,
                base_val,
                e.feature_names,
                options,
            )

            if region_pair is None:
                return None

            x1: NDArray[np.float64] = np.asarray(
                e.region2point(region_pair[0]), dtype=float
            )
            x2: NDArray[np.float64] = np.asarray(
                e.region2point(region_pair[1]), dtype=float
            )

        return self._pair_from_points(booster, leaf_map, columns, flip_set, x1, x2)

    def _solve_via_milp(
        self,
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
        Solve the problem using the MILP (Mixed-Integer Linear Programming) method.

        Args:
            booster (xgb.Booster): The XGBoost booster model to analyze.
            leaf_map (LeafMap): The leaf map of the model.
            columns (list[str]): The list of column names.
            flip_set (tuple[str, ...]): The set of features to flip.
            details_csv (str | None): The path to the details CSV file, or None if not
                                      needed.
            output_gap (tuple[float, float]): The gap in the output values.
            precision (float): The precision of the solver.
            timeout (int): The timeout for the solver in seconds.

        Returns:
            Pair | None: The worst valid pair of points found by the solver, or None if
                         no valid pair is found.
        """

        from ensemble import Ensemble  # type: ignore[import-not-found]
        from milp import milpSolver  # type: ignore[import-not-found]

        feature_indices: list[int] = list(map(columns.index, flip_set))

        with tempfile.TemporaryDirectory() as tmp:
            model_path = str(Path(tmp) / "model.json")
            booster.save_model(model_path)

            options: object = self._build_options(
                "milp",
                model_path,
                feature_indices,
                details_csv,
                precision,
                output_gap,
                timeout,
            )
            e: Ensemble = Ensemble(options)
            e.load(print_vitals=False)

            solver: milpSolver = milpSolver(e, options=options)
            solver.local_sample = None

            captured_points: list[list[float]] = []
            original_region2point: Callable[[object], list[float]] = e.region2point

            def _capturing_region2point(region) -> list[float]:
                point: list[float] = original_region2point(region)
                captured_points.append(point)
                return point

            e.region2point = _capturing_region2point
            try:
                sensitive: bool = solver.attack(options)
            finally:
                e.region2point = original_region2point

            if not sensitive or len(captured_points) < 2:
                return None

            x1: NDArray[np.float64] = np.asarray(captured_points[-2], dtype=float)
            x2: NDArray[np.float64] = np.asarray(captured_points[-1], dtype=float)

        return self._pair_from_points(booster, leaf_map, columns, flip_set, x1, x2)

    def _build_options(
        self,
        solver: str,
        model_path: str,
        features: list[int],
        details_path: str | None,
        precision: float,
        output_gap: tuple[float, float],
        timeout: int,
    ) -> object:
        """
        Build the options for the Ensense solver.

        Args:
            solver (str): The solver to use (e.g., "pb" or "milp").
            model_path (str): The path to the model file.
            features (list[int]): The indices of the features to consider.
            details_path (str | None): The path to the details file.
            precision (float): The precision of the solver.
            output_gap (tuple[float, float]): The gap between the output values.
            timeout (int): The timeout for the solver.

        Returns:
            object: The built options object.
        """

        # imported here, not at module scope, so importing sensei.oracle doesn't
        # require ensense/ to be importable unless this adapter is actually used
        from options import Options  # type: ignore[import-not-found]

        options: Options = Options()
        options.solver = solver
        options.encoding = "pb"
        options.verbosity = 0
        options.in_distro_clauses_file = ""
        options.data_file = ""
        options.model_library = "xgboost"
        options.output_gap = list(output_gap)
        options.local_check_file = None
        options.timeout = timeout
        options.max_trees = None
        options.objective = False
        options.unaffected_cons = False
        options.affected_cons = False
        options.ancestor_cons = False
        options.all_features = False
        options.small_change = False
        options.compute_data_distance = False
        options.plot = False
        options.all_single = False
        options.strong_multi = False
        options.model_file = model_path
        options.details_file = details_path
        options.features = features
        options.precision = precision
        options.debug = False
        options.prob = False
        options.perturb = 0.1
        options.metric = None
        options.pca_data = ""
        options.pca_d = False
        options.local_check_samples = None
        options.lgap = output_gap[0]
        options.ugap = output_gap[1]
        options.truelabel = -2
        options.otherlabel = -2
        options.multiclass = False
        return options

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
        Construct a Pair object from two points and the given parameters.

        Args:
            booster (xgb.Booster): The XGBoost model.
            leaf_map (LeafMap): The leaf map.
            columns (list[str]): The column names.
            flip_set (tuple[str, ...]): The set of features that have been flipped.
            x1 (NDArray[np.float64]): The first point.
            x2 (NDArray[np.float64]): The second point.

        Returns:
            Pair: The constructed Pair object representing the two points and their
                  associated information.
        """

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
