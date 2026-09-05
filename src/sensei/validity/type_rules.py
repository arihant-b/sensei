from sensei.data.bins import Point
from sensei.spec import Spec

_TOL = 1e-6


class TypeRuleChecker:
    """
    Q1 checks: integrality, one-hot, ranges, immutable features.
    """

    @staticmethod
    def is_integral(
        x: Point,
        feature: str,
        feature_bounds: dict[str, tuple[float, float]],
        tol: float = _TOL,
    ) -> bool:
        """
        Check a min-max-scaled [0,1] value corresponds to an integral raw value.

        Args:
            x (Point): The point to check.
            feature (str): The feature to check.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            tol (float, optional): The tolerance for the check. Defaults to _TOL.

        Returns:
            bool: True if the scaled value corresponds to an integral raw value, False
                  otherwise.
        """

        # no scaler info for this feature -- can't verify
        if feature not in feature_bounds or feature not in x:
            return True

        lo, hi = feature_bounds[feature]
        raw: float = lo + float(x[feature]) * (hi - lo)
        return abs(raw - round(raw)) <= tol

    @staticmethod
    def check_integer_features(
        x: Point, spec: Spec, feature_bounds: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Check that all features declared as integer in the spec are indeed integral
        in the given point.

        Args:
            x (Point): The point to check.
            spec (Spec): The specification containing the integer features.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.

        Returns:
            bool: True if all integer features are integral in the point,
                  False otherwise.
        """

        return all(
            map(
                TypeRuleChecker.is_integral,
                [x] * len(spec.integer_features),
                spec.integer_features,
                [feature_bounds] * len(spec.integer_features),
            )
        )

    @staticmethod
    def check_one_hot_groups(x: Point, spec: Spec, tol: float = _TOL) -> bool:
        """
        Check that all one-hot groups declared in the spec are valid in the given point.

        Args:
            x (Point): The point to check.
            spec (Spec): The specification containing the one-hot groups.
            tol (float, optional): The tolerance for floating-point comparisons.
                                   Defaults to _TOL.

        Returns:
            bool: True if all one-hot groups are valid in the point, False otherwise.
        """

        for columns in spec.one_hot_groups.values():
            values: list[float] = list(
                map(lambda c: float(x[c]), filter(lambda c: c in x, columns))
            )

            if len(values) != len(columns):
                continue

            if not all(map(lambda v: abs(v) < tol or abs(v - 1.0) < tol, values)):
                return False

            if abs(sum(values) - 1.0) > tol:
                return False

        return True

    @staticmethod
    def check_ranges(x: Point, spec: Spec, tol: float = 1e-9) -> bool:
        """
        Check that all features declared with ranges in the spec are within those
        ranges in the given point.

        Args:
            x (Point): The point to check.
            spec (Spec): The specification containing the feature constraints.
            tol (float, optional): The tolerance for floating-point comparisons.
                                   Defaults to 1e-9.

        Returns:
            bool: True if all features are within their declared ranges, False
                  otherwise.
        """

        for feature, (lo, hi) in spec.ranges.items():
            if feature not in x:
                continue

            value = float(x[feature])

            if value < lo - tol or value > hi + tol:
                return False

        return True

    @staticmethod
    def check_immutable_unchanged(
        x1: Point, x2: Point, spec: Spec, tol: float = 1e-9
    ) -> bool:
        """
        Check that all immutable features declared in the spec have the same value in
        both points.

        Args:
            x1 (Point): The first point to check.
            x2 (Point): The second point to check.
            spec (Spec): The specification containing the immutable features.
            tol (float, optional): The tolerance for floating-point comparisons.
                                   Defaults to 1e-9.

        Returns:
            bool: True if all immutable features are unchanged between the two points,
                  False otherwise.
        """

        for feature in spec.immutable:
            if (
                feature in x1
                and feature in x2
                and abs(float(x1[feature]) - float(x2[feature])) > tol
            ):
                return False

        return True

    @staticmethod
    def is_type_valid(
        x: Point, spec: Spec, feature_bounds: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Q1 checks on a single point: integrality, one-hot, ranges.

        Args:
            x (Point): The point to check.
            spec (Spec): The specification containing the feature constraints.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.

        Returns:
            bool: True if the point is type-valid, False otherwise.
        """

        return (
            TypeRuleChecker.check_integer_features(x, spec, feature_bounds)
            and TypeRuleChecker.check_one_hot_groups(x, spec)
            and TypeRuleChecker.check_ranges(x, spec)
        )

    @staticmethod
    def is_type_valid_pair(
        x1: Point, x2: Point, spec: Spec, feature_bounds: dict[str, tuple[float, float]]
    ) -> bool:
        """
        Q1 checks on a pair: both points type-valid, plus immutability holds.

        Args:
            x1 (Point): The first point to check.
            x2 (Point): The second point to check.
            spec (Spec): The specification containing the feature constraints.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.

        Returns:
            bool: True if both points are type-valid and all immutable features are
                  unchanged, False otherwise.
        """

        return (
            TypeRuleChecker.is_type_valid(x1, spec, feature_bounds)
            and TypeRuleChecker.is_type_valid(x2, spec, feature_bounds)
            and TypeRuleChecker.check_immutable_unchanged(x1, x2, spec)
        )
