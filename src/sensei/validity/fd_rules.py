import math
import numbers

from sensei.data.bins import Point
from sensei.spec import Spec

_TOL = 1e-6


class FunctionalDependencyChecker:
    """
    A utility class for checking whether a given point satisfies the functional
    dependencies specified in a dataset specification.
    """

    @staticmethod
    def check_functional_deps(x: Point, spec: Spec) -> bool:
        """
        Check if a point satisfies all functional dependencies specified in the dataset
        specification.

        Args:
            x (Point): The point to check, represented as a dictionary mapping feature
                       names to values.
            spec (Spec): The dataset specification containing the functional
                         dependencies.

        Returns:
            bool: True if the point satisfies all functional dependencies, False
                  otherwise.
        """

        for if_col, if_val, then_col, then_val in spec.functional_deps:
            if if_col not in x or then_col not in x:
                continue

            if FunctionalDependencyChecker._matches(
                x[if_col], if_val
            ) and not FunctionalDependencyChecker._matches(x[then_col], then_val):
                return False

        return True

    @staticmethod
    def check_functional_deps_pair(x1: Point, x2: Point, spec: Spec) -> bool:
        """
        Check if a pair of points satisfies all functional dependencies specified in the
        dataset specification.

        Args:
            x1 (Point): The first point to check.
            x2 (Point): The second point to check.
            spec (Spec): The dataset specification containing the functional
                         dependencies.

        Returns:
            bool: True if the pair of points satisfies all functional dependencies,
                  False otherwise.
        """

        return FunctionalDependencyChecker.check_functional_deps(
            x1, spec
        ) and FunctionalDependencyChecker.check_functional_deps(x2, spec)

    @staticmethod
    def _matches(value, target, tol: float = _TOL) -> bool:
        """
        Tolerant equality: float-safe for numbers, exact otherwise.

        Args:
            value (_type_): The value to compare.
            target (_type_): The target value to compare against.
            tol (float, optional): The tolerance for floating-point comparisons.
                                   Defaults to _TOL.

        Returns:
            bool: True if the values are equal within the specified tolerance, False
                  otherwise.
        """

        if isinstance(value, numbers.Real) and isinstance(target, numbers.Real):
            return math.isclose(float(value), float(target), abs_tol=tol)

        return value == target
