import math
import numbers
from typing import Any

from sensei.data.bins import EncodedSample
from sensei.spec import Spec
from sensei.validity.pairwise import pair_check, pair_reason

_TOL = 1e-6


class FunctionalDependencyChecker:
    """Checks a point against `spec.functional_deps` (IF feature=v THEN feature=v')."""

    @staticmethod
    def check_functional_deps(x: EncodedSample, spec: Spec) -> bool:
        """
        `x` satisfies every declared functional dependency in `spec`.

        Args:
            x (EncodedSample): The point to check.
            spec (Spec): Declares `functional_deps`.

        Returns:
            bool: True iff `x` satisfies every dependency.
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
    def check_functional_deps_pair(
        x1: EncodedSample, x2: EncodedSample, spec: Spec
    ) -> bool:
        """
        Both `x1` and `x2` independently satisfy `check_functional_deps`.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares `functional_deps`.

        Returns:
            bool: True iff both points satisfy every dependency.
        """

        return pair_check(
            FunctionalDependencyChecker.check_functional_deps, x1, x2, spec
        )

    @staticmethod
    def func_dep_reason(x: EncodedSample, spec: Spec) -> str | None:
        """
        Diagnostic form of `check_functional_deps`: the first violated
        dependency, named after the IF/THEN feature-value pair it encodes
        (e.g. `"not_if_workclass_3_then_hours-per-week_0"`), or None if `x`
        satisfies every declared functional dependency.

        Args:
            x (EncodedSample): The point to check.
            spec (Spec): Declares `functional_deps`.

        Returns:
            str | None: The first violated dependency's name, or None.
        """

        for if_col, if_val, then_col, then_val in spec.functional_deps:
            if if_col not in x or then_col not in x:
                continue

            if FunctionalDependencyChecker._matches(
                x[if_col], if_val
            ) and not FunctionalDependencyChecker._matches(x[then_col], then_val):
                return f"not_if_{if_col}_{if_val:g}_then_{then_col}_{then_val:g}"

        return None

    @staticmethod
    def func_dep_pair_reason(
        x1: EncodedSample, x2: EncodedSample, spec: Spec
    ) -> str | None:
        """
        Diagnostic form of `check_functional_deps_pair`: the first violated
        dependency, prefixed with which point it came from (`"x1."`/`"x2."`),
        or None if the pair satisfies every declared functional dependency.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares `functional_deps`.

        Returns:
            str | None: The first violation's reason, prefixed with
                `"x1."`/`"x2."`, or None.
        """

        return pair_reason(FunctionalDependencyChecker.func_dep_reason, x1, x2, spec)

    @staticmethod
    def _matches(value: Any, target: Any, tol: float = _TOL) -> bool:
        """
        Tolerant equality: float-safe for numbers, exact otherwise.

        Args:
            value (Any): The value to compare.
            target (Any): The value to compare against.
            tol (float): Absolute tolerance for numeric comparisons.

        Returns:
            bool: True iff `value` matches `target`.
        """

        if isinstance(value, numbers.Real) and isinstance(target, numbers.Real):
            return math.isclose(float(value), float(target), abs_tol=tol)

        return value == target
