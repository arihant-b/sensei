import logging
from collections.abc import Callable

from sensei.data.bins import FrozenBins, Point
from sensei.oracle.types import OracleSaturated, Pair
from sensei.spec import Spec
from sensei.validity.domain_rules import DomainRuleChecker
from sensei.validity.fd_rules import FunctionalDependencyChecker
from sensei.validity.plausibility import PlausibilityChecker
from sensei.validity.type_rules import TypeRuleChecker

MAX_REJECTIONS_PER_ITER = 20

log: logging.Logger = logging.getLogger("sensei.validity.postfilter")


class Postfilter:
    """
    Tier B2 postfilter: Q1 + Q2 + immutability checks on pairs returned by a Tier B1
    oracle call. Rejects invalid pairs, raises OracleSaturated if too many consecutive
    invalid pairs are returned. Not a certificate.

    Raises:
        OracleSaturated: If `max_rejections` consecutive invalid pairs are returned by
                         `pair_source()`.
    """

    @staticmethod
    def validate_pair(
        x1: Point,
        x2: Point,
        spec: Spec,
        bins: FrozenBins,
        theta: float,
        feature_bounds: dict[str, tuple[float, float]],
    ) -> bool:
        """
        Validate a pair of points against the specification, plausibility, and feature
        bounds.

        Args:
            x1 (Point): The first point to check.
            x2 (Point): The second point to check.
            spec (Spec): The specification to check against.
            bins (FrozenBins): The bins to check against.
            theta (float): The threshold for plausibility checks.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.

        Returns:
            bool: True if the pair is valid, False otherwise.
        """

        return (
            TypeRuleChecker.is_type_valid_pair(x1, x2, spec, feature_bounds)
            and FunctionalDependencyChecker.check_functional_deps_pair(x1, x2, spec)
            and DomainRuleChecker.check_domain_rules_pair(x1, x2, spec)
            and PlausibilityChecker.is_plausible_pair(x1, x2, bins, theta)
        )

    @staticmethod
    def find_valid_pair(
        pair_source: Callable[[], Pair | None],
        columns: list[str],
        spec: Spec,
        bins: FrozenBins,
        theta: float,
        feature_bounds: dict[str, tuple[float, float]],
        max_rejections: int = MAX_REJECTIONS_PER_ITER,
    ) -> Pair | None:
        """
        Find a valid pair from the given `pair_source`, validating each pair against the
        specification, plausibility, and feature bounds.

        Args:
            pair_source (Callable[[], Pair | None]): A callable that returns a pair of
                                                     points or None if no more pairs
                                                     are available.
            columns (list[str]): The names of the columns in the points.
            spec (Spec): The specification to check against.
            bins (FrozenBins): The bins to check against.
            theta (float): The threshold for plausibility checks.
            feature_bounds (dict[str, tuple[float, float]]): The bounds for each
                                                             feature.
            max_rejections (int, optional): The maximum number of consecutive invalid
                                            pairs to reject before raising an error.
                                            Defaults to MAX_REJECTIONS_PER_ITER.

        Raises:
            OracleSaturated: If `max_rejections` consecutive invalid pairs are returned
                             by `pair_source()`.

        Returns:
            Pair | None: A valid pair of points if found, or None if no more pairs are
                         available.
        """

        rejections = 0

        while True:
            pair: Pair | None = pair_source()

            if pair is None:
                return None

            x1 = Point(dict(zip(columns, pair.x1, strict=True)))
            x2 = Point(dict(zip(columns, pair.x2, strict=True)))

            if Postfilter.validate_pair(x1, x2, spec, bins, theta, feature_bounds):
                return pair

            rejections += 1
            log.info(
                "Tier B postfilter: rejected invalid pair (%d/%d)",
                rejections,
                max_rejections,
            )

            if rejections >= max_rejections:
                raise OracleSaturated(
                    f"{rejections} consecutive invalid pairs from Tier B -- "
                    f"rejection budget ({max_rejections}) exhausted. Not a certificate."
                )
