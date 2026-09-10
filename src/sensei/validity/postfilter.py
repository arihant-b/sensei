import logging
from collections.abc import Callable

from sensei.data.bins import EncodedSample, FrozenBins
from sensei.oracle.types import OracleDegenerate, OracleSaturated, Pair
from sensei.spec import Spec
from sensei.validity.domain_rules import DomainRuleChecker
from sensei.validity.fd_rules import FunctionalDependencyChecker
from sensei.validity.plausibility import PlausibilityChecker
from sensei.validity.type_rules import TypeRuleChecker

MAX_REJECTIONS_PER_ITER = 20

log: logging.Logger = logging.getLogger("sensei.validity.postfilter")


class Postfilter:
    """
    Q1 + Q2 + immutability checks on pairs returned by an ensense oracle call.
    Rejects invalid pairs, raises OracleSaturated if too many consecutive
    invalid pairs are returned. Not a certificate.

    Raises:
        OracleSaturated: If `max_rejections` consecutive invalid pairs are returned by
                         `pair_source()`.
    """

    @staticmethod
    def validate_pair(
        x1: EncodedSample,
        x2: EncodedSample,
        spec: Spec,
        bins: FrozenBins,
        theta: float,
        feature_bounds: dict[str, tuple[float, float]],
    ) -> bool:
        """
        Q1 + functional-dependency + domain-rule + Q2 checks, in that order.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares every Q1/functional-dependency/domain rule.
            bins (FrozenBins): Frozen plausibility bins for the Q2 check.
            theta (float): Plausibility threshold for the Q2 check.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.

        Returns:
            bool: True iff the pair passes every check.
        """

        return (
            TypeRuleChecker.is_type_valid_pair(x1, x2, spec, feature_bounds)
            and FunctionalDependencyChecker.check_functional_deps_pair(x1, x2, spec)
            and DomainRuleChecker.check_domain_rules_pair(x1, x2, spec)
            and PlausibilityChecker.is_plausible_pair(x1, x2, bins, theta)
        )

    @staticmethod
    def diagnose_pair(
        x1: EncodedSample,
        x2: EncodedSample,
        spec: Spec,
        bins: FrozenBins,
        theta: float,
        feature_bounds: dict[str, tuple[float, float]],
    ) -> str:
        """
        Explain why `validate_pair(x1, x2, ...)` returned False for the same
        arguments: the first failing check, categorized and feature-specific,
        in the SAME order `validate_pair` checks them (type -> functional
        dependency -> domain rule -> plausibility) so the reason returned
        here is always the one that actually caused the rejection --

            "not_type_valid: x1.age_not_integral"
            "not_type_valid: x2.workclass_not_one_hot"
            "not_func_dep: x1.not_if_workclass_3_then_hours-per-week_0"
            "not_domain_rule: x2.1*hours-per-week + -1*age_<=_-18_violated_lhs=5"
            "not_plausible: x1.log_plaus=-42.1090_below_log_theta=-13.8155"

        Only meaningful to call right after `validate_pair` has returned
        False for the same `x1`/`x2` -- each individual check here is
        re-evaluated from scratch, not read off `validate_pair`'s own run.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares every Q1/functional-dependency/domain rule.
            bins (FrozenBins): Frozen plausibility bins for the Q2 check.
            theta (float): Plausibility threshold for the Q2 check.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.

        Returns:
            str: The categorized, feature-specific rejection reason.
        """

        reason: str | None = TypeRuleChecker.type_valid_pair_reason(
            x1, x2, spec, feature_bounds
        )
        if reason is not None:
            return f"not_type_valid: {reason}"

        reason = FunctionalDependencyChecker.func_dep_pair_reason(x1, x2, spec)
        if reason is not None:
            return f"not_func_dep: {reason}"

        reason = DomainRuleChecker.domain_rule_pair_reason(x1, x2, spec)
        if reason is not None:
            return f"not_domain_rule: {reason}"

        reason = PlausibilityChecker.plausibility_pair_reason(x1, x2, bins, theta)
        if reason is not None:
            return f"not_plausible: {reason}"

        return (
            "unknown: validate_pair() failed but diagnose_pair() found no "
            "violation -- the two disagree, which is a bug in one of them"
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
        Call `pair_source()` until it yields a `validate_pair`-passing pair or
        returns None (oracle found nothing). Degenerate pairs and Q1/Q2
        rejections share one consecutive-rejection counter (the rejection
        budget); `max_rejections` in a row raises `OracleSaturated` -- NOT
        the same as `pair_source()` returning None, and never reported as
        UNSAT.

        Args:
            pair_source (Callable[[], Pair | None]): Called repeatedly to
                produce candidate pairs.
            columns (list[str]): All feature names, in model column order.
            spec (Spec): Declares every Q1/functional-dependency/domain rule.
            bins (FrozenBins): Frozen plausibility bins for the Q2 check.
            theta (float): Plausibility threshold for the Q2 check.
            feature_bounds (dict[str, tuple[float, float]]): Raw `(lo, hi)`
                bounds per feature.
            max_rejections (int): Consecutive invalid pairs allowed before
                giving up.

        Returns:
            Pair | None: The first validated pair, or None if `pair_source`
                itself returns None.

        Raises:
            OracleSaturated: `max_rejections` consecutive invalid pairs.
        """

        rejections = 0

        while True:
            try:
                pair: Pair | None = pair_source()
            except OracleDegenerate as e:
                pair = None
                rejections += 1
                log.info(
                    "postfilter: rejected degenerate pair (%d/%d): %s",
                    rejections,
                    max_rejections,
                    e,
                )

                if rejections >= max_rejections:
                    raise OracleSaturated(
                        f"{rejections} consecutive invalid pairs from the ensense "
                        f"oracle -- rejection budget ({max_rejections}) exhausted. "
                        f"Not a certificate."
                    ) from e

                continue

            if pair is None:
                return None

            x1 = EncodedSample(dict(zip(columns, pair.x1, strict=True)))
            x2 = EncodedSample(dict(zip(columns, pair.x2, strict=True)))

            if Postfilter.validate_pair(x1, x2, spec, bins, theta, feature_bounds):
                return pair

            rejections += 1
            log.info(
                "postfilter: rejected invalid pair (%d/%d)",
                rejections,
                max_rejections,
            )
            log.info(
                "postfilter: rejection reason -- %s",
                Postfilter.diagnose_pair(x1, x2, spec, bins, theta, feature_bounds),
            )

            if rejections >= max_rejections:
                raise OracleSaturated(
                    f"{rejections} consecutive invalid pairs from the ensense oracle "
                    f"-- rejection budget ({max_rejections}) exhausted. Not a "
                    f"certificate."
                )
