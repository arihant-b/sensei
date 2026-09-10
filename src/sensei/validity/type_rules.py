from sensei.data.bins import EncodedSample
from sensei.spec import Spec
from sensei.validity.pairwise import pair_check, pair_reason

_TOL = 1e-6
_REL_TOL = 1e-6


class TypeRuleChecker:
    """
    Q1 checks: one-hot, ranges, immutable features.

    Integrality is intentionally NOT one of the pair-level checks here
    (`is_type_valid`/`type_valid_reason`, and therefore `Postfilter`, which
    is Ensense-only -- the sensei oracle enforces integrality directly as a
    hard MILP constraint in `oracle/sensei/encoding.py`, unrelated to this
    module): Ensense-sourced counterexamples are taken as already
    type-valid on this dimension, so this postfilter does not re-derive it.
    `is_integral` itself stays defined and tested as a standalone utility.
    """

    @staticmethod
    def is_integral(
        x: EncodedSample,
        feature: str,
        feature_bounds: dict[str, tuple[float, float]],
        tol: float = _TOL,
    ) -> bool:
        """
        Unscale `x[feature]` back to raw units (`lo + x * (hi - lo)`) and
        check it's within `tol` of a whole number. True if `feature` isn't
        in `feature_bounds`/`x` at all -- nothing to verify.
        """

        # no scaler info for this feature -- can't verify
        if feature not in feature_bounds or feature not in x:
            return True

        lo, hi = feature_bounds[feature]
        raw: float = lo + float(x[feature]) * (hi - lo)
        allowed: float = max(tol, _REL_TOL * abs(raw))
        return abs(raw - round(raw)) <= allowed

    @staticmethod
    def check_one_hot_groups(x: EncodedSample, spec: Spec, tol: float = _TOL) -> bool:
        """Every `spec.one_hot_groups` group is exactly one 1 and the rest 0 in `x`."""

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
    def check_ranges(x: EncodedSample, spec: Spec, tol: float = 1e-9) -> bool:
        """Every `spec.ranges[feature] = (lo, hi)` holds for `x`, within `tol`."""

        for feature, (lo, hi) in spec.ranges.items():
            if feature not in x:
                continue

            value = float(x[feature])

            if value < lo - tol or value > hi + tol:
                return False

        return True

    @staticmethod
    def check_immutable_unchanged(
        x1: EncodedSample, x2: EncodedSample, spec: Spec, tol: float = 1e-9
    ) -> bool:
        """Every `spec.immutable` feature has the same value in x1 and x2."""

        for feature in spec.immutable:
            if (
                feature in x1
                and feature in x2
                and abs(float(x1[feature]) - float(x2[feature])) > tol
            ):
                return False

        return True

    @staticmethod
    def type_valid_reason(
        x: EncodedSample, spec: Spec, feature_bounds: dict[str, tuple[float, float]]
    ) -> str | None:
        """
        Diagnostic form of `is_type_valid`: the first failing rule's reason
        (e.g. `"workclass_not_one_hot"`, `"hours-per-week_not_in_range"`), or
        None if `x` is fully type-valid. Mirrors `is_type_valid`'s checks and
        their order, but walks each rule individually instead of
        short-circuiting on an aggregate bool -- kept separate so the common
        pass-through path (`is_type_valid` itself) stays cheap, and this walk
        only runs when a caller needs to explain a rejection
        (`Postfilter.diagnose_pair`).
        """

        for group, columns in spec.one_hot_groups.items():
            values: list[float] = [float(x[c]) for c in columns if c in x]

            if len(values) != len(columns):
                continue

            if not all(abs(v) < _TOL or abs(v - 1.0) < _TOL for v in values):
                return f"{group}_not_one_hot"

            if abs(sum(values) - 1.0) > _TOL:
                return f"{group}_not_one_hot"

        for feature, (lo, hi) in spec.ranges.items():
            if feature not in x:
                continue

            value = float(x[feature])

            if value < lo - 1e-9 or value > hi + 1e-9:
                return f"{feature}_not_in_range"

        return None

    @staticmethod
    def immutable_reason(
        x1: EncodedSample, x2: EncodedSample, spec: Spec, tol: float = 1e-9
    ) -> str | None:
        """
        Diagnostic form of `check_immutable_unchanged`: the first immutable
        feature that changed between `x1` and `x2`, or None if none did.
        """

        for feature in spec.immutable:
            if (
                feature in x1
                and feature in x2
                and abs(float(x1[feature]) - float(x2[feature])) > tol
            ):
                return f"{feature}_not_immutable"

        return None

    @staticmethod
    def type_valid_pair_reason(
        x1: EncodedSample,
        x2: EncodedSample,
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
    ) -> str | None:
        """
        Diagnostic form of `is_type_valid_pair`: the first failing reason,
        prefixed with which point it came from (`"x1."`/`"x2."`), or None if
        the pair is fully type-valid.
        """

        reason: str | None = pair_reason(
            TypeRuleChecker.type_valid_reason, x1, x2, spec, feature_bounds
        )
        if reason is not None:
            return reason
        return TypeRuleChecker.immutable_reason(x1, x2, spec)

    @staticmethod
    def is_type_valid(
        x: EncodedSample, spec: Spec, feature_bounds: dict[str, tuple[float, float]]
    ) -> bool:
        """Q1 checks on a single point: one-hot, ranges."""

        return TypeRuleChecker.check_one_hot_groups(
            x, spec
        ) and TypeRuleChecker.check_ranges(x, spec)

    @staticmethod
    def is_type_valid_pair(
        x1: EncodedSample,
        x2: EncodedSample,
        spec: Spec,
        feature_bounds: dict[str, tuple[float, float]],
    ) -> bool:
        """Q1 checks on a pair: both points type-valid, plus immutability holds."""

        return pair_check(
            TypeRuleChecker.is_type_valid, x1, x2, spec, feature_bounds
        ) and TypeRuleChecker.check_immutable_unchanged(x1, x2, spec)
