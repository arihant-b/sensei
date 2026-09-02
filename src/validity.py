import math
import numbers


def _matches(value, target) -> bool:
    """Tolerant equality: float-safe for numbers, exact otherwise."""
    if isinstance(value, numbers.Real) and isinstance(target, numbers.Real):
        return math.isclose(value, target, abs_tol=1e-6)

    return value == target


class ValidityChecker:
    """
    Q1 (well-formed) and Q2 (plausible) from the notes.
    Q3 (undesirable) lives in SensitivitySpec, not here.
    """

    def __init__(
        self, spec, bins, cfg,
        feature_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.spec = spec
        self.bins = bins
        self.cfg = cfg
        # feature -> (raw_min, raw_max) pre-scaling, from Dataset.feature_bounds.
        # Needed to check things like integrality against min-max-scaled data.
        self.feature_bounds = feature_bounds or {}

    # ---- Q1: cheap, exact -------------------------------------------------
    def is_type_valid(self, x) -> bool:
        """
        integrality, one-hot sums to 1, ranges, functional deps.

        Trusts that spec.ranges / spec.functional_deps / integer_features
        are declared in the same units/encoding as x -- same posture as
        protected/monotone in spec.py ("declared by a human, never
        inferred"): this checker doesn't know or guess the data's units,
        it enforces exactly what was declared.
        """
        for feature in self.spec.integer_features:
            if feature not in x:
                continue

            bounds: tuple[float, float] | None = self.feature_bounds.get(feature)

            if bounds is None:
                continue    # no scaler info for this feature -- can't verify

            lo, hi = bounds
            raw = lo + float(x[feature]) * (hi - lo)

            if abs(raw - round(raw)) > 1e-6:
                return False

        for columns in self.spec.one_hot_groups.values():
            values = [float(x[c]) for c in columns if c in x]
            if len(values) != len(columns):
                continue
            if not all(abs(v) < 1e-6 or abs(v - 1.0) < 1e-6 for v in values):
                return False
            if abs(sum(values) - 1.0) > 1e-6:
                return False

        for feature, (lo, hi) in self.spec.ranges.items():
            if feature not in x:
                continue
            value = x[feature]
            if value < lo - 1e-9 or value > hi + 1e-9:
                return False

        for if_col, if_val, then_col, then_val in self.spec.functional_deps:
            if if_col not in x or then_col not in x:
                continue
            if _matches(x[if_col], if_val) and not _matches(x[then_col], then_val):
                return False

        return True

    # ---- Q2: costlier, approximate ---------------------------------------
    def plausibility(self, x) -> float:
        """log pi(x). Compare against log(theta)."""
        return self.bins.log_marginal(x)

    def is_plausible(self, x) -> bool:
        return self.plausibility(x) >= math.log(self.cfg.theta)

    # ---- combined ---------------------------------------------------------
    def is_valid_pair(self, x1, x2) -> bool:
        return all([
            self.is_type_valid(x1), self.is_type_valid(x2),
            self.is_plausible(x1), self.is_plausible(x2),
        ])

    def milp_constraints(self) -> list[tuple[dict[str, float], str, float]]:
        """
        Only the genuinely LINEAR pieces of Q1 export here, as
        (coeffs, sense, rhs) triples meaning
        sum_f coeffs[f] * var[f]  <sense>  rhs  (consumed by oracle.py,
        applied once per point). sense is one of "<=", ">=", "==".

        NOT included here, and why -- these are real gaps, not silent
        omissions:
          - integrality (integer_features): even with feature_bounds
            available to unscale a value, "is an integer" is a variable
            TYPE (vtype=GRB.INTEGER on the unscaled quantity), not a
            linear inequality -- has to be applied where the variable is
            created, not exported as a constraint on an existing one.
          - functional deps: "if x[a]==v then x[b]==w" is a conditional,
            not a straight-line inequality. Encoding it needs an
            indicator constraint tied to a binary "x[a]==v" variable,
            which this flat (coeffs, sense, rhs) shape has no room for.
          - Q2 plausibility (sum_f log pi_f(x_f) >= log theta): log_p is
            piecewise-constant per bin (see FrozenBins), so a linear
            encoding needs new per-feature bin-selector binaries, not a
            constraint on variables that already exist -- outside what
            this return shape can express.
        """
        constraints: list[tuple[dict[str, float], str, float]] = []

        for feature, (lo, hi) in self.spec.ranges.items():
            constraints.append(({feature: 1.0}, ">=", float(lo)))
            constraints.append(({feature: 1.0}, "<=", float(hi)))

        for columns in self.spec.one_hot_groups.values():
            constraints.append(({c: 1.0 for c in columns}, "==", 1.0))

        return constraints
