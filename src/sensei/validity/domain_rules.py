from sensei.data.bins import EncodedSample
from sensei.spec import DomainRule, Spec
from sensei.validity.pairwise import pair_check, pair_reason

_TOL = 1e-9


class DomainRuleChecker:
    """Checks a point against `spec.domain_rules` (linear inequalities/equalities)."""

    @staticmethod
    def check_rule(x: EncodedSample, rule: DomainRule, tol: float = _TOL) -> bool:
        """
        `x` satisfies the single linear constraint `rule` (within `tol`).

        Args:
            x (EncodedSample): The point to check.
            rule (DomainRule): The linear constraint to check against.
            tol (float): Numerical tolerance.

        Returns:
            bool: True iff `x` satisfies `rule`.

        Raises:
            ValueError: `rule.sense` isn't one of "<=", ">=", "==".
        """

        # rule doesn't apply to this point's feature set
        if not all(map(lambda f: f in x, rule.coefficients)):
            return True

        lhs: float = sum(coef * float(x[f]) for f, coef in rule.coefficients.items())

        if rule.sense == "<=":
            return lhs <= rule.rhs + tol
        if rule.sense == ">=":
            return lhs >= rule.rhs - tol
        if rule.sense == "==":
            return abs(lhs - rule.rhs) <= tol

        raise ValueError(f"unknown domain rule sense: {rule.sense!r}")

    @staticmethod
    def rule_reason(
        x: EncodedSample, rule: DomainRule, tol: float = _TOL
    ) -> str | None:
        """
        Diagnostic form of `check_rule`: a human-readable description of the
        violated linear constraint -- its coefficients, sense, declared RHS,
        and the point's actual LHS value -- or None if `x` satisfies `rule`.

        Args:
            x (EncodedSample): The point to check.
            rule (DomainRule): The linear constraint to check against.
            tol (float): Numerical tolerance.

        Returns:
            str | None: The violation description, or None if `x`
                satisfies `rule`.

        Raises:
            ValueError: `rule.sense` isn't one of "<=", ">=", "==".
        """

        if not all(f in x for f in rule.coefficients):
            return None

        lhs: float = sum(coef * float(x[f]) for f, coef in rule.coefficients.items())

        if rule.sense == "<=":
            satisfied = lhs <= rule.rhs + tol
        elif rule.sense == ">=":
            satisfied = lhs >= rule.rhs - tol
        elif rule.sense == "==":
            satisfied = abs(lhs - rule.rhs) <= tol
        else:
            raise ValueError(f"unknown domain rule sense: {rule.sense!r}")

        if satisfied:
            return None

        expr: str = " + ".join(
            f"{coef:g}*{feature}" for feature, coef in rule.coefficients.items()
        )
        return f"{expr} {rule.sense} {rule.rhs:g} violated (lhs={lhs:g})"

    @staticmethod
    def domain_rule_reason(x: EncodedSample, spec: Spec) -> str | None:
        """
        Diagnostic form of `check_domain_rules`: the first violated rule's
        reason, or None if `x` satisfies every declared domain rule.

        Args:
            x (EncodedSample): The point to check.
            spec (Spec): Declares `domain_rules`.

        Returns:
            str | None: The first violation's reason, or None.
        """

        for rule in spec.domain_rules:
            reason: str | None = DomainRuleChecker.rule_reason(x, rule)
            if reason is not None:
                return reason

        return None

    @staticmethod
    def domain_rule_pair_reason(
        x1: EncodedSample, x2: EncodedSample, spec: Spec
    ) -> str | None:
        """
        Diagnostic form of `check_domain_rules_pair`: the first violated
        rule's reason, prefixed with which point it came from
        (`"x1."`/`"x2."`), or None if the pair satisfies every declared
        domain rule.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares `domain_rules`.

        Returns:
            str | None: The first violation's reason, prefixed with
                `"x1."`/`"x2."`, or None.
        """

        return pair_reason(DomainRuleChecker.domain_rule_reason, x1, x2, spec)

    @staticmethod
    def check_domain_rules(x: EncodedSample, spec: Spec) -> bool:
        """
        `x` satisfies every declared domain rule in `spec`.

        Args:
            x (EncodedSample): The point to check.
            spec (Spec): Declares `domain_rules`.

        Returns:
            bool: True iff `x` satisfies every rule.
        """

        return all(
            map(
                DomainRuleChecker.check_rule,
                [x] * len(spec.domain_rules),
                spec.domain_rules,
            )
        )

    @staticmethod
    def check_domain_rules_pair(
        x1: EncodedSample, x2: EncodedSample, spec: Spec
    ) -> bool:
        """
        Both `x1` and `x2` independently satisfy `check_domain_rules`.

        Args:
            x1 (EncodedSample): The first point.
            x2 (EncodedSample): The second point.
            spec (Spec): Declares `domain_rules`.

        Returns:
            bool: True iff both points satisfy every rule.
        """

        return pair_check(DomainRuleChecker.check_domain_rules, x1, x2, spec)
