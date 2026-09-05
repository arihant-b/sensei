from sensei.data.bins import Point
from sensei.spec import DomainRule, Spec

_TOL = 1e-9


class DomainRuleChecker:
    """
    A utility class for checking whether a given point satisfies the domain rules
    specified in a dataset specification.

    Raises:
        ValueError: If an unknown sense is encountered in a domain rule.
    """

    @staticmethod
    def check_rule(x: Point, rule: DomainRule, tol: float = _TOL) -> bool:
        """
        Check if a point satisfies a single domain rule.

        Args:
            x (Point): The point to check, represented as a dictionary mapping feature
                       names to values.
            rule (DomainRule): The domain rule to check against.
            tol (float, optional): The tolerance for floating-point comparisons.
                                   Defaults to _TOL.

        Raises:
            ValueError: If an unknown sense is encountered in a domain rule.

        Returns:
            bool: True if the point satisfies the domain rule, False otherwise.
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
    def check_domain_rules(x: Point, spec: Spec) -> bool:
        """
        Check if a point satisfies all domain rules specified in the dataset
        specification.

        Args:
            x (Point): The point to check, represented as a dictionary mapping feature
                       names to values.
            spec (Spec): The dataset specification containing the domain rules.

        Returns:
            bool: True if the point satisfies all domain rules, False otherwise.
        """

        return all(
            map(
                DomainRuleChecker.check_rule,
                [x] * len(spec.domain_rules),
                spec.domain_rules,
            )
        )

    @staticmethod
    def check_domain_rules_pair(x1: Point, x2: Point, spec: Spec) -> bool:
        """
        Check if a pair of points satisfies all domain rules specified in the dataset
        specification.

        Args:
            x1 (Point): The first point to check, represented as a dictionary mapping
                        feature names to values.
            x2 (Point): The second point to check, represented as a dictionary mapping
                        feature names to values.
            spec (Spec): The dataset specification containing the domain rules.

        Returns:
            bool: True if both points satisfy all domain rules, False otherwise.
        """

        return DomainRuleChecker.check_domain_rules(
            x1, spec
        ) and DomainRuleChecker.check_domain_rules(x2, spec)
