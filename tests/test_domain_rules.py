import sys
from pathlib import Path

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.bins import Point  # noqa: E402
from sensei.spec import DomainRule  # noqa: E402
from sensei.validity.domain_rules import DomainRuleChecker  # noqa: E402


def test_le_rule() -> None:
    rule = DomainRule(coefficients={"exp": 1.0, "age": -1.0}, sense="<=", rhs=-18.0)

    assert DomainRuleChecker.check_rule(Point({"exp": 20, "age": 40}), rule)
    assert not DomainRuleChecker.check_rule(Point({"exp": 25, "age": 40}), rule)


def test_eq_rule_with_tolerance() -> None:
    rule = DomainRule(coefficients={"a": 1.0, "b": -1.0}, sense="==", rhs=0.0)

    assert DomainRuleChecker.check_rule(Point({"a": 5.0, "b": 5.0000000001}), rule)
    assert not DomainRuleChecker.check_rule(Point({"a": 5.0, "b": 5.1}), rule)


def test_rule_skipped_when_feature_missing():
    rule = DomainRule(coefficients={"nonexistent": 1.0}, sense="<=", rhs=0.0)
    assert DomainRuleChecker.check_rule(Point({"a": 1.0}), rule)
