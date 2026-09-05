import sys
from pathlib import Path

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.data.bins import Point  # noqa: E402
from sensei.data.loader import Dataset  # noqa: E402
from sensei.spec import Spec, load_spec  # noqa: E402
from sensei.validity.fd_rules import FunctionalDependencyChecker  # noqa: E402
from sensei.validity.type_rules import TypeRuleChecker  # noqa: E402


def _adult_malformed_point() -> tuple[Point, dict]:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.feature_bounds is not None

    edu_num_lo, edu_num_hi = ds.feature_bounds["education-num"]
    edu_lo, edu_hi = ds.feature_bounds["education"]

    education_num_scaled: float = (13.500002 - edu_num_lo) / (edu_num_hi - edu_num_lo)
    education_scaled: float = (10.0 - edu_lo) / (edu_hi - edu_lo)

    point = Point(
        {"education-num": education_num_scaled, "education": education_scaled}
    )
    return point, ds.feature_bounds


def test_malformed_education_num_fails_integrality() -> None:
    point, feature_bounds = _adult_malformed_point()

    assert not TypeRuleChecker.is_integral(point, "education-num", feature_bounds), (
        "education-num=13.500002 must fail the integrality check -- if this "
        "passes, is_integral() is broken, not the example"
    )


def test_malformed_point_fails_functional_dependency() -> None:
    point, _ = _adult_malformed_point()
    spec: Spec = load_spec("adult")

    assert not FunctionalDependencyChecker.check_functional_deps(point, spec), (
        "education=Doctorate with education-num=13.500002 contradicts spec.adult.yaml's"
        " declared functional_deps (education==Doctorate implies education-num==16) -- "
        "if this passes, check_functional_deps() is broken, not the example"
    )


def test_a_genuinely_valid_point_passes_both_checks() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.feature_bounds is not None

    edu_num_lo, edu_num_hi = ds.feature_bounds["education-num"]
    edu_lo, edu_hi = ds.feature_bounds["education"]

    point = Point(
        {
            "education-num": (16.0 - edu_num_lo) / (edu_num_hi - edu_num_lo),
            "education": (10.0 - edu_lo) / (edu_hi - edu_lo),
        }
    )
    spec: Spec = load_spec("adult")

    assert TypeRuleChecker.is_integral(point, "education-num", ds.feature_bounds)
    assert FunctionalDependencyChecker.check_functional_deps(point, spec)
