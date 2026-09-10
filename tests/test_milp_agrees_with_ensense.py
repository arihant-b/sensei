import dataclasses
import sys
import tempfile
from pathlib import Path

from xgboost import Booster

from sensei.oracle.types import Pair

_ROOT: Path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from sensei.config import load_defaults  # noqa: E402
from sensei.data.loader import Dataset  # noqa: E402
from sensei.model.leaves import LeafMap  # noqa: E402
from sensei.model.train import Trainer  # noqa: E402
from sensei.oracle.sensei.solve import SenseiOracle  # noqa: E402

_ENSENSE_SRC: Path = _ROOT / "ensense" / "src"

if str(_ENSENSE_SRC) not in sys.path:
    sys.path.insert(0, str(_ENSENSE_SRC))


def _ensense_milp_sensitive(
    booster,
    columns: list[str],
    flip_set: tuple[str, ...],
    output_gap: tuple[float, float],
    timeout: int = 60,
) -> bool:
    from ensemble import Ensemble  # type: ignore[import-not-found]
    from milp import milpSolver  # type: ignore[import-not-found]
    from options import Options  # type: ignore[import-not-found]

    with tempfile.TemporaryDirectory() as tmp:
        model_path = str(Path(tmp) / "model.json")
        booster.save_model(model_path)

        options = Options()
        options.solver = "milp"
        options.model_file = model_path
        options.model_library = "xgboost"
        options.details_file = None
        options.features = [columns.index(f) for f in flip_set]
        options.output_gap = list(output_gap)
        options.lgap = output_gap[0]
        options.ugap = output_gap[1]
        options.timeout = timeout
        options.truelabel = -2
        options.otherlabel = -2
        options.multiclass = False
        options.max_trees = None
        options.prob = False
        options.objective = False
        options.verbosity = 0
        options.local_check_samples = None

        e = Ensemble(options)
        e.load(print_vitals=False)

        solver = milpSolver(e, options=options)
        solver.local_sample = None
        return bool(solver.attack(options))


def _our_oracle_sensitive(
    booster,
    columns: list[str],
    feature_bounds,
    flip_set: tuple[str, ...],
    eps: float,
    timeout: int = 60,
) -> bool:
    leaf_map = LeafMap(booster)
    defaults = load_defaults()
    settings = dataclasses.replace(
        defaults,
        sensitivity=dataclasses.replace(defaults.sensitivity, eps=eps),
        oracle=dataclasses.replace(defaults.oracle, time_limit_s=timeout, mip_gap=0.05),
        seed=42,
    )
    pair: Pair | None = SenseiOracle().worst_valid_pair(
        booster,
        leaf_map,
        columns,
        feature_bounds,
        spec=None,
        flip_set=flip_set,
        direction="protected",
        mode="feasibility",
        settings=settings,
        enforce_validity=False,
    )
    return pair is not None


def test_sat_agreement_on_a_model_with_a_real_violation() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.y_train is not None
    assert ds.columns is not None and ds.feature_bounds is not None

    booster: Booster = Trainer.train_baseline(
        ds.X_train, ds.y_train, n_estimators=50, max_depth=5, seed=42
    )

    ensense_sensitive: bool = _ensense_milp_sensitive(
        booster, ds.columns, ("sex",), output_gap=(0.45, 0.55)
    )
    ours_sensitive: bool = _our_oracle_sensitive(
        booster, ds.columns, ds.feature_bounds, ("sex",), eps=0.05
    )

    assert ensense_sensitive is True, (
        "expected Ensense core to find a violation here (known from prior runs)"
    )
    assert ours_sensitive is True, (
        "our encoding disagrees with Ensense core -- fix ours"
    )


def test_unsat_agreement_on_a_tiny_model_with_no_real_effect() -> None:
    ds: Dataset = Dataset("adult", eval_holdout=0.2, seed=42).load()

    assert ds.X_train is not None and ds.y_train is not None
    assert ds.columns is not None and ds.feature_bounds is not None

    booster: Booster = Trainer.train_baseline(
        ds.X_train, ds.y_train, n_estimators=10, max_depth=3, seed=42
    )

    ensense_sensitive: bool = _ensense_milp_sensitive(
        booster, ds.columns, ("sex",), output_gap=(0.45, 0.55)
    )
    ours_sensitive: bool = _our_oracle_sensitive(
        booster, ds.columns, ds.feature_bounds, ("sex",), eps=0.1
    )

    assert ensense_sensitive is False, (
        "expected Ensense core to find nothing here (known from prior runs)"
    )
    assert ours_sensitive is False, (
        "our encoding disagrees with Ensense core -- fix ours"
    )
