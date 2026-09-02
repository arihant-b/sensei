"""
Guards the two things CLAUDE.md calls out about repair.py's cut
construction: shared leaves must cancel to coefficient 0 and drop out
of the constraint entirely, and the constant term coef . v0 must shift
the bound so the cut lands where the pre-repair gap actually was.
"""

import numpy as np
from fixtures.synthetic import (
    make_planted_counterexample,
    make_synthetic_dataset,
    make_synthetic_model,
)

from oracle import Counterexample
from repair import LeafRepairQP


def _qp(seed: int = 0) -> LeafRepairQP:
    model = make_synthetic_model(seed=seed)
    X, _ = make_synthetic_dataset(seed=seed)
    return LeafRepairQP(model, X, model.cfg)


def test_shared_leaves_cancel_out_of_the_cut():
    qp = _qp()
    shared = 1
    ce = Counterexample(
        x1=np.zeros(3), x2=np.zeros(3), gap=1.0,
        leaves1=np.array([0, shared, 2]),
        leaves2=np.array([shared, 3, 4]),
    )
    qp.add_cut(ce)
    qp.m.update()   # Gurobi stages addConstr; names/counts need a flush to see it

    constr = qp.m.getConstrByName("cut_hi_0")
    assert constr is not None
    row = qp.m.getRow(constr)
    row_var_names = {row.getVar(i).VarName for i in range(row.size())}

    assert qp.d[shared].VarName not in row_var_names, "shared leaf must cancel out"
    assert qp.d[0].VarName in row_var_names
    assert qp.d[3].VarName in row_var_names
    assert shared not in qp.cuts[0]["leaves"]


def test_cut_constant_matches_pre_repair_gap():
    """coef . v0 must shift the bound to where the ORIGINAL gap was."""
    model = make_synthetic_model(seed=0)
    X, _ = make_synthetic_dataset(seed=0)
    qp = LeafRepairQP(model, X, model.cfg)
    ce = make_planted_counterexample(model)
    assert model.v0 is not None

    qp.add_cut(ce)

    expected_gap = float(model.v0[ce.leaves1].sum() - model.v0[ce.leaves2].sum())
    assert abs(qp.cuts[0]["const"] - expected_gap) < 1e-9


def test_n_cuts_only_grows():
    """Cuts are never removed: n_cuts must be monotonically increasing."""
    qp = _qp()
    assert qp.n_cuts == 0

    seen = []
    for i in range(4):
        ce = Counterexample(
            x1=np.zeros(3), x2=np.zeros(3), gap=1.0,
            leaves1=np.array([i % qp.n]),
            leaves2=np.array([(i + 1) % qp.n]),
        )
        qp.add_cut(ce)
        seen.append(qp.n_cuts)

    assert seen == [1, 2, 3, 4]
    qp.m.update()
    assert qp.m.NumConstrs == 8      # 2 per cut (hi/lo), none ever removed
    assert len(qp.cuts) == 4
