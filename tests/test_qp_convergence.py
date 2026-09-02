"""
The repair QP's core promise: after adding a cut for a real violation
and solving, the gap it targeted must land inside the epsilon tube, and
that must keep holding as more cuts are layered on (cuts are never
removed -- see CLAUDE.md invariant 2).
"""

import numpy as np
from fixtures.synthetic import (
    make_planted_counterexample,
    make_synthetic_dataset,
    make_synthetic_model,
)

from oracle import Counterexample
from repair import LeafRepairQP


def test_solve_pulls_the_planted_gap_inside_epsilon():
    model = make_synthetic_model(seed=0)
    X, _ = make_synthetic_dataset(seed=0)
    qp = LeafRepairQP(model, X, model.cfg)
    ce = make_planted_counterexample(model)

    assert abs(ce.gap) > model.cfg.epsilon, "fixture should start as a real violation"

    qp.add_cut(ce)
    v_new = qp.solve()

    new_gap = float(v_new[ce.leaves1].sum() - v_new[ce.leaves2].sum())
    assert abs(new_gap) <= model.cfg.epsilon + 1e-6


def test_solve_stays_close_to_v0_under_drift_penalty():
    """mu penalises drift from v0 -- repair should not rewrite the model."""
    model = make_synthetic_model(seed=0)
    X, _ = make_synthetic_dataset(seed=0)
    qp = LeafRepairQP(model, X, model.cfg)
    ce = make_planted_counterexample(model)

    qp.add_cut(ce)
    v_new = qp.solve()

    assert np.all(np.isfinite(v_new))
    assert np.max(np.abs(v_new - model.v0)) < abs(ce.gap)


def test_earlier_cuts_stay_satisfied_after_later_cuts_and_resolves():
    """
    A repaired violation cannot reappear because its constraint is still
    binding -- adding and solving a second cut must not loosen the first.
    """
    model = make_synthetic_model(seed=0)
    X, _ = make_synthetic_dataset(seed=0)
    qp = LeafRepairQP(model, X, model.cfg)

    first = make_planted_counterexample(model)
    qp.add_cut(first)
    qp.solve()

    second = Counterexample(
        x1=np.zeros(3), x2=np.zeros(3), gap=1.0,
        leaves1=np.array([(int(first.leaves1[0]) + 2) % qp.n]),
        leaves2=np.array([(int(first.leaves2[0]) + 2) % qp.n]),
    )
    qp.add_cut(second)
    v_final = qp.solve()

    assert qp.n_cuts == 2
    assert np.all(np.isfinite(v_final))

    first_gap = float(v_final[first.leaves1].sum() - v_final[first.leaves2].sum())
    assert abs(first_gap) <= model.cfg.epsilon + 1e-6
