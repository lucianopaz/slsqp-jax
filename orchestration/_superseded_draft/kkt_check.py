"""Independent convergence judge, shared by ALL solvers (sqpdax, scipy SLSQP, trust-constr).

Convention: minimise f(x) s.t. c_eq(x) = 0, c_in(x) >= 0, lb <= x <= ub.
The orchestrator uses ONLY this module to decide `converged_independent`, so an implementer
cannot 'fix' a failure by loosening the solver's own stopping test.
"""

import numpy as np
from scipy.optimize import lsq_linear

TOL_STAT, TOL_FEAS, TOL_COMP, ACT_TOL = 1e-6, 1e-6, 1e-6, 1e-7


def kkt_report(x, grad_f, c_eq, J_eq, c_in, J_in, lb, ub):
    """All arrays are numpy. Use empty arrays (shape (0,), (0, n)) when a block is absent."""
    n = x.size
    # Fold finite bounds into the inequality block.
    rows, vals = [J_in], [c_in]
    for i in np.where(np.isfinite(lb))[0]:
        e = np.zeros((1, n))
        e[0, i] = 1.0
        rows.append(e)
        vals.append(np.array([x[i] - lb[i]]))
    for i in np.where(np.isfinite(ub))[0]:
        e = np.zeros((1, n))
        e[0, i] = -1.0
        rows.append(e)
        vals.append(np.array([ub[i] - x[i]]))
    J_all, c_all = np.vstack(rows), np.concatenate(vals)

    viol = max(
        np.max(np.abs(c_eq), initial=0.0), np.max(np.maximum(-c_all, 0.0), initial=0.0)
    )

    active = c_all <= ACT_TOL
    A = np.hstack([J_eq.T, J_all[active].T])  # n x (m_eq + n_active)
    lo = np.concatenate([np.full(c_eq.size, -np.inf), np.zeros(active.sum())])
    hi = np.full(lo.size, np.inf)
    if A.shape[1] > 0:
        sol = lsq_linear(A, grad_f, bounds=(lo, hi))  # min ||g - A y||, y_ineq >= 0
        resid = grad_f - A @ sol.x
        mu = sol.x[c_eq.size :]
    else:
        resid, mu = grad_f, np.zeros(0)
    stat = np.max(np.abs(resid), initial=0.0) / max(
        1.0, np.max(np.abs(grad_f), initial=0.0)
    )

    # Complementarity on inactive constraints is satisfied by construction (mu = 0 there);
    # report slack*multiplier on the active set as a sanity check.
    comp = float(np.max(np.abs(mu * c_all[active]), initial=0.0)) if mu.size else 0.0
    return dict(
        kkt_stationarity=float(stat),
        max_violation=float(viol),
        complementarity=comp,
        converged_independent=bool(
            stat <= TOL_STAT and viol <= TOL_FEAS and comp <= TOL_COMP
        ),
    )
