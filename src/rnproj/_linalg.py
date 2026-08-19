"""Weighted least-squares solvers for the projection estimator."""

from __future__ import annotations

import numpy as np

__all__ = ["wls", "wls_nonnegative"]


def wls(
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    *,
    ridge: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Solve ``min_b sum_i w_i * (X_i b - Y_i)^2``, optionally with ridge.

    Parameters
    ----------
    X : (n, p) design matrix.
    Y : (n,) or (n, m) target(s); one solve per column.
    w : (n,) strictly positive weights.
    ridge : float
        Scale-invariant Tikhonov strength: the normal equations use
        ``G + ridge * mean(diag(G)) * I`` with ``G = Xw' Xw``.

    Returns
    -------
    beta : (p,) or (p, m)
    cond : float
        Condition number of the row-weighted design matrix ``sqrt(w) X``.
    """
    sw = np.sqrt(w)[:, None]
    Xw = X * sw
    Yw = np.asarray(Y) * (sw if np.asarray(Y).ndim == 2 else sw[:, 0])

    if ridge > 0.0:
        G = Xw.T @ Xw
        G[np.diag_indices_from(G)] += ridge * float(np.mean(np.diag(G)))
        beta = np.linalg.solve(G, Xw.T @ Yw)
        cond = float(np.linalg.cond(Xw))
        return beta, cond

    beta, _, rank, sv = np.linalg.lstsq(Xw, Yw, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    return beta, cond


def wls_nonnegative(
    X: np.ndarray,
    Y: np.ndarray,
    w: np.ndarray,
    *,
    thin: int = 1,
) -> tuple[np.ndarray, float]:
    """Weighted least squares subject to a nonnegative fitted payoff.

    Solves ``min_b sum_i w_i (X_i b - Y_i)^2  s.t.  X b >= 0`` (the fitted
    replicating payoff is nonnegative in every grid state), the Python
    analogue of the Matlab ``lsqlin(X, Y, -X, 0)`` variant. Only supports a
    single target column.

    Parameters
    ----------
    thin : int
        Enforce the constraint on every ``thin``-th grid row (the payoff is
        piecewise linear between strikes, so a modest thinning is exact as
        long as all kink locations remain represented; ``1`` = every row).
    """
    from scipy.optimize import LinearConstraint, minimize

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 1:
        raise ValueError("wls_nonnegative supports a single target column.")
    sw = np.sqrt(w)[:, None]
    Xw = X * sw
    Yw = Y * sw[:, 0]

    beta0, cond = wls(X, Y, w)
    Xc = X[::thin]
    if np.all(Xc @ beta0 >= 0.0):
        return beta0, cond

    G = Xw.T @ Xw
    c = Xw.T @ Yw

    res = minimize(
        lambda b: 0.5 * b @ G @ b - c @ b,
        x0=beta0,
        jac=lambda b: G @ b - c,
        hess=lambda b: G,
        method="trust-constr",
        constraints=[LinearConstraint(Xc, 0.0, np.inf)],
        options={"gtol": 1e-10, "xtol": 1e-12, "maxiter": 2000},
    )
    if not res.success:
        raise RuntimeError(f"nonnegative-payoff WLS failed to converge: {res.message}")
    return res.x, cond
