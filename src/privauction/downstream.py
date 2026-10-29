"""Downstream stage: capacity-constrained Cournot competition with linear demand.

Firms face the inverse demand P = a - Q, where P is the *margin* over a common
constant marginal cost (so profits are pi_i = P * q_i).  Firm i can produce at
most k_i (capacity; ``np.inf`` means unconstrained).  Consumer surplus is
CS = Q^2 / 2.  All quantities are in normalised units (pre-auction industry
output Q0 = 1); ``calibration`` rescales them to money.

Equilibrium (Kreps-Scheinkman-type capacity game, simultaneous quantities):
unconstrained firms all produce the same quantity q_u = P; constrained firms
produce k_c.  The active set is found by the monotone elimination algorithm in
``cournot_capacity`` (proved correct in the paper's appendix).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Outcome:
    q: np.ndarray          # equilibrium quantities of the strategic firms
    price: float           # equilibrium margin P
    Q: float               # strategic output + fringe output
    profit: np.ndarray     # pi_i = P q_i
    cs: float              # Q^2 / 2
    hhi: float             # Herfindahl-Hirschman index, 0-10000 (fringe atomistic)
    shares: np.ndarray     # q_i / Q


def cournot_capacity(a: float, k: np.ndarray, fringe: float = 0.0) -> Outcome:
    """Equilibrium of the linear Cournot game with capacities ``k``.

    ``fringe`` is the fixed aggregate output of a price-taking atomistic fringe
    (it shifts residual demand: a_eff = a - fringe).
    """
    k = np.asarray(k, dtype=float)
    n = k.size
    a_eff = a - fringe
    constrained = np.zeros(n, dtype=bool)
    while True:
        u = ~constrained
        n_u = int(u.sum())
        if n_u == 0:
            q = k.copy()
            break
        q_u = (a_eff - k[constrained].sum()) / (n_u + 1)
        newly = u & (k < q_u - 1e-15)
        if not newly.any():
            q = np.where(constrained, k, q_u)
            break
        constrained |= newly
    Q_strat = float(q.sum())
    price = a_eff - Q_strat
    Q = Q_strat + fringe
    profit = price * q
    shares = q / Q
    return Outcome(q=q, price=price, Q=Q, profit=profit, cs=0.5 * Q * Q,
                   hhi=float(10000.0 * np.sum(shares ** 2)), shares=shares)


def brute_force_cournot(a: float, k: np.ndarray, fringe: float = 0.0,
                        iters: int = 20000) -> np.ndarray:
    """Independent check: damped sequential best-response iteration."""
    k = np.asarray(k, dtype=float)
    q = np.minimum(k, 0.1)
    a_eff = a - fringe
    for _ in range(iters):
        for i in range(k.size):
            others = q.sum() - q[i]
            q[i] = min(k[i], max(0.0, 0.5 * (a_eff - others)))
    return q
