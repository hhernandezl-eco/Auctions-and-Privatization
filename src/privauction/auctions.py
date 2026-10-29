"""Auction stage: dominant-strategy handicap auctions, set-aside and Myerson benchmark.

Bidders: D (dominant incumbent) and E (challenger).  Each bidder i has a
*willingness to pay* w_i = m_i * xi_i, where m_i is the public strategic premium
derived from the downstream game (``calibration.structural_premia``) and
xi_i ~ LogNormal(-sigma^2/2, sigma) is a private multiplicative shock (mean 1).

Because the sale is committed to the winner and externalities are public, all
standard-auction theory applies in terms of w_i (Jehiel-Moldovanu 2000; see the
paper, Lemma 1).  Mechanisms that rank bidders by a monotone score and charge
the *critical bid* are dominant-strategy incentive compatible, hence bids equal
w_i and the simulation applies the equilibrium bid functions b_i*(w_i) = w_i.

Outcome codes: 0 = no sale (asset retained, status quo), 1 = D wins, 2 = E wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

NO_SALE, D_WINS, E_WINS = 0, 1, 2


# ----------------------------------------------------------------------------
# Valuation distributions
# ----------------------------------------------------------------------------
def lognormal_unit_mean(sigma: float):
    """Frozen scipy lognormal with mean 1 and log-sd ``sigma``."""
    return stats.lognorm(s=sigma, scale=np.exp(-0.5 * sigma ** 2))


def draw_shocks(rng: np.random.Generator, n: int, sigma: float, corr: float = 0.0):
    """Mean-one lognormal shocks (xi_D, xi_E).

    ``corr`` is the correlation of the underlying normals (0 = independent private values).
    Values remain *private* (each bidder knows its own w_i), so truthful bidding in the
    handicap auctions stays weakly dominant for any joint distribution; only the Myerson
    benchmark, which assumes independence, is not applicable when ``corr > 0``.
    """
    z = rng.standard_normal((2, n))
    z2 = corr * z[0] + np.sqrt(max(1.0 - corr ** 2, 0.0)) * z[1]
    xi_D = np.exp(sigma * z[0] - 0.5 * sigma ** 2)
    xi_E = np.exp(sigma * z2 - 0.5 * sigma ** 2)
    return xi_D, xi_E


# ----------------------------------------------------------------------------
# Mechanisms
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Result:
    winner: np.ndarray   # outcome codes
    price: np.ndarray    # payment of the winner (0 if no sale)


def _finish(wD, wE, qD, qE, d_wins_both, priceD_both, priceE_both, r) -> Result:
    both = qD & qE
    only_d = qD & ~qE
    only_e = qE & ~qD
    winner = np.zeros(wD.shape, dtype=np.int8)
    price = np.zeros(wD.shape)
    winner[only_d] = D_WINS
    price[only_d] = r
    winner[only_e] = E_WINS
    price[only_e] = r
    dw = both & d_wins_both
    ew = both & ~d_wins_both
    winner[dw] = D_WINS
    price[dw] = np.maximum(r, priceD_both[dw])
    winner[ew] = E_WINS
    price[ew] = np.maximum(r, priceE_both[ew])
    return Result(winner, price)


def handicap_multiplicative(wD, wE, beta: float, r: float) -> Result:
    """Second-price/English auction in which D's bid counts only ``beta`` times.

    beta = 1: standard auction (Vickrey/English).  D wins iff beta*b_D > b_E and
    pays the critical bid b_E / beta; E wins otherwise and pays beta * b_D.
    """
    qD, qE = wD >= r, wE >= r
    d_wins = beta * wD > wE
    return _finish(wD, wE, qD, qE, d_wins, wE / beta, beta * wD, r)


def handicap_additive(wD, wE, delta: float, r: float) -> Result:
    """Second-price auction with an additive bid credit ``delta`` for E."""
    qD, qE = wD >= r, wE >= r
    d_wins = wD - delta > wE
    return _finish(wD, wE, qD, qE, d_wins, wE + delta, wD - delta, r)


def set_aside(wD, wE, r: float) -> Result:
    """D is barred; E is the only bidder and buys at the reserve price r if w_E >= r."""
    winner = np.where(wE >= r, E_WINS, NO_SALE).astype(np.int8)
    price = np.where(wE >= r, r, 0.0)
    return Result(winner, price)


class VirtualValue:
    """psi(w) = w - (1 - F(w)) / f(w) for w = m * xi, xi ~ lognormal (mean 1)."""

    def __init__(self, m: float, sigma: float, grid_pts: int = 4001):
        self.m, self.sigma = m, sigma
        self.dist = lognormal_unit_mean(sigma)
        lo, hi = self.dist.ppf(1e-6), self.dist.ppf(1 - 1e-6)
        self.xgrid = np.linspace(lo, hi, grid_pts) * m
        self.psi_grid = self(self.xgrid)

    def __call__(self, w):
        x = np.asarray(w, dtype=float) / self.m
        surv = self.dist.sf(x)
        dens = self.dist.pdf(x) / self.m
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(w, dtype=float) - surv / dens

    def is_regular(self) -> bool:
        return bool(np.all(np.diff(self.psi_grid) > 0))

    def inverse(self, y):
        """psi^{-1}(y); psi is strictly increasing on the grid (checked)."""
        return np.interp(y, self.psi_grid, self.xgrid, left=self.xgrid[0], right=np.inf)


def myerson(wD, wE, psiD: VirtualValue, psiE: VirtualValue, kappa: float,
            dCS_over: float, r: float) -> Result:
    """Myerson-optimal allocation for planner weights (R + kappa * CS).

    Scores (relative to the status quo, which coincides with 'D wins'):
        s_D = psi_D(w_D),   s_E = psi_E(w_E) + kappa * dCS,
    where dCS = CS(E wins) - CS(D wins).  The asset is sold iff max score >= r
    (retention value r); the winner pays the critical value.
    """
    sD = psiD(wD)
    sE = psiE(wE) + kappa * dCS_over
    best = np.maximum(sD, sE)
    sell = best >= r
    d_wins = sell & (sD >= sE)
    e_wins = sell & (sE > sD)
    winner = np.zeros(wD.shape, dtype=np.int8)
    winner[d_wins] = D_WINS
    winner[e_wins] = E_WINS
    price = np.zeros(wD.shape)
    # critical value: lowest w with psi_i(w) + shift_i >= max(r, s_j)
    thr_D = np.maximum(r, sE)
    price[d_wins] = psiD.inverse(thr_D[d_wins])
    thr_E = np.maximum(r, sD) - kappa * dCS_over
    price[e_wins] = psiE.inverse(thr_E[e_wins])
    return Result(winner, price)


# ----------------------------------------------------------------------------
# Welfare accounting
# ----------------------------------------------------------------------------
def summarise(res: Result, cs_D: float, cs_E: float, kappa: float, r0: float,
              pD: dict | None = None) -> dict:
    """Expected outcomes and Monte-Carlo standard errors (all USD million, PV).

    ``r0`` is the State's retention value of the asset.  Fiscal proceeds are
    F = payment if sold, r0 if retained; ``net`` = F - r0 is the gain over
    retention.  Planner's welfare: W = F + kappa * CS (bidders' surplus has
    weight zero), with CS(no sale) = CS(D wins) = status quo.
    """
    n = res.winner.size
    cs = np.where(res.winner == E_WINS, cs_E, cs_D)
    fiscal = np.where(res.winner == NO_SALE, r0, res.price)
    net = fiscal - r0
    welf = fiscal + kappa * cs
    out = {
        "net": net.mean(), "net_se": net.std(ddof=1) / np.sqrt(n),
        "cs_gain": (cs - cs_D).mean(), "cs_gain_se": cs.std(ddof=1) / np.sqrt(n),
        "welfare": welf.mean(), "welfare_se": welf.std(ddof=1) / np.sqrt(n),
        "p_D": (res.winner == D_WINS).mean(),
        "p_E": (res.winner == E_WINS).mean(),
        "p_nosale": (res.winner == NO_SALE).mean(),
    }
    if pD is not None:
        hhi = np.where(res.winner == E_WINS, pD["hhi_E"], pD["hhi_D"])
        out["hhi"] = hhi.mean()
        ts = np.where(res.winner == E_WINS, pD["total_surplus_E"], pD["total_surplus_D"])
        out["total_surplus_gain"] = (ts - pD["total_surplus_D"]).mean()
    return out
