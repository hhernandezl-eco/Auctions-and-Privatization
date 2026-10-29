"""Calibration of the structural model to MinTIC data (4Q-2020).

Pipeline
--------
1. Observed shares s_i (listed operators) + atomistic fringe -> HHI, IMSS.
2. Pre-auction equilibrium = Claro (D) unconstrained, every other listed
   operator capacity-constrained at its observed output (``Section 4``).
   With Q0 = 1 this pins down the demand intercept (see ``build_market``).
3. Money scale: variable profit of the industry = mu * observed revenue, and the
   asset's value is the present value (annuity) of the change in annual profit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .downstream import Outcome, cournot_capacity

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw"

COP_PER_USD_2020 = 3693.36   # average TRM 2020 (Banco de la Republica); see data/README.md


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_mintic() -> pd.DataFrame:
    return pd.read_csv(DATA / "mintic_q4_2020.csv")


def shares(segment: str, df: pd.DataFrame | None = None,
           total: float | None = None) -> pd.Series:
    """Shares of listed operators in ``segment``; residual is the atomistic fringe.

    ``total`` defaults to the TOTAL row if present, else the sum of listed
    operators (no fringe).
    """
    df = load_mintic() if df is None else df
    d = df[df.segment == segment]
    listed = d[d.operator != "TOTAL"].set_index("operator")["value"].astype(float)
    if total is None:
        tot = d[d.operator == "TOTAL"]["value"]
        total = float(tot.iloc[0]) if len(tot) else float(listed.sum())
    return listed / total


def combined_revenue_shares(df: pd.DataFrame | None = None) -> pd.Series:
    """Mobile revenue shares (voice + data), listed operators only."""
    df = load_mintic() if df is None else df
    v = df[df.segment.isin(["mobile_voice_revenue", "mobile_data_revenue"])]
    s = v.groupby("operator")["value"].sum().astype(float)
    return s / s.sum()


def annual_mobile_revenue_usd_m(df: pd.DataFrame | None = None) -> float:
    """Annualised (4 x Q4-2020) mobile voice+data revenue, USD million."""
    df = load_mintic() if df is None else df
    v = df[df.segment.isin(["mobile_voice_revenue", "mobile_data_revenue"])]
    cop_bn_q = float(v["value"].sum())
    return 4.0 * cop_bn_q * 1e3 / COP_PER_USD_2020   # COP bn -> COP mn -> USD mn


def hhi(s: pd.Series | np.ndarray) -> float:
    return float(10000.0 * np.sum(np.asarray(s, dtype=float) ** 2))


def imss(s: pd.Series, delta: float) -> float:
    """Melnik-Shy-Stenbacka (2008) dominance threshold S^d = 1/2 [1 - delta (s1^2 - s2^2)].

    A firm is dominant if its share exceeds the threshold.
    """
    top = np.sort(np.asarray(s, dtype=float))[::-1]
    return 0.5 * (1.0 - delta * (top[0] ** 2 - top[1] ** 2))


# ----------------------------------------------------------------------------
# Structural calibration
# ----------------------------------------------------------------------------
@dataclass
class Market:
    names: list[str]
    k0: np.ndarray            # pre-auction capacities (inf for the dominant firm)
    a: float                  # demand intercept (normalised, Q0 = 1)
    fringe: float
    dom: int                  # index of the dominant incumbent
    base: Outcome
    scale: float = 1.0        # money per unit of normalised profit/CS (USD mn per year)
    gamma: float = 1.0        # annuity factor
    meta: dict = field(default_factory=dict)


def build_market(s_listed: pd.Series, dominant: str, fringe_share: float | None = None,
                 mu: float = 0.35, revenue_usd_m: float | None = None,
                 rho: float = 0.10, horizon: int = 20) -> Market:
    """Pre-auction equilibrium reproducing observed shares.

    The dominant firm is the only unconstrained one: in the capacity-Cournot
    equilibrium its output equals the margin P, so q_D = s_D and
    a_eff = Q0 + P = 1 + s_D with Q0 = 1 (listed + fringe).
    """
    names = list(s_listed.index)
    s = s_listed.to_numpy(float)
    fr = (1.0 - s.sum()) if fringe_share is None else fringe_share
    fr = max(fr, 0.0)
    # renormalise so that listed + fringe = 1
    s = s / (s.sum() + fr) if (s.sum() + fr) > 0 else s
    fr = fr / (s_listed.sum() + fr) if (s_listed.sum() + fr) > 0 else 0.0
    d = names.index(dominant)
    k0 = s.copy()
    k0[d] = np.inf
    a_eff = 1.0 - fr + s[d]            # strategic output (1-fr) + margin P = s_D
    a = a_eff + fr
    base = cournot_capacity(a, k0, fringe=fr)
    assert abs(base.Q - 1.0) < 1e-9 and np.allclose(base.q, s), "calibration failed"
    revenue_usd_m = annual_mobile_revenue_usd_m() if revenue_usd_m is None else revenue_usd_m
    profit_usd_m = mu * revenue_usd_m                 # observed industry variable profit
    scale = profit_usd_m / float(base.profit.sum())    # USD mn per normalised unit
    gamma = (1.0 - (1.0 + rho) ** (-horizon)) / rho
    return Market(names=names, k0=k0, a=a, fringe=fr, dom=d, base=base, scale=scale,
                  gamma=gamma, meta=dict(mu=mu, rho=rho, horizon=horizon,
                                         revenue_usd_m=revenue_usd_m,
                                         implied_elasticity=s[d] / mu))


def winner_outcomes(m: Market, challenger: str, delta_k: float):
    """Downstream outcomes when D wins, E wins (and the status quo)."""
    e = m.names.index(challenger)
    k_e = m.k0.copy()
    k_e[e] = k_e[e] + delta_k
    k_d = m.k0.copy()
    k_d[m.dom] = np.inf                 # extra capacity is slack for D
    out_E = cournot_capacity(m.a, k_e, fringe=m.fringe)
    out_D = cournot_capacity(m.a, k_d, fringe=m.fringe)
    return out_D, out_E, e


def structural_premia(m: Market, challenger: str, delta_k: float) -> dict:
    """Willingness-to-pay premia (public components) in USD million (PV).

    w_D = PV[pi_D(D wins) - pi_D(E wins)]      (pure preemption)
    w_E = PV[pi_E(E wins) - pi_E(D wins)]      (productive gain + denial)
    dCS = PV[CS(E wins) - CS(D wins)], dHHI = HHI(E wins) - HHI(D wins).
    """
    out_D, out_E, e = winner_outcomes(m, challenger, delta_k)
    f = m.scale * m.gamma
    return dict(
        m_D=f * (out_D.profit[m.dom] - out_E.profit[m.dom]),
        m_E=f * (out_E.profit[e] - out_D.profit[e]),
        cs_D=f * out_D.cs, cs_E=f * out_E.cs, dCS=f * (out_E.cs - out_D.cs),
        hhi_D=out_D.hhi, hhi_E=out_E.hhi, dHHI=out_E.hhi - out_D.hhi,
        Q_D=out_D.Q, Q_E=out_E.Q,
        total_surplus_D=f * (out_D.cs + out_D.profit.sum()),
        total_surplus_E=f * (out_E.cs + out_E.profit.sum()),
    )


def three_bidder_premia(m: Market, challengers: tuple[str, str], delta_k: float) -> pd.DataFrame:
    """Pairwise willingness-to-pay premia with one dominant firm and two constrained challengers.

    Entry (i, j) is bidder i's gain from winning rather than losing to j,
    ``PV[pi_i(i wins) - pi_i(j wins)]`` in USD million, i.e. the externality structure of
    Jehiel and Moldovanu (2000) when all three operators bid.  Also returns the equilibrium
    margin and the consumer-surplus effect of each winner.
    """
    bidders = [m.names[m.dom], *challengers]
    f = m.scale * m.gamma
    outcome = {}
    for w in bidders:
        k = m.k0.copy()
        if w == m.names[m.dom]:
            k[m.dom] = np.inf                       # slack for the dominant firm
        else:
            k[m.names.index(w)] += delta_k
        outcome[w] = cournot_capacity(m.a, k, fringe=m.fringe)
    idx = {b: m.names.index(b) for b in bidders}
    rows = []
    for i in bidders:
        row = {"Bidder": i}
        for j in bidders:
            row[j] = np.nan if i == j else f * (outcome[i].profit[idx[i]] - outcome[j].profit[idx[i]])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Bidder")
    df.attrs["margin"] = {w: outcome[w].price for w in bidders}
    df.attrs["cs"] = {w: f * outcome[w].cs for w in bidders}
    df.attrs["hhi"] = {w: outcome[w].hhi for w in bidders}
    return df
