"""Experiment drivers: scenarios, optimisation of the handicap, sensitivity grids."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import stats

from . import auctions as A
from . import calibration as C


# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Params:
    dominant: str = "Claro"
    challenger: str = "Movistar"
    segment: str = "mobile_lines"      # mobile_lines | mobile_revenue | mobile_internet
    mu: float = 0.35                   # variable-profit margin on revenue
    rho: float = 0.10                  # discount rate
    horizon: int = 20                  # licence/asset horizon (years)
    delta_k: float = 0.10              # capacity of the asset, share of pre-auction output
    sigma: float = 0.30                # log-sd of private valuation shocks
    corr: float = 0.0                  # correlation of the private shocks (0 = independent)
    lam: float = 0.30                  # marginal cost of public funds
    reserve_ratio: float = 0.625       # retention value / E[standard price]  (2006: 533.6/853.6)

    @property
    def kappa(self) -> float:
        """Weight of consumer surplus relative to public revenue, 1/(1+lambda)."""
        return 1.0 / (1.0 + self.lam)


def _segment_shares(segment: str) -> pd.Series:
    if segment == "mobile_lines":
        return C.shares("mobile_lines")
    if segment == "mobile_internet":
        return C.shares("mobile_internet_accesses")
    if segment == "mobile_revenue":
        return C.combined_revenue_shares()
    raise ValueError(segment)


@dataclass
class Setup:
    p: Params
    market: C.Market
    prem: dict
    r0: float                       # State's retention value (= reserve in the baseline)


def expected_standard_price(m_D: float, m_E: float, sigma: float, n: int = 1500,
                            corr: float = 0.0) -> float:
    """E[price] of the standard auction without reserve (quadrature if independent, else seeded MC)."""
    if corr != 0.0:
        xD, xE = A.draw_shocks(np.random.default_rng(99), 4_000_000, sigma, corr)
        return float(np.minimum(m_D * xD, m_E * xE).mean())
    u = (np.arange(n) + 0.5) / n
    z = stats.norm.ppf(u)
    xi = np.exp(sigma * z - 0.5 * sigma ** 2)
    wD = m_D * xi[:, None]
    wE = m_E * xi[None, :]
    return float(np.minimum(wD, wE).mean())


def build(p: Params) -> Setup:
    s = _segment_shares(p.segment)
    market = C.build_market(s, p.dominant, mu=p.mu, rho=p.rho, horizon=p.horizon)
    prem = C.structural_premia(market, p.challenger, p.delta_k)
    price0 = expected_standard_price(prem["m_D"], prem["m_E"], p.sigma, corr=p.corr)
    return Setup(p=p, market=market, prem=prem, r0=p.reserve_ratio * price0)


def draw(p: Params, n: int, seed: int):
    rng = np.random.default_rng(seed)
    return A.draw_shocks(rng, n, p.sigma, p.corr)


# ----------------------------------------------------------------------------
def _summ(st: Setup, res: A.Result, kappa=None) -> dict:
    k = st.p.kappa if kappa is None else kappa
    return A.summarise(res, st.prem["cs_D"], st.prem["cs_E"], k, st.r0, st.prem)


def mechanism(st: Setup, name: str, wD, wE, beta: float = 1.0, reserve: float | None = None,
              psi=None, kappa=None) -> A.Result:
    rho = st.r0 if reserve is None else reserve
    if name == "standard":
        return A.handicap_multiplicative(wD, wE, 1.0, rho)
    if name == "preference":
        return A.handicap_multiplicative(wD, wE, beta, rho)
    if name == "setaside":
        return A.set_aside(wD, wE, rho)
    if name == "myerson":
        k = st.p.kappa if kappa is None else kappa
        pD, pE = psi
        return A.myerson(wD, wE, pD, pE, k, st.prem["dCS"], st.r0)
    raise ValueError(name)


BETA_GRID = np.round(np.arange(0.05, 1.0001, 0.01), 2)


def optimise_beta(st: Setup, wD, wE, kappa=None, reserve=None) -> tuple[float, np.ndarray]:
    """Grid search of the multiplicative handicap maximising expected welfare (CRN)."""
    k = st.p.kappa if kappa is None else kappa
    vals = np.array([_summ(st, A.handicap_multiplicative(
        wD, wE, b, st.r0 if reserve is None else reserve), k)["welfare"] for b in BETA_GRID])
    return float(BETA_GRID[int(np.argmax(vals))]), vals


def scenarios(st: Setup, n_eval: int = 10_000, n_opt: int = 200_000,
              seed_eval: int = 20201, seed_opt: int = 7, kappa=None) -> pd.DataFrame:
    """Baseline comparison: standard, preference (beta*), set-aside, Myerson benchmark."""
    p = st.p
    k = p.kappa if kappa is None else kappa
    xD, xE = draw(p, n_opt, seed_opt)
    beta_star, _ = optimise_beta(st, st.prem["m_D"] * xD, st.prem["m_E"] * xE, kappa=k)
    xD, xE = draw(p, n_eval, seed_eval)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    psi = (A.VirtualValue(st.prem["m_D"], p.sigma), A.VirtualValue(st.prem["m_E"], p.sigma))
    rows = []
    spec = [("S1 Standard auction", "standard", 1.0),
            ("S2 Bid preference (beta*)", "preference", beta_star),
            ("S3 Set-aside (D barred)", "setaside", 1.0)]
    if p.corr == 0.0:       # the Myerson formula assumes independent types
        spec.append(("Myerson benchmark", "myerson", np.nan))
    for label, name, b in spec:
        res = mechanism(st, name, wD, wE, beta=b, psi=psi, kappa=k)
        d = _summ(st, res, kappa=k)
        d.update(scenario=label, beta=b)
        rows.append(d)
    return pd.DataFrame(rows).set_index("scenario")


def welfare_curve(st: Setup, kappas, n: int = 200_000, seed: int = 7) -> pd.DataFrame:
    xD, xE = draw(st.p, n, seed)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    out = {"beta": BETA_GRID}
    for k in kappas:
        _, vals = optimise_beta(st, wD, wE, kappa=k)
        out[f"W_kappa={k:.2f}"] = vals - vals[-1]          # relative to beta = 1
    cols = []
    for b in BETA_GRID:
        s = _summ(st, A.handicap_multiplicative(wD, wE, b, st.r0))
        cols.append((s["net"], s["cs_gain"], s["p_E"]))
    arr = np.array(cols)
    out["net"], out["cs_gain"], out["p_E"] = arr[:, 0], arr[:, 1], arr[:, 2]
    return pd.DataFrame(out)


def sensitivity(base: Params, grid: dict, n_opt: int = 200_000, n_eval: int = 10_000,
                seed_opt: int = 7, seed_eval: int = 20201) -> pd.DataFrame:
    """Sensitivity of beta* and welfare gains, with the *same* protocol as ``scenarios``.

    beta* is chosen on ``n_opt`` draws (seed ``seed_opt``) and every outcome is evaluated on the
    same ``n_eval`` draws (seed ``seed_eval``) used in the main experiment, so the baseline row
    coincides exactly with the baseline scenarios table.  ``grid`` maps a Params field name to a
    list of values; all combinations are run.
    """
    import itertools
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        p = replace(base, **dict(zip(keys, combo)))
        st = build(p)
        xD, xE = draw(p, n_opt, seed_opt)
        b, _ = optimise_beta(st, st.prem["m_D"] * xD, st.prem["m_E"] * xE)
        xD, xE = draw(p, n_eval, seed_eval)
        wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
        std = _summ(st, mechanism(st, "standard", wD, wE))
        pref = _summ(st, mechanism(st, "preference", wD, wE, beta=b))
        sa = _summ(st, mechanism(st, "setaside", wD, wE))
        row = dict(zip(keys, combo))
        row.update(beta_star=b, m_D=st.prem["m_D"], m_E=st.prem["m_E"], dCS=st.prem["dCS"],
                   dHHI=st.prem["dHHI"], r0=st.r0,
                   W_gain_pref=pref["welfare"] - std["welfare"],
                   W_gain_setaside=sa["welfare"] - std["welfare"],
                   net_std=std["net"], net_pref=pref["net"], net_sa=sa["net"],
                   pE_std=std["p_E"], pE_pref=pref["p_E"],
                   hhi_std=std["hhi"], hhi_pref=pref["hhi"], hhi_sa=sa["hhi"])
        rows.append(row)
    return pd.DataFrame(rows)


def optimal_reserve_table(st: Setup, n: int = 150_000, seed: int = 7,
                          rho_mult=np.linspace(1.0, 2.0, 21)) -> pd.DataFrame:
    """Re-optimise the policy reserve for each scenario (retention value fixed at r0)."""
    xD, xE = draw(st.p, n, seed)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    rows = []
    for name in ["standard", "preference", "setaside"]:
        best = None
        for mult in rho_mult:
            rho = st.r0 * mult
            if name == "preference":
                b, vals = optimise_beta(st, wD, wE, reserve=rho)
                w = vals.max()
            else:
                b = 1.0
                w = _summ(st, mechanism(st, name, wD, wE, reserve=rho))["welfare"]
            if best is None or w > best[0]:
                best = (w, rho, b)
        w, rho, b = best
        res = mechanism(st, name, wD, wE, beta=b, reserve=rho)
        d = _summ(st, res)
        d.update(scenario=name, reserve_opt=rho, reserve_mult=rho / st.r0, beta=b)
        rows.append(d)
    return pd.DataFrame(rows).set_index("scenario")
