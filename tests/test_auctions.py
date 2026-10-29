import numpy as np
import pytest

from privauction import auctions as A
from privauction import informed as I
from privauction.experiments import Params, build, expected_standard_price, draw, mechanism


def uniform_draws(n, uD, uE, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, uD, n), rng.uniform(0, uE, n)


def test_standard_auction_uniform_closed_form():
    """Draft's baseline: E[second-highest] = k(alpha/2 - alpha^2/6) (correct, Section 5 of the draft)."""
    for alpha in [0.3, 0.6, 0.9]:
        wD, wE = uniform_draws(2_000_000, 1.0, alpha)
        res = A.handicap_multiplicative(wD, wE, 1.0, 0.0)
        assert res.price.mean() == pytest.approx(alpha / 2 - alpha ** 2 / 6, abs=2e-3)


def test_optimal_additive_handicap_uniform_matches_myerson():
    """With U[0,u_i] valuations, psi_i = 2w - u_i, so the optimal credit is (u_D - u_E)/2."""
    uD, uE = 1.0, 0.6
    wD, wE = uniform_draws(1_000_000, uD, uE, seed=3)
    deltas = np.linspace(0, 0.6, 61)
    revs = [A.handicap_additive(wD, wE, d, 0.0).price.mean() for d in deltas]
    d_star = deltas[int(np.argmax(revs))]
    assert d_star == pytest.approx((uD - uE) / 2, abs=0.02)
    # revenue at the optimum equals the expected positive virtual surplus (Myerson 1981)
    virt = np.maximum.reduce([2 * wD - uD, 2 * wE - uE, np.zeros_like(wD)])
    # no reserve in the handicap auction => compare with max(psiD, psiE) when winner is always selected
    virt_committed = np.maximum(2 * wD - uD, 2 * wE - uE)
    assert max(revs) == pytest.approx(virt_committed.mean(), abs=3e-3)
    assert max(revs) > A.handicap_additive(wD, wE, 0.0, 0.0).price.mean()


def test_handicap_is_dominant_strategy_truthful():
    """Given any rival bid, bidding the true w maximises i's payoff w - price (if win)."""
    rng = np.random.default_rng(1)
    beta = 0.55
    for _ in range(300):
        wD, wE = rng.uniform(0.2, 2.0, 2)
        bids = np.linspace(0.05, 3.0, 600)
        # D's payoff as a function of its own bid, E bids truthfully
        payoffs = []
        for b in bids:
            res = A.handicap_multiplicative(np.array([b]), np.array([wE]), beta, 0.0)
            payoffs.append((wD - res.price[0]) if res.winner[0] == A.D_WINS else 0.0)
        u_truth = ((wD - A.handicap_multiplicative(np.array([wD]), np.array([wE]), beta, 0.0).price[0])
                   if A.handicap_multiplicative(np.array([wD]), np.array([wE]), beta, 0.0).winner[0] == A.D_WINS else 0.0)
        assert max(payoffs) <= u_truth + 1e-9
        # same for E
        payoffs = []
        for b in bids:
            res = A.handicap_multiplicative(np.array([wD]), np.array([b]), beta, 0.0)
            payoffs.append((wE - res.price[0]) if res.winner[0] == A.E_WINS else 0.0)
        rt = A.handicap_multiplicative(np.array([wD]), np.array([wE]), beta, 0.0)
        u_truth = (wE - rt.price[0]) if rt.winner[0] == A.E_WINS else 0.0
        assert max(payoffs) <= u_truth + 1e-9


def test_set_aside_is_limit_of_preference_with_reserve():
    st = build(Params())
    xD, xE = draw(st.p, 20_000, 1)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    a = A.handicap_multiplicative(wD, wE, 1e-9, st.r0)
    b = A.set_aside(wD, wE, st.r0)
    # same winner except when E fails to qualify (then D, if qualified, wins at r in `a`)
    qual = wE >= st.r0
    assert np.all(a.winner[qual] == b.winner[qual])
    assert np.allclose(a.price[qual], b.price[qual])


def test_myerson_payment_identity_and_optimality():
    """E[payment] = E[virtual value of the winner] and Myerson weakly dominates the other rules."""
    st = build(Params(sigma=0.3))
    p = st.p
    xD, xE = draw(p, 400_000, 11)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    psi = (A.VirtualValue(st.prem["m_D"], p.sigma), A.VirtualValue(st.prem["m_E"], p.sigma))
    assert psi[0].is_regular() and psi[1].is_regular()
    kappa = 0.0
    res = A.myerson(wD, wE, psi[0], psi[1], kappa, st.prem["dCS"], st.r0)
    sold = res.winner != A.NO_SALE
    vv = np.where(res.winner == A.D_WINS, psi[0](wD), psi[1](wE))
    assert res.price[sold].sum() / len(wD) == pytest.approx(vv[sold].sum() / len(wD), rel=0.01)
    for k in [0.0, 0.77, 1.0]:
        Wm = A.summarise(A.myerson(wD, wE, psi[0], psi[1], k, st.prem["dCS"], st.r0),
                         st.prem["cs_D"], st.prem["cs_E"], k, st.r0)["welfare"]
        for name, beta in [("standard", 1.0), ("preference", 0.5), ("preference", 0.8), ("setaside", 1.0)]:
            r = mechanism(st, name, wD, wE, beta=beta)
            W = A.summarise(r, st.prem["cs_D"], st.prem["cs_E"], k, st.r0)["welfare"]
            assert Wm >= W - 0.5      # US$ million, MC tolerance


def test_monte_carlo_matches_quadrature():
    st = build(Params(sigma=0.3))
    xD, xE = draw(st.p, 1_000_000, 5)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    mc = np.minimum(wD, wE).mean()
    assert mc == pytest.approx(expected_standard_price(st.prem["m_D"], st.prem["m_E"], st.p.sigma), rel=0.003)


def test_insider_first_price_benchmark():
    rng = np.random.default_rng(2)
    X, s = 100.0, 60.0
    sim = I.simulate_first_price(X, s, 1_000_000, rng)
    assert sim["revenue"] == pytest.approx(I.revenue_first_price(X, s), abs=0.1)
    assert sim["insider"] == pytest.approx(I.insider_rent(s), abs=0.1)
    assert sim["outsider"] == pytest.approx(0.0, abs=0.1)
    assert I.revenue_full_information(X, s) - I.revenue_first_price(X, s) == pytest.approx(s / 6)


def test_correlated_shocks_have_requested_correlation_and_unit_mean():
    rng = np.random.default_rng(5)
    xD, xE = A.draw_shocks(rng, 2_000_000, 0.3, corr=0.6)
    assert xD.mean() == pytest.approx(1.0, abs=0.003) and xE.mean() == pytest.approx(1.0, abs=0.003)
    assert np.corrcoef(np.log(xD), np.log(xE))[0, 1] == pytest.approx(0.6, abs=0.003)
    # independence is the default
    xD, xE = A.draw_shocks(np.random.default_rng(5), 2_000_000, 0.3)
    assert abs(np.corrcoef(np.log(xD), np.log(xE))[0, 1]) < 0.003


def test_uniform_myerson_welfare_regimes_proposition4c():
    """E[max{psi_D, psi_E + theta}] for U[0,1] x U[0,alpha]: three geometric regimes in theta."""
    rng = np.random.default_rng(0)
    n = 2_000_000
    for alpha, theta in [(0.5, 0.0), (0.5, 0.4), (0.5, 0.5), (0.5, 0.9), (0.5, 1.2), (0.5, 1.7), (0.8, 0.1), (0.8, 0.6)]:
        v, w = rng.uniform(0, 1, n), rng.uniform(0, alpha, n)
        mc = np.maximum(2 * v - 1, 2 * w - alpha + theta).mean()
        if theta <= 1 - alpha:
            f = (alpha ** 2 + 3 * theta ** 2 + 6 * theta + 3) / 12
        elif theta <= 1 + alpha:
            f = (alpha ** 3 - 3 * alpha ** 2 * theta + 3 * alpha ** 2 + 3 * alpha * theta ** 2 + 18 * alpha * theta
                 + 3 * alpha - theta ** 3 + 3 * theta ** 2 - 3 * theta + 1) / (24 * alpha)
        else:
            f = theta
        assert mc == pytest.approx(f, abs=4e-3)
