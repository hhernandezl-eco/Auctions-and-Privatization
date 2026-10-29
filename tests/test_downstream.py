import numpy as np
import pytest

from privauction import calibration as C
from privauction.downstream import brute_force_cournot, cournot_capacity


def test_active_set_matches_best_response_iteration():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = rng.integers(2, 6)
        k = rng.uniform(0.02, 0.6, n)
        k[rng.integers(n)] = np.inf
        a = rng.uniform(1.0, 2.0)
        fr = rng.uniform(0, 0.1)
        eq = cournot_capacity(a, k, fringe=fr)
        bf = brute_force_cournot(a, k, fringe=fr, iters=3000)
        assert np.allclose(eq.q, bf, atol=1e-7)


def test_closed_form_duopoly_premia():
    """Proposition 1 (duopoly, a = 1, k_E = x, slot large enough for E to be unconstrained)."""
    for x in [0.0, 0.05, 0.1, 0.2, 0.3, 0.33]:
        base_D = cournot_capacity(1.0, np.array([np.inf, x]))
        E_wins = cournot_capacity(1.0, np.array([np.inf, 1.0 / 3.0 + 1e-9]))
        w_D = base_D.profit[0] - E_wins.profit[0]
        w_E = E_wins.profit[1] - base_D.profit[1]
        assert w_D == pytest.approx((1 - x) ** 2 / 4 - 1 / 9, abs=1e-6)
        assert w_E == pytest.approx(1 / 9 - x * (1 - x) / 2, abs=1e-6)
        assert w_D >= w_E - 1e-12                              # preemption dominates
        # consumer surplus and total surplus are higher when E wins
        assert E_wins.cs > base_D.cs - 1e-12
        W_D = base_D.cs + base_D.profit.sum()
        W_E = E_wins.cs + E_wins.profit.sum()
        assert W_E >= W_D - 1e-12
        assert E_wins.hhi <= base_D.hhi + 1e-9


def test_threshold_one_third():
    # w_D - w_E = (1 - x^2)/4 - 2/9 > 0 iff x < 1/3
    for x in [0.1, 0.3, 0.3333]:
        assert (1 - x ** 2) / 4 - 2 / 9 > 0
    assert (1 - 0.34 ** 2) / 4 - 2 / 9 < 0


def test_calibration_reproduces_shares_and_draft_table3():
    s = C.shares("mobile_lines")
    m = C.build_market(s, "Claro")
    assert np.allclose(m.base.shares, m.base.q / m.base.Q)
    assert m.base.Q == pytest.approx(1.0)
    # the draft's Table 3 (HHI mobile 3325.05; mobile Internet 3864.01) is reproduced from MinTIC
    assert C.hhi(s) == pytest.approx(3325.05, rel=0.01)
    assert C.hhi(C.shares("mobile_internet_accesses")) == pytest.approx(3864.01, rel=0.01)
    assert C.imss(s, 1.0) == pytest.approx(0.41, abs=0.005)
    assert C.imss(s, 0.5) == pytest.approx(0.455, abs=0.005)
    # Claro exceeds the dominance threshold
    assert s["Claro"] > C.imss(s, 1.0)


def test_extra_capacity_is_slack_for_dominant_firm():
    s = C.shares("mobile_lines")
    m = C.build_market(s, "Claro")
    out_D, out_E, e = C.winner_outcomes(m, "Tigo", 0.10)
    assert np.allclose(out_D.q, m.base.q)
    assert out_E.Q > m.base.Q


def test_three_bidder_premia_consistent_with_two_bidder_case():
    """Pairwise premia reduce to the two-bidder premia and reveal mutual denial among challengers."""
    s = C.shares("mobile_lines")
    m = C.build_market(s, "Claro")
    df = C.three_bidder_premia(m, ("Movistar", "Tigo"), 0.10)
    for ch in ["Movistar", "Tigo"]:
        p = C.structural_premia(m, ch, 0.10)
        assert df.loc["Claro", ch] == pytest.approx(p["m_D"])
        assert df.loc[ch, "Claro"] == pytest.approx(p["m_E"])
    # a constrained challenger loses more when the other challenger wins than when Claro wins
    assert df.loc["Movistar", "Tigo"] > df.loc["Movistar", "Claro"]
    assert df.loc["Tigo", "Movistar"] > df.loc["Tigo", "Claro"]
