"""Numerical audit of the closed-form results in the original draft (SSRN 3951613, Section 5).

Draft setting: vA ~ U[0,k], vB ~ U[0,alpha k]; bidder B is favoured by a factor beta.  The draft states
    pi_E(beta) = alpha k /(6 beta) * (3 - 2 alpha/beta + alpha)       [p. 22]
    beta*      = 4 alpha / (3 + alpha)                              [p. 23]
    "pi_E(beta*) beats the unrestricted auction iff alpha < 0.8912"  [p. 23]

Findings (k = 1; symbolic derivation in the paper's appendix, numerical confirmation here):
  1. pi_E(beta) is the *correct* expected revenue of the dominant-strategy auction in which A's bid
     counts beta times (so the draft's formula and beta* are right for this special case);
  2. pi_E(beta*) = (3 + alpha)^2 / 48, hence the gain over the standard auction is 3 (1 - alpha)^2 / 16 > 0
     for EVERY alpha < 1: the threshold 0.8912 is an algebra slip (the inequality 11 alpha^2/3 - 10 alpha + 6 > 0
     does not follow from the displayed one);
  3. the Myerson-optimal (additive-credit) auction yields (3 + alpha^2)/12, a gain of (1 - alpha)^2 / 4:
     the multiplicative preference recovers exactly 75% of the optimal discrimination gain for every alpha.
Outputs: results/tables/audit_original.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def draft_pi(alpha, beta):
    return alpha / (6 * beta) * (3 - 2 * alpha / beta + alpha)


def baseline(alpha):
    return alpha / 2 - alpha ** 2 / 6


def true_revenue(alpha, beta, n=3000):
    """E[price]: A ~ U[0,1] counts beta times, B ~ U[0,alpha]; winner pays the critical bid."""
    u = (np.arange(n) + 0.5) / n
    vA = u[:, None]
    vB = alpha * u[None, :]
    a_wins = beta * vA > vB
    return np.where(a_wins, vB / beta, beta * vA).mean()


def draft_claim_inequality(alpha):
    """The draft's reduced inequality 11 alpha^2/3 - 10 alpha + 6 > 0 (true iff alpha < 0.8912)."""
    return 11 * alpha ** 2 / 3 - 10 * alpha + 6 > 0


def main():
    rows = []
    for alpha in [0.3, 0.5, 0.7, 0.85, 0.89, 0.92, 0.95, 0.99]:
        b = 4 * alpha / (3 + alpha)
        gain_draft_formula = draft_pi(alpha, b) - baseline(alpha)
        rows.append(dict(
            alpha=alpha,
            beta_star=b,
            draft_pi_at_beta_star=draft_pi(alpha, b),
            true_revenue_at_beta_star=true_revenue(alpha, b),
            closed_form_rev=(3 + alpha) ** 2 / 48,
            gain_true=true_revenue(alpha, b) - true_revenue(alpha, 1.0),
            closed_form_gain=3 * (1 - alpha) ** 2 / 16,
            draft_inequality_holds=draft_claim_inequality(alpha),
            gain_is_positive=gain_draft_formula > 0,
            myerson_revenue=(3 + alpha ** 2) / 12,
            myerson_gain=(1 - alpha) ** 2 / 4,
            share_of_myerson_gain=(3 * (1 - alpha) ** 2 / 16) / ((1 - alpha) ** 2 / 4),
        ))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "audit_original.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(df.round(4).to_string(index=False))
    # draft's own formula == true revenue; claimed threshold is wrong
    assert np.allclose(df.draft_pi_at_beta_star, df.true_revenue_at_beta_star, atol=2e-4)
    assert (df.gain_is_positive).all()
    assert (df.draft_inequality_holds == (df.alpha < 0.8912)).all()


if __name__ == "__main__":
    main()
