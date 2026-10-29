"""Benchmark for an informational advantage (Engelbrecht-Wiggans, Milgrom, Weber 1983).

Common value V = X + Y with X public and Y ~ U[0, s] known only to the insider.
In the sealed first-price auction:
  * the insider bids  b(y) = X + E[Y | Y <= y] = X + y/2,
  * the outsider (zero expected profit) draws its bid from the distribution of b(Y),
  * expected revenue R_FP = X + s/3, insider rent s/6, outsider rent 0.
Under public disclosure of Y (symmetric information) Bertrand competition gives
price = V, so R_FI = X + s/2.  The revenue loss from the informational advantage
is therefore s/6 (one third of the expected surplus above X).
"""
from __future__ import annotations

import numpy as np


def revenue_first_price(X: float, s: float) -> float:
    return X + s / 3.0


def revenue_full_information(X: float, s: float) -> float:
    return X + s / 2.0


def insider_rent(s: float) -> float:
    return s / 6.0


def implied_s(R: float, X: float) -> float:
    """Dispersion s that rationalises revenue R under the first-price insider benchmark."""
    return 3.0 * (R - X)


def simulate_first_price(X: float, s: float, n: int, rng: np.random.Generator) -> dict:
    """Monte-Carlo check of the equilibrium (insider bids X + y/2, outsider mimics)."""
    y = rng.uniform(0.0, s, n)
    b_ins = X + y / 2.0
    b_out = X + rng.uniform(0.0, s, n) / 2.0          # same distribution as the insider's bid
    win_ins = b_ins >= b_out
    price = np.where(win_ins, b_ins, b_out)
    V = X + y
    ins_profit = np.where(win_ins, V - b_ins, 0.0)
    out_profit = np.where(~win_ins, V - b_out, 0.0)
    return dict(revenue=price.mean(), insider=ins_profit.mean(), outsider=out_profit.mean())
