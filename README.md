# Privatising under dominance: bid preferences in telecommunications auctions with downstream competition

Replication package for the paper. It reproduces every number, table and figure from raw public data.

## What the paper does

1. A downstream capacity-constrained Cournot model in which dominance is capacity slack; the dominant firm's willingness to pay is pure *preemption* (Prop. 1).
2. The welfare-weighted Myerson mechanism for general valuation distributions (Prop. 2), handicap auctions (Prop. 3), and closed forms for the uniform case (Prop. 4: a multiplicative preference recovers exactly 3/4 of the optimal gain).
3. Extensions: two challengers (pairwise premia, Jehiel & Moldovanu 2000) and a benchmark for informational advantage (Engelbrecht-Wiggans, Milgrom & Weber 1983; Prop. 5).
4. Calibration to MinTIC 4Q-2020 data and a Monte Carlo of 10,000 auctions under three counterfactuals: standard auction, optimal bid preference, set-aside of the dominant firm (+ Myerson benchmark), with sensitivity analysis.

## Repository layout

```
src/privauction/  downstream.py  capacity-Cournot equilibrium
                  calibration.py data -> shares, HHI, IMSS, structural premia
                  auctions.py    mechanisms (handicap, set-aside, Myerson), welfare accounting
                  informed.py    insider benchmark (Prop. 5)
                  experiments.py scenarios, optimisation of beta, sensitivity grids
scripts/          run_all.py 
tests/            pytest suite (closed forms, brute-force checks, incentive compatibility, Prop. 4 regimes)
data/raw/         MinTIC 4Q-2020 figures and COLTEL-2006 facts, each with its source
results/          CSV outputs of the last run
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                                  # 15 tests, ~1 min
python scripts/run_all.py                               # full run, ~10 min; --quick for a smoke test
```

`run_all.py` writes the LaTeX tables, figures and `numbers.tex`. Random seeds are fixed (`seed_eval=20201`, `seed_opt=7`), so results are reproducible bit-for-bit on the same library versions (tested with Python 3.9, numpy 2.0, scipy 1.13, pandas 2.3).

## Data

* MinTIC, *Boletín Trimestral de las TIC, 4T-2020* (April 2021): <https://colombiatic.mintic.gov.co/679/articles-172261_archivo_pdf.pdf>.
* COLTEL 2006: Telefónica press release (7 April 2006), Infobae (2006-04-08) and Razón Pública, "Otra capitalización para Telefónica".
* COP/USD: 2020 average market rate ≈ 3,693 (approximate; see `src/privauction/calibration.py`); 2,313 for 2006.

## Assumptions to be aware of

`σ` (dispersion of private values) is not identified by market data; margin `μ`, asset size `ΔK`, horizon and discount rate are assumptions. All are varied in the sensitivity analysis (Table 5 of the paper).
