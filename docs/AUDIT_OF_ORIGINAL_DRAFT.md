# Audit of the original draft (Documentos FCE-CID N.° 122, SSRN 3951613) and what changed

This file records, for each deficiency identified by the reviewing professor **and** each additional problem found while
re-checking primary sources, what was wrong and where it is fixed in the new paper (`paper/main.tex`).
Everything numerical below can be re-derived with `python scripts/audit_original.py` and `pytest`.

## A. Factual errors found in primary sources (not in the professor's list)

| # | Original draft | Verified fact | Source | Fix |
|---|---|---|---|---|
| 1 | Reserve price of the 2006 COLTEL auction = "533.6 million" (treated as US$ million) | COP 533,577 **million** (≈ US$231 m), set on 31 Mar 2006 from Citigroup Markets and McKinsey valuations | Infobae 2006-04-08; Razón Pública | `data/raw/coltel_2006.csv`, Sec. 3 |
| 2 | Winning bid by Telefónica = US$ 622 m | COP 853,577 million (≈ US$368 m) | Infobae 2006-04-08; Razón Pública | idem |
| 3 | The rival of Telefónica in the auction was Telmex/Slim, who "knew" ... | The rival bidder was **CANTV** (Venezuela). Telmex only had a **2005 MOU** (US$350 m investment, ~US$90 m in Telmex shares) that was suspended after the Comptroller's warning | Infobae; Razón Pública | Sec. 3; Sec. 6 (information treated as a stand-alone benchmark, *not* asserted for 2006) |
| 4 | Slim's earlier offer quoted as both US$350 m (p. 17) and US$322 m (p. 26) | US$350 m (MOU) | Razón Pública | removed |
| 5 | Auction described as "ascending first-price" | Ascending multi-round auction with a minimum price | Infobae | Sec. 3 |
| 6 | "By 1987 the theory had hit a ceiling (McAfee & McMillan, 1987)" | The 1980s–90s saw affiliated values (Milgrom & Weber 1982), FCC multi-object auctions, etc. | professor; literature | statement removed; Sec. 2 rewritten |

## B. The professor's critique, item by item

### A. Literature
* Myerson (1981), Maskin & Riley (2000), Milgrom & Weber (1982), Jehiel & Moldovanu (2000), Engelbrecht-Wiggans et al. (1983), Klemperer (1998) etc. are now cited and used: Props. 2–5. The bibliography has 58 entries, 50 of them scholarly articles/books (up from 3 peer-reviewed articles); Colombian laws, CONPES documents and press reports are cited only as primary sources for institutional facts.
* **Please verify every bibliographic detail against the publisher records before submission** (they were written from memory of standard references; none was auto-generated from a database).

### B. Mathematics
* *Generality / Myerson regularity*: Prop. 2 gives the welfare-weighted Myerson mechanism for general continuous regular `F_i` (not only uniform), with virtual values `ψ_i(v)=v-(1-F_i)/f_i`; the original uniform result is Prop. 4.
* *Inconsistent use of IPV for two different phenomena*: split. Market dominance → downstream capacity-Cournot game with endogenous willingness to pay (Prop. 1, after Gilbert & Newbery 1982 and Jehiel & Moldovanu 2000). Information → Engelbrecht-Wiggans–Milgrom–Weber benchmark (Prop. 5).
* *Notation `E(v_A)`, `E(v_B)`*: eliminated. Expected proceeds are written `R(β)` / `F`.
* *Welfare weights*: the planner maximises `F + κ·CS`, `κ = 1/(1+λ)` with `λ` the marginal cost of public funds (the professor's `W = π_E + λ·CS` is the same objective after a rescaling; `λ` there is the CS weight, which is `κ` here).
* **Important positive finding**: the closed form `π_E(β)` and `β* = 4α/(3+α)` of the original draft are **correct** for the uniform case with a multiplicative preference (verified by exact integration and Monte Carlo; `results/tables/audit_original.csv`). What is wrong is the claim "the preference helps iff α < 0.8912": the gain is `3(1-α)²k/16 > 0` for **every** α < 1 (the displayed reduction `11α²/3 − 10α + 6 > 0` does not follow). The optimal-discrimination (Myerson) gain is `(1-α)²k/4`, so the multiplicative preference recovers exactly 75 % of it (Prop. 4).

### C. Simulation (Section 6 and the R annex)
1. *Not an auction simulation*: confirmed from the annex: `VA`, `VB` are generated and never used; the loop evaluates the closed-form revenue at 100 draws of α. → Replaced by a Monte Carlo of N = 10,000 **auctions** per scenario: draw private shocks, form willingness to pay, apply equilibrium bids (`b*_i(w_i)=w_i`, dominant strategies), record winner/price (`src/privauction/auctions.py`, `scripts/run_all.py`).
2. *`maxB` vs `maxb`*: R is case-sensitive, so `VB <- runif(100, 533.6, maxb)` fails. Moot now (R code removed; the replication is in Python).
3. *Mis-scaled overlaid densities (`par(new=TRUE)`)*: replaced by figures with shared axes (ECDF, welfare curves, frontier).
4. *Reserve price inconsistency*: the closed form assumed `r = 0` while the inputs started at 533.6. → Reserve/retention value `r_0` enters every mechanism and the welfare accounting (`r_0 = 0.625 × expected standard price`, the 2006 reserve-to-price ratio).

### Roadmap
* Step 1 (downstream interaction, endogenous valuations, Myerson with welfare weights): Sec. 4–5.
* Step 2 (2006 as an informational problem): Sec. 6 benchmark; we stress the documentary record does not establish an informational advantage.
* Step 3 (structural calibration, N = 10,000, reserve > 0, three counterfactuals): Sec. 7–8. HHI/IMSS are now *connected* to the model: shares fix capacities, which fix `m_D`, `m_E`, ΔCS, ΔHHI.
* Step 4 (condense Sections 2–3 into ≤ 2 pages; English; Proposition/Proof format): Sec. 3 and the paper as a whole.

## C. What is still the author's call
* `σ` (dispersion of private values) is **not identified**; results are reported over [0.1, 0.6]. Bid data would identify it.
* `μ`, `Δ`, horizon, discount rate are assumptions with sensitivity analysis.
* Use of Movistar/Tigo as the "challenger" (no WOM in the Dec-2020 MinTIC data; WOM entered in 2021).
* Journal choice and formatting (the manuscript uses the Elsevier `elsarticle` class).

## D. Referee round (Journal of Industry, Competition and Trade)
The referee report was addressed in `paper/response_to_referee.pdf`. Main changes: the audit of the 2021 draft was removed from the paper (kept only here); the informational benchmark became an extension (Sec. 8.3); microfoundation of the downstream stage (Sec. 4.1); two-challenger discussion with pairwise premia (Sec. 8.2, Table 7); justification and sensitivity of the reserve anchor (Sec. 6); sensitivity protocol harmonised with the main experiment; post-2020 market developments; official press release for 2006 facts.

### Second referee round
Prop. 4(c) now states the three geometric regimes of the uniform Myerson benchmark (theta <= (1-alpha)k; up to (1+alpha)k; beyond), verified symbolically and by Monte Carlo (`tests/test_auctions.py`). A sensitivity panel on the correlation of the private shocks was added (Table 5): correlation *strengthens* the case for a preference (challenger wins 3 % of standard auctions at correlation 0.75; beta* = 0.54). Declarations completed; Razon Publica author identified; README/CITATION need the final repository URL (`epourl` in `paper/main.tex`).
