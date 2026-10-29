"""Reproduce every number, table and figure of the paper.

    python scripts/run_all.py            # full run (about 5-10 minutes)
    python scripts/run_all.py --quick    # smaller Monte-Carlo, for smoke tests

Writes
    results/*.csv                      machine-readable results
    paper/tables/*.tex                 LaTeX tables (booktabs)
    paper/figures/*.pdf                figures
    paper/numbers.tex                  \\newcommand macros used in the text
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from privauction import auctions as A
from privauction import calibration as C
from privauction import experiments as X
from privauction import informed as I

RES = ROOT / "results"
TAB = ROOT / "paper" / "tables"
FIG = ROOT / "paper" / "figures"
for d in (RES, TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

# colour-blind-safe (Okabe-Ito) and distinct line styles so figures survive greyscale printing
OI = dict(black="#000000", orange="#E69F00", sky="#56B4E9", green="#009E73",
          yellow="#F0E442", blue="#0072B2", verm="#D55E00", pink="#CC79A7")
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.4,
    "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})

NUM: dict[str, str] = {}


def macro(name: str, value, fmt: str = "{:.1f}") -> None:
    NUM[name] = fmt.format(value) if not isinstance(value, str) else value


def tex_table(df: pd.DataFrame, path: Path, fmt: dict | None = None, colfmt: str | None = None,
              index: bool = False, header: list[str] | None = None) -> None:
    fmt = fmt or {}
    cols = list(df.columns)
    if index:
        df = df.reset_index()
        cols = list(df.columns)
    hdr = header or cols
    colfmt = colfmt or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{tabular}{" + colfmt + "}", r"\toprule", " & ".join(hdr) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = fmt.get(c)
            if isinstance(v, str) or f is None and not isinstance(v, (float, np.floating)):
                cells.append(str(v))
            else:
                cells.append(f.format(v) if f else "{:.2f}".format(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def savefig(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def table_market_structure() -> None:
    df = C.load_mintic()
    rows = []
    for label, s in [("Mobile lines", C.shares("mobile_lines", df)),
                     ("Mobile Internet accesses", C.shares("mobile_internet_accesses", df)),
                     ("Mobile revenue (voice+data)", C.combined_revenue_shares(df)),
                     ("Fixed Internet accesses", C.shares("fixed_internet_accesses", df))]:
        top = s.sort_values(ascending=False)
        rows.append(dict(
            Segment=label, Leader=top.index[0].replace(" (Comcel)", ""),
            s1=100 * top.iloc[0], s2=100 * top.iloc[1], HHI=C.hhi(s),
            d1=100 * C.imss(s, 1.0), d05=100 * C.imss(s, 0.5),
            Dominant="Yes" if top.iloc[0] > C.imss(s, 1.0) else "No"))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "market_structure.csv", index=False)
    tex_table(out, TAB / "market_structure.tex",
              fmt=dict(s1="{:.1f}", s2="{:.1f}", HHI="{:,.0f}", d1="{:.1f}", d05="{:.1f}"),
              colfmt="llrrrrrc",
              header=["Segment", "Leader", "$s_1$ (\\%)", "$s_2$ (\\%)", "HHI",
                      "$S^d$, $\\delta{=}1$", "$S^d$, $\\delta{=}0.5$", "$s_1>S^d$"])
    ml = out.iloc[0]
    macro("hhiMobile", ml.HHI, "{:,.0f}")
    macro("claroShare", ml.s1, "{:.1f}")
    macro("claroThreshold", ml.d1, "{:.1f}")
    macro("annualRevUSD", C.annual_mobile_revenue_usd_m() / 1000.0, "{:.2f}")


def table_premia(base: X.Params) -> dict:
    rows, keep = [], {}
    for ch in ["Movistar", "Tigo"]:
        st = X.build(replace(base, challenger=ch))
        p = st.prem
        keep[ch] = st
        rows.append(dict(
            Challenger=ch,
            sE=100 * st.market.base.shares[st.market.names.index(ch)],
            mD=p["m_D"], mE=p["m_E"], dCS=p["dCS"], hhiD=p["hhi_D"], hhiE=p["hhi_E"],
            dHHI=p["dHHI"], r0=st.r0))
    out = pd.DataFrame(rows)
    out.to_csv(RES / "premia.csv", index=False)
    tex_table(out, TAB / "premia.tex",
              fmt=dict(sE="{:.1f}", mD="{:,.0f}", mE="{:,.0f}", dCS="{:,.0f}", hhiD="{:,.0f}",
                       hhiE="{:,.0f}", dHHI="{:,.0f}", r0="{:,.0f}"),
              colfmt="lrrrrrrrr",
              header=["Challenger $E$", "$s_E$ (\\%)", "$m_D$", "$m_E$", "$\\Delta CS$",
                      "HHI$^{D}$", "HHI$^{E}$", "$\\Delta$HHI", "$r_0$"])
    m = keep["Movistar"]
    macro("mD", m.prem["m_D"], "{:,.0f}")
    macro("mE", m.prem["m_E"], "{:,.0f}")
    macro("dCS", m.prem["dCS"], "{:,.0f}")
    macro("dHHI", -m.prem["dHHI"], "{:,.0f}")
    macro("rZero", m.r0, "{:,.0f}")
    macro("mDoverE", m.prem["m_D"] / m.prem["m_E"], "{:.2f}")
    macro("impliedElasticity", m.market.meta["implied_elasticity"], "{:.2f}")
    macro("scaleFactor", m.market.scale * m.market.gamma, "{:,.0f}")
    macro("gammaFactor", m.market.gamma, "{:.2f}")
    macro("kappaBase", base.kappa, "{:.2f}")
    return keep


def table_scenarios(st: X.Setup, name: str, n_eval: int, n_opt: int) -> pd.DataFrame:
    sc = X.scenarios(st, n_eval=n_eval, n_opt=n_opt)
    sc.to_csv(RES / f"scenarios_{name}.csv")
    std = sc.loc["S1 Standard auction"]
    out = sc.copy()
    out["dW"] = out["welfare"] - std["welfare"]
    out["dW_se"] = np.sqrt(out["welfare_se"] ** 2 + std["welfare_se"] ** 2)
    pretty = pd.DataFrame({
        "Scenario": out.index,
        "beta": [("--" if (np.isnan(b) or n.startswith("S3") or n.startswith("Myer")) else f"{b:.2f}")
                 for n, b in zip(out.index, out["beta"])],
        "Net proceeds": [f"{v:,.0f} ({s:.0f})" for v, s in zip(out["net"], out["net_se"])],
        "Pr(D wins)": out["p_D"], "Pr(E wins)": out["p_E"], "Pr(no sale)": out["p_nosale"],
        "CS gain": [f"{v:,.0f} ({s:.0f})" for v, s in zip(out["cs_gain"], out["cs_gain_se"])],
        "HHI": out["hhi"],
        "dW": [f"{v:,.0f} ({s:.0f})" for v, s in zip(out["dW"], out["dW_se"])],
    })
    tex_table(pretty, TAB / f"scenarios_{name}.tex",
              fmt={"Pr(D wins)": "{:.2f}", "Pr(E wins)": "{:.2f}", "Pr(no sale)": "{:.2f}", "HHI": "{:,.0f}"},
              colfmt="lcrrrrrrr",
              header=["Scenario", "$\\beta$", "Net proceeds", "Pr($D$)", "Pr($E$)", "Pr(none)",
                      "CS gain", "E[HHI]", "$\\Delta W$ vs.\\ S1"])
    return sc


def fig_premia_and_densities(st: X.Setup, keep: dict) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    names = ["Movistar", "Tigo"]
    xs = np.arange(len(names))
    w = 0.26
    ax[0].bar(xs - w, [keep[n].prem["m_D"] for n in names], w, color=OI["verm"], label="$m_D$ (Claro, preemption)")
    ax[0].bar(xs, [keep[n].prem["m_E"] for n in names], w, color=OI["blue"], label="$m_E$ (challenger)")
    ax[0].bar(xs + w, [keep[n].prem["dCS"] for n in names], w, color=OI["green"], hatch="///",
              edgecolor="white", label="$\\Delta CS$ (E wins minus D wins)")
    ax[0].set_xticks(xs, [f"E = {n}" for n in names])
    ax[0].set_ylabel("US\\$ million (PV)")
    ax[0].set_ylim(0, 1550)
    ax[0].legend(fontsize=6.5, loc="upper left", ncol=1)
    ax[0].set_title("(a) Structural premia", fontsize=9, loc="left")
    # densities at the baseline
    p = st.p
    grid = np.linspace(0.01, 2.6, 600)
    dist = A.lognormal_unit_mean(p.sigma)
    for lab, m, col, ls in [("Claro ($D$)", st.prem["m_D"], OI["verm"], "-"),
                            (f"{p.challenger} ($E$)", st.prem["m_E"], OI["blue"], "--")]:
        ax[1].plot(grid * st.prem["m_D"], dist.pdf(grid * st.prem["m_D"] / m) / m, color=col, ls=ls, label=lab)
    ax[1].axvline(st.r0, color="black", lw=0.8, ls=":")
    ax[1].text(st.r0, ax[1].get_ylim()[1] * 0.92, " retention value $r_0$", fontsize=7, ha="left", va="top")
    ax[1].set_xlabel("willingness to pay $w_i$ (US\\$ million)")
    ax[1].set_ylabel("density")
    ax[1].legend(fontsize=7)
    ax[1].set_title(f"(b) Distributions ($\\sigma={p.sigma}$)", fontsize=9, loc="left")
    fig.tight_layout()
    savefig(fig, "fig_premia_densities")


def fig_welfare_vs_beta(st: X.Setup, n: int) -> None:
    kaps = [0.0, 0.5, st.p.kappa, 1.0]
    cur = X.welfare_curve(st, kaps, n=n)
    cur.to_csv(RES / "welfare_curve.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    styles = [("-", OI["black"]), ("--", OI["sky"]), ("-", OI["blue"]), (":", OI["verm"])]
    for k, (ls, col) in zip(kaps, styles):
        y = cur[f"W_kappa={k:.2f}"]
        ax[0].plot(cur.beta, y, ls=ls, color=col, label=f"$\\kappa={k:.2f}$")
        i = int(np.argmax(y.values))
        ax[0].plot(cur.beta[i], y.iloc[i], "o", color=col, ms=4)
    ax[0].axhline(0, color="black", lw=0.5)
    ax[0].set_xlabel("handicap factor $\\beta$ (1 = standard)")
    ax[0].set_ylabel("welfare gain vs. $\\beta=1$ (US\\$ m)")
    ax[0].invert_xaxis()
    ax[0].legend(fontsize=7)
    ax[0].set_title("(a) Planner's welfare", fontsize=9, loc="left")
    ax[1].plot(cur.beta, cur.net - cur.net.iloc[-1], color=OI["verm"], label="net proceeds")
    ax[1].plot(cur.beta, cur.cs_gain - cur.cs_gain.iloc[-1], color=OI["green"], ls="--", label="consumer surplus")
    ax[1].axhline(0, color="black", lw=0.5)
    ax[1].set_xlabel("handicap factor $\\beta$ (1 = standard)")
    ax[1].set_ylabel("change vs. $\\beta=1$ (US\\$ m)")
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Components", fontsize=9, loc="left")
    fig.tight_layout()
    savefig(fig, "fig_welfare_beta")


def fig_frontier(st: X.Setup, sc: pd.DataFrame, n: int) -> None:
    cur = pd.read_csv(RES / "welfare_curve.csv")
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.plot(cur.net, cur.cs_gain, color="#999999", lw=1.0, label="preference path ($\\beta$ from 1 to 0.05)")
    marks = [("S1 Standard auction", "o", OI["black"], "S1"), ("S2 Bid preference (beta*)", "s", OI["blue"], "S2 ($\\beta^*$)"),
             ("S3 Set-aside (D barred)", "^", OI["verm"], "S3"), ("Myerson benchmark", "D", OI["green"], "Myerson")]
    for key, mk, col, lab in marks:
        r = sc.loc[key]
        ax.plot(r["net"], r["cs_gain"], mk, color=col, ms=6, label=lab)
    ax.set_xlabel("net proceeds over retention value (US\\$ m)")
    ax.set_ylabel("expected consumer-surplus gain (US\\$ m)")
    ax.set_ylim(100, 1000)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout()
    savefig(fig, "fig_frontier")


def fig_ecdf(st: X.Setup, sc: pd.DataFrame, n_eval: int) -> None:
    p = st.p
    xD, xE = X.draw(p, n_eval, 20201)
    wD, wE = st.prem["m_D"] * xD, st.prem["m_E"] * xE
    b = float(sc.loc["S2 Bid preference (beta*)", "beta"])
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    series = [("S1 standard", X.mechanism(st, "standard", wD, wE), OI["black"], "-"),
              (f"S2 preference ($\\beta^*={b:.2f}$)", X.mechanism(st, "preference", wD, wE, beta=b), OI["blue"], "--"),
              ("S3 set-aside", X.mechanism(st, "setaside", wD, wE), OI["verm"], ":")]
    for lab, res, col, ls in series:
        fiscal = np.where(res.winner == A.NO_SALE, st.r0, res.price) - st.r0
        x = np.sort(fiscal)
        ax[0].plot(x, np.arange(1, x.size + 1) / x.size, color=col, ls=ls, label=lab)
        cs = np.where(res.winner == A.E_WINS, st.prem["cs_E"], st.prem["cs_D"]) - st.prem["cs_D"]
        ax[1].bar(["S1", "S2", "S3"][[s[0][:2] for s in series].index(lab[:2])], cs.mean(), color=col, alpha=0.9,
                  hatch={"-": "", "--": "//", ":": ".."}[ls], edgecolor="white")
    ax[0].set_xlabel("net proceeds per auction (US\\$ m)")
    ax[0].set_ylabel("empirical CDF ($N=10{,}000$)")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[0].set_title("(a) Distribution of proceeds", fontsize=9, loc="left")
    ax[1].set_ylabel("mean CS gain (US\\$ m)")
    ax[1].set_title("(b) Consumer surplus", fontsize=9, loc="left")
    fig.tight_layout()
    savefig(fig, "fig_ecdf")


def sensitivity_block(base: X.Params, n_opt: int, n_eval: int) -> pd.DataFrame:
    blocks = [
        ("sigma", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60], "Private dispersion $\\sigma$"),
        ("lam", [0.0, 0.30, 1.0, 3.0, 1e9], "Cost of public funds $\\lambda$"),
        ("delta_k", [0.05, 0.10, 0.15, 0.20], "Asset size $\\Delta K$"),
        ("mu", [0.20, 0.30, 0.35, 0.40, 0.50], "Margin $\\mu$"),
        ("reserve_ratio", [0.30, 0.50, 0.625, 0.80], "Reserve ratio $r_0/\\bar p$"),
        ("horizon", [10, 15, 20], "Horizon $T$ (years)"),
        ("challenger", ["Movistar", "Tigo"], "Challenger"),
        ("corr", [0.0, 0.25, 0.50, 0.75], "Correlation $r_\\xi$ of the private shocks"),
        ("segment", ["mobile_lines", "mobile_internet", "mobile_revenue"], "Market definition"),
    ]
    frames = []
    for key, vals, label in blocks:
        df = X.sensitivity(base, {key: vals}, n_opt=n_opt, n_eval=n_eval)
        df.insert(0, "param", label)
        df["value"] = df[key].astype(str)
        frames.append(df.drop(columns=[key]))
    return pd.concat(frames, ignore_index=True)


def table_sensitivity(sens: pd.DataFrame) -> None:
    sens.to_csv(RES / "sensitivity_oneway.csv", index=False)

    def fv(param: str, v: str) -> str:
        if param.startswith("Cost"):
            x = float(v)
            if x < 1e8:
                return f"$\\lambda{{=}}{x:g}$ ($\\kappa{{=}}{1/(1+x):.2f}$)"
            return "$\\lambda\\to\\infty$ ($\\kappa{=}0$)"
        if param.startswith("Challenger"):
            return v
        if param.startswith("Market"):
            return {"mobile_lines": "mobile lines", "mobile_internet": "mobile Internet accesses",
                    "mobile_revenue": "mobile revenue"}[v]
        x = float(v)
        return f"{x:g}" if x != int(x) else f"{int(x)}"

    cols = ["beta_star", "pE_std", "pE_pref", "W_gain_pref", "W_gain_setaside", "net_std", "net_pref",
            "hhi_std", "hhi_pref"]
    f = ["{:.2f}", "{:.2f}", "{:.2f}", "{:,.0f}", "{:,.0f}", "{:,.0f}", "{:,.0f}", "{:,.0f}", "{:,.0f}"]
    lines = ["\\begin{tabular}{lrrrrrrrrr}", "\\toprule",
             "Value & $\\beta^*$ & Pr($E$) S1 & Pr($E$) S2 & $\\Delta W$ S2 & $\\Delta W$ S3 "
             "& Net S1 & Net S2 & HHI S1 & HHI S2 \\\\"]
    for lab, g in sens.groupby("param", sort=False):
        lines.append("\\midrule")
        lines.append("\\multicolumn{10}{l}{\\textit{" + lab + "}} \\\\")
        for _, r in g.iterrows():
            cells = [fv(lab, r["value"])] + [fm.format(r[c]) for c, fm in zip(cols, f)]
            lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "sensitivity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fig_heatmap(base: X.Params, n_opt: int, n_eval: int) -> pd.DataFrame:
    sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    lams = [0.0, 0.3, 1.0, 3.0, 1e9]
    df = X.sensitivity(base, {"sigma": sigmas, "lam": lams}, n_opt=n_opt, n_eval=n_eval)
    df.to_csv(RES / "sensitivity_sigma_lambda.csv", index=False)
    piv = df.pivot(index="lam", columns="sigma", values="beta_star").loc[lams]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    im = ax.imshow(piv.values, cmap="Blues_r", vmin=0.0, vmax=1.0, aspect="auto", origin="lower")
    ax.set_xticks(range(len(sigmas)), [f"{s:.1f}" for s in sigmas])
    ax.set_yticks(range(len(lams)), [("$\\kappa=0$" if l > 1e8 else f"$\\kappa={1/(1+l):.2f}$") for l in lams])
    for i in range(len(lams)):
        for j in range(len(sigmas)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v < 0.5 else "black")
    ax.set_xlabel("private dispersion $\\sigma$")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.set_label("$\\beta^*$", fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig_heatmap_beta")
    return df


def table_reserve(st: X.Setup, name: str) -> None:
    t = X.optimal_reserve_table(st)
    t.to_csv(RES / f"optimal_reserve_{name}.csv")
    pretty = pd.DataFrame({
        "Scenario": ["S1 Standard", "S2 Preference", "S3 Set-aside"],
        "rho": [f"{m:.2f}" for m in t["reserve_mult"]],
        "beta": [f"{b:.2f}" if s in ("preference", "standard") else "--" for s, b in zip(t.index, t["beta"])],
        "Net": [f"{v:,.0f}" for v in t["net"]], "pE": t["p_E"], "pNo": t["p_nosale"],
        "CS": [f"{v:,.0f}" for v in t["cs_gain"]],
        "W": [f"{v:,.0f}" for v in t["welfare"] - st.prem["cs_D"] * st.p.kappa - st.r0],
    })
    tex_table(pretty, TAB / f"optimal_reserve_{name}.tex", fmt={"pE": "{:.2f}", "pNo": "{:.2f}"},
              colfmt="lcccccrr",
              header=["Scenario", "$\\rho/r_0$", "$\\beta$", "Net proceeds", "Pr($E$)", "Pr(none)", "CS gain", "$W-W_0$"])


def table_three_bidder(base: X.Params) -> None:
    """Pairwise willingness-to-pay premia when Claro, Movistar and Tigo all bid (extension)."""
    st = X.build(base)
    df = C.three_bidder_premia(st.market, ("Movistar", "Tigo"), base.delta_k)
    df.to_csv(RES / "three_bidder_premia.csv")
    cols = ["Claro", "Movistar", "Tigo"]
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Bidder $i$ & if Claro wins & if Movistar wins & if Tigo wins \\\\", "\\midrule"]
    for i in cols:
        cells = [i] + ["--" if i == j else f"{df.loc[i, j]:,.0f}" for j in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "three_bidder.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    macro("pairMvsC", df.loc["Movistar", "Claro"], "{:,.0f}")
    macro("pairMvsT", df.loc["Movistar", "Tigo"], "{:,.0f}")
    macro("pairTvsC", df.loc["Tigo", "Claro"], "{:,.0f}")
    macro("pairTvsM", df.loc["Tigo", "Movistar"], "{:,.0f}")
    macro("pairCvsE", df.loc["Claro", "Movistar"], "{:,.0f}")


def table_informed() -> None:
    # COLTEL 2006 (COP million): reserve 533,577 ; winning bid 853,577 ; about 2,313 COP/USD (Infobae)
    X0, R = 533_577.0, 853_577.0
    s = I.implied_s(R, X0)
    fx = 2313.0
    rows = [("Reserve (State valuation, $X$)", X0, X0 / fx / 1e3 * 1e3),
            ("Realised price ($R_{FP}$)", R, R / fx),
            ("Implied dispersion $s=3(R-X)$", s, s / fx),
            ("Insider rent $s/6$", I.insider_rent(s), I.insider_rent(s) / fx),
            ("Counterfactual price under disclosure $X+s/2$", I.revenue_full_information(X0, s),
             I.revenue_full_information(X0, s) / fx)]
    df = pd.DataFrame(rows, columns=["Item", "COP million", "USD million"])
    df.to_csv(RES / "coltel_informed_benchmark.csv", index=False)
    tex_table(df, TAB / "informed.tex", fmt={"COP million": "{:,.0f}", "USD million": "{:,.0f}"}, colfmt="lrr")
    macro("coltelLossUSD", I.insider_rent(s) / fx, "{:,.0f}")
    macro("coltelLossPct", 100 * I.insider_rent(s) / R, "{:.0f}")
    macro("coltelS", s / fx, "{:,.0f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_eval, n_opt = (10_000, 200_000) if not args.quick else (10_000, 40_000)

    base = X.Params()
    table_market_structure()
    keep = table_premia(base)
    st = keep["Movistar"]

    sc = table_scenarios(st, "movistar", n_eval, n_opt)
    sc_t = table_scenarios(keep["Tigo"], "tigo", n_eval, n_opt)
    for nm, key in [("S", "S2 Bid preference (beta*)")]:
        macro("betaStar", sc.loc[key, "beta"], "{:.2f}")
        macro("betaStarTigo", sc_t.loc[key, "beta"], "{:.2f}")
    std, pref, sa, my = (sc.loc[k] for k in ["S1 Standard auction", "S2 Bid preference (beta*)",
                                              "S3 Set-aside (D barred)", "Myerson benchmark"])
    macro("pDStd", 100 * std.p_D, "{:.0f}")
    macro("pDPref", 100 * pref.p_D, "{:.0f}")
    macro("netDropPref", std.net - pref.net, "{:,.0f}")
    macro("csGainPrefVsStd", pref.cs_gain - std.cs_gain, "{:,.0f}")
    macro("pEStd", 100 * std.p_E, "{:.0f}")
    macro("pEPref", 100 * pref.p_E, "{:.0f}")
    macro("pESA", 100 * sa.p_E, "{:.0f}")
    macro("pNoSA", 100 * sa.p_nosale, "{:.0f}")
    macro("netStd", std.net, "{:,.0f}")
    macro("netPref", pref.net, "{:,.0f}")
    macro("netSA", sa.net, "{:,.0f}")
    macro("netMy", my.net, "{:,.0f}")
    macro("csStd", std.cs_gain, "{:,.0f}")
    macro("csPref", pref.cs_gain, "{:,.0f}")
    macro("csSA", sa.cs_gain, "{:,.0f}")
    macro("csMy", my.cs_gain, "{:,.0f}")
    macro("dWPref", pref.welfare - std.welfare, "{:,.0f}")
    macro("dWSA", sa.welfare - std.welfare, "{:,.0f}")
    macro("dWMy", my.welfare - std.welfare, "{:,.0f}")
    macro("hhiStd", std.hhi, "{:,.0f}")
    macro("hhiPref", pref.hhi, "{:,.0f}")
    macro("hhiSA", sa.hhi, "{:,.0f}")
    macro("netLossPrefPct", 100 * (1 - pref.net / std.net), "{:.0f}")
    macro("shareMyerson", 100 * (pref.welfare - std.welfare) / (my.welfare - std.welfare), "{:.0f}")
    macro("nEval", n_eval, "{:,.0f}")

    fig_premia_and_densities(st, keep)
    fig_welfare_vs_beta(st, n_opt)
    fig_frontier(st, sc, n_opt)
    fig_ecdf(st, sc, n_eval)

    # revenue-only planner (kappa = 0): preference raises revenue (Myerson 1981 / Maskin-Riley 2000)
    st0 = X.build(replace(base, lam=1e9))
    sc0 = X.scenarios(st0, n_eval=n_eval, n_opt=n_opt)
    sc0.to_csv(RES / "scenarios_movistar_kappa0.csv")
    macro("betaStarRev", sc0.loc["S2 Bid preference (beta*)", "beta"], "{:.2f}")
    macro("revGainPct", 100 * (sc0.loc["S2 Bid preference (beta*)", "net"] / sc0.loc["S1 Standard auction", "net"] - 1), "{:.0f}")
    macro("revGainAbs", sc0.loc["S2 Bid preference (beta*)", "net"] - sc0.loc["S1 Standard auction", "net"], "{:,.0f}")

    sens = sensitivity_block(base, n_opt, n_eval)
    table_sensitivity(sens)
    fig_heatmap(base, n_opt, n_eval)
    table_reserve(st, "movistar")
    table_informed()
    table_three_bidder(base)

    # regularity of the virtual values at the baseline
    pD, pE = A.VirtualValue(st.prem["m_D"], base.sigma), A.VirtualValue(st.prem["m_E"], base.sigma)
    macro("regular", "yes" if (pD.is_regular() and pE.is_regular()) else "no")

    # sensitivity summary numbers for the text
    macro("betaMin", sens.beta_star.min(), "{:.2f}")
    macro("betaMax", sens.beta_star.max(), "{:.2f}")
    lam_rows = sens[sens.param.str.startswith("Cost")]
    macro("lamZeroBeta", lam_rows[lam_rows.value.astype(float) > 1e8].beta_star.iloc[0], "{:.2f}")
    # numbers quoted in the sensitivity discussion
    def _row(prefix: str, val: float) -> pd.Series:
        g = sens[sens.param.str.startswith(prefix)]
        return g.iloc[(g.value.astype(float) - val).abs().argmin()]

    macro("betaKappaOne", _row("Cost", 0.0).beta_star, "{:.2f}")
    macro("betaSigmaLo", _row("Private", 0.1).beta_star, "{:.2f}")
    macro("betaSigmaHi", _row("Private", 0.6).beta_star, "{:.2f}")
    macro("betaDKLo", _row("Asset", 0.05).beta_star, "{:.2f}")
    macro("betaDKHi", _row("Asset", 0.20).beta_star, "{:.2f}")
    mk = sens[sens.param == "Market definition"].set_index("value")
    macro("betaInternet", mk.loc["mobile_internet", "beta_star"], "{:.2f}")
    macro("betaRevenue", mk.loc["mobile_revenue", "beta_star"], "{:.2f}")
    r1 = _row("Cost", 0.0)
    macro("gapSAKappaOne", r1.W_gain_pref - r1.W_gain_setaside, "{:,.0f}")
    rr = sens[sens.param.str.startswith("Reserve")]
    macro("reserveBetaMin", rr.beta_star.min(), "{:.2f}")
    macro("reserveBetaMax", rr.beta_star.max(), "{:.2f}")
    macro("reserveWMin", rr.W_gain_pref.min(), "{:,.0f}")
    macro("reserveWMax", rr.W_gain_pref.max(), "{:,.0f}")
    rc = _row("Correlation", 0.75)
    macro("betaCorrHi", rc.beta_star, "{:.2f}")
    macro("wGainCorrHi", rc.W_gain_pref, "{:,.0f}")
    macro("pEStdCorrHi", 100 * rc.pE_std, "{:.0f}")
    macro("netPrefCorrHi", rc.net_pref, "{:,.0f}")
    macro("sigmaMin", 0.1, "{:.1f}")
    macro("sigmaMax", 0.6, "{:.1f}")

    (ROOT / "paper" / "numbers.tex").write_text(
        "% Auto-generated by scripts/run_all.py -- do not edit by hand\n" +
        "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(NUM.items())) + "\n", encoding="utf-8")
    (RES / "key_numbers.json").write_text(json.dumps(NUM, indent=2), encoding="utf-8")
    print("done:", len(NUM), "macros")


if __name__ == "__main__":
    main()
