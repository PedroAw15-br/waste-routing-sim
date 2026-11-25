"""Analisa experiments_real: casos de Santa Helena e São Miguel do Iguaçu com os parâmetros
declarados ao SINISA 2023, métrica S, viabilidade e limiar de variabilidade dos sensores.

Uso:  python -m wrsim.analysis_real --res results_real --data results --fig figures
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import Params
from .city import build_city
from .economics import Costs, operating_cost, breakeven_labor_share, npv, annual_savings
from .analysis import metrics, ci95, paired, C, INK, INK2
from .experiments_real import CASES, CVS, case_params

NOME = {"santa_helena": "Santa Helena", "sao_miguel": "São Miguel do Iguaçu"}
# declarado ao SINISA 2023 (coleta indiferenciada urbana)
SINISA = {"santa_helena": {"massa_t_ano": 3859.8, "pop_urbana": 13842, "compactadores_contratada": 2,
                           "destino": "Aterro Sanitário Paraná Ambiental, Cascavel (PR)", "distancia_km": 131.6},
          "sao_miguel": {"pop_urbana": 18818, "compactadores_publicos": 3,
                         "destino": "Aterro controlado municipal", "distancia_km": 11.7}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_real")
    ap.add_argument("--data", default="results")
    ap.add_argument("--fig", default="figures")
    ap.add_argument("--extra", default=None, help="resultados de experiments_real_extra")
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
    if a.extra:
        sim += json.load(open(os.path.join(a.extra, "sim_results.json")))
    bs = np.random.default_rng(7)
    cal = json.load(open(os.path.join(a.data, "calibracao.json")))
    G = defaultdict(lambda: defaultdict(list))
    for r in sim:
        G[r["tag"]][r["policy"]].append(r)
    for g in G.values():
        for pol in g:
            g[pol].sort(key=lambda r: r["rep"])
    base = Params(); costs = Costs(); out = {}
    cities = {}
    for name in CASES:
        p = case_params(name, a.data); city = build_city(p); cities[name] = (p, city)
        annual_t = float(city.mean_kg.sum()) * 365 / 1000; n_nodes = int(city.n_cont.sum())

        MC = {}

        def M(tag, pol):
            if (tag, pol) not in MC:
                MC[(tag, pol)] = [metrics(r, base) for r in G[f"{name}|{tag}"][pol]]
            return MC[(tag, pol)]

        def annual(tag, pol, idx=None):
            ms = M(tag, pol)
            if idx is not None:
                ms = [ms[i] for i in idx]
            d = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
            return {"fuel_l": d["fuel_t"] * annual_t, "km": d["km_t"] * annual_t, "crew_h": d["crew_h_t"] * annual_t,
                    "shifts": d["shifts_t"] * annual_t, "ot_h": d["ot_h_t"] * annual_t}

        def lam(tag, b, pr, sysn, u="hour", idx=None):
            return breakeven_labor_share(costs, n_nodes, annual(tag, b, idx), annual(tag, pr, idx), u, sysn)

        pols = list(G[f"{name}|main"].keys())
        desc = {pol: {k: ci95([m[k] for m in M("main", pol)])[0] for k in M("main", pol)[0]} for pol in pols}
        comp = {f"{pa}_vs_{pb}": {k: paired([m[k] for m in M("main", pa)], [m[k] for m in M("main", pb)]) for k in
                                  ("crew_h_d", "ovf_month", "fuel_d", "km_d", "shifts_d", "trips_d")}
                for pa, pb in (("P0", "B2"), ("P2", "B2"), ("P2", "P0"), ("P0S", "B2"), ("P1", "B2"), ("P2", "B1"))}
        opc = {pol: operating_cost(costs, annual("main", pol)) for pol in ("B2", "P0", "P2")}
        curve = sorted((np.mean([m["ovf_month"] for m in M(t, "P0")]), operating_cost(costs, annual(t, "P0")))
                       for t in ("z=1.645", "main", "z=3.09"))
        target = desc["P2"]["ovf_month"]
        ext = bool(target < curve[0][0] or target > curve[-1][0])
        c_p0 = float(np.interp(target, [c[0] for c in curve], [c[1] for c in curve]))
        S = (c_p0 - opc["P2"]) / (opc["B2"] - opc["P2"]) if opc["B2"] > opc["P2"] else float("nan")
        econ = {u: {"P2_vs_B2": lam("main", "B2", "P2", "iot", u), "P0_vs_B2": lam("main", "B2", "P0", "p0", u),
                    "P2_vs_P0": lam("main", "P0", "P2", "sensors", u)} for u in ("hour", "shift")}
        econ["capex_iot"] = float(n_nodes * 290 + 9000 + 25000)
        sweep = []
        for cv in CVS:
            t = f"cv={cv}"
            hs = {pol: np.mean([m["crew_h_d"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            os_ = {pol: np.mean([m["ovf_month"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            nr = len(M(t, "B2"))
            bl, bh = [], []
            for _ in range(a.boot):
                idx = bs.integers(0, nr, nr)
                bl.append(lam(t, "P0", "P2", "sensors", idx=idx))
                hb = {pol: np.mean([M(t, pol)[i]["crew_h_d"] for i in idx]) for pol in ("B2", "P0")}
                bh.append(100 * (hb["P0"] / hb["B2"] - 1))
            tr = {pol: np.mean([m["trips_d"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            fu = {pol: np.mean([m["fuel_d"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            sweep.append({"cv": cv, "h": hs, "ovf": os_, "n_reps": nr, "trips": tr, "fuel": fu,
                          "lam_sens_hour_ci": [float(np.percentile(np.minimum(bl, 50), 5)), float(np.percentile(np.minimum(bl, 50), 95))],
                          "h_P0_B2_ci": [float(np.percentile(bh, 5)), float(np.percentile(bh, 95))],
                          "_boot_h": bh, "_boot_l": bl,
                          "h_P0_B2": 100 * (hs["P0"] / hs["B2"] - 1), "h_P2_B2": 100 * (hs["P2"] / hs["B2"] - 1),
                          "lam_sens_hour": lam(t, "P0", "P2", "sensors"), "lam_sens_shift": lam(t, "P0", "P2", "sensors", "shift"),
                          "lam_P2_B2_hour": lam(t, "B2", "P2", "iot"), "lam_P0_B2_hour": lam(t, "B2", "P0", "p0")})

        def cross(key, level):
            xs = [s["cv"] for s in sweep]; ys = [min(s[key], 5.0) for s in sweep]
            for i in range(len(xs) - 1):
                if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
                    return xs[i] + (level - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            return None
        def cross_boot(kb, level):
            vals = []
            for j in range(a.boot):
                xs = [s["cv"] for s in sweep]; ys = [min(s[kb][j], 5.0) for s in sweep]
                v = None
                for i in range(len(xs) - 1):
                    if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
                        v = xs[i] + (level - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i]); break
                vals.append(np.nan if v is None else v)
            vals = np.array(vals)
            ok = ~np.isnan(vals)
            return {"p5": float(np.percentile(vals[ok], 5)) if ok.any() else None,
                    "p95": float(np.percentile(vals[ok], 95)) if ok.any() else None, "frac_def": float(ok.mean())}
        thr_ci = {"P0_melhor_que_B2_horas": cross_boot("_boot_h", 0.0), "sens_hour_100": cross_boot("_boot_l", 1.0)}
        for s_ in sweep:
            s_.pop("_boot_h"); s_.pop("_boot_l")
        # robustez às premissas sem dado público
        rob = {}
        for tg in sorted({k.split("|", 2)[2] for k in G if k.startswith(f"{name}|rob|")}):
            t = f"rob|{tg}"
            hs = {pol: np.mean([m["crew_h_d"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            os_ = {pol: np.mean([m["ovf_month"] for m in M(t, pol)]) for pol in ("B2", "P0", "P2")}
            rob[tg] = {"h_P0_B2": 100 * (hs["P0"] / hs["B2"] - 1), "h_P2_B2": 100 * (hs["P2"] / hs["B2"] - 1),
                       "o_P0_B2": 100 * (os_["P0"] / os_["B2"] - 1), "o_P2_B2": 100 * (os_["P2"] / os_["B2"] - 1),
                       "lam_P0_B2_hour": lam(t, "B2", "P0", "p0"), "lam_sens_hour": lam(t, "P0", "P2", "sensors"),
                       "n_reps": len(M(t, "B2"))}
        # decomposição da economia anual frente a B2 (λ = 100%, por hora)
        decomp = {}
        for pol in ("P0", "P2"):
            qa, qb = annual("main", "B2"), annual("main", pol)
            sv = annual_savings(costs, qa, qb, 1.0, "hour")
            nv = npv(costs, n_nodes, sv["total"], {"P0": "p0", "P2": "iot"}[pol])
            decomp[pol] = {**{k: float(v) for k, v in sv.items()}, "crew_h": float(qa["crew_h"] - qb["crew_h"]),
                           "capex": float(nv["capex"]), "opex": float(nv["opex"]), "npv_100": float(nv["npv"]),
                           "ot_h": float(qa["ot_h"] - qb["ot_h"]), "km": float(qa["km"] - qb["km"])}
        thr = {"sens_hour_100": cross("lam_sens_hour", 1.0), "sens_hour_50": cross("lam_sens_hour", 0.5),
               "sens_shift_100": cross("lam_sens_shift", 1.0), "P0_melhor_que_B2_horas": cross("h_P0_B2", 0.0)}
        model_t = desc["B2"]["dstock_t"] * 0 + annual_t
        out[name] = {"city": {"pontos": int(city.n), "conteineres": n_nodes, "pop": float(city.mean_kg.sum()) / base.per_capita_kg,
                              "massa_t_ano": annual_t},
                     "sinisa": SINISA[name], "desc": desc, "comp": comp, "S": S, "S_extrapolado": ext,
                     "opcost": opc, "econ": econ, "sweep": sweep, "limiar": thr, "limiar_ci": thr_ci, "rob": rob, "decomp": decomp, "n_reps": len(G[f"{name}|main"]["B2"])}

    # figura: limiar de variabilidade
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=False)
    t_lo = min(out[n]["limiar"]["P0_melhor_que_B2_horas"] for n in CASES)
    t_hi = max(out[n]["limiar"]["P0_melhor_que_B2_horas"] for n in CASES)
    cv_w = cal["cv"]["mediana"]
    for ax in axes:
        ax.axvspan(t_lo, t_hi, color="#d9d9d9", alpha=0.6, lw=0, zorder=0)
        ax.axvline(0.30, color=INK2, lw=0.8, ls=":", zorder=1)
        ax.axvline(cv_w, color=INK, lw=0.9, ls="--", zorder=1)
    axes[1].text(0.29, 0.98, "cenário-\nbase", transform=axes[1].get_xaxis_transform(), fontsize=5.8, color=INK2, va="top", ha="right")
    axes[1].text(cv_w - 0.01, 0.02, "Wyndham", transform=axes[1].get_xaxis_transform(), fontsize=5.8, color=INK, ha="right", va="bottom")
    axes[1].text((t_lo + t_hi) / 2, 0.98, "limiar", transform=axes[1].get_xaxis_transform(), fontsize=5.8, color=INK, ha="center", va="top")
    for j, name in enumerate(CASES):
        sw = out[name]["sweep"]; x = [s["cv"] for s in sw]
        ax = axes[0]
        ax.fill_between(x, [100 * min(s["lam_sens_hour_ci"][0], 3) for s in sw], [100 * min(s["lam_sens_hour_ci"][1], 3) for s in sw],
                        color=C[[3, 0][j]], alpha=0.15, lw=0)
        ax.plot(x, [100 * min(s["lam_sens_hour"], 3) for s in sw], marker="o", color=C[[3, 0][j]], lw=1.6, label=NOME[name])
        ax = axes[1]
        ax.fill_between(x, [s["h_P0_B2_ci"][0] for s in sw], [s["h_P0_B2_ci"][1] for s in sw], color=C[[3, 0][j]], alpha=0.15, lw=0)
        ax.plot(x, [s["h_P0_B2"] for s in sw], marker="v", color=C[[3, 0][j]], lw=1.4, ls="--", label=f"P0 × B2, {NOME[name]}")
        ax.plot(x, [s["h_P2_B2"] for s in sw], marker="o", color=C[[3, 0][j]], lw=1.4, label=f"P2 × B2, {NOME[name]}")
    axes[0].axhline(100, color=INK2, lw=0.8, ls=":"); axes[0].set_ylim(0, 300)
    axes[0].set_xlabel("CV diário da geração"); axes[0].set_ylabel("λ∗ dos sensores sobre P0 (%)")
    axes[0].set_title("Fração mínima de conversão das horas", fontsize=8)
    axes[1].axhline(0, color=INK2, lw=0.8); axes[1].set_xlabel("CV diário da geração"); axes[1].set_ylabel("Horas de equipe × B2 (%)")
    axes[1].set_title("Horas de equipe em relação a B2", fontsize=8)
    h_, l_ = axes[1].get_legend_handles_labels()
    fig.legend(h_, [s.replace(" × B2, ", " × B2 – ") for s in l_], loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1)); os.makedirs(a.fig, exist_ok=True)
    fig.savefig(os.path.join(a.fig, "fig11_limiar_variabilidade.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)

    # figura: as duas malhas reais
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.6))
    for ax, name in zip(axes, CASES):
        p, c = cities[name]
        xy = c.coords / 1000
        for e0, e1 in c.edges:
            ax.plot(xy[[e0, e1], 0], xy[[e0, e1], 1], color="#c9c8c2", lw=0.35, zorder=1)
        pts = xy[c.point_node]
        for s, col, mk in ((0, C[1], "D"), (1, C[0], "o"), (2, C[2], "s")):
            m = c.sector == s
            ax.scatter(pts[m, 0], pts[m, 1], s=3 + 3 * c.n_cont[m], c=col, marker=mk, linewidths=0, zorder=3)
        ax.scatter(*xy[c.garage_node], s=45, marker="P", c=INK, zorder=4)
        ax.scatter(*xy[c.exit_node], s=45, marker="X", c=C[7], zorder=4)
        allp = np.vstack([pts, xy[[c.garage_node, c.exit_node]]])
        ax.set_xlim(allp[:, 0].min() - 0.3, allp[:, 0].max() + 0.3); ax.set_ylim(allp[:, 1].min() - 0.3, allp[:, 1].max() + 0.3)
        ax.set_aspect("equal"); ax.grid(False); ax.set_title(NOME[name], fontsize=8.5); ax.set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    fig.text(0.99, 0.0, "Malhas: © colaboradores do OpenStreetMap (ODbL)", ha="right", fontsize=6, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.fig, "fig9_casos_reais.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)

    json.dump(out, open(os.path.join(a.res, "summary_real.json"), "w"), indent=1, default=float)
    for name, o in out.items():
        print("==", name, "pontos", o["city"]["pontos"], "pop", round(o["city"]["pop"]), "massa t/ano", round(o["city"]["massa_t_ano"]), "| SINISA", o["sinisa"])
        print("  horas", {k: round(v["crew_h_d"]["rel_pct"], 1) for k, v in o["comp"].items()})
        print("  transb", {k: round(v["ovf_month"]["rel_pct"], 1) for k, v in o["comp"].items()})
        print("  diesel", {k: round(v["fuel_d"]["rel_pct"], 1) for k, v in o["comp"].items()})
        print("  turnos/dia", {p: round(o["desc"][p]["shifts_d"], 2) for p in o["desc"]}, "viagens", {p: round(o["desc"][p]["trips_d"], 2) for p in o["desc"]})
        print("  S", round(o["S"], 2), "extrap", o["S_extrapolado"], "| lam", {u: {k: round(v, 2) for k, v in o["econ"][u].items()} for u in ("hour", "shift")})
        print("  limiar", o["limiar"], "| IC", o["limiar_ci"])
        print("  decomp", {p: {k: round(v) for k, v in d.items()} for p, d in o["decomp"].items()})
        for k, v in o["rob"].items():
            print("   rob", k, {kk: round(vv, 2) for kk, vv in v.items()})
        for s in o["sweep"]:
            print("   cv", s["cv"], "hP0", round(s["h_P0_B2"], 1), "hP2", round(s["h_P2_B2"], 1), "n", s["n_reps"], "lam_sens h/t", round(min(s["lam_sens_hour"], 99), 2), [round(min(v, 99), 2) for v in s["lam_sens_hour_ci"]], round(min(s["lam_sens_shift"], 99), 2), "ovf", {k: round(v) for k, v in s["ovf"].items()}, "viag", {k: round(v, 2) for k, v in s["trips"].items()}, "diesel", {k: round(v) for k, v in s["fuel"].items()})


if __name__ == "__main__":
    main()
