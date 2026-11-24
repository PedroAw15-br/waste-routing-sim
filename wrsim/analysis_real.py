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
                           "destino": "Aterro sanitário em Cascavel (PR)", "distancia_km": 114.2},
          "sao_miguel": {"pop_urbana": 18818, "compactadores_publicos": 3,
                         "destino": "Aterro controlado municipal", "distancia_km": 11.7}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_real")
    ap.add_argument("--data", default="results")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
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

        def M(tag, pol):
            return [metrics(r, base) for r in G[f"{name}|{tag}"][pol]]

        def annual(tag, pol):
            ms = M(tag, pol); d = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
            return {"fuel_l": d["fuel_t"] * annual_t, "km": d["km_t"] * annual_t, "crew_h": d["crew_h_t"] * annual_t,
                    "shifts": d["shifts_t"] * annual_t, "ot_h": d["ot_h_t"] * annual_t}

        def lam(tag, b, pr, sysn, u="hour"):
            return breakeven_labor_share(costs, n_nodes, annual(tag, b), annual(tag, pr), u, sysn)

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
            sweep.append({"cv": cv, "h": hs, "ovf": os_,
                          "h_P0_B2": 100 * (hs["P0"] / hs["B2"] - 1), "h_P2_B2": 100 * (hs["P2"] / hs["B2"] - 1),
                          "lam_sens_hour": lam(t, "P0", "P2", "sensors"), "lam_sens_shift": lam(t, "P0", "P2", "sensors", "shift"),
                          "lam_P2_B2_hour": lam(t, "B2", "P2", "iot"), "lam_P0_B2_hour": lam(t, "B2", "P0", "p0")})

        def cross(key, level):
            xs = [s["cv"] for s in sweep]; ys = [min(s[key], 5.0) for s in sweep]
            for i in range(len(xs) - 1):
                if (ys[i] - level) * (ys[i + 1] - level) <= 0 and ys[i] != ys[i + 1]:
                    return xs[i] + (level - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
            return None
        thr = {"sens_hour_100": cross("lam_sens_hour", 1.0), "sens_hour_50": cross("lam_sens_hour", 0.5),
               "sens_shift_100": cross("lam_sens_shift", 1.0), "P0_melhor_que_B2_horas": cross("h_P0_B2", 0.0)}
        model_t = desc["B2"]["dstock_t"] * 0 + annual_t
        out[name] = {"city": {"pontos": int(city.n), "conteineres": n_nodes, "pop": float(city.mean_kg.sum()) / base.per_capita_kg,
                              "massa_t_ano": annual_t},
                     "sinisa": SINISA[name], "desc": desc, "comp": comp, "S": S, "S_extrapolado": ext,
                     "opcost": opc, "econ": econ, "sweep": sweep, "limiar": thr, "n_reps": len(G[f"{name}|main"]["B2"])}

    # figura: limiar de variabilidade
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), sharey=False)
    for j, name in enumerate(CASES):
        sw = out[name]["sweep"]; x = [s["cv"] for s in sw]
        ax = axes[0]
        ax.plot(x, [100 * min(s["lam_sens_hour"], 3) for s in sw], marker="o", color=C[[3, 0][j]], lw=1.6, label=NOME[name])
        ax = axes[1]
        ax.plot(x, [s["h_P0_B2"] for s in sw], marker="v", color=C[[3, 0][j]], lw=1.4, ls="--", label=f"P0 × B2, {NOME[name]}")
        ax.plot(x, [s["h_P2_B2"] for s in sw], marker="o", color=C[[3, 0][j]], lw=1.4, label=f"P2 × B2, {NOME[name]}")
    axes[0].axhline(100, color=INK2, lw=0.8, ls=":"); axes[0].set_ylim(0, 300)
    axes[0].set_xlabel("CV diário da geração"); axes[0].set_ylabel("λ∗ dos sensores sobre P0 (%)")
    axes[0].set_title("Fração mínima de conversão das horas", fontsize=8); axes[0].legend(fontsize=6.8)
    axes[1].axhline(0, color=INK2, lw=0.8); axes[1].set_xlabel("CV diário da geração"); axes[1].set_ylabel("Horas de equipe × B2 (%)")
    axes[1].set_title("Horas de equipe em relação a B2", fontsize=8); axes[1].legend(fontsize=5.8)
    fig.tight_layout(); os.makedirs(a.fig, exist_ok=True)
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
        print("  limiar", o["limiar"])
        for s in o["sweep"]:
            print("   cv", s["cv"], "hP0", round(s["h_P0_B2"], 1), "hP2", round(s["h_P2_B2"], 1), "lam_sens h/t", round(min(s["lam_sens_hour"], 99), 2), round(min(s["lam_sens_shift"], 99), 2), "ovf", {k: round(v) for k, v in s["ovf"].items()})


if __name__ == "__main__":
    main()
