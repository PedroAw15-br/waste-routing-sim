"""Consolida o experimento na malha real (OpenStreetMap): estatísticas, S, economia e figura.

Uso:  python -m wrsim.analysis_osm --res results_osm --fig figures
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import Params
from .economics import Costs, annual_savings, npv, breakeven_labor_share, operating_cost
from .analysis import metrics, ci95, paired, C, INK


def fig_city_osm(cityj, path):
    xy = np.array(cityj["coords"]) / 1000
    E = np.array(cityj["edges"])
    pn = np.array(cityj["point_node"]); sec = np.array(cityj["sector"]); nc = np.array(cityj["n_cont"])
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    ax.grid(False)
    for a, b in E:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#c9c8c2", lw=0.6, zorder=1)
    labels = {0: "Centro comercial", 1: "Setor A", 2: "Setor B"}
    cols = {0: C[1], 1: C[0], 2: C[2]}
    marks = {0: "D", 1: "o", 2: "s"}
    for s in (0, 1, 2):
        m = sec == s
        ax.scatter(xy[pn[m], 0], xy[pn[m], 1], s=8 + 10 * nc[m], c=cols[s], marker=marks[s],
                   edgecolor="white", linewidth=0.5, label=labels[s], zorder=3)
    ax.scatter(*xy[cityj["garage"]], s=110, marker="P", c=INK, zorder=4, label="Garagem")
    ax.scatter(*xy[cityj["exit"]], s=110, marker="X", c=C[7], zorder=4, label="Saída para o aterro")
    pts = np.vstack([xy[pn], xy[[cityj["garage"], cityj["exit"]]]])
    ax.set_xlim(pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4)
    ax.set_ylim(pts[:, 1].min() - 0.4, pts[:, 1].max() + 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=7.5)
    ax.text(0.99, 0.01, "Malha viária: © colaboradores do OpenStreetMap (ODbL)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.5, color="#52514e")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_osm")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
    cityj = json.load(open(os.path.join(a.res, "city.json")))
    base = Params()
    groups = defaultdict(lambda: defaultdict(list))
    for r in sim:
        groups[r["tag"]][r["policy"]].append(r)
    for g in groups.values():
        for pol in g:
            g[pol].sort(key=lambda r: r["rep"])
    out = {"city": cityj["summary"], "groups": {}}
    PAIRS = (("P2", "B2"), ("P2", "B1"), ("P0", "B2"), ("P0", "B1"), ("P0S", "B2"), ("P2", "P0"),
             ("P2I", "P2"), ("P2M", "P2"), ("P1", "B2"), ("B2", "B1"))
    for tag, g in groups.items():
        M = {pol: [metrics(r, base) for r in rs] for pol, rs in g.items()}
        desc = {pol: {k: dict(zip(("mean", "sd", "ci"), ci95([m[k] for m in ms]))) for k in ms[0]}
                for pol, ms in M.items()}
        comps = {f"{pa}_vs_{pb}": {k: paired([m[k] for m in M[pa]], [m[k] for m in M[pb]])
                                   for k in M[pa][0] if k != "est_mae"}
                 for pa, pb in PAIRS if pa in M and pb in M}
        out["groups"][tag] = {"desc": desc, "comp": comps}
    main_g = out["groups"]["main"]
    sd = {}
    for pol in groups["main"]:
        sh = [r["days_over2"] / r["collect_days"] for r in groups["main"][pol]]
        sd[pol] = float(np.mean(sh))
    out["days_over2_share"] = sd

    costs = Costs()
    n_nodes = cityj["summary"]["conteineres"]
    annual_t = cityj["summary"]["geracao_media_kg_dia"] * 365 / 1000

    def annual(pol, tag="main"):
        d = out["groups"][tag]["desc"][pol]
        return {"fuel_l": d["fuel_t"]["mean"] * annual_t, "km": d["km_t"]["mean"] * annual_t,
                "crew_h": d["crew_h_t"]["mean"] * annual_t, "shifts": d["shifts_t"]["mean"] * annual_t,
                "ot_h": d["ot_h_t"]["mean"] * annual_t}

    econ = {"n_nodes": n_nodes, "cases": {}}
    for pol in main_g["desc"]:
        econ[f"opcost_{pol}"] = operating_cost(costs, annual(pol), "hour")
    for name, (b, pr, sysn) in {"P2_vs_B2": ("B2", "P2", "iot"), "P0_vs_B2": ("B2", "P0", "p0"),
                                "P2_vs_P0": ("P0", "P2", "sensors"), "P2_vs_B1": ("B1", "P2", "iot")}.items():
        A, P = annual(b), annual(pr)
        econ["cases"][name] = {u: {"breakeven": breakeven_labor_share(costs, n_nodes, A, P, u, sysn),
                                   "npv1": npv(costs, n_nodes, annual_savings(costs, A, P, 1.0, u)["total"], sysn)["npv"],
                                   "npv0": npv(costs, n_nodes, annual_savings(costs, A, P, 0.0, u)["total"], sysn)["npv"]}
                               for u in ("hour", "shift")}
    out["economics"] = econ
    zs = {1.645: "z30=1.645", 2.33: "main", 3.09: "z30=3.09"}
    curve = sorted((out["groups"][t]["desc"]["P0"]["ovf_month"]["mean"],
                    operating_cost(costs, annual("P0", t), "hour"), z) for z, t in zs.items())
    target = main_g["desc"]["P2"]["ovf_month"]["mean"]
    c_p0 = float(np.interp(target, [c[0] for c in curve], [c[1] for c in curve]))
    c_b2, c_p2 = econ["opcost_B2"], econ["opcost_P2"]
    out["S_metric"] = {"C_B2": c_b2, "C_P2": c_p2, "C_P0_at_P2_service": c_p0, "P2_ovf_target": target,
                       "extrapolated": bool(target < curve[0][0] or target > curve[-1][0]),
                       "P0_curve": curve, "S": (c_p0 - c_p2) / (c_b2 - c_p2)}
    os.makedirs(a.fig, exist_ok=True)
    fig_city_osm(cityj, os.path.join(a.fig, "fig9_santa_helena.png"))
    with open(os.path.join(a.res, "summary_osm.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("ok")


if __name__ == "__main__":
    main()
