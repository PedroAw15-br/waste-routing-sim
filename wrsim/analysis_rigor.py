"""Analisa experiments_rigor: 100 replicações, cenário calibrado e sensibilidade global (PRCC).

Uso:  python -m wrsim.analysis_rigor --res results_rigor --data results --fig figures
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .params import Params
from .city import build_city
from .economics import Costs, operating_cost, breakeven_labor_share
from .analysis import metrics, ci95, paired, C, INK2
from .experiments_rigor import SH

LABEL = {"cv_res": "Variabilidade diária da geração (CV)", "weekend_factor": "Fator de fim de semana",
         "peak_mult": "Multiplicador dos picos", "landfill_km": "Distância ao aterro",
         "gen_scale": "Geração total", "t_stop_min": "Tempo por parada", "t_container_min": "Tempo por contêiner",
         "shift_fixed_min": "Tempo fixo por turno", "v_urban_kmh": "Velocidade urbana",
         "sensor_sigma": "Erro de leitura do sensor", "sensor_miss": "Leituras perdidas",
         "sizing_safety": "Folga dos contêineres", "truck_capacity_kg": "Carga útil do caminhão"}


def prcc(X, y):
    """Correlação parcial de postos de cada coluna de X com y."""
    R = np.column_stack([stats.rankdata(c) for c in X.T]); ry = stats.rankdata(y)
    out = []
    for j in range(R.shape[1]):
        o = np.column_stack([np.ones(len(ry)), np.delete(R, j, axis=1)])
        rx = R[:, j] - o @ np.linalg.lstsq(o, R[:, j], rcond=None)[0]
        rr = ry - o @ np.linalg.lstsq(o, ry, rcond=None)[0]
        out.append(float(np.corrcoef(rx, rr)[0, 1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_rigor")
    ap.add_argument("--data", default="results")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
    design = json.load(open(os.path.join(a.res, "design.json")))
    G = defaultdict(lambda: defaultdict(list))
    for r in sim:
        G[r["tag"]][r["policy"]].append(r)
    for g in G.values():
        for pol in g:
            g[pol].sort(key=lambda r: r["rep"])
    base = Params(); costs = Costs()
    sh = {k: v.replace("results/", a.data.rstrip("/") + "/") for k, v in SH.items()}
    cities = {"syn": build_city(base), "sh": build_city(replace(base, **sh))}
    out = {}

    def M(tag, pol, n=None):
        rs = G[tag][pol][:n] if n else G[tag][pol]
        return [metrics(r, base) for r in rs]

    # 1) 100 replicações e comparação com as 30 primeiras
    for name, tag in (("syn", "syn100"), ("sh", "sh100")):
        res = {}
        for pa, pb in (("P0", "B2"), ("P2", "B2"), ("P2", "P0"), ("P0S", "B2"), ("P1", "B2")):
            for k in ("crew_h_d", "ovf_month", "fuel_d", "km_d", "shifts_d"):
                c100 = paired([m[k] for m in M(tag, pa)], [m[k] for m in M(tag, pb)])
                c30 = paired([m[k] for m in M(tag, pa, 30)], [m[k] for m in M(tag, pb, 30)])
                res[f"{pa}_vs_{pb}|{k}"] = {"n100": c100, "n30": c30}
        prec = {pol: {k: 100 * ci95([m[k] for m in M(tag, pol)])[2] / ci95([m[k] for m in M(tag, pol)])[0]
                      for k in ("crew_h_d", "ovf_month", "fuel_d")} for pol in ("B2", "P0", "P2")}
        out[f"rep100_{name}"] = {"comp": res, "precision_pct": prec, "n": len(G[tag]["B2"])}

    # 2) cenário calibrado (CV de Wyndham)
    for name in ("syn", "sh"):
        city = cities[name]; annual_t = float(city.mean_kg.sum()) * 365 / 1000; n_nodes = int(city.n_cont.sum())

        def annual(tag, pol):
            ms = M(tag, pol); d = {k: np.mean([m[k] for m in ms]) for k in ms[0]}
            return {"fuel_l": d["fuel_t"] * annual_t, "km": d["km_t"] * annual_t, "crew_h": d["crew_h_t"] * annual_t,
                    "shifts": d["shifts_t"] * annual_t, "ot_h": d["ot_h_t"] * annual_t}
        t0 = f"cal_{name}|z=2.33"
        opc = {p: operating_cost(costs, annual(t0, p)) for p in ("B2", "P0", "P2")}
        curve = sorted((np.mean([m["ovf_month"] for m in M(t, "P0")]), operating_cost(costs, annual(t, "P0")))
                       for t in (f"cal_{name}|z=1.645", t0, f"cal_{name}|z=3.09"))
        target = np.mean([m["ovf_month"] for m in M(t0, "P2")])
        c_p0 = float(np.interp(target, [c[0] for c in curve], [c[1] for c in curve]))
        comp = {f"{pa}_vs_{pb}|{k}": paired([m[k] for m in M(t0, pa)], [m[k] for m in M(t0, pb)])["rel_pct"]
                for pa, pb in (("P0", "B2"), ("P2", "B2"), ("P2", "P0")) for k in ("crew_h_d", "ovf_month", "fuel_d")}
        out[f"cal_{name}"] = {"S": (c_p0 - opc["P2"]) / (opc["B2"] - opc["P2"]),
                              "extrapolated": bool(target < curve[0][0] or target > curve[-1][0]),
                              "comp": comp,
                              "lambda_sensores_hora": breakeven_labor_share(costs, n_nodes, annual(t0, "P0"), annual(t0, "P2"), "hour", "sensors"),
                              "lambda_sensores_turno": breakeven_labor_share(costs, n_nodes, annual(t0, "P0"), annual(t0, "P2"), "shift", "sensors"),
                              "lambda_P0_hora": breakeven_labor_share(costs, n_nodes, annual(t0, "B2"), annual(t0, "P0"), "hour", "p0"),
                              "ovf": {p: float(np.mean([m["ovf_month"] for m in M(t0, p)])) for p in ("B2", "P0", "P2")}}

    # 3) sensibilidade global
    names = [d[0] for d in design["lhs_def"]]
    X, Y = [], defaultdict(list)
    for i, smp in enumerate(design["lhs_samples"]):
        tag = f"lhs={i}"
        if not all(p in G[tag] for p in ("B2", "P0", "P2")):
            continue
        h = {p: np.mean([m["crew_h_d"] for m in M(tag, p)]) for p in ("B2", "P0", "P2")}
        o = {p: np.mean([m["ovf_month"] for m in M(tag, p)]) for p in ("B2", "P0", "P2")}
        X.append([smp[n] for n in names])
        Y["horas_P0_B2"].append(100 * (h["P0"] / h["B2"] - 1)); Y["horas_P2_B2"].append(100 * (h["P2"] / h["B2"] - 1))
        Y["horas_P2_P0"].append(100 * (h["P2"] / h["P0"] - 1)); Y["transb_P0_B2"].append(100 * (o["P0"] / o["B2"] - 1))
        Y["transb_P2_B2"].append(100 * (o["P2"] / o["B2"] - 1))
        Y["ordem_horas"].append(float(h["P2"] < h["P0"] < h["B2"]))
    X = np.array(X)
    glob = {"n": int(len(X)), "params": names, "outputs": {}}
    for k, y in Y.items():
        y = np.array(y)
        ent = {"p5": float(np.percentile(y, 5)), "p50": float(np.percentile(y, 50)), "p95": float(np.percentile(y, 95)),
               "mean": float(y.mean()), "frac_negativo": float((y < 0).mean())}
        if k != "ordem_horas":
            ent["prcc"] = dict(zip(names, prcc(X, y)))
        glob["outputs"][k] = ent
    out["global"] = glob

    # figura: PRCC das horas de P0 e de P2 em relação a B2 e de P2 em relação a P0
    keys = [("horas_P0_B2", "Horas P0 × B2"), ("horas_P2_B2", "Horas P2 × B2"), ("horas_P2_P0", "Horas P2 × P0")]
    order = np.argsort([-max(abs(glob["outputs"][k]["prcc"][n]) for k, _ in keys) for n in names])
    fig, ax = plt.subplots(figsize=(6.3, 4.6))
    yy = np.arange(len(names))[::-1]
    for j, (k, lab) in enumerate(keys):
        v = [glob["outputs"][k]["prcc"][names[i]] for i in order]
        ax.barh(yy + (1 - j) * 0.26, v, height=0.25, color=C[[6, 3, 0][j]], label=lab)
    ax.set_yticks(yy); ax.set_yticklabels([LABEL[names[i]] for i in order], fontsize=7.5)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_xlim(-1, 1)
    ax.set_xlabel("Correlação parcial de postos (PRCC) com a diferença relativa de horas de equipe")
    ax.legend(fontsize=7.3, loc="lower right"); ax.grid(axis="y", visible=False)
    os.makedirs(a.fig, exist_ok=True)
    fig.savefig(os.path.join(a.fig, "fig10_sensibilidade_global.png"), bbox_inches="tight", facecolor="white"); plt.close(fig)
    json.dump(out, open(os.path.join(a.res, "summary_rigor.json"), "w"), indent=1, default=float)
    print(json.dumps({k: (v if k != "global" else {kk: {x: y for x, y in vv.items() if x != "prcc"} for kk, vv in v["outputs"].items()})
                      for k, v in out.items() if not k.startswith("rep100")}, indent=1, default=float)[:3500])
    for nm in ("syn", "sh"):
        r = out[f"rep100_{nm}"]["comp"]
        print(nm, {k: (round(v['n100']['rel_pct'], 1), round(v['n30']['rel_pct'], 1)) for k, v in r.items() if k.split('|')[1] in ('crew_h_d', 'ovf_month')})


if __name__ == "__main__":
    main()
