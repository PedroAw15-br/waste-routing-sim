"""Resume a robustez de P0 ao histórico e à mudança inesperada na geração (experiments_hist).

Uso:  python -m wrsim.analysis_hist --res results_hist --data results
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace

from .params import Params
from .city import build_city
from .economics import Costs, operating_cost, breakeven_labor_share
from .analysis import metrics, ci95, paired
from .experiments_hist import CONFIGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_hist")
    ap.add_argument("--data", default="results")
    a = ap.parse_args()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
    G = defaultdict(lambda: defaultdict(list))
    for r in sim:
        G[r["tag"]][r["policy"]].append(r)
    base = Params()
    costs = Costs()
    out = {}
    for cfg, kw in CONFIGS.items():
        kw = {k: v.replace("results/", a.data.rstrip("/") + "/") for k, v in kw.items()}
        city = build_city(replace(base, **kw))
        annual_t = float(city.mean_kg.sum()) * 365 / 1000
        n_nodes = int(city.n_cont.sum())
        for scen in ("hist", "mudanca"):
            g = G[f"{cfg}|{scen}"]
            for pol in g:
                g[pol].sort(key=lambda r: r["rep"])
            M = {pol: [metrics(r, base) for r in rs] for pol, rs in g.items()}
            mean = {pol: {k: ci95([m[k] for m in ms])[0] for k in ms[0]} for pol, ms in M.items()}

            def annual(pol):
                d = mean[pol]
                return {"fuel_l": d["fuel_t"] * annual_t, "km": d["km_t"] * annual_t, "crew_h": d["crew_h_t"] * annual_t,
                        "shifts": d["shifts_t"] * annual_t, "ot_h": d["ot_h_t"] * annual_t}
            opc = {pol: operating_cost(costs, annual(pol)) for pol in M}
            res = {}
            for pol in M:
                row = {"crew_h_d": mean[pol]["crew_h_d"], "ovf_month": mean[pol]["ovf_month"],
                       "days_over2_pct": mean[pol]["days_over2_pct"], "opcost": opc[pol]}
                if pol != "B2":
                    c = paired([m["crew_h_d"] for m in M[pol]], [m["crew_h_d"] for m in M["B2"]])
                    o = paired([m["ovf_month"] for m in M[pol]], [m["ovf_month"] for m in M["B2"]])
                    row.update(h_vs_B2=c["rel_pct"], h_vs_B2_ci=(c["rel_lo"], c["rel_hi"]), ovf_vs_B2=o["rel_pct"])
                    sysn = "iot" if pol.startswith("P2") else "p0"
                    row["lambda_hour_vs_B2"] = breakeven_labor_share(costs, n_nodes, annual("B2"), annual(pol), "hour", sysn)
                if pol.startswith("P0"):
                    # parcela do ganho de P2 que esse P0 não alcança (sem igualar o nível de serviço)
                    row["share_not_reached"] = (opc[pol] - opc["P2"]) / (opc["B2"] - opc["P2"])
                res[pol] = row
            out[f"{cfg}|{scen}"] = {"annual_t": annual_t, "n_nodes": n_nodes, "n_reps": len(M["B2"]), "pol": res}
    json.dump(out, open(os.path.join(a.res, "summary_hist.json"), "w"), indent=1, default=float)
    for key, v in out.items():
        print(key)
        for pol, r in v["pol"].items():
            print(f"  {pol:6s} h={r['crew_h_d']:.2f} ovf={r['ovf_month']:.1f} >2t={r['days_over2_pct']:.1f}% "
                  + (f"dh={r['h_vs_B2']:+.1f}% dovf={r['ovf_vs_B2']:+.1f}% lam={r['lambda_hour_vs_B2']:.2f} " if 'h_vs_B2' in r else "")
                  + (f"share={r['share_not_reached']:.2f}" if 'share_not_reached' in r else ""))


if __name__ == "__main__":
    main()
