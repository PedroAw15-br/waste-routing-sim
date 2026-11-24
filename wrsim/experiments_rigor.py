"""Rigor estatístico e dependência de premissas:

1. experimento principal com 100 replicações (cidade sintética e Santa Helena);
2. cenário com a variabilidade diária calibrada nos dados de Wyndham (CV = 0,88),
   com a curva em z de P0 e P2 para recalcular S;
3. sensibilidade global: amostragem por hipercubo latino de 13 premissas, variadas
   juntas, para B2, P0 e P2 (análise por correlação parcial de postos em analysis_rigor).

Uso:  python -m wrsim.experiments_rigor --out results_rigor --workers 12
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .params import Params
from .experiments import _task, MAIN_POLICIES

SH = dict(osm_file="results/osm_santa_helena.json", census_file="results/setores_santa_helena_2022.geojson")

# premissas da sensibilidade global: (nome, mínimo, máximo)
LHS = [("cv_res", 0.20, 1.00), ("weekend_factor", 0.90, 1.20), ("peak_mult", 1.00, 1.60),
       ("landfill_km", 5.0, 30.0), ("gen_scale", 0.80, 1.20), ("t_stop_min", 0.25, 1.00),
       ("t_container_min", 0.60, 1.50), ("shift_fixed_min", 15.0, 45.0), ("v_urban_kmh", 12.0, 24.0),
       ("sensor_sigma", 0.00, 0.10), ("sensor_miss", 0.00, 0.20), ("sizing_safety", 1.00, 1.50),
       ("truck_capacity_kg", 5000.0, 9000.0)]


def lhs_samples(n, seed=2026):
    rng = np.random.default_rng(seed)
    u = np.empty((n, len(LHS)))
    for j in range(len(LHS)):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return [{nm: lo + u[i, j] * (hi - lo) for j, (nm, lo, hi) in enumerate(LHS)} for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_rigor")
    ap.add_argument("--data", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps-main", type=int, default=100)
    ap.add_argument("--reps-cal", type=int, default=30)
    ap.add_argument("--lhs", type=int, default=150)
    ap.add_argument("--lhs-reps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sh = {k: v.replace("results/", a.data.rstrip("/") + "/") for k, v in SH.items()}
    syn, shp = Params(), replace(Params(), **sh)
    tasks = []
    for tag, base in (("syn100", syn), ("sh100", shp)):
        tasks += [(tag, base.to_dict(), pol, r, False) for pol in MAIN_POLICIES for r in range(1, a.reps_main + 1)]
    cal = dict(cv_res=0.88, cv_cen=0.88, weekend_factor=1.0)
    for name, base in (("syn", syn), ("sh", shp)):
        b = replace(base, **cal)
        tasks += [(f"cal_{name}|z=2.33", b.to_dict(), pol, r, False) for pol in ("B2", "P0", "P2") for r in range(1, a.reps_cal + 1)]
        for z in (1.645, 3.09):
            tasks += [(f"cal_{name}|z={z}", replace(b, z_service=z).to_dict(), pol, r, False)
                      for pol in ("P0", "P2") for r in range(1, a.reps_cal + 1)]
    samples = lhs_samples(a.lhs)
    for i, smp in enumerate(samples):
        b = replace(syn, cv_cen=min(1.5 * smp["cv_res"], 1.5), **smp)
        tasks += [(f"lhs={i}", b.to_dict(), pol, r, False) for pol in ("B2", "P0", "P2") for r in range(1, a.lhs_reps + 1)]
    json.dump({"lhs_def": LHS, "lhs_samples": samples, "cal": cal}, open(os.path.join(a.out, "design.json"), "w"), indent=1)
    t0 = time.time()
    print(f"{len(tasks)} simulações", flush=True)
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(a.out, "sim_results.json"), "w"))
    print(f"concluído em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
