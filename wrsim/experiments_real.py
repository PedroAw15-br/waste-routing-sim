"""Casos reais com os parâmetros de operação declarados ao SINISA 2023:

- Santa Helena (PR): coleta indiferenciada terceirizada, 2 caminhões compactadores, destino
  no aterro sanitário de Cascavel (114 km por estrada, 66 km/h em média pelo OSRM);
- São Miguel do Iguaçu (PR): coleta por equipe própria, destino no aterro controlado
  municipal (11,7 km, 58 km/h).

Para cada caso: experimento principal, curva em z de P0 e P2 (métrica S) e varredura da
variabilidade diária da geração (CV), usada para estimar o limiar a partir do qual os
sensores se pagam.

Uso:  python -m wrsim.experiments_real --out results_real --workers 12
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

from .params import Params
from .experiments import _task

CASES = {
    "santa_helena": dict(osm_file="results/osm_santa_helena.json", census_file="results/setores_santa_helena_2022.geojson",
                         census_mun="4123501", osm_center=(-24.8591735, -54.3328813),
                         landfill_latlon=(-24.9555, -53.4552), landfill_km=110.0, v_hwy_kmh=66.0),
    "sao_miguel": dict(osm_file="results/osm_sao_miguel.json", census_file="results/setores_sao_miguel_2022.geojson",
                       census_mun="4125704", osm_center=(-25.349171, -54.240498),
                       landfill_latlon=(-25.2626907, -54.2377393), landfill_km=10.0, v_hwy_kmh=58.0),
}
CVS = (0.20, 0.30, 0.45, 0.60, 0.75, 0.90)


def case_params(name, data="results"):
    kw = {k: (v.replace("results/", data.rstrip("/") + "/") if isinstance(v, str) else v) for k, v in CASES[name].items()}
    return replace(Params(), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_real")
    ap.add_argument("--data", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--reps-cv", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for name in CASES:
        base = case_params(name, a.data)
        R = range(1, a.reps + 1)
        tasks += [(f"{name}|main", base.to_dict(), pol, r, False) for pol in ("B1", "B2", "P1", "P0", "P0S", "P2") for r in R]
        for z in (1.645, 3.09):
            tasks += [(f"{name}|z={z}", replace(base, z_service=z).to_dict(), pol, r, False) for pol in ("P0", "P2") for r in R]
        for cv in CVS:
            b = replace(base, cv_res=cv, cv_cen=min(1.5 * cv, 1.35))
            tasks += [(f"{name}|cv={cv}", b.to_dict(), pol, r, False) for pol in ("B2", "P0", "P2") for r in range(1, a.reps_cv + 1)]
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
