"""Executa todos os experimentos do artigo e grava os resultados em results/.

Uso:  python -m wrsim.experiments [--reps 30] [--sens-reps 10] [--workers 4]
                                  [--skip-bench]
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

from .params import Params
from .city import build_city, city_summary
from .sim import static_routes, simulate
from .benchmark import run_benchmark

_CACHE = {}

MAIN_POLICIES = ("B1", "B2", "P1", "P0", "P0S", "P2", "P2I", "P2M")
CMP = ("B1", "B2", "P0", "P2")          # políticas das sensibilidades


def _city_and_routes(p: Params):
    key = (p.city_seed, p.n_points, p.urban_pop, p.grid_nx, p.grid_ny, p.gen_scale,
           p.density_kg_m3, p.sizing_safety, p.center_cap_mult, p.pop_sigma, p.osm_file, p.census_file, p.census_mun, tuple(p.osm_center), tuple(p.landfill_latlon), tuple(p.exit_latlon or ()), tuple(p.garage_latlon or ()), tuple(p.center_latlon or ()))
    if key not in _CACHE:
        c = build_city(p)
        _CACHE[key] = (c, static_routes(c, p))
    return _CACHE[key]


def _task(args):
    tag, pdict, policy, rep, record = args
    p = Params(**pdict)
    city, routes = _city_and_routes(p)
    r = simulate(city, p, policy, rep, routes, record_daily=record)
    r["tag"] = tag
    return r


def scenarios(base: Params):
    """Lista de (tag, params, políticas, replicações) das análises de sensibilidade."""
    S = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        S.append((f"thr={thr}", replace(base, threshold=thr), ("P1", "P2")))
    S.append(("z=1.28", replace(base, z_service=1.28), ("P0", "P2")))
    for lab, kw in (("sens=ruim", dict(sensor_sigma=0.10, sensor_miss=0.20)),
                    ("sem_calendario", dict(use_calendar=False))):
        S.append((lab, replace(base, **kw), ("P0", "P2")))
    for k in (0.25, 0.5, 0.75):
        S.append((f"cobertura={k}", replace(base, sensor_share=k), ("P2",)))
    S.append(("cap_centro=1.2", replace(base, center_cap_mult=1.2), ("B2", "P0", "P2")))
    for lab, kw in (("geracao=0.8", dict(gen_scale=0.8)), ("geracao=1.2", dict(gen_scale=1.2)),
                    ("parada=0.25", dict(t_stop_min=0.25)), ("parada=1.0", dict(t_stop_min=1.0)),
                    ("conteiner=0.6", dict(t_container_min=0.6)), ("conteiner=1.5", dict(t_container_min=1.5)),
                    ("fixo=15", dict(shift_fixed_min=15.0)), ("fixo=45", dict(shift_fixed_min=45.0)),
                    ("cv=baixo", dict(cv_res=0.20, cv_cen=0.30)), ("cv=alto", dict(cv_res=0.40, cv_cen=0.60)),
                    ("sem_picos", dict(use_peaks=False)), ("aquecimento=14", dict(warmup_days=14)),
                    ("porte=10mil", dict(urban_pop=8500.0, n_points=140, grid_nx=16, grid_ny=14))):
        S.append((lab, replace(base, **kw), CMP))
    for cs in (2027, 2028, 2029, 2030, 2031):
        S.append((f"cidade={cs}", replace(base, city_seed=cs), CMP))
    S.append(("frota=2", replace(base, fleet_max_shifts=2), ("P0", "P2")))
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--sens-reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bench-time", type=int, default=20)
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = Params()
    t0 = time.time()

    city, routes = _city_and_routes(base)
    with open(os.path.join(a.out, "city.json"), "w") as f:
        json.dump({"summary": city_summary(city), "params": base.to_dict(),
                   "coords": city.coords.tolist(), "edges": city.edges.tolist(),
                   "point_node": city.point_node.tolist(), "garage": city.garage_node,
                   "exit": city.exit_node, "sector": city.sector.tolist(),
                   "n_cont": city.n_cont.tolist(), "mean_kg": city.mean_kg.tolist(),
                   "cap_kg": city.cap_kg.tolist(),
                   "route_B1": routes["B1"].tolist()}, f)

    reps = range(1, a.reps + 1)
    sreps = range(1, a.sens_reps + 1)
    tasks = []
    # 1) experimento principal (30 replicações, números aleatórios comuns)
    for pol in MAIN_POLICIES:
        for r in reps:
            tasks.append(("main", base.to_dict(), pol, r, True))
    # 2) mesmo nível de serviço: curva z para P0 e P2 com 30 replicações
    for z in (1.645, 3.09):
        for pol in ("P0", "P2"):
            for r in reps:
                tasks.append((f"z30={z}", replace(base, z_service=z).to_dict(), pol, r, False))
    # distância ao aterro com 30 replicações
    for d in (5.0, 30.0):
        for pol in CMP:
            for r in reps:
                tasks.append((f"aterro={int(d)}", replace(base, landfill_km=d).to_dict(), pol, r, False))
    # 3) sensibilidades (10 replicações)
    for tag, pp, pols in scenarios(base):
        for pol in pols:
            for r in sreps:
                tasks.append((tag, pp.to_dict(), pol, r, False))

    print(f"{len(tasks)} simulações", flush=True)
    with Pool(a.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "sim_results.json"), "w") as f:
        json.dump(results, f)
    print(f"simulações concluídas em {time.time() - t0:.0f}s", flush=True)

    # 4) comparação de algoritmos de sequenciamento
    if not a.skip_bench:
        rows = run_benchmark(city, base, ref_time=a.bench_time)
        with open(os.path.join(a.out, "benchmark.json"), "w") as f:
            json.dump(rows, f)
    print(f"tudo concluído em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
