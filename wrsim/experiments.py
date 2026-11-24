"""Executa todos os experimentos do artigo e grava os resultados em results/.

Uso:  python -m wrsim.experiments [--reps 30] [--sens-reps 10] [--workers 2]
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


def _city_and_routes(p: Params):
    key = (p.city_seed, p.n_points, p.gen_scale, p.density_kg_m3)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--sens-reps", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bench-time", type=int, default=20)
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
    # 1) experimento principal
    for pol in ("B1", "B2", "P1", "P2"):
        for r in reps:
            tasks.append(("main", base.to_dict(), pol, r, True))
    # 2) sensibilidade: limiar (P1 e P2)
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        for pol in ("P1", "P2"):
            for r in sreps:
                tasks.append((f"thr={thr}", replace(base, threshold=thr).to_dict(), pol, r, False))
    # 3) sensibilidade: nível de serviço da regra preditiva
    for z in (1.28, 1.645, 2.33, 3.09):
        for r in sreps:
            tasks.append((f"z={z}", replace(base, z_service=z).to_dict(), "P2", r, False))
    # 4) qualidade do sensoriamento e uso do calendário
    for lab, kw in (("sens=ideal", dict(sensor_sigma=0.0, sensor_miss=0.0)),
                    ("sens=base", dict()),
                    ("sens=ruim", dict(sensor_sigma=0.10, sensor_miss=0.20)),
                    ("sem_calendario", dict(use_calendar=False))):
        for r in sreps:
            tasks.append((lab, replace(base, **kw).to_dict(), "P2", r, False))
    # 5) distância ao aterro e 6) taxa de geração (equivale a variar a densidade)
    for lab, kw in (("aterro=5", dict(landfill_km=5.0)), ("aterro=15", dict()),
                    ("aterro=30", dict(landfill_km=30.0)),
                    ("geracao=0.8", dict(gen_scale=0.8)), ("geracao=1.2", dict(gen_scale=1.2))):
        for pol in ("B1", "B2", "P2"):
            for r in sreps:
                tasks.append((lab, replace(base, **kw).to_dict(), pol, r, False))
    # 7) robustez: outras cinco malhas sintéticas
    for cs in (2027, 2028, 2029, 2030, 2031):
        for pol in ("B1", "B2", "P2"):
            for r in sreps:
                tasks.append((f"cidade={cs}", replace(base, city_seed=cs).to_dict(), pol, r, False))

    print(f"{len(tasks)} simulações", flush=True)
    with Pool(a.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "sim_results.json"), "w") as f:
        json.dump(results, f)
    print(f"simulações concluídas em {time.time() - t0:.0f}s", flush=True)

    # 8) comparação de algoritmos de sequenciamento
    rows = run_benchmark(city, base, ref_time=a.bench_time)
    with open(os.path.join(a.out, "benchmark.json"), "w") as f:
        json.dump(rows, f)
    print(f"tudo concluído em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
