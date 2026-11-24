"""Experimento principal sobre a malha viária real de Santa Helena (PR), do OpenStreetMap,
opcionalmente com a população dos setores censitários do Censo 2022 (IBGE).

Uso:  python -m wrsim.experiments_osm --osm results/osm_santa_helena.json \
          --census results/setores_santa_helena_2022.geojson --out results_santa_helena
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

from .params import Params
from .city import build_city, city_summary
from .experiments import _task, _city_and_routes, MAIN_POLICIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default="results/osm_santa_helena.json")
    ap.add_argument("--census", default="", help="setores censitários (GeoJSON/GeoPackage do IBGE)")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results_osm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = replace(Params(), osm_file=a.osm, census_file=a.census)
    t0 = time.time()
    city, routes = _city_and_routes(base)
    with open(os.path.join(a.out, "city.json"), "w") as f:
        json.dump({"summary": city_summary(city), "params": base.to_dict(),
                   "coords": city.coords.tolist(), "edges": city.edges.tolist(),
                   "point_node": city.point_node.tolist(), "garage": city.garage_node,
                   "exit": city.exit_node, "sector": city.sector.tolist(),
                   "n_cont": city.n_cont.tolist()}, f)
    reps = range(1, a.reps + 1)
    tasks = [("main", base.to_dict(), pol, r, False) for pol in MAIN_POLICIES for r in reps]
    for z in (1.645, 3.09):
        tasks += [(f"z30={z}", replace(base, z_service=z).to_dict(), pol, r, False)
                  for pol in ("P0", "P2") for r in reps]
    print(f"{len(tasks)} simulações", flush=True)
    with Pool(a.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            results.append(r)
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "sim_results.json"), "w") as f:
        json.dump(results, f)
    print(f"concluído em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
