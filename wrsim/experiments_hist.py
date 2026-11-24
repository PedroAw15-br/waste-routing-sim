"""Robustez da política sem sensores (P0) à qualidade do histórico, nas três configurações
de rede e demanda: P0 com taxa estimada de 7, 30 e 90 dias de pesagem por ponto e um cenário
de mudança inesperada na geração (20% dos pontos com ±30% a partir da avaliação).

Uso:  python -m wrsim.experiments_hist --out results_hist --workers 12
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

from .params import Params
from .experiments import _task

CONFIGS = {
    "sintetica": dict(),
    "osm": dict(osm_file="results/osm_santa_helena.json"),
    "santa_helena": dict(osm_file="results/osm_santa_helena.json",
                         census_file="results/setores_santa_helena_2022.geojson"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results_hist")
    ap.add_argument("--data", default="results", help="pasta com a malha OSM e os setores do Censo")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tasks = []
    for cfg, kw in CONFIGS.items():
        kw = {k: v.replace("results/", a.data.rstrip("/") + "/") for k, v in kw.items()}
        base = replace(Params(), **kw)
        for r in range(1, a.reps + 1):
            for pol in ("B2", "P0", "P2", "P0H7", "P0H30", "P0H90"):
                tasks.append((f"{cfg}|hist", base.to_dict(), pol, r, False))
            for pol in ("B2", "P0", "P2"):
                tasks.append((f"{cfg}|mudanca", replace(base, drift_share=0.2).to_dict(), pol, r, False))
    print(f"{len(tasks)} simulações", flush=True)
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            res.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)}  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(a.out, "sim_results.json"), "w"))
    print(f"concluído em {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
