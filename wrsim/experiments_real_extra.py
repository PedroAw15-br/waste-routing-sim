"""Rodada complementar dos casos reais:

1. mais replicações (21 a 60) na varredura do CV diário da geração, para estabilizar a curva
   de λ∗ (em Santa Helena, com uma viagem por turno, 20 replicações deixavam a curva oscilar);
2. robustez às premissas que não têm dado público: posição dos pontos dentro dos setores
   (quatro sorteios alternativos), densidade de pontos (cerca de 85 e 43 habitantes por ponto)
   e heterogeneidade da geração entre pontos (pop_sigma 0 e 0,7).

Uso:  python -m wrsim.experiments_real_extra --out results_real_extra --data results --workers 12
"""
import argparse
import json
import os
import time
from dataclasses import replace
from multiprocessing import Pool

from .experiments import _task
from .experiments_real import CASES, CVS, case_params

ROB = {"seed=2027": dict(city_seed=2027), "seed=2028": dict(city_seed=2028),
       "seed=2029": dict(city_seed=2029), "seed=2030": dict(city_seed=2030),
       "dens=85": dict(n_points=200), "dens=43": dict(n_points=400),
       "sigma=0": dict(pop_sigma=0.0), "sigma=0.7": dict(pop_sigma=0.7)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_real_extra")
    ap.add_argument("--data", default="results")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cv-from", type=int, default=21)
    ap.add_argument("--cv-to", type=int, default=60)
    ap.add_argument("--reps-rob", type=int, default=15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    for name in CASES:
        base = case_params(name, a.data)
        for cv in CVS:
            b = replace(base, cv_res=cv, cv_cen=min(1.5 * cv, 1.35))
            tasks += [(f"{name}|cv={cv}", b.to_dict(), pol, r, False)
                      for pol in ("B2", "P0", "P2") for r in range(a.cv_from, a.cv_to + 1)]
        for lab, kw in ROB.items():
            b = replace(base, **kw)
            tasks += [(f"{name}|rob|{lab}", b.to_dict(), pol, r, False)
                      for pol in ("B2", "P0", "P2") for r in range(1, a.reps_rob + 1)]
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
