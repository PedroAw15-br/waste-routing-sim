"""Roda de novo um cenário de sensibilidade e substitui os resultados dele em sim_results.json."""
import json, sys
from multiprocessing import Pool
from wrsim.params import Params
from wrsim.experiments import scenarios, _task

if __name__ == "__main__":
    res_path, tag_alvo, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
    base = Params()
    tasks = []
    for tag, pp, pols in scenarios(base):
        if tag == tag_alvo:
            for pol in pols:
                for r in range(1, reps + 1):
                    tasks.append((tag, pp.to_dict(), pol, r, False))
    assert tasks, "cenário não encontrado"
    with Pool(10) as pool:
        novos = pool.map(_task, tasks)
    sim = json.load(open(res_path))
    sim = [r for r in sim if r["tag"] != tag_alvo] + novos
    json.dump(sim, open(res_path, "w"))
    for pol in sorted({r["policy"] for r in novos}):
        rs = [r for r in novos if r["policy"] == pol]
        print(tag_alvo, pol, "ovf/mês", round(sum(r["overflow_events"] for r in rs) / len(rs) * 30 / 49, 1))
