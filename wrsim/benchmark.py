"""Comparação dos algoritmos de sequenciamento (TSP) contra uma referência
obtida com o OR-Tools (Guided Local Search com tempo limite)."""
import time
import numpy as np

from .routing import nearest_neighbor, two_opt_tour, tour_cost, ga_tsp, ga_tsp_memetic


def ortools_tsp(Dsub, depot, time_limit_s=30):
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    M = np.rint(Dsub * 1000).astype(int)  # metros
    n = M.shape[0]
    man = pywrapcp.RoutingIndexManager(n, 1, depot)
    model = pywrapcp.RoutingModel(man)

    def cb(i, j):
        return int(M[man.IndexToNode(i), man.IndexToNode(j)])

    ti = model.RegisterTransitCallback(cb)
    model.SetArcCostEvaluatorOfAllVehicles(ti)
    sp = pywrapcp.DefaultRoutingSearchParameters()
    sp.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    sp.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    sp.time_limit.seconds = time_limit_s
    sol = model.SolveWithParameters(sp)
    idx = model.Start(0)
    tour = []
    while not model.IsEnd(idx):
        node = man.IndexToNode(idx)
        if node != depot:
            tour.append(node)
        idx = sol.Value(model.NextVar(idx))
    t = np.array(tour, dtype=np.int64)
    return t, tour_cost(t, Dsub, depot)


def run_benchmark(city, p, sizes=(60, 100, 140, 200, 280), per_size=4, seed=7, ref_time=30):
    rng = np.random.default_rng(seed)
    rows = []
    # aquecimento da compilação JIT
    sub = np.concatenate([np.arange(20), [city.G]])
    Ds = city.D[np.ix_(sub, sub)]
    ga_tsp(Ds, 20, 20, 10, 2, 2, 3, .9, .02, 2, 1)
    ga_tsp_memetic(Ds, 20, 20, 6, 2, 2, 3, .9, .02, 2, .2, 1)
    for k in sizes:
        for inst in range(per_size):
            S = np.sort(rng.choice(city.n, size=k, replace=False)) if k < city.n else np.arange(city.n)
            sub = np.concatenate([S, [city.G]])
            Dsub = city.D[np.ix_(sub, sub)]
            dep = k
            res = {}
            t0 = time.perf_counter(); t = nearest_neighbor(Dsub, dep, k)
            res["VMP"] = (tour_cost(t, Dsub, dep), time.perf_counter() - t0)
            t0 = time.perf_counter(); t2 = two_opt_tour(t.copy(), Dsub, dep)
            res["VMP+2-opt"] = (tour_cost(t2, Dsub, dep), time.perf_counter() - t0 + res["VMP"][1])
            t0 = time.perf_counter()
            tg, cg, _ = ga_tsp(Dsub, dep, k, p.ga_pop, p.ga_gens, p.ga_stall, p.ga_tk,
                               p.ga_pc, p.ga_pm, p.ga_elite, 100 + inst)
            res["AG puro"] = (cg, time.perf_counter() - t0)
            t0 = time.perf_counter()
            tm, cm, _ = ga_tsp_memetic(Dsub, dep, k, p.mga_pop, p.mga_gens, p.mga_stall, p.ga_tk,
                                       p.ga_pc, p.ga_pm, p.ga_elite, p.mga_ls_prob, 100 + inst)
            res["AG memético"] = (cm, time.perf_counter() - t0)
            t0 = time.perf_counter()
            to, co = ortools_tsp(Dsub, dep, ref_time)
            res["OR-Tools GLS"] = (co, time.perf_counter() - t0)
            ref = min(v[0] for v in res.values())
            for m, (c, s) in res.items():
                rows.append(dict(n=k, inst=inst, method=m, cost_km=c, gap_pct=100 * (c / ref - 1), time_s=s))
    return rows
