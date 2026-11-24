"""Roteirização: Algoritmo Genético (permutação, OX, swap, torneio, elitismo),
busca local 2-opt, heurística do vizinho mais próximo e procedimento Split
(rota-primeiro, agrupa-depois) para respeitar a capacidade do caminhão e as
viagens de descarga ao aterro.

Todas as funções numéricas são compiladas com Numba.
"""
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Custos e heurísticas construtivas
# ---------------------------------------------------------------------------
@njit(cache=True)
def tour_cost(t, D, depot):
    n = t.shape[0]
    if n == 0:
        return 0.0
    c = D[depot, t[0]] + D[t[n - 1], depot]
    for i in range(n - 1):
        c += D[t[i], t[i + 1]]
    return c


@njit(cache=True)
def nearest_neighbor(D, depot, n):
    visited = np.zeros(n, dtype=np.bool_)
    tour = np.empty(n, dtype=np.int64)
    cur = depot
    for k in range(n):
        best = -1
        bd = 1e18
        for j in range(n):
            if not visited[j] and D[cur, j] < bd:
                bd = D[cur, j]
                best = j
        tour[k] = best
        visited[best] = True
        cur = best
    return tour


@njit(cache=True)
def two_opt_path(path, D):
    """2-opt de primeira melhoria sobre um caminho com extremidades fixas."""
    m = path.shape[0]
    improved = True
    while improved:
        improved = False
        for i in range(1, m - 2):
            a = path[i - 1]
            b = path[i]
            dab = D[a, b]
            for j in range(i + 1, m - 1):
                c = path[j]
                d = path[j + 1]
                delta = D[a, c] + D[b, d] - dab - D[c, d]
                if delta < -1e-10:
                    lo, hi = i, j
                    while lo < hi:
                        tmp = path[lo]
                        path[lo] = path[hi]
                        path[hi] = tmp
                        lo += 1
                        hi -= 1
                    improved = True
                    b = path[i]
                    dab = D[a, b]
    return path


@njit(cache=True)
def two_opt_tour(t, D, depot):
    n = t.shape[0]
    if n < 3:
        return t
    path = np.empty(n + 2, dtype=np.int64)
    path[0] = depot
    path[n + 1] = depot
    path[1:n + 1] = t
    path = two_opt_path(path, D)
    return path[1:n + 1].copy()


# ---------------------------------------------------------------------------
# Algoritmo Genético
# ---------------------------------------------------------------------------
@njit(cache=True)
def _ox(p1, p2, n, out, used):
    a = np.random.randint(0, n)
    b = np.random.randint(0, n)
    if a > b:
        a, b = b, a
    used[:] = False
    for k in range(a, b + 1):
        out[k] = p1[k]
        used[p1[k]] = True
    pos = (b + 1) % n
    for k in range(n):
        g = p2[(b + 1 + k) % n]
        if not used[g]:
            out[pos] = g
            pos = (pos + 1) % n


@njit(cache=True)
def _tournament(fit, k):
    best = np.random.randint(0, fit.shape[0])
    for _ in range(k - 1):
        c = np.random.randint(0, fit.shape[0])
        if fit[c] < fit[best]:
            best = c
    return best


@njit(cache=True)
def ga_tsp(D, depot, n, pop_size, gens, stall, tk, pc, pm, elite, seed):
    """AG para o TSP sobre os pontos 0..n-1 com depósito `depot`.

    Retorna (melhor_tour, custo, gerações_executadas)."""
    np.random.seed(seed)
    if n <= 3:
        t = nearest_neighbor(D, depot, n)
        return t, tour_cost(t, D, depot), 0
    pop = np.empty((pop_size, n), dtype=np.int64)
    new = np.empty((pop_size, n), dtype=np.int64)
    fit = np.empty(pop_size)
    pop[0] = nearest_neighbor(D, depot, n)
    for i in range(1, pop_size):
        pop[i] = np.random.permutation(n)
    for i in range(pop_size):
        fit[i] = tour_cost(pop[i], D, depot)
    best_i = np.argmin(fit)
    best = pop[best_i].copy()
    best_c = fit[best_i]
    used = np.zeros(n, dtype=np.bool_)
    child = np.empty(n, dtype=np.int64)
    no_imp = 0
    g = 0
    for g in range(gens):
        order = np.argsort(fit)
        for e in range(elite):
            new[e] = pop[order[e]]
        for i in range(elite, pop_size):
            i1 = _tournament(fit, tk)
            i2 = _tournament(fit, tk)
            if np.random.random() < pc:
                _ox(pop[i1], pop[i2], n, child, used)
            else:
                child[:] = pop[i1]
            for q in range(n):
                if np.random.random() < pm:
                    r = np.random.randint(0, n)
                    tmp = child[q]
                    child[q] = child[r]
                    child[r] = tmp
            new[i] = child
        for i in range(pop_size):
            pop[i] = new[i]
            fit[i] = tour_cost(pop[i], D, depot)
        bi = np.argmin(fit)
        if fit[bi] < best_c - 1e-12:
            best_c = fit[bi]
            best = pop[bi].copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= stall:
                break
    return best, best_c, g + 1


@njit(cache=True)
def ga_tsp_memetic(D, depot, n, pop_size, gens, stall, tk, pc, pm, elite, ls_prob, seed):
    """AG memético: população inicial e parte dos filhos refinados por 2-opt."""
    np.random.seed(seed)
    if n <= 3:
        t = nearest_neighbor(D, depot, n)
        return t, tour_cost(t, D, depot), 0
    pop = np.empty((pop_size, n), dtype=np.int64)
    new = np.empty((pop_size, n), dtype=np.int64)
    fit = np.empty(pop_size)
    pop[0] = two_opt_tour(nearest_neighbor(D, depot, n), D, depot)
    for i in range(1, pop_size):
        pop[i] = two_opt_tour(np.random.permutation(n), D, depot)
    for i in range(pop_size):
        fit[i] = tour_cost(pop[i], D, depot)
    best_i = np.argmin(fit)
    best = pop[best_i].copy()
    best_c = fit[best_i]
    used = np.zeros(n, dtype=np.bool_)
    child = np.empty(n, dtype=np.int64)
    no_imp = 0
    g = 0
    for g in range(gens):
        order = np.argsort(fit)
        for e in range(elite):
            new[e] = pop[order[e]]
        for i in range(elite, pop_size):
            i1 = _tournament(fit, tk)
            i2 = _tournament(fit, tk)
            if np.random.random() < pc:
                _ox(pop[i1], pop[i2], n, child, used)
            else:
                child[:] = pop[i1]
            for q in range(n):
                if np.random.random() < pm:
                    r = np.random.randint(0, n)
                    tmp = child[q]
                    child[q] = child[r]
                    child[r] = tmp
            if np.random.random() < ls_prob:
                new[i] = two_opt_tour(child.copy(), D, depot)
            else:
                new[i] = child
        for i in range(pop_size):
            pop[i] = new[i]
            fit[i] = tour_cost(pop[i], D, depot)
        bi = np.argmin(fit)
        if fit[bi] < best_c - 1e-12:
            best_c = fit[bi]
            best = pop[bi].copy()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= stall:
                break
    return best, best_c, g + 1


def solve_tsp(Dsub, depot, p, seed, local_search=None):
    """Motor de roteirização adotado: AG memético (OX + swap + 2-opt).

    `local_search=False` executa o AG puro com os hiperparâmetros originais
    (usado apenas no experimento de comparação de algoritmos)."""
    n = Dsub.shape[0] - 1
    if n == 0:
        return np.empty(0, dtype=np.int64)
    ls = p.local_search if local_search is None else local_search
    if ls:
        t, c, _ = ga_tsp_memetic(Dsub, depot, n, p.mga_pop, p.mga_gens, p.mga_stall, p.ga_tk,
                                 p.ga_pc, p.ga_pm, p.ga_elite, p.mga_ls_prob, seed)
        return t
    t, c, _ = ga_tsp(Dsub, depot, n, p.ga_pop, p.ga_gens, p.ga_stall, p.ga_tk,
                     p.ga_pc, p.ga_pm, p.ga_elite, seed)
    return t


# ---------------------------------------------------------------------------
# Split: divide o tour gigante em viagens que respeitam a capacidade
# ---------------------------------------------------------------------------
@njit(cache=True)
def split(seq, loads, D, G, X, L, Q):
    """Divisão ótima (Prins, 2004; Beasley, 1983) do tour gigante `seq`.

    Primeira viagem parte da garagem G; as demais partem do aterro (entrada
    pela saída X); toda viagem termina no aterro (X + L km)."""
    n = seq.shape[0]
    V = np.full(n + 1, 1e18)
    P = np.zeros(n + 1, dtype=np.int64)
    V[0] = 0.0
    for i in range(n):
        load = 0.0
        dist = 0.0
        for j in range(i, n):
            load += loads[j]
            if load > Q and j > i:
                break
            if j == i:
                if i == 0:
                    dist = D[G, seq[i]]
                else:
                    dist = L + D[X, seq[i]]
            else:
                dist += D[seq[j - 1], seq[j]]
            cost = dist + D[seq[j], X] + L
            if V[i] + cost < V[j + 1]:
                V[j + 1] = V[i] + cost
                P[j + 1] = i
    # reconstrução dos cortes
    cuts = []
    j = n
    while j > 0:
        cuts.append(j)
        j = P[j]
    cuts.append(0)
    out = np.empty(len(cuts), dtype=np.int64)
    for k in range(len(cuts)):
        out[k] = cuts[len(cuts) - 1 - k]
    return out


# ---------------------------------------------------------------------------
# Avaliação operacional de um dia de coleta
# ---------------------------------------------------------------------------
def _trip_metrics(pts, start, city, p):
    """Quilometragem urbana/rodoviária de uma viagem que começa em `start`
    ('G' ou 'U') e termina no aterro."""
    D = city.D
    if start == "G":
        urban = D[city.G, pts[0]]
        hwy = p.landfill_km
    else:
        urban = D[city.X, pts[0]]
        hwy = 2 * p.landfill_km
    urban += D[pts[:-1], pts[1:]].sum() + D[pts[-1], city.X]
    return urban, hwy


def evaluate_day(seq, loads_kg, city, p, reopt_trips=True):
    """Executa um dia de coleta para a sequência `seq` (índices globais de pontos).

    Retorna dicionário com quilometragem, tempo, turnos, combustível etc."""
    out = dict(km_urban=0.0, km_hwy=0.0, route_min=0.0, shifts=0, trips=0,
               stops=0, containers=0, mass_kg=0.0, fuel_l=0.0, overtime_min=0.0)
    if len(seq) == 0:
        return out
    seq = np.asarray(seq, dtype=np.int64)
    loads = np.asarray(loads_kg, dtype=np.float64)
    cuts = split(seq, loads, city.D, city.G, city.X, p.landfill_km, p.truck_capacity_kg)
    trips = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        pts = seq[a:b].copy()
        if reopt_trips and len(pts) >= 3:
            # 2-opt com extremidades fixas (X -> pontos -> X)
            path = np.concatenate([[city.X], pts, [city.X]]).astype(np.int64)
            path = two_opt_path(path, city.D)
            pts = path[1:-1].copy()
        trips.append(pts)

    vu, vh = p.v_urban_kmh / 60.0, p.v_hwy_kmh / 60.0  # km/min
    ret_urban, ret_hwy = city.D[city.X, city.G], p.landfill_km

    def trip_time(pts, start):
        u, h = _trip_metrics(pts, start, city, p)
        serv = len(pts) * p.t_stop_min + city.n_cont[pts].sum() * p.t_container_min
        return u, h, u / vu + h / vh + serv + p.t_unload_min

    shift_t = None
    for pts in trips:
        if shift_t is None:
            u, h, t = trip_time(pts, "G")
            shift_t = t
            out["shifts"] += 1
        else:
            u2, h2, t2 = trip_time(pts, "U")
            back = ret_urban / vu + ret_hwy / vh
            if shift_t + t2 + back <= p.shift_route_min:
                u, h, t = u2, h2, t2
                shift_t += t
            else:
                # fecha o turno atual e abre outro partindo da garagem
                out["km_urban"] += ret_urban
                out["km_hwy"] += ret_hwy
                shift_t += back
                out["route_min"] += shift_t
                out["overtime_min"] += max(0.0, shift_t - p.shift_route_min)
                u, h, t = trip_time(pts, "G")
                shift_t = t
                out["shifts"] += 1
        out["km_urban"] += u
        out["km_hwy"] += h
        out["trips"] += 1
        out["stops"] += len(pts)
        out["containers"] += int(city.n_cont[pts].sum())
    back = ret_urban / vu + ret_hwy / vh
    out["km_urban"] += ret_urban
    out["km_hwy"] += ret_hwy
    shift_t += back
    out["route_min"] += shift_t
    out["overtime_min"] += max(0.0, shift_t - p.shift_route_min)
    out["mass_kg"] = float(loads.sum())
    out["fuel_l"] = (out["km_urban"] * p.fuel_urban_l_km + out["km_hwy"] * p.fuel_hwy_l_km
                     + out["stops"] * p.fuel_stop_l + out["containers"] * p.fuel_container_l
                     + out["trips"] * p.fuel_unload_l)
    return out
