"""Simulação de eventos diários: geração estocástica, sensoriamento imperfeito,
políticas de coleta (fixas e dinâmicas), roteirização e indicadores.

Linha do tempo de cada dia t (0 = segunda-feira):
  manhã  -> leituras dos sensores; se for dia de coleta, a política escolhe os
            pontos e a frota executa as rotas (pontos coletados ficam vazios);
  noite  -> resíduos do dia t são depositados; volume acima da capacidade
            é contabilizado como transbordo.
"""
import numpy as np

from .params import Params
from .routing import solve_tsp, evaluate_day

HOLIDAYS = (35, 52)                       # segunda (sem. 6) e quinta (sem. 8)
PEAKS = ((25, 26, 27), (33, 34, 35), (52, 53, 54, 55))
POLICIES = ("B1", "B2", "P1", "P2")
POLICY_NAMES = {
    "B1": "Rota fixa diária (todos os pontos)",
    "B2": "Rota fixa setorizada (centro diário, bairros em dias alternados)",
    "P1": "Dinâmica por limiar fixo",
    "P2": "Dinâmica preditiva (limiar + risco de transbordo)",
}


def horizon(p: Params):
    return p.warmup_days + p.measure_days


def is_collection_day(t):
    return (t % 7) != 6 and t not in HOLIDAYS


def next_collection_day(t, T):
    s = t + 1
    while s < T + 7 and not is_collection_day(s):
        s += 1
    return s


def make_demand(city, p: Params, rep: int):
    """Massa gerada por ponto e por dia (kg), com sazonalidade semanal e picos."""
    T = horizon(p)
    rng = np.random.default_rng([rep, 1])
    wk = np.ones(7)
    wk[5:] = p.weekend_factor
    wk = wk / wk.mean()
    cv = np.where(city.central, p.cv_cen, p.cv_res)
    k = 1.0 / cv ** 2
    noise = rng.gamma(shape=k, scale=1.0 / k, size=(T, city.n))
    gen = city.mean_kg[None, :] * p.gen_scale * wk[np.arange(T) % 7][:, None] * noise
    for ev in PEAKS:
        for d in ev:
            if d < T:
                gen[d, city.central] *= p.peak_mult
    return gen


def calendar_factors(city, p: Params, T: int):
    """Fatores multiplicativos esperados da geração por noite (perfil semanal e
    calendário de eventos conhecido pela prefeitura). O multiplicador de evento
    usado na previsão (p.event_mult_forecast) é propositalmente menor que o real."""
    wk = np.ones(7)
    wk[5:] = p.weekend_factor
    wk = wk / wk.mean()
    f = np.ones((T, city.n))
    if p.use_calendar:
        f *= wk[np.arange(T) % 7][:, None]
        for ev in PEAKS:
            for d in ev:
                if d < T:
                    f[d, city.central] *= p.event_mult_forecast
    return f


def make_sensor_draws(city, p: Params, rep: int):
    T = horizon(p)
    rng = np.random.default_rng([rep, 2])
    return rng.standard_normal((T, city.n)), rng.random((T, city.n))


def static_routes(city, p: Params, seeds=(11, 12, 13)):
    """Sequências fixas (otimizadas uma única vez com o mesmo AG + 2-opt)."""
    def best_tour(idx):
        idx = np.asarray(idx)
        sub = np.concatenate([idx, [city.G]])
        Dsub = city.D[np.ix_(sub, sub)]
        best, bc = None, np.inf
        for s in seeds:
            t = solve_tsp(Dsub, len(idx), p, s, local_search=True)
            c = Dsub[len(idx), t[0]] + Dsub[t[:-1], t[1:]].sum() + Dsub[t[-1], len(idx)]
            if c < bc:
                best, bc = t, c
        return idx[best]

    allp = np.arange(city.n)
    cenA = np.where((city.sector == 0) | (city.sector == 1))[0]
    cenB = np.where((city.sector == 0) | (city.sector == 2))[0]
    return {"B1": best_tour(allp), "B2A": best_tour(cenA), "B2B": best_tour(cenB)}


def _dynamic_route(S, city, p, seed):
    sub = np.concatenate([S, [city.G]])
    Dsub = city.D[np.ix_(sub, sub)]
    t = solve_tsp(Dsub, len(S), p, seed)
    return S[t]


def simulate(city, p: Params, policy: str, rep: int, routes: dict,
             gen=None, sensor=None, record_daily=False):
    T = horizon(p)
    n = city.n
    cap = city.cap_kg
    if gen is None:
        gen = make_demand(city, p, rep)
    if sensor is None:
        sensor = make_sensor_draws(city, p, rep)
    zs, us = sensor

    level = np.clip(city.mean_kg * 1.5 / cap, 0, 0.95)   # estado inicial (domingo)
    over_prev = np.zeros(n, dtype=bool)
    # estimadores do sistema (aprendidos a partir das leituras)
    rate_m = np.full(n, 0.25)
    rate_v = np.full(n, 0.01)
    anc_lvl = level.copy()
    anc_day = np.full(n, -1)

    acc = {k: 0.0 for k in ("km_urban", "km_hwy", "route_min", "shifts", "trips", "stops",
                            "containers", "mass_kg", "fuel_l", "overtime_min")}
    acc.update(spill_kg=0.0, overflow_events=0, overflow_nights=0, gen_kg=0.0,
               fill_sum=0.0, collect_days=0, max_shifts=0, est_abs_err=0.0, est_n=0,
               deferrals=0)
    pending_sectors = set()
    cal = calendar_factors(city, p, T + 7)
    stock0 = None
    daily = []
    a = p.ewma_alpha

    for t in range(T):
        measuring = t >= p.warmup_days
        if t == p.warmup_days:
            stock0 = float((level * cap).sum())
        # ---------------- manhã: leituras ----------------
        r = np.clip(level + p.sensor_sigma * zs[t], 0.0, 1.1)
        valid = us[t] >= p.sensor_miss
        nights = t - anc_day
        upd = valid & (nights >= 1) & (r < 0.97) & (anc_lvl < 0.97)
        obs = np.where(upd, (r - anc_lvl) / np.maximum(nights, 1), 0.0)
        d = obs - rate_m
        rate_m = np.where(upd, rate_m + a * d, rate_m)
        rate_v = np.where(upd, (1 - a) * (rate_v + a * d * d), rate_v)
        anc_lvl = np.where(valid, r, anc_lvl)
        anc_day = np.where(valid, t, anc_day)
        est = np.where(valid, r, anc_lvl + rate_m * (t - anc_day))

        day = None
        if (not is_collection_day(t)) and (t % 7) != 6 and policy == "B2":
            pending_sectors.add("A" if (t % 7) in (0, 2, 4) else "B")
        if is_collection_day(t):
            pol = policy if measuring else "B1"
            dow = t % 7
            if pol == "B1":
                seq = routes["B1"]
                static = True
            elif pol == "B2":
                # setor do dia + setor cuja coleta caiu em feriado (transferida)
                todays = "A" if dow in (0, 2, 4) else "B"
                sectors = {todays} | pending_sectors
                pending_sectors = set()
                if sectors == {"A", "B"}:
                    seq = routes["B1"]
                else:
                    seq = routes["B2" + todays]
                static = True
            else:
                g = next_collection_day(t, T) - t
                f = cal[t:t + g]                     # fatores de calendário das próximas g noites
                fs = f.sum(axis=0)
                fs2 = (f ** 2).sum(axis=0)
                pred = est + rate_m * fs + p.z_service * np.sqrt(np.maximum(rate_v, 1e-6) * fs2)
                if pol == "P1":
                    mask = est >= p.threshold
                    prio = est
                elif pol == "P2":
                    must = pred > 1.0
                    mask = (est >= p.threshold) | must
                    prio = pred + must * 10.0
                else:
                    raise ValueError(policy)
                S = np.where(mask)[0]
                seq = _dynamic_route(S, city, p, seed=rep * 1000 + t) if len(S) else S
                static = False
            loads = level[seq] * cap[seq]
            day = evaluate_day(seq, loads, city, p, reopt_trips=not static)
            # restrição de frota: adia os pontos de menor prioridade
            if (not static) and p.fleet_max_shifts > 0:
                it = 0
                while day["shifts"] > p.fleet_max_shifts and len(S) > 1 and it < 20:
                    keep = int(np.floor(len(S) * 0.92))
                    S = S[np.argsort(-prio[S])][:keep]
                    seq = _dynamic_route(S, city, p, seed=rep * 1000 + t + 7 * it)
                    loads = level[seq] * cap[seq]
                    day = evaluate_day(seq, loads, city, p, reopt_trips=True)
                    it += 1
                    if measuring:
                        acc["deferrals"] += 1
            if measuring:
                for k in ("km_urban", "km_hwy", "route_min", "shifts", "trips", "stops",
                          "containers", "mass_kg", "fuel_l", "overtime_min"):
                    acc[k] += day[k]
                acc["fill_sum"] += float(level[seq].sum())
                acc["collect_days"] += 1
                acc["max_shifts"] = max(acc["max_shifts"], day["shifts"])
                if pol in ("P1", "P2"):
                    acc["est_abs_err"] += float(np.abs(est - level).sum())
                    acc["est_n"] += n
            level[seq] = 0.0
            over_prev[seq] = False
            anc_lvl[seq] = 0.0
            anc_day[seq] = t

        # ---------------- noite: geração e transbordo ----------------
        level = level + gen[t] / cap
        over = level > 1.0 + 1e-9
        spill = np.where(over, (level - 1.0) * cap, 0.0)
        level = np.minimum(level, 1.0)
        if measuring:
            acc["gen_kg"] += float(gen[t].sum())
            acc["spill_kg"] += float(spill.sum())
            ev = int((over & ~over_prev).sum())
            acc["overflow_events"] += ev
            acc["overflow_nights"] += int(over.sum())
            if record_daily:
                daily.append(dict(t=t, collect=day is not None,
                                  km=(day["km_urban"] + day["km_hwy"]) if day else 0.0,
                                  stops=day["stops"] if day else 0,
                                  shifts=day["shifts"] if day else 0,
                                  overflow_events=ev, spill_kg=float(spill.sum())))
        over_prev = over

    stock1 = float((level * cap).sum())
    # balanço de massa: gerado = coletado + transbordado + variação de estoque
    bal = acc["gen_kg"] - (acc["mass_kg"] + acc["spill_kg"] + (stock1 - stock0))
    assert abs(bal) < 1e-6 * max(1.0, acc["gen_kg"]), f"balanço de massa violado: {bal}"

    res = dict(acc)
    res["stock_start_kg"] = stock0
    res["stock_end_kg"] = stock1
    res["fill_at_collection"] = acc["fill_sum"] / max(acc["stops"], 1)
    res["est_mae"] = acc["est_abs_err"] / acc["est_n"] if acc["est_n"] else float("nan")
    res["policy"] = policy
    res["rep"] = rep
    if record_daily:
        res["daily"] = daily
    return res
