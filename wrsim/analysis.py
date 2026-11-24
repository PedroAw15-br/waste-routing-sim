"""Consolida os resultados: estatísticas, diferenças pareadas, análise econômica,
tabelas (JSON) e figuras (PNG, 300 dpi).

Uso:  python -m wrsim.analysis [--res results] [--fig figures]
"""
import argparse
import json
import os
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from .params import Params
from .economics import (Costs, annual_savings, npv, breakeven_labor_share, capex, opex,
                        operating_cost)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POL_COLOR = {"B1": C[0], "B2": C[1], "P1": C[2], "P0": C[6], "P0S": C[4], "P2": C[3],
             "P2I": C[5], "P2M": C[7]}
POL_MARK = {"B1": "s", "B2": "D", "P1": "^", "P0": "v", "P0S": "<", "P2": "o", "P2I": "P", "P2M": "X"}
POL_LABEL = {"B1": "B1 – fixa diária", "B2": "B2 – fixa setorizada",
             "P1": "P1 – limiar fixo", "P0": "P0 – preditiva sem sensores",
             "P0S": "P0S – sem sensores, taxa por setor", "P2": "P2 – preditiva com sensores",
             "P2I": "P2 com sensores ideais", "P2M": "P2 com inclusão oportunista"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "figure.dpi": 100,
})


# ---------------------------------------------------------------------------
# Métricas por replicação
# ---------------------------------------------------------------------------
def metrics(r, p: Params):
    nd = r["collect_days"]
    col_t = max(r["mass_kg"], 1.0) / 1000.0           # massa coletada (t)
    crew_min = r["route_min"] + r["shifts"] * p.shift_fixed_min
    crew_h = crew_min / 60
    ot_h = r.get("extra_shifts", 0) * (crew_h / max(r["shifts"], 1))
    month = 30.0 / p.measure_days
    km = r["km_urban"] + r["km_hwy"]
    ovs = r.get("overflow_by_sector", [np.nan] * 3)
    return {
        "km_d": km / nd, "km_urb_d": r["km_urban"] / nd, "km_hwy_d": r["km_hwy"] / nd,
        "crew_h_d": crew_h / nd, "shifts_d": r["shifts"] / nd, "max_shifts": r["max_shifts"],
        "stops_d": r["stops"] / nd, "trips_d": r["trips"] / nd,
        "fuel_d": r["fuel_l"] / nd, "co2_d": r["fuel_l"] * p.co2_kg_l / nd,
        "fill": r["fill_at_collection"],
        "ovf_month": r["overflow_events"] * month,
        "ovf_cen_month": ovs[0] * month, "ovf_A_month": ovs[1] * month, "ovf_B_month": ovs[2] * month,
        "spill_pct": 100 * r["spill_kg"] / r["gen_kg"],
        "spill_t_month": r["spill_kg"] / 1000 * month,
        "dstock_t": (r["stock_end_kg"] - r["stock_start_kg"]) / 1000,
        "days_over2_pct": 100 * r.get("days_over2", 0) / nd,
        "extra_shifts_d": r.get("extra_shifts", 0) / nd,
        # por tonelada COLETADA (anualização sem viés de estoque no fim do horizonte)
        "fuel_t": r["fuel_l"] / col_t, "km_t": km / col_t, "crew_h_t": crew_h / col_t,
        "shifts_t": r["shifts"] / col_t, "ot_h_t": ot_h / col_t,
        "est_mae": r.get("est_mae", float("nan")),
    }


def br(v, dec=1, sign=False):
    t = f"{v:+.{dec}f}" if sign else f"{v:.{dec}f}"
    return t.replace(".", ",").replace("-", "−")


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0, 0.0
    s = x.std(ddof=1)
    h = stats.t.ppf(0.975, len(x) - 1) * s / np.sqrt(len(x))
    return m, s, h


def paired(xp, xb):
    xp, xb = np.asarray(xp, float), np.asarray(xb, float)
    d = xp - xb
    m, s, h = ci95(d)
    base = xb.mean()
    try:
        pw = stats.wilcoxon(xp, xb).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        pw = 1.0
    same_sign = bool(np.all(d > 0) or np.all(d < 0))
    return {"diff": m, "rel_pct": 100 * m / base if base else float("nan"),
            "rel_lo": 100 * (m - h) / base if base else float("nan"),
            "rel_hi": 100 * (m + h) / base if base else float("nan"),
            "p_wilcoxon": float(pw), "all_same_sign": same_sign, "n": int(len(d))}


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=9,
            fontweight="bold", color=ec)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.085 - i * 0.042, ln, ha="center", va="top", fontsize=7.6, color=INK)


def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _box(ax, 0.1, 0.70, 0.8, 0.27, "CAMADA DE PERCEPÇÃO",
         ["Nó sensor em cada contêiner: ESP32 + rádio LoRa (SX1276/SX1262)",
          "Sensor ultrassônico à prova d'água · bateria com sono profundo",
          "Leituras periódicas do nível de enchimento (% da altura útil)"], "#e8f1fc", "#1c5cab")
    _box(ax, 0.1, 0.375, 0.8, 0.25, "CAMADA DE COMUNICAÇÃO",
         ["LoRaWAN classe A · gateway em ponto elevado + gateway redundante",
          "Servidor de rede de código aberto · leituras perdidas tratadas na aplicação"],
         "#e6f5ee", "#0e7a52")
    _box(ax, 0.1, 0.03, 0.8, 0.29, "CAMADA DE APLICAÇÃO",
         ["Estimação do nível e da taxa de enchimento (média móvel exponencial)",
          "Seleção dos pontos: limiar + risco previsto de transbordo (calendário)",
          "Dijkstra (matriz de distâncias) + AG memético + Split (capacidade, aterro)",
          "Sem sensores (P0): taxa histórica por ponto no lugar da leitura"], "#fdf0e8", "#b4471b")
    for y0, y1 in ((0.70, 0.625), (0.375, 0.32)):
        ax.add_patch(FancyArrowPatch((0.33, y0), (0.33, y1), arrowstyle="-|>", mutation_scale=10, color=INK2))
        ax.add_patch(FancyArrowPatch((0.67, y1), (0.67, y0), arrowstyle="-|>", mutation_scale=10, color=INK2))
        ax.text(0.345, (y0 + y1) / 2, "telemetria", fontsize=7, color=INK2, va="center", style="italic")
        ax.text(0.685, (y0 + y1) / 2, "configuração / rotas", fontsize=7, color=INK2, va="center", style="italic")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_flow(path):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [("1. Leitura dos sensores\n(P0: última coleta)", "#e8f1fc", "#1c5cab"),
             ("2. Estimação do nível\ne da taxa (EWMA)", "#e8f1fc", "#1c5cab"),
             ("3. Previsão até a\npróxima coleta", "#fdf0e8", "#b4471b"),
             ("4. Seleção: limiar ou\nrisco de transbordo", "#fdf0e8", "#b4471b"),
             ("5. Sequenciamento\n(AG memético)", "#fdf0e8", "#b4471b"),
             ("6. Split por capacidade\ne turnos da frota", "#e6f5ee", "#0e7a52")]
    pos = [(0.02, 0.58), (0.36, 0.58), (0.70, 0.58), (0.70, 0.10), (0.36, 0.10), (0.02, 0.10)]
    w, h = 0.28, 0.30
    for (t, fc, ec), (x, y) in zip(steps, pos):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8, color=INK)
    arrows = [((0.30, 0.73), (0.36, 0.73)), ((0.64, 0.73), (0.70, 0.73)), ((0.84, 0.58), (0.84, 0.40)),
              ((0.70, 0.25), (0.64, 0.25)), ((0.36, 0.25), (0.30, 0.25))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=INK2))
    ax.add_patch(FancyArrowPatch((0.16, 0.40), (0.16, 0.58), arrowstyle="-|>", mutation_scale=10,
                                 color="#0e7a52", ls="--"))
    ax.text(0.18, 0.49, "novo ciclo (dia seguinte)", fontsize=7, color="#0e7a52", style="italic", va="center")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_city(cityj, path):
    xy = np.array(cityj["coords"]) / 1000
    E = np.array(cityj["edges"])
    pn = np.array(cityj["point_node"]); sec = np.array(cityj["sector"]); nc = np.array(cityj["n_cont"])
    fig, ax = plt.subplots(figsize=(6.3, 5.4))
    ax.grid(False)
    for a, b in E:
        ax.plot(xy[[a, b], 0], xy[[a, b], 1], color="#c9c8c2", lw=0.7, zorder=1)
    labels = {0: "Centro comercial (coleta diária em B2)", 1: "Setor A (seg/qua/sex em B2)",
              2: "Setor B (ter/qui/sáb em B2)"}
    cols = {0: C[1], 1: C[0], 2: C[2]}
    marks = {0: "D", 1: "o", 2: "s"}
    for s in (0, 1, 2):
        m = sec == s
        ax.scatter(xy[pn[m], 0], xy[pn[m], 1], s=10 + 12 * nc[m], c=cols[s], marker=marks[s],
                   edgecolor="white", linewidth=0.6, label=labels[s], zorder=3)
    g, x = cityj["garage"], cityj["exit"]
    ax.scatter(*xy[g], s=110, marker="P", c=INK, zorder=4, label="Garagem")
    ax.scatter(*xy[x], s=110, marker="X", c=C[7], zorder=4, label="Saída para o aterro (15 km)")
    ax.annotate("", xy=(xy[x, 0] + 0.25, xy[x, 1]), xytext=xy[x],
                arrowprops=dict(arrowstyle="-|>", color=C[7], lw=1.5))
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=7.5, markerscale=0.9)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_daily(main, p, path, pols=("B2", "P1", "P0", "P2")):
    T0 = p.warmup_days
    x = np.arange(1, p.measure_days + 1)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    for pol in pols:
        S = np.full((len(main[pol]), p.measure_days), np.nan)
        O = np.zeros((len(main[pol]), p.measure_days))
        for i, r in enumerate(main[pol]):
            for d in r["daily"]:
                j = d["t"] - T0
                if d["collect"]:
                    S[i, j] = d["stops"]
                O[i, j] = d["overflow_events"]
        y = np.nanmean(S, axis=0)
        m = ~np.isnan(y)
        axes[0].plot(x[m], y[m], color=POL_COLOR[pol], marker=POL_MARK[pol], ms=3, lw=1.3,
                     label=POL_LABEL[pol])
        axes[1].plot(x, O.mean(axis=0), color=POL_COLOR[pol], lw=1.3, label=POL_LABEL[pol])
    from .sim import PEAKS, HOLIDAYS
    for ax in axes:
        for ev in PEAKS:
            ax.axvspan(ev[0] - T0 + 0.5, ev[-1] - T0 + 1.5, color="#f3e6c8", alpha=0.6, lw=0)
        for h in HOLIDAYS:
            ax.axvline(h - T0 + 1, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("Pontos visitados por\ndia de coleta")
    axes[1].set_ylabel("Eventos de transbordo\npor noite")
    axes[1].set_xlabel("Dia do período de avaliação (faixas: eventos de pico; pontilhado: feriado)")
    axes[0].legend(ncol=2, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, 1.30))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_forest(comp, path):
    keys = [("km_d", "Distância percorrida"), ("fuel_d", "Consumo de diesel"),
            ("crew_h_d", "Horas de equipe"), ("shifts_d", "Turnos de caminhão"),
            ("stops_d", "Pontos visitados"), ("ovf_month", "Eventos de transbordo"),
            ("spill_t_month", "Massa transbordada")]
    sets = (("P0_vs_B2", "P0 (sem sensores) em relação a B2", POL_COLOR["P0"], "v", 0.24),
            ("P2_vs_B2", "P2 (com sensores) em relação a B2", POL_COLOR["P2"], "o", 0.0),
            ("P2_vs_P0", "P2 em relação a P0 (efeito da leitura dos sensores)", INK2, "s", -0.24))
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ys = np.arange(len(keys))[::-1]
    for key_c, lab_c, col, mk, off in sets:
        for y, (key, lab) in zip(ys, keys):
            c = comp[key_c][key]
            ax.errorbar(c["rel_pct"], y + off, xerr=[[c["rel_pct"] - c["rel_lo"]], [c["rel_hi"] - c["rel_pct"]]],
                        fmt=mk, color=col, ms=4.5, capsize=2, lw=1.1, label=lab_c if y == ys[0] else None)
            ax.text(c["rel_hi"] + 1.5 if c["rel_pct"] >= 0 else c["rel_lo"] - 1.5, y + off,
                    br(c['rel_pct'], 1, True) + "%", va="center", ha="left" if c["rel_pct"] >= 0 else "right",
                    fontsize=6.5, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in keys])
    ax.set_xlabel("Diferença relativa média (%) com IC 95% (30 replicações pareadas)")
    lo = min(comp[k][kk]["rel_lo"] for k, *_ in sets for kk, _ in keys)
    hi = max(comp[k][kk]["rel_hi"] for k, *_ in sets for kk, _ in keys)
    ax.set_xlim(lo - 12, hi + 12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=1, fontsize=7.3)
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_frontier(points, path):
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for pol, pts in points.items():
        xs = [q["x"] for q in pts]; ys_ = [q["y"] for q in pts]
        if len(pts) > 1:
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys_)[o], color=POL_COLOR[pol], lw=1.2, alpha=0.9)
        ax.scatter(xs, ys_, color=POL_COLOR[pol], marker=POL_MARK[pol], s=34, zorder=3,
                   edgecolor="white", linewidth=0.8, label=POL_LABEL[pol])
        for q in pts:
            if q.get("lab"):
                ax.annotate(q["lab"], (q["x"], q["y"]), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.4, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Horas de equipe por dia de coleta")
    ax.set_ylabel("Eventos de transbordo por mês (escala log)")
    ax.legend(fontsize=6.8, loc="upper right", ncol=2)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_landfill(sens, path):
    dists = [5, 15, 30]
    keys = [("km_d", "Distância"), ("fuel_d", "Diesel"), ("crew_h_d", "Horas de equipe")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for ax, (cmpk, title) in zip(axes, (("P0_vs_B2", "P0 (sem sensores) em relação a B2"),
                                        ("P2_vs_B2", "P2 (com sensores) em relação a B2"))):
        for k, (key, lab) in enumerate(keys):
            ys_ = [sens[f"aterro={d}"][cmpk][key]["rel_pct"] for d in dists]
            ax.plot(dists, ys_, color=C[k], marker="os^"[k], lw=1.5, ms=5, label=lab)
            ax.annotate(br(ys_[-1], 0, True) + "%", (dists[-1], ys_[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, va="center", color=INK2)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Distância da saída urbana ao aterro (km)")
        ax.set_xticks(dists); ax.set_xlim(3, 35)
    axes[0].set_ylabel("Diferença relativa (%)")
    axes[0].legend(fontsize=7.3, loc="lower left")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_npv(curves, path):
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    sh = np.linspace(0, 1, 101)
    for lab, col, v, be in curves:
        ax.plot(sh * 100, np.array(v) / 1000, color=col, lw=1.8, label=lab)
        if 0 <= be <= 1:
            ax.scatter([be * 100], [0], color=col, s=28, zorder=3)
            ax.annotate(br(be * 100, 0) + "%", (be * 100, 0), xytext=(4, -12), textcoords="offset points",
                        fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("Parcela das horas de equipe liberadas convertida em economia, λ (%)")
    ax.set_ylabel("VPL em 5 anos (mil R$)")
    ax.legend(fontsize=7.0, loc="upper left")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--fig", default="figures")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    base = Params()
    sim = json.load(open(os.path.join(a.res, "sim_results.json")))
    cityj = json.load(open(os.path.join(a.res, "city.json")))
    bench = json.load(open(os.path.join(a.res, "benchmark.json")))

    groups = defaultdict(lambda: defaultdict(list))
    for r in sim:
        groups[r["tag"]][r["policy"]].append(r)
    for g in groups.values():
        for pol in g:
            g[pol].sort(key=lambda r: r["rep"])

    def params_for(tag):
        if tag.startswith("fixo="):
            return replace(base, shift_fixed_min=float(tag[5:]))
        if tag.startswith("aquecimento="):
            return replace(base, warmup_days=int(tag[12:]))
        return base

    out = {"city": cityj["summary"], "params": base.to_dict(), "groups": {}}
    M = {}
    PAIRS = (("P2", "B2"), ("P2", "B1"), ("P0", "B2"), ("P0", "B1"), ("P0S", "B2"), ("P2", "P0"),
             ("P2", "P0S"), ("P2I", "P2"), ("P2I", "B2"), ("P2M", "P2"), ("P2M", "B2"), ("P1", "B2"),
             ("P1", "B1"), ("B2", "B1"))
    for tag, g in groups.items():
        pp = params_for(tag)
        M[tag] = {pol: [metrics(r, pp) for r in rs] for pol, rs in g.items()}
        desc = {}
        for pol, ms in M[tag].items():
            desc[pol] = {k: dict(zip(("mean", "sd", "ci"), ci95([m[k] for m in ms]))) for k in ms[0]}
            desc[pol]["n_reps"] = len(ms)
            rs = g[pol]
            desc[pol]["n_sensor_nodes"] = rs[0].get("n_sensor_nodes", 0)
            desc[pol]["n_sensor_points"] = rs[0].get("n_sensor_points", 0)
            desc[pol]["maygo_added_d"] = float(np.mean([r.get("maygo_added", 0) / r["collect_days"] for r in rs]))
        comps = {}
        for pa, pb in PAIRS:
            if pa in M[tag] and pb in M[tag] and len(M[tag][pa]) == len(M[tag][pb]):
                comps[f"{pa}_vs_{pb}"] = {k: paired([m[k] for m in M[tag][pa]], [m[k] for m in M[tag][pb]])
                                          for k in M[tag][pa][0] if k not in ("est_mae",)}
        out["groups"][tag] = {"desc": desc, "comp": comps}

    main_g = out["groups"]["main"]
    dist = {}
    for pol in groups["main"]:
        sh = [d["shifts"] for r in groups["main"][pol] for d in r["daily"] if d["collect"]]
        vals, cnt = np.unique(sh, return_counts=True)
        dist[pol] = {"share": {str(int(v)): float(c) / len(sh) for v, c in zip(vals, cnt)},
                     "p95": float(np.percentile(sh, 95)), "mean": float(np.mean(sh))}
    out["shift_distribution"] = dist
    out["replication_precision"] = {
        pol: {k: 100 * main_g["desc"][pol][k]["ci"] / abs(main_g["desc"][pol][k]["mean"])
              for k in ("km_d", "fuel_d", "crew_h_d", "ovf_month")}
        for pol in main_g["desc"]}

    # ---------------- análise econômica ----------------
    costs = Costs()
    n_nodes = cityj["summary"]["conteineres"]
    annual_t = base.urban_pop * base.per_capita_kg * 365 / 1000

    def annual(pol, tag="main"):
        d = out["groups"][tag]["desc"][pol]
        return {"fuel_l": d["fuel_t"]["mean"] * annual_t, "km": d["km_t"]["mean"] * annual_t,
                "crew_h": d["crew_h_t"]["mean"] * annual_t, "shifts": d["shifts_t"]["mean"] * annual_t,
                "ot_h": d["ot_h_t"]["mean"] * annual_t}

    econ = {"costs": costs.to_dict(), "n_nodes": n_nodes, "annual_t": annual_t,
            "capex_iot": capex(costs, n_nodes), "opex_iot": opex(costs, n_nodes),
            "capex_p0": capex(costs, n_nodes, "p0"), "opex_p0": opex(costs, n_nodes, "p0"),
            "capex_sensors": capex(costs, n_nodes, "sensors"), "opex_sensors": opex(costs, n_nodes, "sensors")}
    for pol in main_g["desc"]:
        q = annual(pol)
        econ[f"annual_{pol}"] = q
        econ[f"opcost_{pol}"] = operating_cost(costs, q, "hour")
        econ[f"opcost_shift_{pol}"] = operating_cost(costs, q, "shift")

    cases = {  # nome: (referência, proposta, sistema)
        "P2_vs_B2": ("B2", "P2", "iot"), "P2_vs_B1": ("B1", "P2", "iot"),
        "P0_vs_B2": ("B2", "P0", "p0"), "P0_vs_B1": ("B1", "P0", "p0"),
        "P0S_vs_B2": ("B2", "P0S", "p0"), "P2_vs_P0": ("P0", "P2", "sensors"),
        "P2_vs_P0S": ("P0S", "P2", "sensors"),
    }
    econ["cases"] = {}
    for name, (b, pr, sysn) in cases.items():
        A, P = annual(b), annual(pr)
        e = {}
        for unit in ("hour", "shift"):
            s0 = annual_savings(costs, A, P, 0.0, unit)
            s1 = annual_savings(costs, A, P, 1.0, unit)
            e[unit] = {"savings0": s0, "savings1": s1,
                       "npv0": npv(costs, n_nodes, s0["total"], sysn),
                       "npv1": npv(costs, n_nodes, s1["total"], sysn),
                       "breakeven": breakeven_labor_share(costs, n_nodes, A, P, unit, sysn),
                       "curve": [npv(costs, n_nodes, annual_savings(costs, A, P, s, unit)["total"], sysn)["npv"]
                                 for s in np.linspace(0, 1, 101)]}
        e["system"] = sysn
        econ["cases"][name] = e
    # custo de operação diária do sistema: 0, 10 e 20 mil R$/ano
    econ["ops_sensitivity"] = {}
    for ops in (0.0, 10000.0, 20000.0):
        cc = replace(costs, ops_rs_yr=ops)
        econ["ops_sensitivity"][str(int(ops))] = {
            n: breakeven_labor_share(cc, n_nodes, annual(b), annual(pr), "hour", s)
            for n, (b, pr, s) in cases.items()}
    # tornado (P2 x B2, λ = 50%, valoração por hora)
    tor = {}
    A, P = annual("B2"), annual("P2")
    ref = npv(costs, n_nodes, annual_savings(costs, A, P, 0.5)["total"])["npv"]
    for name, lo, hi in (("diesel_rs_l", 0.75, 1.25), ("maint_rs_km", 0.7, 1.3), ("crew_rs_h", 0.7, 1.3),
                         ("node_hw_rs", 0.6, 1.8), ("node_replace_rate", 0.5, 2.0), ("discount", 0.6, 1.4),
                         ("ops_rs_yr", 0.0, 2.0)):
        vals = []
        for f in (lo, hi):
            cc = replace(costs, **{name: getattr(costs, name) * f})
            vals.append(npv(cc, n_nodes, annual_savings(cc, A, P, 0.5)["total"])["npv"])
        tor[name] = {"lo_factor": lo, "hi_factor": hi, "npv_lo": vals[0], "npv_hi": vals[1]}
    econ["npv_ref_half"] = ref
    econ["tornado"] = tor
    # distância ao aterro (30 replicações)
    econ["landfill"] = {}
    for d in (5, 30):
        tag = f"aterro={d}"
        econ["landfill"][str(d)] = {
            n: breakeven_labor_share(costs, n_nodes, annual(b, tag), annual(pr, tag), "hour", s)
            for n, (b, pr, s) in cases.items() if b in out["groups"][tag]["desc"] and pr in out["groups"][tag]["desc"]}
    # cobertura parcial de sensores (P2 com k% dos pontos; restante como P0)
    cov = {}
    for k in (0.25, 0.5, 0.75):
        tag = f"cobertura={k}"
        d = out["groups"][tag]["desc"]["P2"]
        nn = d["n_sensor_nodes"]
        Pk = annual("P2", tag)
        for b in ("B2",):
            A = annual(b)
            cov[str(k)] = {"n_nodes": nn, "crew_h_d": d["crew_h_d"]["mean"], "ovf_month": d["ovf_month"]["mean"],
                           "breakeven_vs_B2": breakeven_labor_share(costs, nn, A, Pk, "hour", "iot"),
                           "npv1_vs_B2": npv(costs, nn, annual_savings(costs, A, Pk, 1.0)["total"], "iot")["npv"]}
    out["coverage"] = cov
    out["economics"] = econ

    # ---------------- métrica S (pré-registro) ----------------
    # custo de P0 interpolado no nível de transbordos de P2 (z = 2,33)
    zs = {1.645: "z30=1.645", 2.33: "main", 3.09: "z30=3.09"}
    curve = sorted(((out["groups"][t]["desc"]["P0"]["ovf_month"]["mean"],
                     operating_cost(costs, annual("P0", t), "hour"), z) for z, t in zs.items()))
    target = main_g["desc"]["P2"]["ovf_month"]["mean"]
    xo = [c[0] for c in curve]; yc = [c[1] for c in curve]
    c_p0_interp = float(np.interp(target, xo, yc))
    c_b2 = econ["opcost_B2"]; c_p2 = econ["opcost_P2"]
    out["S_metric"] = {"C_B2": c_b2, "C_P2": c_p2, "C_P0_raw": econ["opcost_P0"],
                       "C_P0_at_P2_service": c_p0_interp, "P2_ovf_target": target,
                       "P0_curve": [{"ovf_month": c[0], "opcost": c[1], "z": c[2]} for c in curve],
                       "extrapolated": bool(target < min(xo) or target > max(xo)),
                       "S": (c_p0_interp - c_p2) / (c_b2 - c_p2),
                       "S_raw": (econ["opcost_P0"] - c_p2) / (c_b2 - c_p2)}
    # mesmo nível de serviço para P2 (curva em z)
    out["P2_curve"] = [{"z": z, "ovf_month": out["groups"][t]["desc"]["P2"]["ovf_month"]["mean"],
                        "crew_h_d": out["groups"][t]["desc"]["P2"]["crew_h_d"]["mean"],
                        "opcost": operating_cost(costs, annual("P2", t), "hour")} for z, t in zs.items()]
    out["P0_curve"] = [{"z": z, "ovf_month": out["groups"][t]["desc"]["P0"]["ovf_month"]["mean"],
                        "crew_h_d": out["groups"][t]["desc"]["P0"]["crew_h_d"]["mean"],
                        "opcost": operating_cost(costs, annual("P0", t), "hour")} for z, t in zs.items()]

    # ---------------- benchmark ----------------
    bt = defaultdict(lambda: defaultdict(list))
    for row in bench:
        bt[row["method"]][row["n"]].append(row)
    out["benchmark"] = {m: {str(n): {"gap_mean": float(np.mean([r["gap_pct"] for r in rows])),
                                     "gap_max": float(np.max([r["gap_pct"] for r in rows])),
                                     "time_mean": float(np.mean([r["time_s"] for r in rows]))}
                            for n, rows in d.items()} for m, d in bt.items()}

    # ---------------- figuras ----------------
    fig_architecture(os.path.join(a.fig, "fig1_arquitetura.png"))
    fig_flow(os.path.join(a.fig, "fig2_fluxo.png"))
    fig_city(cityj, os.path.join(a.fig, "fig3_cidade.png"))
    fig_forest(main_g["comp"], os.path.join(a.fig, "fig4_comparacao.png"))
    fig_daily({pol: groups["main"][pol] for pol in ("B2", "P1", "P0", "P2")}, base,
              os.path.join(a.fig, "fig5_serie_diaria.png"))
    pts = defaultdict(list)
    for pol in ("B1", "B2", "P0S", "P2I", "P2M"):
        d = main_g["desc"][pol]
        pts[pol].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"], "lab": ""})
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        d = out["groups"][f"thr={thr}"]["desc"]["P1"]
        pts["P1"].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"], "lab": "θ = " + br(thr, 1)})
    for pol in ("P0", "P2"):
        for z, t in sorted(zs.items()):
            d = out["groups"][t]["desc"][pol]
            pts[pol].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"],
                             "lab": ("z = " + str(z).replace(".", ",")) if pol == "P2" else ""})
    order = ("B1", "B2", "P1", "P0", "P0S", "P2", "P2I", "P2M")
    fig_frontier({k: pts[k] for k in order if k in pts}, os.path.join(a.fig, "fig6_fronteira.png"))
    sens_land = {"aterro=15": main_g["comp"], "aterro=5": out["groups"]["aterro=5"]["comp"],
                 "aterro=30": out["groups"]["aterro=30"]["comp"]}
    fig_landfill(sens_land, os.path.join(a.fig, "fig7_aterro.png"))
    cs = econ["cases"]
    fig_npv([("P2 (sensores) em relação a B2", POL_COLOR["P2"], cs["P2_vs_B2"]["hour"]["curve"], cs["P2_vs_B2"]["hour"]["breakeven"]),
             ("P0 (sem sensores) em relação a B2", POL_COLOR["P0"], cs["P0_vs_B2"]["hour"]["curve"], cs["P0_vs_B2"]["hour"]["breakeven"]),
             ("Sensores sobre P0 (P2 em relação a P0)", INK2, cs["P2_vs_P0"]["hour"]["curve"], cs["P2_vs_P0"]["hour"]["breakeven"])],
            os.path.join(a.fig, "fig8_vpl.png"))

    with open(os.path.join(a.res, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("ok")


if __name__ == "__main__":
    main()
