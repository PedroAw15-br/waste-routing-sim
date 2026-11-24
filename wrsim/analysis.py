"""Consolida os resultados: estatísticas, testes pareados, análise econômica,
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
from .economics import Costs, annual_savings, npv, breakeven_labor_share, capex, opex

# paleta categórica (ordem fixa) e tokens de texto
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POL_COLOR = {"B1": C[0], "B2": C[1], "P1": C[2], "P2": C[3]}
POL_MARK = {"B1": "s", "B2": "D", "P1": "^", "P2": "o"}
POL_LABEL = {"B1": "B1 – fixa diária", "B2": "B2 – fixa setorizada",
             "P1": "P1 – dinâmica por limiar", "P2": "P2 – dinâmica preditiva"}
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
    gen_t = r["gen_kg"] / 1000.0
    crew_min = r["route_min"] + r["shifts"] * p.shift_fixed_min
    month = 30.0 / p.measure_days
    km = r["km_urban"] + r["km_hwy"]
    return {
        "km_d": km / nd, "km_urb_d": r["km_urban"] / nd, "km_hwy_d": r["km_hwy"] / nd,
        "route_h_d": r["route_min"] / 60 / nd, "crew_h_d": crew_min / 60 / nd,
        "shifts_d": r["shifts"] / nd, "max_shifts": r["max_shifts"],
        "stops_d": r["stops"] / nd, "cont_d": r["containers"] / nd, "trips_d": r["trips"] / nd,
        "fuel_d": r["fuel_l"] / nd, "co2_d": r["fuel_l"] * p.co2_kg_l / nd,
        "fill": r["fill_at_collection"],
        "ovf_month": r["overflow_events"] * month,
        "ovf_nights_month": r["overflow_nights"] * month,
        "spill_pct": 100 * r["spill_kg"] / r["gen_kg"],
        "spill_t_month": r["spill_kg"] / 1000 * month,
        "fuel_t": r["fuel_l"] / gen_t, "km_t": km / gen_t, "crew_h_t": crew_min / 60 / gen_t,
        "shifts_t": r["shifts"] / gen_t,
        "overtime_h_month": r["overtime_min"] / 60 * month,
        "deferrals": r.get("deferrals", 0), "est_mae": r.get("est_mae", float("nan")),
    }


def br(v, dec=1, sign=False):
    t = f"{v:+.{dec}f}" if sign else f"{v:.{dec}f}"
    return t.replace(".", ",").replace("-", "\u2212")


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
    dz = m / s if s > 0 else float("inf")
    return {"diff": m, "rel_pct": 100 * m / base, "rel_lo": 100 * (m - h) / base,
            "rel_hi": 100 * (m + h) / base, "p_wilcoxon": float(pw), "dz": float(dz)}


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
          "Rotas para a frota e registro de indicadores"], "#fdf0e8", "#b4471b")
    for y0, y1 in ((0.70, 0.625), (0.375, 0.32)):
        ax.add_patch(FancyArrowPatch((0.33, y0), (0.33, y1), arrowstyle="-|>", mutation_scale=10, color=INK2))
        ax.add_patch(FancyArrowPatch((0.67, y1), (0.67, y0), arrowstyle="-|>", mutation_scale=10, color=INK2))
        ax.text(0.345, (y0 + y1) / 2, "telemetria", fontsize=7, color=INK2, va="center", style="italic")
        ax.text(0.685, (y0 + y1) / 2, "configuração / rotas", fontsize=7, color=INK2, va="center", style="italic")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_flow(path):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    steps = [("1. Leitura dos\nsensores", "#e8f1fc", "#1c5cab"),
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


def fig_daily(main, p, path):
    T0 = p.warmup_days
    x = np.arange(1, p.measure_days + 1)
    fig, axes = plt.subplots(2, 1, figsize=(6.3, 4.6), sharex=True)
    for pol in ("B1", "B2", "P2"):
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
        axes[0].plot(x[m], y[m], color=POL_COLOR[pol], marker=POL_MARK[pol], ms=3, lw=1.4,
                     label=POL_LABEL[pol])
        axes[1].plot(x, O.mean(axis=0), color=POL_COLOR[pol], lw=1.4, label=POL_LABEL[pol])
    from .sim import PEAKS, HOLIDAYS
    for ax in axes:
        for ev in PEAKS:
            ax.axvspan(ev[0] - T0 + 0.5, ev[-1] - T0 + 1.5, color="#f3e6c8", alpha=0.6, lw=0)
        for h in HOLIDAYS:
            ax.axvline(h - T0 + 1, color=INK2, lw=0.8, ls=":")
    axes[0].set_ylabel("Pontos visitados por\ndia de coleta")
    axes[1].set_ylabel("Eventos de transbordo\npor noite")
    axes[1].set_xlabel("Dia do período de avaliação (faixas: eventos de pico; pontilhado: feriado)")
    axes[0].legend(ncol=3, fontsize=7.3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_forest(comp, path):
    keys = [("km_d", "Distância percorrida"), ("fuel_d", "Consumo de diesel"),
            ("crew_h_d", "Horas de equipe"), ("shifts_d", "Turnos de caminhão"),
            ("stops_d", "Paradas (pontos visitados)"), ("ovf_month", "Eventos de transbordo"),
            ("spill_t_month", "Massa transbordada")]
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    ys = np.arange(len(keys))[::-1]
    for k, (b, off) in enumerate((("B2", 0.17), ("B1", -0.17))):
        col = POL_COLOR[b]
        for y, (key, lab) in zip(ys, keys):
            c = comp[f"P2_vs_{b}"][key]
            ax.errorbar(c["rel_pct"], y + off, xerr=[[c["rel_pct"] - c["rel_lo"]], [c["rel_hi"] - c["rel_pct"]]],
                        fmt=POL_MARK[b], color=col, ms=5, capsize=2, lw=1.2,
                        label=f"P2 em relação a {POL_LABEL[b]}" if y == ys[0] else None)
            ax.text(c["rel_hi"] + 2 if c["rel_pct"] >= 0 else c["rel_lo"] - 2, y + off,
                    br(c['rel_pct'], 1, True) + "%", va="center", ha="left" if c["rel_pct"] >= 0 else "right",
                    fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.9)
    ax.set_yticks(ys); ax.set_yticklabels([l for _, l in keys])
    ax.set_xlabel("Diferença relativa média (%) com IC 95% (30 replicações pareadas)")
    lo = min(min(comp[f"P2_vs_{b}"][k]["rel_lo"] for k, _ in keys) for b in ("B1", "B2"))
    hi = max(max(comp[f"P2_vs_{b}"][k]["rel_hi"] for k, _ in keys) for b in ("B1", "B2"))
    ax.set_xlim(lo - 14, hi + 14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=1, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_frontier(points, path):
    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    for pol in ("B1", "B2", "P1", "P2"):
        pts = points[pol]
        xs = [q["x"] for q in pts]; ys_ = [q["y"] for q in pts]
        if len(pts) > 1:
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys_)[o], color=POL_COLOR[pol], lw=1.2, alpha=0.9)
        ax.scatter(xs, ys_, color=POL_COLOR[pol], marker=POL_MARK[pol], s=36, zorder=3,
                   edgecolor="white", linewidth=0.8, label=POL_LABEL[pol])
        for q in pts:
            if q.get("lab"):
                ax.annotate(q["lab"], (q["x"], q["y"]), xytext=(4, 4), textcoords="offset points",
                            fontsize=6.8, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("Horas de equipe por dia de coleta")
    ax.set_ylabel("Eventos de transbordo por mês (escala log)")
    ax.legend(fontsize=7.3, loc="upper right")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_landfill(sens, path):
    dists = [5, 15, 30]
    keys = [("km_d", "Distância"), ("fuel_d", "Diesel"), ("crew_h_d", "Horas de equipe")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
    for ax, b in zip(axes, ("B2", "B1")):
        for k, (key, lab) in enumerate(keys):
            ys_ = [sens[f"aterro={d}"][f"P2_vs_{b}"][key]["rel_pct"] for d in dists]
            ax.plot(dists, ys_, color=C[k], marker="os^"[k], lw=1.5, ms=5, label=lab)
            ax.annotate(br(ys_[-1], 0, True) + "%", (dists[-1], ys_[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, va="center", color=INK2)
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_title(f"P2 em relação a {b}", fontsize=8.5)
        ax.set_xlabel("Distância da saída urbana ao aterro (km)")
        ax.set_xticks(dists); ax.set_xlim(3, 35)
    axes[0].set_ylabel("Diferença relativa (%)")
    axes[0].legend(fontsize=7.3, loc="lower left")
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_npv(econ, path):
    fig, ax = plt.subplots(figsize=(6.3, 3.0))
    sh = np.linspace(0, 1, 101)
    for b in ("B2", "B1"):
        v = np.array(econ[f"npv_curve_{b}"]) / 1000
        ax.plot(sh * 100, v, color=POL_COLOR[b], lw=1.8, label=f"P2 em relação a {POL_LABEL[b]}")
        be = econ[f"breakeven_share_{b}"]
        if 0 <= be <= 1:
            ax.scatter([be * 100], [0], color=POL_COLOR[b], s=30, zorder=3)
            ax.annotate(br(be * 100, 0) + "%", (be * 100, 0), xytext=(4, -12), textcoords="offset points",
                        fontsize=7, color=INK2)
    ax.axhline(0, color=INK2, lw=0.9)
    ax.set_xlabel("Parcela dos turnos-equipe liberados convertida em economia (%)")
    ax.set_ylabel("VPL em 5 anos (mil R$)")
    ax.legend(fontsize=7.3, loc="upper left")
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
        if tag.startswith("thr="):
            return replace(base, threshold=float(tag[4:]))
        if tag.startswith("z="):
            return replace(base, z_service=float(tag[2:]))
        if tag.startswith("aterro="):
            return replace(base, landfill_km=float(tag[7:]))
        if tag.startswith("geracao="):
            return replace(base, gen_scale=float(tag[8:]))
        return base

    out = {"city": cityj["summary"], "params": base.to_dict(), "groups": {}}
    M = {}
    for tag, g in groups.items():
        pp = params_for(tag)
        M[tag] = {pol: [metrics(r, pp) for r in rs] for pol, rs in g.items()}
        desc = {}
        for pol, ms in M[tag].items():
            desc[pol] = {k: dict(zip(("mean", "sd", "ci"), ci95([m[k] for m in ms]))) for k in ms[0]}
            desc[pol]["n_reps"] = len(ms)
        comps = {}
        for pa, pb in (("P2", "B2"), ("P2", "B1"), ("P1", "B2"), ("P1", "B1"), ("B2", "B1")):
            if pa in M[tag] and pb in M[tag] and len(M[tag][pa]) == len(M[tag][pb]):
                comps[f"{pa}_vs_{pb}"] = {k: paired([m[k] for m in M[tag][pa]], [m[k] for m in M[tag][pb]])
                                          for k in M[tag][pa][0] if k not in ("est_mae",)}
        out["groups"][tag] = {"desc": desc, "comp": comps}

    main_g = out["groups"]["main"]
    dist = {}
    for pol in ("B1", "B2", "P1", "P2"):
        sh = [d["shifts"] for r in groups["main"][pol] for d in r["daily"] if d["collect"]]
        vals, cnt = np.unique(sh, return_counts=True)
        dist[pol] = {"share": {str(int(v)): float(c) / len(sh) for v, c in zip(vals, cnt)},
                     "p95": float(np.percentile(sh, 95)), "mean": float(np.mean(sh))}
    out["shift_distribution"] = dist
    # adequação do número de replicações: semiamplitude relativa do IC95%
    out["replication_precision"] = {
        pol: {k: 100 * main_g["desc"][pol][k]["ci"] / abs(main_g["desc"][pol][k]["mean"])
              for k in ("km_d", "fuel_d", "crew_h_d", "ovf_month")}
        for pol in ("B1", "B2", "P1", "P2")}

    # ---------------- análise econômica ----------------
    costs = Costs()
    n_nodes = cityj["summary"]["conteineres"]
    annual_t = base.urban_pop * base.per_capita_kg * 365 / 1000

    def annual(pol, tag="main"):
        d = out["groups"][tag]["desc"][pol]
        return {"fuel_l": d["fuel_t"]["mean"] * annual_t, "km": d["km_t"]["mean"] * annual_t,
                "crew_h": d["crew_h_t"]["mean"] * annual_t, "shifts": d["shifts_t"]["mean"] * annual_t}

    econ = {"costs": costs.to_dict(), "n_nodes": n_nodes, "annual_t": annual_t,
            "capex": capex(costs, n_nodes), "opex": opex(costs, n_nodes)}
    for pol in ("B1", "B2", "P2"):
        econ[f"annual_{pol}"] = annual(pol)
    for b in ("B2", "B1"):
        A, P = annual(b), annual("P2")
        econ[f"savings0_{b}"] = annual_savings(costs, A, P, 0.0)
        econ[f"savings1_{b}"] = annual_savings(costs, A, P, 1.0)
        econ[f"npv0_{b}"] = npv(costs, n_nodes, econ[f"savings0_{b}"]["total"])
        econ[f"npv1_{b}"] = npv(costs, n_nodes, econ[f"savings1_{b}"]["total"])
        econ[f"breakeven_share_{b}"] = breakeven_labor_share(costs, n_nodes, A, P)
        econ[f"npv_curve_{b}"] = [npv(costs, n_nodes, annual_savings(costs, A, P, s)["total"])["npv"]
                                  for s in np.linspace(0, 1, 101)]
    # sensibilidade econômica (VPL com 50% das horas convertidas, P2 x B2)
    tor = {}
    A, P = annual("B2"), annual("P2")
    ref = npv(costs, n_nodes, annual_savings(costs, A, P, 0.5)["total"])["npv"]
    for name, lo, hi in (("diesel_rs_l", 0.75, 1.25), ("maint_rs_km", 0.7, 1.3), ("crew_rs_h", 0.7, 1.3),
                         ("node_hw_rs", 0.6, 1.8), ("node_replace_rate", 0.5, 2.0), ("discount", 0.6, 1.4)):
        vals = []
        for f in (lo, hi):
            cc = replace(costs, **{name: getattr(costs, name) * f})
            vals.append(npv(cc, n_nodes, annual_savings(cc, A, P, 0.5)["total"])["npv"])
        tor[name] = {"lo_factor": lo, "hi_factor": hi, "npv_lo": vals[0], "npv_hi": vals[1]}
    for d in (5, 30):
        tag = f"aterro={d}"
        Ad, Pd = annual("B2", tag), annual("P2", tag)
        tor[f"aterro_{d}km"] = {"npv": npv(costs, n_nodes, annual_savings(costs, Ad, Pd, 0.5)["total"])["npv"]}
    econ["npv_ref_half"] = ref
    econ["tornado"] = tor
    # custo do nó que zera o VPL (P2 x B1, sem horas convertidas)
    out["economics"] = econ

    # ---------------- benchmark ----------------
    bt = defaultdict(lambda: defaultdict(list))
    for row in bench:
        bt[row["method"]][row["n"]].append(row)
    out["benchmark"] = {m: {str(n): {"gap_mean": float(np.mean([r["gap_pct"] for r in rows])),
                                     "gap_max": float(np.max([r["gap_pct"] for r in rows])),
                                     "time_mean": float(np.mean([r["time_s"] for r in rows]))}
                            for n, rows in d.items()} for m, d in bt.items()}
    inst = defaultdict(dict)
    for row in bench:
        inst[(row["n"], row["inst"])][row["method"]] = row["cost_km"]
    out["benchmark_counts"] = {
        "n_inst": len(inst),
        "ga_improved_nn": sum(1 for v in inst.values() if v["AG puro"] < v["VMP"] - 1e-9),
        "ga_max_improvement_pct": max(100 * (1 - v["AG puro"] / v["VMP"]) for v in inst.values()),
        "memetic_best": sum(1 for v in inst.values() if v["AG memético"] <= min(v.values()) + 1e-9),
        "ortools_best": sum(1 for v in inst.values() if v["OR-Tools GLS"] <= min(v.values()) + 1e-9),
    }
    out["benchmark_overall"] = {m: {"gap_mean": float(np.mean([r["gap_pct"] for rows in d.values() for r in rows])),
                                    "time_mean": float(np.mean([r["time_s"] for rows in d.values() for r in rows]))}
                                for m, d in bt.items()}

    # ---------------- figuras ----------------
    fig_architecture(os.path.join(a.fig, "fig1_arquitetura.png"))
    fig_flow(os.path.join(a.fig, "fig2_fluxo.png"))
    fig_city(cityj, os.path.join(a.fig, "fig3_cidade.png"))
    raw_main = {pol: groups["main"][pol] for pol in ("B1", "B2", "P2")}
    fig_daily(raw_main, base, os.path.join(a.fig, "fig4_serie_diaria.png"))
    fig_forest(main_g["comp"], os.path.join(a.fig, "fig5_comparacao.png"))
    pts = {"B1": [], "B2": [], "P1": [], "P2": []}
    for pol in ("B1", "B2"):
        d = main_g["desc"][pol]
        pts[pol].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"], "lab": ""})
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        d = out["groups"][f"thr={thr}"]["desc"]["P1"]
        pts["P1"].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"], "lab": "θ = " + br(thr, 1)})
    for z in (1.28, 1.645, 2.33, 3.09):
        d = out["groups"][f"z={z}"]["desc"]["P2"]
        pts["P2"].append({"x": d["crew_h_d"]["mean"], "y": d["ovf_month"]["mean"], "lab": "z = " + str(z).replace(".", ",")})
    fig_frontier(pts, os.path.join(a.fig, "fig6_fronteira.png"))
    sens_land = {f"aterro={d}": out["groups"][f"aterro={d}"]["comp"] for d in (5, 15, 30)}
    fig_landfill(sens_land, os.path.join(a.fig, "fig7_aterro.png"))
    fig_npv(econ, os.path.join(a.fig, "fig8_vpl.png"))

    with open(os.path.join(a.res, "summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("ok")


if __name__ == "__main__":
    main()
