"""Figura do compromisso entre horas de equipe e transbordos (cidade sintética), em dois painéis:
(a) visão geral, em escala log, com todas as políticas; (b) ampliação linear da região em que
estão a coleta setorizada (B2) e as regras preditivas, com rótulos diretos nos pontos.

Uso:  python -m wrsim.fig_fronteira --summary results/summary.json --out figures/fig6_fronteira.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .analysis import POL_COLOR, POL_MARK, INK, INK2

NOME = {"B1": "B1 – coleta diária de todos os pontos", "B2": "B2 – coleta setorizada (prática atual)",
        "P1": "P1 – limiar fixo de enchimento", "P0": "P0 – preditiva sem sensores",
        "P2": "P2 – preditiva com sensores", "P0S": "P0S – sem sensores, taxa do setor",
        "P2I": "P2 com sensores ideais", "P2M": "P2 com inclusão oportunista"}
CURTO = {"B2": "B2\n(prática atual)", "P0S": "P0S", "P2I": "P2 ideal", "P2M": "P2 oportunista"}


def load(path):
    s = open(path, encoding="utf-8").read().replace("NaN", "null").replace("-Infinity", "-1e999").replace("Infinity", "1e999")
    return json.loads(s)


def pt(S, tag, pol):
    d = S["groups"][tag]["desc"][pol]
    return d["crew_h_d"]["mean"], d["ovf_month"]["mean"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.json")
    ap.add_argument("--out", default="figures/fig6_fronteira.png")
    a = ap.parse_args()
    S = load(a.summary)
    single = {p: pt(S, "main", p) for p in ("B1", "B2", "P0S", "P2I", "P2M")}
    p1 = [(thr, *pt(S, f"thr={thr}", "P1")) for thr in (0.5, 0.6, 0.7, 0.8, 0.9)]
    ztags = [("1,645", "z30=1.645"), ("2,33", "main"), ("3,09", "z30=3.09")]
    curves = {p: [(z, *pt(S, t, p)) for z, t in ztags] for p in ("P0", "P2")}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    # ---------------- (a) visão geral
    ax0.plot([q[1] for q in p1], [q[2] for q in p1], color=POL_COLOR["P1"], lw=1.4, marker=POL_MARK["P1"], ms=5)
    for thr, x, y in (p1[0], p1[-1]):
        ax0.annotate(f"θ = {thr:.1f}".replace(".", ","), (x, y), xytext=(5, -2), textcoords="offset points", fontsize=6.5, color=INK2)
    ax0.text(p1[2][1] + 0.3, p1[2][2] * 1.05, "P1", fontsize=7.5, color=POL_COLOR["P1"], weight="bold")
    for p in ("B1", "B2"):
        ax0.scatter(*single[p], color=POL_COLOR[p], marker=POL_MARK[p], s=38, zorder=3, edgecolor="white", lw=0.7)
        ax0.annotate(p, single[p], xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=POL_COLOR[p], weight="bold")
    for p in ("P0", "P2"):
        ax0.plot([q[1] for q in curves[p]], [q[2] for q in curves[p]], color=POL_COLOR[p], lw=1.2, marker=POL_MARK[p], ms=4)
    xs = [single[p][0] for p in ("B2", "P0S", "P2I", "P2M")] + [q[1] for c in curves.values() for q in c]
    ys = [single[p][1] for p in ("B2", "P0S", "P2I", "P2M")] + [q[2] for c in curves.values() for q in c]
    x0, x1, y0, y1 = min(xs) - 0.4, max(xs) + 0.4, min(ys) * 0.85, max(ys) * 1.12
    ax0.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=0.9, ec=INK2))
    ax0.text(x0, y1 * 1.06, "ampliado em (b)", fontsize=6.3, color=INK2, va="bottom")
    ax0.set_yscale("log")
    ax0.set_xlabel("Horas de equipe por dia de coleta")
    ax0.set_ylabel("Transbordos por mês (escala log)")
    ax0.set_title("(a) Todas as políticas", fontsize=8.5)

    # ---------------- (b) ampliação linear
    ax1.axvline(single["B2"][0], color=POL_COLOR["B2"], lw=0.7, ls=":", zorder=0)
    ax1.axhline(single["B2"][1], color=POL_COLOR["B2"], lw=0.7, ls=":", zorder=0)
    for p in ("P0", "P2"):
        c = curves[p]
        ax1.plot([q[1] for q in c], [q[2] for q in c], color=POL_COLOR[p], lw=1.6, marker=POL_MARK[p], ms=6, zorder=3)
        for z, x, y in c:
            ax1.annotate(f"z = {z}", (x, y), xytext=(6, -9) if p == "P0" else (-6, -10), ha="left" if p == "P0" else "right",
                         textcoords="offset points", fontsize=6.2, color=INK2)
        xm, ym = (c[0][1] + c[1][1]) / 2, (c[0][2] + c[1][2]) / 2
        ax1.text(xm + (0.08 if p == "P0" else -0.08), ym, p, fontsize=8.5, color=POL_COLOR[p], weight="bold",
                 ha="left" if p == "P0" else "right", va="center")
    for p in ("B2", "P0S", "P2I", "P2M"):
        ax1.scatter(*single[p], color=POL_COLOR[p], marker=POL_MARK[p], s=46, zorder=4, edgecolor="white", lw=0.7)
        off = {"B2": (-4, 7), "P0S": (6, 3), "P2I": (6, -3), "P2M": (6, -2)}[p]
        ax1.annotate(CURTO[p], single[p], xytext=off, textcoords="offset points", fontsize=7, color=POL_COLOR[p],
                     weight="bold" if p == "B2" else "normal", ha="right" if p == "B2" else "left", va="bottom" if p == "B2" else "center")
    ax1.set_xlim(x0, x1); ax1.set_ylim(y0, y1)
    ax1.annotate("", xy=(x0 + 0.02 * (x1 - x0), y0 + 0.02 * (y1 - y0)), xytext=(x0 + 0.12 * (x1 - x0), y0 + 0.10 * (y1 - y0)),
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax1.text(x0 + 0.13 * (x1 - x0), y0 + 0.07 * (y1 - y0), "melhor: menos horas\ne menos transbordos", fontsize=6.4, color=INK, va="center")
    ax1.set_xlabel("Horas de equipe por dia de coleta")
    ax1.set_ylabel("Transbordos por mês")
    ax1.set_title("(b) Região da prática atual e das regras preditivas", fontsize=8.5)
    for ax in (ax0, ax1):
        ax.grid(alpha=0.35)
    h = [plt.Line2D([], [], color=POL_COLOR[p], marker=POL_MARK[p], ls="-" if p in ("P1", "P0", "P2") else "", ms=5, label=NOME[p])
         for p in ("B1", "B2", "P1", "P0", "P2", "P0S", "P2I", "P2M")]
    fig.legend(handles=h, loc="lower center", ncol=2, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    main()
