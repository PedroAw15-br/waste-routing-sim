"""Figura dos casos reais (mapa): (a) Santa Helena e (b) São Miguel do Iguaçu na mesma escala,
com a área urbana do Censo 2022, a malha do OpenStreetMap, os pontos de coleta por setor da
coleta setorizada (B2), a garagem, a saída e o início da rota real até o aterro; (c) situação
regional, com as rotas do OSRM até os aterros declarados ao SINISA.

Uso:  python -m wrsim.fig_casos_reais --data results --out figures/fig9_casos_reais.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from .city import build_city, _census_sectors
from .experiments_real import case_params

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
ROAD, URB = "#d6d5cf", "#efeee9"
SECT = {0: ("#eb6834", "D", "Centro comercial: coleta diária"),
        1: ("#2a78d6", "o", "Setor A: seg., qua. e sex."),
        2: ("#1baf7a", "s", "Setor B: ter., qui. e sáb.")}
INFO = {
    "santa_helena": dict(nome="Santa Helena", garagem="Garagem\n(base da contratada)", saida="Saída\n(PR-495)",
                         destino="Aterro Sanitário\nParaná Ambiental (Cascavel)", curto="Aterro Paraná Ambiental,\nCascavel", km=131.6,
                         # rótulos (dx, dy em km; alinhamento), posicionados à mão para não colidir
                         g_lab=(-0.35, 0.40, "right", "bottom"), e_lab=(-0.30, 0.05, "right", "bottom"),
                         r_lab=(-0.25, 0.05, "right", "bottom")),
    "sao_miguel": dict(nome="São Miguel do Iguaçu", garagem="Garagem\n(Pátio de Máquinas)", saida="Saída\n(PR-497)",
                       destino="Aterro controlado\nmunicipal", curto="Aterro controlado\nmunicipal", km=11.7,
                       g_lab=(0.30, -0.20, "left", "top"), e_lab=(0.30, -0.10, "left", "top"),
                       r_lab=(0.30, -0.02, "left", "top")),
}


def _route_xy(path, p):
    r = json.load(open(path))["routes"][0]
    g = np.array(r["geometry"]["coordinates"])
    lat0, lon0 = p.osm_center
    kx = 111320.0 * np.cos(np.radians(lat0))
    return np.c_[(g[:, 0] - lon0) * kx, (g[:, 1] - lat0) * 110540.0], g, r["distance"] / 1000


def _clip_after(xy, start, box):
    """Trecho da rota a partir do ponto mais próximo da saída até sair do quadro."""
    i = int(np.argmin(np.linalg.norm(xy - start, axis=1)))
    seg = [start]
    for q in xy[i + 1:]:
        seg.append(q)
        if not (box[0] <= q[0] <= box[1] and box[2] <= q[1] <= box[3]):
            break
    return np.array(seg)


def extent(name, data):
    """Retângulo (km) que contém a área urbana, a garagem e a saída."""
    p = case_params(name, data)
    c = build_city(p)
    S = _census_sectors(p)
    bx = np.array([g.bounds for g, _ in S]) / 1000.0
    xs = np.r_[bx[:, 0], bx[:, 2], c.coords[[c.garage_node, c.exit_node], 0] / 1000]
    ys = np.r_[bx[:, 1], bx[:, 3], c.coords[[c.garage_node, c.exit_node], 1] / 1000]
    return p, c, S, (xs.min(), xs.max(), ys.min(), ys.max())


def city_panel(ax, name, data, half_w, half_h, letter, pre):
    p, c, S, ext = pre
    info = INFO[name]
    xy = c.coords / 1000.0
    # janela: centrada no conjunto área urbana + garagem + saída, com a mesma escala nos dois painéis
    cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    box = (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
    # área urbana (setores censitários urbanos)
    for g, _ in S:
        polys = getattr(g, "geoms", [g])
        for pg in polys:
            x, y = pg.exterior.xy
            ax.fill(np.array(x) / 1000, np.array(y) / 1000, color=URB, ec="white", lw=0.6, zorder=0)
    # malha viária
    segs = xy[c.edges]
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=ROAD, linewidths=0.45, zorder=1))
    # raio do centro comercial
    cen = None
    if p.center_latlon:
        lat0, lon0 = p.osm_center
        cen = np.array([(p.center_latlon[1] - lon0) * 111.32 * np.cos(np.radians(lat0)),
                        (p.center_latlon[0] - lat0) * 110.54])
        ax.add_patch(Circle(cen, p.central_radius_m / 1000, fill=False, ec=SECT[0][0], lw=0.8, alpha=0.8, zorder=2))
    # rota real a partir da saída
    rpath = os.path.join(data, f"rota_osrm_{name}.json")
    if os.path.exists(rpath):
        rxy, _, _ = _route_xy(rpath, p)
        seg = _clip_after(rxy / 1000, xy[c.exit_node], box)
        ax.plot(seg[:, 0], seg[:, 1], color=INK, lw=1.6, zorder=3, solid_capstyle="round")
        if len(seg) >= 2:
            a, b = seg[-2], seg[-1]
            d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            tip = np.array([np.clip(b[0], box[0] + 0.05, box[1] - 0.05), np.clip(b[1], box[2] + 0.05, box[3] - 0.05)])
            ax.annotate("", xy=tip, xytext=tip - 0.35 * d, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4), zorder=4)
            kmtxt = f"{info['km']:.1f}".replace(".", ",")
            lab = f"{info['curto']}:\n{kmtxt} km por estrada"
            dx, dy, ha, va = info["r_lab"]
            ax.text(tip[0] + dx, tip[1] + dy, lab, fontsize=6.3, color=INK, ha=ha, va=va,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    # pontos de coleta por setor (tamanho ~ contêineres)
    pts = xy[c.point_node]
    for s, (col, mk, _) in SECT.items():
        m = c.sector == s
        ax.scatter(pts[m, 0], pts[m, 1], s=4 + 3.2 * c.n_cont[m], c=col, marker=mk, edgecolors="white", linewidths=0.35, zorder=4)
    # garagem e saída
    g, e = xy[c.garage_node], xy[c.exit_node]
    ax.scatter(*g, s=70, marker="s", c=INK, edgecolors="white", linewidths=1.0, zorder=6)
    ax.scatter(*e, s=70, marker="o", c="white", edgecolors=INK, linewidths=1.6, zorder=6)

    def place(pt, text, spec):
        dx, dy, ha, va = spec
        ax.annotate(text, pt, xytext=(pt[0] + dx, pt[1] + dy),
                    fontsize=6.4, color=INK, weight="bold", ha=ha, va=va, zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6))
    place(g, info["garagem"], info["g_lab"])
    place(e, info["saida"], info["e_lab"])
    # escala e norte
    sx, sy = box[0] + 0.25, box[2] + 0.25
    ax.plot([sx, sx + 1.0], [sy, sy], color=INK, lw=2.2, solid_capstyle="butt", zorder=7)
    ax.text(sx + 0.5, sy + 0.08, "1 km", fontsize=6.5, color=INK, ha="center", va="bottom", zorder=7)
    nx_, ny_ = box[1] - 0.25, box[3] - 0.55
    ax.annotate("", xy=(nx_, ny_ + 0.35), xytext=(nx_, ny_), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    ax.text(nx_, ny_ + 0.4, "N", fontsize=7.5, color=INK, ha="center", va="bottom", weight="bold")
    ax.set_xlim(box[0], box[1]); ax.set_ylim(box[2], box[3])
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7"); sp.set_linewidth(0.6)
    ax.set_title(f"({letter}) {info['nome']}: {c.n} pontos, {int(c.n_cont.sum())} contêineres", fontsize=8, loc="left")
    return c


def region_panel(ax, data):
    LAB = {  # posições dos rótulos (graus) escolhidas para não cruzar rotas nem marcadores
        "santa_helena": dict(cid=(0.025, 0.012, "left", "bottom"), aterro=(0.0, -0.03, "center", "top"),
                             km=(-53.86, -25.135, "center", "top")),
        "sao_miguel": dict(cid=(0.025, -0.006, "left", "top"), aterro=(0.025, 0.0, "left", "center"),
                           km=(-54.275, -25.305, "right", "center")),
    }
    for name, col in (("santa_helena", INK), ("sao_miguel", INK2)):
        p = case_params(name, data)
        _, g, km = _route_xy(os.path.join(data, f"rota_osrm_{name}.json"), p)
        ax.plot(g[:, 0], g[:, 1], color=col, lw=1.3, zorder=2)
        lat, lon = p.center_latlon or p.osm_center
        ax.scatter(lon, lat, s=34, c=INK2, edgecolors="white", linewidths=0.8, zorder=4)
        dx, dy, ha, va = LAB[name]["cid"]
        ax.text(lon + dx, lat + dy, INFO[name]["nome"], fontsize=6.8, ha=ha, va=va, color=INK, weight="bold", zorder=6)
        la, lo = p.landfill_latlon
        ax.scatter(lo, la, s=55, marker="^", c=INK, edgecolors="white", linewidths=0.8, zorder=5)
        dx, dy, ha, va = LAB[name]["aterro"]
        ax.text(lo + dx, la + dy, INFO[name]["destino"], fontsize=6.2, ha=ha, va=va, color=INK, zorder=6)
        x, y, ha, va = LAB[name]["km"]
        ax.text(x, y, f"{km:.1f} km".replace(".", ","), fontsize=6.4, color=INK2, ha=ha, va=va, zorder=6)
    ax.scatter(-53.4552, -24.9555, s=18, c=MUTED, zorder=3)
    ax.text(-53.4552, -24.9555 + 0.02, "Cascavel", fontsize=6.3, color=INK2, ha="center", va="bottom")
    ax.set_xlim(-54.42, -53.12); ax.set_ylim(-25.40, -24.80)
    ax.set_aspect(1 / np.cos(np.radians(25.1)))
    ax.set_xlabel("Longitude (°)", fontsize=7); ax.set_ylabel("Latitude (°)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.grid(color="#e1e0d9", lw=0.4)
    for sp in ax.spines.values():
        sp.set_color("#c3c2b7"); sp.set_linewidth(0.6)
    ax.set_title("(c) Situação regional: rotas do OSRM até os aterros declarados ao SINISA", fontsize=8, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results")
    ap.add_argument("--out", default="figures/fig9_casos_reais.png")
    a = ap.parse_args()
    render(a.data, a.out)


def render(data, out):
    a = argparse.Namespace(data=data, out=out)
    pre = {n: extent(n, a.data) for n in ("santa_helena", "sao_miguel")}
    half_w = max((e[1] - e[0]) / 2 for *_, e in pre.values()) + 0.45
    half_h = max((e[3] - e[2]) / 2 for *_, e in pre.values()) + 0.9
    fig = plt.figure(figsize=(7.0, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.14, wspace=0.05)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    city_panel(ax1, "santa_helena", a.data, half_w, half_h, "a", pre["santa_helena"])
    city_panel(ax2, "sao_miguel", a.data, half_w, half_h, "b", pre["sao_miguel"])
    ax3 = fig.add_subplot(gs[1, :])
    region_panel(ax3, a.data)
    h = [Line2D([], [], marker=mk, ls="", color=col, mec="white", ms=6, label=lab) for col, mk, lab in SECT.values()]
    h += [Line2D([], [], marker="s", ls="", color=INK, mec="white", ms=6.5, label="Garagem"),
          Line2D([], [], marker="o", ls="", mfc="white", mec=INK, mew=1.4, ms=6.5, label="Saída da cidade"),
          Line2D([], [], color=INK, lw=1.6, label="Rota real até o aterro (OSRM)"),
          Line2D([], [], marker="^", ls="", color=INK, ms=6.5, label="Aterro (coordenada do SINISA)"),
          Line2D([], [], marker="o", ls="", color=INK2, mec="white", ms=5.5, label="Sede municipal, em (c)"),
          Patch(fc=URB, ec="#c3c2b7", lw=0.5, label="Setores urbanos do Censo 2022"),
          Line2D([], [], color=ROAD, lw=1.2, label="Vias (OpenStreetMap)"),
          Line2D([], [], marker="o", ls="", mfc="none", mec=SECT[0][0], ms=9, label="Raio de 450 m do centro"),
          Line2D([], [], marker="o", ls="", color=MUTED, ms=2.5, label="Tamanho ∝ nº de contêineres")]
    fig.legend(handles=h, loc="lower center", ncol=3, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, 0.025),
               columnspacing=1.2, handletextpad=0.5)
    fig.text(0.5, 0.0, "Malhas e rotas: © colaboradores do OpenStreetMap (ODbL); OSRM. Setores: IBGE (2024). Aterros: SINISA 2023.",
             ha="center", va="bottom", fontsize=5.6, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.17)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
