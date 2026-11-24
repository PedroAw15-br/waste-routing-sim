"""Geração da cidade sintética: malha viária, pontos de coleta e matriz de distâncias.

A malha é uma grade ortogonal perturbada (quadras de ~110 m) com contorno
irregular e remoção aleatória de trechos, preservando a conectividade.
As distâncias mínimas são calculadas com o Algoritmo de Dijkstra
(scipy.sparse.csgraph.dijkstra, fila de prioridade).
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

from .params import Params


@dataclass
class City:
    coords: np.ndarray        # (V,2) coordenadas dos nós (m)
    edges: np.ndarray         # (E,2) pares de nós
    edge_len: np.ndarray      # (E,) comprimento (m)
    point_node: np.ndarray    # (N,) nó de cada ponto de coleta
    garage_node: int
    exit_node: int            # saída urbana para o aterro
    central: np.ndarray       # (N,) bool: ponto no centro comercial
    sector: np.ndarray        # (N,) 0=centro, 1=setor A (oeste), 2=setor B (leste)
    mean_kg: np.ndarray       # (N,) geração média diária (kg)
    n_cont: np.ndarray        # (N,) contêineres por ponto
    cap_kg: np.ndarray        # (N,) capacidade do ponto (kg)
    D: np.ndarray             # (N+2,N+2) distâncias mínimas (km); índice N=garagem, N+1=saída

    @property
    def n(self):
        return len(self.point_node)

    @property
    def G(self):
        return self.n

    @property
    def X(self):
        return self.n + 1


def _build_graph(p: Params, rng):
    nx_, ny_ = p.grid_nx, p.grid_ny
    W, H = (nx_ - 1) * p.block_m, (ny_ - 1) * p.block_m
    ii, jj = np.meshgrid(np.arange(nx_), np.arange(ny_), indexing="ij")
    base = np.stack([ii.ravel() * p.block_m, jj.ravel() * p.block_m], axis=1)
    coords = base + rng.uniform(-1, 1, size=base.shape) * p.jitter * p.block_m

    # contorno irregular: superelipse com raio ruidoso (cantos arredondados)
    cx, cy = W / 2, H / 2
    u = (base[:, 0] - cx) / (W / 2)
    v = (base[:, 1] - cy) / (H / 2)
    r = (np.abs(u) ** 4 + np.abs(v) ** 4) ** 0.25
    keep = r <= 1.04 + rng.normal(0, 0.05, size=len(r))
    idx = -np.ones(nx_ * ny_, dtype=int)
    idx[keep] = np.arange(keep.sum())

    edges = []
    for i in range(nx_):
        for j in range(ny_):
            a = i * ny_ + j
            if idx[a] < 0:
                continue
            if i + 1 < nx_ and idx[(i + 1) * ny_ + j] >= 0:
                edges.append((idx[a], idx[(i + 1) * ny_ + j]))
            if j + 1 < ny_ and idx[i * ny_ + j + 1] >= 0:
                edges.append((idx[a], idx[i * ny_ + j + 1]))
    edges = np.array(edges)
    coords = coords[keep]
    V = len(coords)

    # mantém apenas a maior componente conexa (descarta nós isolados no contorno)
    A0 = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(V, V))
    _, lab = connected_components(A0, directed=False)
    main = lab == np.bincount(lab).argmax()
    remap = -np.ones(V, dtype=int)
    remap[main] = np.arange(main.sum())
    edges = edges[main[edges[:, 0]] & main[edges[:, 1]]]
    edges = remap[edges]
    coords = coords[main]
    V = len(coords)

    # remoção aleatória de trechos mantendo o grafo conexo
    n_remove = int(p.edge_removal * len(edges))
    alive = np.ones(len(edges), dtype=bool)
    removed = 0
    for e in rng.permutation(len(edges)):
        if removed >= n_remove:
            break
        alive[e] = False
        ee = edges[alive]
        A = csr_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])), shape=(V, V))
        if connected_components(A, directed=False)[0] == 1:
            removed += 1
        else:
            alive[e] = True
    edges = edges[alive]
    seg = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    edge_len = seg * rng.uniform(1.0, 1.08, size=len(seg))  # sinuosidade
    return coords, edges, edge_len


def build_city(p: Params) -> City:
    rng = np.random.default_rng(p.city_seed)
    coords, edges, edge_len = _build_graph(p, rng)
    V = len(coords)
    center = coords.mean(axis=0)

    # garagem na periferia sudoeste; saída para o aterro na periferia leste
    garage = int(np.argmin(coords[:, 0] + coords[:, 1]))
    exit_ = int(np.argmax(coords[:, 0] - 0.3 * np.abs(coords[:, 1] - center[1])))

    candidates = np.setdiff1d(np.arange(V), [garage, exit_])
    point_node = np.sort(rng.choice(candidates, size=p.n_points, replace=False))
    pc = coords[point_node]
    central = np.linalg.norm(pc - center, axis=1) <= p.central_radius_m
    sector = np.where(central, 0, np.where(pc[:, 0] < np.median(pc[~central, 0]), 1, 2))

    # geração média por ponto (kg/dia): população heterogênea + centro comercial
    w = rng.lognormal(0.0, p.pop_sigma, size=p.n_points)
    w = w * np.where(central, p.central_mult, 1.0)
    total = p.urban_pop * p.per_capita_kg
    mean_kg = total * w / w.sum()

    # dimensionamento dos contêineres para a prática setorizada:
    # centro coletado diariamente (maior intervalo = 2 noites: sáb->seg)
    # bairros em dias alternados (maior intervalo = 3 noites)
    c1 = p.container_l / 1000.0 * p.density_kg_m3
    gap = np.where(central, 2, 3)
    n_cont = np.clip(np.ceil(p.sizing_safety * mean_kg * gap / c1), 1, p.max_containers).astype(int)
    cap_kg = n_cont * c1

    # matriz de distâncias mínimas (Dijkstra) entre pontos, garagem e saída
    A = csr_matrix((edge_len, (edges[:, 0], edges[:, 1])), shape=(V, V))
    sources = np.concatenate([point_node, [garage, exit_]])
    dist = dijkstra(A, directed=False, indices=sources)  # (N+2, V) em metros
    D = dist[:, sources] / 1000.0
    D = 0.5 * (D + D.T)

    return City(coords, edges, edge_len, point_node, garage, exit_, central, sector,
                mean_kg, n_cont, cap_kg, np.ascontiguousarray(D))


def city_summary(city: City) -> dict:
    deg = np.bincount(city.edges.ravel(), minlength=len(city.coords))
    xs, ys = city.coords[:, 0], city.coords[:, 1]
    return {
        "vertices": int(len(city.coords)),
        "arestas": int(len(city.edges)),
        "grau_medio": float(deg.mean()),
        "extensao_viaria_km": float(city.edge_len.sum() / 1000),
        "trecho_medio_m": float(city.edge_len.mean()),
        "area_envoltoria_km2": float((xs.max() - xs.min()) * (ys.max() - ys.min()) / 1e6),
        "pontos": int(city.n),
        "pontos_centro": int(city.central.sum()),
        "pontos_setor_A": int((city.sector == 1).sum()),
        "pontos_setor_B": int((city.sector == 2).sum()),
        "conteineres": int(city.n_cont.sum()),
        "conteineres_por_ponto": float(city.n_cont.mean()),
        "geracao_media_kg_dia": float(city.mean_kg.sum()),
        "taxa_enchimento_media_dia": float((city.mean_kg / city.cap_kg).mean()),
        "taxa_enchimento_p90_dia": float(np.percentile(city.mean_kg / city.cap_kg, 90)),
        "dist_garagem_saida_km": float(city.D[city.G, city.X]),
    }
