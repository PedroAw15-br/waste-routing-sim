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


URBAN_WAYS = {"residential", "living_street", "tertiary", "unclassified", "secondary"}


def _build_graph_osm(p: Params):
    """Malha viária real a partir de um JSON do Overpass (OpenStreetMap).

    Vias de serviço (estacionamentos, acessos internos) são descartadas; o grafo é
    tratado como não orientado, como na cidade sintética. Devolve também os
    cruzamentos de ruas urbanas (candidatos a ponto de coleta) e os nós de rodovia."""
    import json
    el = json.load(open(p.osm_file, encoding="utf-8"))["elements"]
    lat0, lon0 = p.osm_center
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in el if e["type"] == "node"}
    ways = [e for e in el if e["type"] == "way" and e["tags"].get("highway") != "service"]
    ids = sorted({n for w in ways for n in w["nodes"]})
    ix = {k: i for i, k in enumerate(ids)}
    kx = 111320.0 * np.cos(np.radians(lat0))
    coords = np.array([[(nodes[k][1] - lon0) * kx, (nodes[k][0] - lat0) * 110540.0] for k in ids])
    E, urban, road = set(), set(), set()
    for w in ways:
        hw = w["tags"].get("highway", "")
        for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
            if a != b:
                E.add((min(ix[a], ix[b]), max(ix[a], ix[b])))
        (urban if hw in URBAN_WAYS else road if hw.startswith(("primary", "trunk")) else set()).update(ix[n] for n in w["nodes"])
    edges = np.array(sorted(E))
    V = len(coords)
    A = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(V, V))
    _, lab = connected_components(A, directed=False)
    main = lab == np.bincount(lab).argmax()
    remap = -np.ones(V, dtype=int)
    remap[main] = np.arange(main.sum())
    edges = remap[edges[main[edges[:, 0]] & main[edges[:, 1]]]]
    keep = np.where(main)[0]
    coords = coords[main]
    urban = np.array(sorted(remap[i] for i in urban if main[i]))
    road = np.array(sorted(remap[i] for i in road if main[i]))
    edge_len = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)
    deg = np.bincount(edges.ravel(), minlength=len(coords))
    cross = urban[deg[urban] >= 3]
    return coords, edges, edge_len, cross, road


def build_city(p: Params) -> City:
    rng = np.random.default_rng(p.city_seed)
    if p.osm_file:
        coords, edges, edge_len, cross, road = _build_graph_osm(p)
        V = len(coords)
        center = np.zeros(2)                      # origem = centro urbano informado
        # saída para o aterro: ponto em que a rodovia principal (na direção do seu
        # trecho mais longo) cruza a borda urbana, tomada como 1,1 × o percentil 95
        # da distância dos cruzamentos urbanos ao centro;
        # garagem: cruzamento urbano mais periférico no lado oposto
        dr = np.linalg.norm(coords[road], axis=1)
        far = coords[road[np.argmax(dr)]] / dr.max()
        same = road[(coords[road] @ far) / np.maximum(dr, 1e-9) > 0.8]
        edge_r = 1.1 * np.percentile(np.linalg.norm(coords[cross], axis=1), 95)
        exit_ = int(same[np.argmin(np.abs(np.linalg.norm(coords[same], axis=1) - edge_r))])
        pop_w, pop_total = None, None
        if p.census_file:
            # pontos só em cruzamentos dentro dos setores urbanos do Censo 2022, cerca de
            # um ponto para cada (urban_pop / n_points) habitantes de cada setor
            setores = _census_sectors(p)
            hab_por_ponto = p.urban_pop / p.n_points
            pts, wts = [], []
            for poly, pop in setores:
                inside = cross[_contains(poly, coords[cross])]
                if pop <= 0 or len(inside) == 0:
                    continue
                k = int(min(len(inside), max(1, round(pop / hab_por_ponto))))
                sel = rng.choice(inside, size=k, replace=False)
                pts.extend(sel.tolist()); wts.extend([pop / k] * k)
            urban_cross = np.array(sorted(set(pts)))
            opp = urban_cross[(coords[urban_cross] @ coords[exit_]) < 0]
            garage = int(opp[np.argmax(np.linalg.norm(coords[opp], axis=1))])
            keep = [i for i, n in enumerate(pts) if n not in (garage, exit_)]
            order = np.argsort(np.array(pts)[keep])
            point_node = np.array(pts)[keep][order]
            pop_w = np.array(wts)[keep][order]
            pop_total = float(sum(pop for _, pop in setores))
            return _finish_city(p, rng, coords, edges, edge_len, point_node, garage, exit_, center,
                                pop_w=pop_w, pop_total=pop_total)
        opp = cross[(coords[cross] @ coords[exit_]) < 0]
        garage = int(opp[np.argmax(np.linalg.norm(coords[opp], axis=1))])
        candidates = np.setdiff1d(cross, [garage, exit_])
        point_node = np.sort(rng.choice(candidates, size=p.n_points, replace=False))
        return _finish_city(p, rng, coords, edges, edge_len, point_node, garage, exit_, center)
    coords, edges, edge_len = _build_graph(p, rng)
    V = len(coords)
    center = coords.mean(axis=0)

    # garagem na periferia sudoeste; saída para o aterro na periferia leste
    garage = int(np.argmin(coords[:, 0] + coords[:, 1]))
    exit_ = int(np.argmax(coords[:, 0] - 0.3 * np.abs(coords[:, 1] - center[1])))

    candidates = np.setdiff1d(np.arange(V), [garage, exit_])
    point_node = np.sort(rng.choice(candidates, size=p.n_points, replace=False))
    return _finish_city(p, rng, coords, edges, edge_len, point_node, garage, exit_, center)


def _census_sectors(p):
    """Setores urbanos do município (Censo 2022) na área da malha: [(polígono em metros, população)]."""
    import sqlite3
    from shapely import wkb
    from shapely.ops import transform
    lat0, lon0 = p.osm_center
    kx = 111320.0 * np.cos(np.radians(lat0))
    if p.census_file.endswith((".geojson", ".json")):
        import json
        from shapely.geometry import shape
        fc = json.load(open(p.census_file, encoding="utf-8"))
        rows = [(shape(f["geometry"]), f["properties"]["v0001"]) for f in fc["features"]
                if f["properties"]["CD_MUN"] == p.census_mun and f["properties"]["SITUACAO"] == "Urbana"]
    else:
        con = sqlite3.connect(p.census_file)
        tab = con.execute("select table_name from gpkg_contents").fetchone()[0]
        raw = con.execute(f'select geom, v0001 from "{tab}" where CD_MUN=? and SITUACAO="Urbana"', (p.census_mun,)).fetchall()
        rows = []
        for g, pop in raw:
            env = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(g[3] >> 1) & 7]
            rows.append((wkb.loads(bytes(g[8 + env:])), pop))
    out = []
    for g, pop in rows:
        geom = transform(lambda x, y, z=None: ((x - lon0) * kx, (y - lat0) * 110540.0), g)
        c = geom.centroid
        if abs(c.x) < 4000 and abs(c.y) < 4000:          # sede urbana (exclui outros distritos)
            out.append((geom, int(pop or 0)))
    return out


def _contains(poly, xy):
    from shapely import contains_xy
    return contains_xy(poly, xy[:, 0], xy[:, 1])


def _finish_city(p, rng, coords, edges, edge_len, point_node, garage, exit_, center,
                 pop_w=None, pop_total=None):
    V = len(coords)
    pc = coords[point_node]
    central = np.linalg.norm(pc - center, axis=1) <= p.central_radius_m
    sector = np.where(central, 0, np.where(pc[:, 0] < np.median(pc[~central, 0]), 1, 2))

    # geração média por ponto (kg/dia): população heterogênea + centro comercial
    w = rng.lognormal(0.0, p.pop_sigma, size=len(point_node))
    if pop_w is not None:                       # população do setor censitário de cada ponto
        w = pop_w * w
    w = w * np.where(central, p.central_mult, 1.0)
    total = (pop_total if pop_total is not None else p.urban_pop) * p.per_capita_kg
    mean_kg = total * w / w.sum()

    # dimensionamento dos contêineres para a prática setorizada:
    # centro coletado diariamente (maior intervalo = 2 noites: sáb->seg)
    # bairros em dias alternados (maior intervalo = 3 noites)
    c1 = p.container_l / 1000.0 * p.density_kg_m3
    gap = np.where(central, 2, 3)
    n_cont = np.clip(np.ceil(p.sizing_safety * mean_kg * gap / c1), 1, p.max_containers).astype(int)
    cap_kg = n_cont * c1 * np.where(central, p.center_cap_mult, 1.0)

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
