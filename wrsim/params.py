"""Parâmetros do modelo de simulação.

Cada parâmetro tem um valor de referência; as fontes e hipóteses estão
documentadas no artigo (Tabela de parâmetros). Os experimentos de
sensibilidade alteram cópias deste objeto com dataclasses.replace().
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Params:
    # ---------------- Cidade sintética ----------------
    city_seed: int = 2026
    osm_file: str = ""         # malha viária real (JSON do Overpass/OpenStreetMap); vazio = cidade sintética
    census_file: str = ""      # setores censitários (GeoPackage do IBGE, Censo 2022); vazio = população hipotética
    census_mun: str = "4123501" # código IBGE do município (Santa Helena, PR)
    landfill_latlon: tuple = ()  # (lat, lon) do aterro; define a direção da saída urbana na malha real
    osm_center: tuple = (-24.8591735, -54.3328813)  # (lat, lon) do centro urbano usado no recorte
    grid_nx: int = 22            # nós na direção x da malha base
    grid_ny: int = 20            # nós na direção y
    block_m: float = 110.0       # comprimento médio da quadra (m)
    jitter: float = 0.12         # perturbação das coordenadas (fração da quadra)
    edge_removal: float = 0.10   # fração de arestas removidas (vias sem conexão)
    n_points: int = 280          # pontos de coleta conteinerizados
    urban_pop: float = 17000.0   # população urbana atendida (≈85% de 20 mil hab.)
    per_capita_kg: float = 0.78  # kg/hab/dia (região Sul, ABREMA 2024)
    central_radius_m: float = 450.0
    central_mult: float = 1.6    # geração relativa dos pontos do centro comercial
    pop_sigma: float = 0.35      # heterogeneidade (lognormal) entre pontos
    container_l: float = 1000.0  # volume de cada contêiner (L)
    density_kg_m3: float = 180.0 # massa específica dos resíduos soltos no contêiner
    sizing_safety: float = 1.2   # folga no dimensionamento dos contêineres
    center_cap_mult: float = 1.0 # capacidade extra no centro (alternativa sem sensores)
    max_containers: int = 4

    # ---------------- Geração de resíduos ----------------
    cv_res: float = 0.30         # coeficiente de variação diário (residencial)
    cv_cen: float = 0.45         # coeficiente de variação diário (centro)
    weekend_factor: float = 1.10 # geração relativa em sábados e domingos
    peak_mult: float = 1.40      # multiplicador nos eventos de pico (centro)
    use_peaks: bool = True       # False = período sem eventos de pico (sensibilidade)
    gen_scale: float = 1.0       # escala global (análise de sensibilidade)
    drift_share: float = 0.0     # fração dos pontos cuja geração muda sem aviso no início da avaliação
    drift_mag: float = 0.30      # tamanho da mudança (±30% na média do ponto, sinal sorteado)

    # ---------------- Calendário ----------------
    warmup_days: int = 7
    measure_days: int = 49       # 7 semanas medidas

    # ---------------- Frota e operação ----------------
    truck_capacity_kg: float = 7000.0
    garage_latlon: tuple = None  # (lat, lon) da garagem informada (Google Maps); None = extremo oposto à saída
    exit_latlon: tuple = None    # ponto em que a rota real ao destino cruza a borda urbana (OSRM)
    landfill_km: float = 15.0    # distância (só ida) da saída urbana ao aterro
    v_urban_kmh: float = 18.0
    v_hwy_kmh: float = 50.0
    t_stop_min: float = 0.5      # parada e reposicionamento por ponto
    t_container_min: float = 1.0 # basculamento por contêiner
    t_unload_min: float = 15.0   # pesagem e descarga no aterro
    shift_route_min: float = 410.0   # tempo de rota disponível por turno
    shift_fixed_min: float = 30.0    # tempo fixo por turno (inspeção, deslocamentos internos)

    # ---------------- Consumo e emissões ----------------
    fuel_urban_l_km: float = 0.50
    fuel_hwy_l_km: float = 0.35
    fuel_stop_l: float = 0.03
    fuel_container_l: float = 0.02
    fuel_unload_l: float = 0.30
    co2_kg_l: float = 2.65       # balanço de carbono do diesel fóssil

    # ---------------- Sensoriamento ----------------
    sensor_sigma: float = 0.05   # desvio-padrão do erro de leitura (fração da altura)
    sensor_miss: float = 0.05    # prob. de não haver leitura válida na manhã
    ewma_alpha: float = 0.25
    sensor_share: float = 1.0    # fração dos pontos com sensor em P2 (os demais seguem P0)

    # ---------------- Políticas dinâmicas ----------------
    threshold: float = 0.70
    z_service: float = 2.33      # quantil normal da regra preditiva (~99% nominal)
    fleet_max_shifts: int = 0    # 0 = sem limite; >0 = turnos (caminhões) disponíveis por dia
    use_calendar: bool = True    # previsão considera perfil semanal e eventos agendados
    event_mult_forecast: float = 1.30  # estimativa do planejador para o pico (real = 1.40)
    maygo_fill: float = 0.70     # P2M: nível previsto mínimo para inclusão oportunista
    maygo_km: float = 0.30       # P2M: distância máxima (km) a um ponto já selecionado

    # ---------------- Algoritmo Genético ----------------
    ga_pop: int = 100
    ga_gens: int = 300
    ga_stall: int = 50
    ga_tk: int = 3
    ga_pc: float = 0.90
    ga_pm: float = 0.02
    ga_elite: int = 2
    local_search: bool = True    # True = AG memético (adotado); False = AG puro
    mga_pop: int = 30            # AG memético: população
    mga_gens: int = 150
    mga_stall: int = 25
    mga_ls_prob: float = 0.20    # prob. de aplicar 2-opt a cada filho

    def to_dict(self):
        return asdict(self)
