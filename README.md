# waste-routing-sim

Simulador de coleta conteinerizada de resíduos sólidos urbanos (RSU) que compara rotas fixas com rotas dinâmicas apoiadas por sensores IoT de nível de enchimento (ESP32, sensor ultrassônico, LoRaWAN) em um município de pequeno porte.

Este repositório acompanha o artigo *"Sensoriamento IoT e Roteirização Dinâmica Preditiva para a Coleta Conteinerizada de Resíduos Sólidos Urbanos em Municípios de Pequeno Porte: Avaliação por Simulação com Linhas de Base Realistas"* (Werner; Vasconcelos, 2026). Todos os números, tabelas e figuras do artigo são gerados por este código a partir das sementes fixadas em `wrsim/params.py`.

## O que o modelo representa

- **Cidade sintética**: malha viária em grade perturbada (399 vértices, 667 arestas, 76,6 km de vias), 280 pontos de coleta e 378 contêineres de 1.000 L. O centro comercial tem geração 1,6 vez maior. A garagem fica no sudoeste, e a saída para o aterro fica a leste.
- **Demanda**: 17 mil habitantes atendidos e 0,78 kg/hab./dia (média da Região Sul segundo a ABREMA). A geração diária é estocástica (Gama), com perfil de fim de semana, dois feriados e três eventos de pico no centro.
- **Sensores**: leitura com erro gaussiano (desvio de 5% da altura útil) e 5% de leituras perdidas. As taxas de enchimento são estimadas por EWMA.
- **Operação**: caminhão compactador de 7 t, Split ótimo da sequência em viagens ao aterro, 2-opt em cada viagem e jornada de 410 min de rota. Turnos extras são usados quando necessário.
- **Políticas**:
  - **B1** – rota fixa diária, todos os pontos.
  - **B2** – rota fixa setorizada: centro diariamente, setor A às segundas, quartas e sextas, setor B às terças, quintas e sábados.
  - **P1** – rota dinâmica por limiar fixo (θ = 70%).
  - **P0** – preditiva **sem sensores**: mesma regra de P2, com o nível estimado pela taxa média histórica de cada ponto e pelo calendário, a partir da última coleta.
  - **P0S** – como P0, mas só com a taxa média do setor.
  - **P2** – rota dinâmica preditiva com sensores: coleta o ponto se o limiar for atingido **ou** se o risco de transbordo até a próxima oportunidade de coleta superar o quantil *z*, considerando o calendário. `sensor_share < 1` sensoriza só parte dos pontos.
  - **P2I** – P2 com sensores ideais (sem erro e sem perda).
  - **P2M** – P2 com inclusão oportunista (*may-go*) de pontos próximos.
- **Pré-registro**: `PREREGISTRO.md` descreve a métrica *S* (fração do ganho que depende dos sensores) e as faixas de interpretação, fixadas antes de rodar os experimentos.
- **Roteirização**: distâncias pelo Algoritmo de Dijkstra (SciPy), sequência por algoritmo genético memético (OX, mutação por troca, torneio, elitismo e 2-opt nos filhos) e divisão em viagens pelo procedimento Split (Beasley, 1983; Prins, 2004). Para validação, o AG é comparado ao OR-Tools (*Guided Local Search*).
- **Verificação**: cada simulação confere o balanço de massa (gerado = coletado + transbordado + variação do estoque).
- **Economia**: custo total de propriedade e VPL em 5 anos (10% a.a.), com a fração λ dos turnos-equipe liberados que é convertida em economia e o ponto de equilíbrio λ\*.

## Caso real: Santa Helena (PR) e qualidade do histórico

Além da cidade sintética, o modelo roda sobre a malha viária real da sede urbana de Santa Helena (PR):

- `results/osm_santa_helena.json`: vias trafegáveis baixadas pela API Overpass em 24/11/2026. Malha viária © colaboradores do OpenStreetMap, sob a licença ODbL (https://www.openstreetmap.org/copyright).
- `results/setores_santa_helena_2022.geojson`: setores censitários do município, com a população (`v0001`) do Censo Demográfico 2022, extraídos da malha com atributos do IBGE (`PR_setores_CD2022.gpkg`).

Com os setores, os pontos de coleta ficam em cruzamentos dentro dos setores urbanos da sede, cerca de 1 por 60 habitantes, e a geração de cada ponto é proporcional à população do setor. A posição real dos contêineres, a garagem e a saída para o aterro não são públicas, por isso foram estimadas.

```bash
# malha real com demanda hipotética (280 pontos, 17 mil hab.)
python -m wrsim.experiments_osm --osm results/osm_santa_helena.json --out results_osm_demanda_hipotetica
# malha real e população do Censo 2022
python -m wrsim.experiments_osm --osm results/osm_santa_helena.json --census results/setores_santa_helena_2022.geojson --out results_santa_helena
python -m wrsim.analysis_osm --res results_santa_helena --fig figures
# P0 com 7, 30 e 90 dias de histórico e mudança inesperada na geração, nas três configurações
python -m wrsim.experiments_hist --out results_hist
python -m wrsim.analysis_hist --res results_hist
```

Políticas extras: `P0H<dias>` é P0 com a taxa de cada ponto estimada a partir de `<dias>` dias de pesagem, com erro amostral. O parâmetro `drift_share` muda sem aviso a geração de uma fração dos pontos a partir do início da avaliação.

## Calibração com dados públicos e rigor estatístico

- `results/wyndham_daily.csv`: nível diário de enchimento de 32 lixeiras públicas de Wyndham (Victoria, Austrália), 2018-2021, da base *Wyndham Smart Bin Daily Fill level* (Wyndham City Council, data.gov.au, licença CC BY 2.5 AU). `wrsim/calibracao.py` estima a taxa diária, o coeficiente de variação da geração (pelo enchimento acumulado em cada ciclo entre coletas, descontando o arredondamento em degraus de 20%) e o fator de fim de semana. Também verifica a regra de previsão: com 30 ou 7 dias de histórico, mede com que frequência o nível real supera a previsão com z = 1,645, 2,33 e 3,09, com e sem a leitura do dia.
- `wrsim/experiments_rigor.py` / `analysis_rigor.py`: experimento principal com 100 replicações (cidade sintética e Santa Helena), cenário com o CV calibrado (0,88) e sensibilidade global com 150 combinações de 13 premissas por hipercubo latino, analisadas por correlação parcial de postos (PRCC).

```bash
python -m wrsim.calibracao --data results/wyndham_daily.csv --out results/calibracao.json
python -m wrsim.experiments_rigor --out results_rigor
python -m wrsim.analysis_rigor --res results_rigor --fig figures
```

## Casos reais com dados do SINISA 2023 e limiar de variabilidade

`wrsim/experiments_real.py` / `analysis_real.py` rodam dois municípios do oeste do Paraná com destino, distância e frota declarados ao SINISA (Diagnóstico do Manejo de RSU, ano de referência 2023):

- Santa Helena: coleta contratada, aterro sanitário em Cascavel a 114 km por estrada (OSRM), sem estação de transbordo;
- São Miguel do Iguaçu: equipe própria, aterro controlado municipal a 11,7 km (`results/osm_sao_miguel.json`, `results/setores_sao_miguel_2022.geojson`).

Para cada caso: experimento principal (30 replicações), curva em z de P0 e P2 para a métrica *S* e uma varredura do CV diário da geração (0,20 a 0,90, 20 replicações por ponto). O resultado `limiar` em `summary_real.json` é o CV a partir do qual os sensores se pagam sobre P0 (λ∗ = 100%, mão de obra valorada por hora).

```bash
python -m wrsim.experiments_real --out results_real --workers 8
python -m wrsim.analysis_real --res results_real --data results --fig figures
```

`wrsim/experiments_real_extra.py` completa a varredura do CV até 60 replicações por ponto e testa a robustez às premissas sem dado público: quatro sorteios alternativos da posição dos pontos, densidade de pontos (cerca de 85 e 43 hab./ponto) e heterogeneidade da geração entre pontos (`pop_sigma` 0 e 0,7). A análise junta as duas rodadas, calcula intervalos por reamostragem (bootstrap) e decompõe a economia anual (horas normais, horas em turnos extras, diesel e manutenção).

```bash
python -m wrsim.experiments_real_extra --out results_real_extra --workers 8
python -m wrsim.analysis_real --res results_real --extra results_real_extra --data results --fig figures
```

A saída urbana (`exit_latlon`) é o ponto em que a rota do OSRM até o destino cruza a borda urbana do modelo, e `landfill_km` é o restante dessa rota (Santa Helena: PR-495, ao sul, 111,6 km; São Miguel do Iguaçu: PR-497, ao norte, 8,1 km). A garagem, cuja localização não é pública, fica no extremo oposto à saída.

## Estrutura

```
wrsim/
  params.py       parâmetros e sementes (dataclass imutável)
  city.py         geração da malha, pontos, setores, contêineres e matriz de distâncias
  routing.py      AG puro e memético, 2-opt, Split e avaliação do dia (Numba)
  sim.py          simulação diária: sensores, EWMA, políticas, geração e transbordos
  benchmark.py    comparação AG × OR-Tools
  experiments.py  experimento principal e análises de sensibilidade (600 simulações)
  economics.py    custos, VPL e ponto de equilíbrio
  analysis.py     indicadores, IC95%, diferenças pareadas, economia, figuras e summary.json
  experiments_osm.py / analysis_osm.py    malha real (OpenStreetMap) e população do Censo 2022
  experiments_hist.py / analysis_hist.py  P0 com histórico limitado e mudança na geração
  calibracao.py   calibração e verificação da regra com dados públicos (Wyndham)
  experiments_rigor.py / analysis_rigor.py  100 replicações, cenário calibrado e sensibilidade global
results/          saídas brutas (JSON) da execução usada no artigo
figures/          figuras do artigo (PNG)
```

## Como reproduzir

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
python -m wrsim.experiments --out results --workers 12   # 1.560 simulações + benchmark (~21 min com 12 processos)
python -m wrsim.analysis --res results --fig figures     # tabelas (summary.json) e figuras
```

Opções de `wrsim.experiments`:

- `--reps`: replicações do experimento principal (padrão 30).
- `--sens-reps`: replicações por cenário de sensibilidade (padrão 10).
- `--workers`: processos paralelos (padrão 4).
- `--bench-time`: tempo do OR-Tools por instância, em segundos (padrão 20).
- `--skip-bench`: não roda o benchmark (reaproveita um `benchmark.json` existente).

`rerun_tag.py` roda de novo um único cenário de sensibilidade e substitui os resultados dele em `sim_results.json`.

Os resultados são determinísticos dadas as sementes. Variações mínimas podem ocorrer apenas no benchmark do OR-Tools, que tem limite de tempo de relógio.

Versões usadas na execução que gerou `results/` e `figures/`: Python 3.13.18, NumPy 2.5.3, SciPy 1.18.1, Numba 0.68.1, Matplotlib 3.11.2 e OR-Tools 9.15.6755.

## Saídas

- `results/city.json`: resumo da malha e dos pontos.
- `results/sim_results.json`: indicadores diários e totais de cada simulação.
- `results/benchmark.json`: AG × OR-Tools por instância.
- `results/summary.json`: estatísticas usadas no texto (médias, IC95%, diferenças pareadas, Wilcoxon, *d_z*, economia e sensibilidades).

## Limitações

A cidade é sintética, e os custos (nós sensores, *gateways*, diesel, mão de obra) são hipóteses documentadas em `wrsim/economics.py` e na Tabela 4 do artigo. Para aplicar o modelo a um município real, substitua a malha por dados do OpenStreetMap, a geração por dados de pesagem e os custos por cotações locais.

## Licença

MIT. Veja `LICENSE`.
