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
  - **P2** – rota dinâmica preditiva: coleta o ponto se o limiar for atingido **ou** se o risco de transbordo até a próxima oportunidade de coleta superar o quantil *z*, considerando o calendário.
- **Roteirização**: distâncias pelo Algoritmo de Dijkstra (SciPy), sequência por algoritmo genético memético (OX, mutação por troca, torneio, elitismo e 2-opt nos filhos) e divisão em viagens pelo procedimento Split (Beasley, 1983; Prins, 2004). Para validação, o AG é comparado ao OR-Tools (*Guided Local Search*).
- **Verificação**: cada simulação confere o balanço de massa (gerado = coletado + transbordado + variação do estoque).
- **Economia**: custo total de propriedade e VPL em 5 anos (10% a.a.), com a fração λ dos turnos-equipe liberados que é convertida em economia e o ponto de equilíbrio λ\*.

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
  analysis.py     indicadores, IC95%, Wilcoxon, tamanho de efeito, figuras e summary.json
results/          saídas brutas (JSON) da execução usada no artigo
figures/          figuras do artigo (PNG)
```

## Como reproduzir

Requer Python 3.10 ou superior.

```bash
pip install -r requirements.txt
python -m wrsim.experiments --out results            # 600 simulações + benchmark (~11 min com 2 núcleos)
python -m wrsim.analysis --res results --fig figures  # tabelas (summary.json) e figuras
```

Opções de `wrsim.experiments`:

- `--reps`: replicações do experimento principal (padrão 30).
- `--sens-reps`: replicações por cenário de sensibilidade (padrão 10).
- `--workers`: processos paralelos (padrão 2).
- `--bench-time`: tempo do OR-Tools por instância, em segundos (padrão 20).

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
