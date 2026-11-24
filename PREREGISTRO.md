# Pré-registro da regra de enquadramento — 23/11/2026, antes de rodar os experimentos novos

Escrito antes de qualquer resultado de P0, P0S, P2I ou P2M existir.

## Métrica

Custo operacional anual C (R$/ano) = horas de equipe × R$ 85/h + diesel × R$ 7,35/L + km × R$ 1,10/km
(horas valoradas por hora trabalhada; turnos acima de 2 por dia com adicional de 50%).

S = (C_P0 − C_P2) / (C_B2 − C_P2)

- C_P0 é interpolado (linear em z ∈ {1,645; 2,33; 3,09}) no nível de transbordos (eventos/mês) que P2 obtém com z = 2,33.
- S é a fração do ganho de P2 sobre B2 que depende da leitura dos sensores (valor da informação).

## Faixas

| S | Tese do artigo |
|---|---|
| S < 0,30 | O agendamento preditivo por calendário, sem IoT, capta a maior parte do ganho; o sensor é complemento e só se paga em condições específicas. Título muda. |
| 0,30 ≤ S ≤ 0,60 | Decomposição do valor da informação: duas contribuições lado a lado (agendamento preditivo e sensor). |
| S > 0,60 | Mantém a tese do sensor; λ* recalculado sobre P2 − P0. |

P0S (taxa por setor, sem sensor) entra como contraprova realista em qualquer faixa.

## Políticas e cenários fixados

Principal (30 replicações, números aleatórios comuns): B1, B2, P1, P0, P0S, P2, P2I, P2M.
Curva de serviço (30 replicações): P0 e P2 com z ∈ {1,645; 2,33; 3,09}.
Aterro a 5 e 30 km (30 replicações): B1, B2, P0, P2.
Sensibilidades (10 replicações): limiar; z = 1,28; sensores degradados; sem calendário; cobertura parcial
(25/50/75%); geração ±20%; tempos de serviço; CV; sem picos; aquecimento de 14 dias; porte de 10 mil hab.;
cinco outras malhas; frota limitada a 2 turnos; B2 com +20% de capacidade no centro.
