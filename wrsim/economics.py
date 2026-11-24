"""Análise econômica: custo total de propriedade dos sistemas e economia
operacional anual em relação a cada cenário de referência.

Dois sistemas são custeados:
  "iot"  sensores em cada contêiner + gateways LoRaWAN + software de despacho (P2);
  "p0"   apenas software de despacho com taxas históricas e calendário (P0).
A mão de obra é valorada por hora de equipe efetivamente trabalhada (principal)
ou por turno-equipe inteiro (sensibilidade); horas em turnos acima de dois por
dia recebem adicional de hora extra.

Todos os valores monetários estão em reais (R$) de 2026. Os parâmetros sem fonte
oficial são hipóteses explícitas, submetidas à análise de sensibilidade.
"""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class Costs:
    diesel_rs_l: float = 7.35          # ANP, média nacional do diesel B S10 (27/09-03/10/2026)
    maint_rs_km: float = 1.10          # pneus, freios, manutenção variável (hipótese)
    crew_rs_h: float = 85.0            # equipe (motorista + 2 coletores) com encargos (hipótese)
    shift_h: float = 7.33              # jornada diária paga por turno-equipe (44 h/semana em 6 dias)
    overtime_premium: float = 0.50     # adicional sobre as horas em turnos acima de dois por dia
    node_hw_rs: float = 250.0          # ESP32 + rádio LoRa + ultrassônico + bateria + caixa IP67 (hipótese)
    node_install_rs: float = 40.0
    gateways: int = 2                  # um principal + um redundante
    gateway_rs: float = 4500.0
    integration_rs: float = 25000.0    # software de despacho, integração e capacitação (hipótese)
    node_replace_rate: float = 0.07    # fração de nós substituídos ao ano (falha/vandalismo)
    battery_rs_node_yr: float = 15.0
    server_rs_yr: float = 1200.0       # servidor de rede LoRaWAN de código aberto + hospedagem
    support_rs_yr: float = 6000.0      # manutenção técnica do software
    ops_rs_yr: float = 10000.0         # despacho diário e acompanhamento das leituras/alertas
    discount: float = 0.10
    horizon_yr: int = 5

    def to_dict(self):
        return asdict(self)


def capex(c: Costs, n_nodes: int, system: str = "iot") -> float:
    if system == "p0":
        return c.integration_rs
    if system == "sensors":        # incremento de P2 sobre P0: só sensores e gateways
        return n_nodes * (c.node_hw_rs + c.node_install_rs) + c.gateways * c.gateway_rs
    return n_nodes * (c.node_hw_rs + c.node_install_rs) + c.gateways * c.gateway_rs + c.integration_rs


def opex(c: Costs, n_nodes: int, system: str = "iot") -> float:
    nodes = n_nodes * (c.node_hw_rs * c.node_replace_rate + c.battery_rs_node_yr) + c.server_rs_yr
    if system == "p0":
        return c.support_rs_yr + c.ops_rs_yr
    if system == "sensors":
        return nodes
    return nodes + c.support_rs_yr + c.ops_rs_yr


def labor_cost(c: Costs, q: dict, unit: str = "hour") -> float:
    """Custo anual de mão de obra de uma política (quantidades anuais em `q`)."""
    if unit == "shift":
        return q["shifts"] * c.shift_h * c.crew_rs_h
    return (q["crew_h"] + c.overtime_premium * q["ot_h"]) * c.crew_rs_h


def operating_cost(c: Costs, q: dict, unit: str = "hour") -> float:
    """Custo operacional anual: mão de obra + diesel + manutenção variável."""
    return labor_cost(c, q, unit) + q["fuel_l"] * c.diesel_rs_l + q["km"] * c.maint_rs_km


def annual_savings(c: Costs, base: dict, prop: dict, labor_share: float, unit: str = "hour"):
    """Economia anual (R$) da política proposta em relação à de referência.

    `labor_share` = fração da mão de obra liberada que vira economia efetiva
    (0 = equipes fixas sem realocação; 1 = horas integralmente realocadas ou evitadas)."""
    fuel = (base["fuel_l"] - prop["fuel_l"]) * c.diesel_rs_l
    maint = (base["km"] - prop["km"]) * c.maint_rs_km
    labor = (labor_cost(c, base, unit) - labor_cost(c, prop, unit)) * labor_share
    return dict(fuel=fuel, maint=maint, labor=labor, total=fuel + maint + labor)


def _df(c: Costs):
    return sum((1 + c.discount) ** -t for t in range(1, c.horizon_yr + 1))


def npv(c: Costs, n_nodes: int, saving_yr: float, system: str = "iot"):
    k0 = capex(c, n_nodes, system)
    op = opex(c, n_nodes, system)
    net = saving_yr - op
    v = -k0 + net * _df(c)
    payback = k0 / net if net > 0 else float("inf")
    return dict(npv=v, capex=k0, opex=op, net_yr=net, payback_yr=payback)


def breakeven_labor_share(c: Costs, n_nodes: int, base: dict, prop: dict,
                          unit: str = "hour", system: str = "iot"):
    """Fração mínima da mão de obra liberada que precisa virar economia para VPL = 0.
    Valores acima de 1 indicam que o sistema não se paga nem com conversão integral."""
    s0 = annual_savings(c, base, prop, 0.0, unit)["total"]
    s1 = annual_savings(c, base, prop, 1.0, unit)["labor"]
    need = capex(c, n_nodes, system) / _df(c) + opex(c, n_nodes, system) - s0
    if s1 <= 0:
        return float("inf") if need > 0 else 0.0
    return max(need / s1, 0.0)
