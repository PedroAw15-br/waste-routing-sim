"""Análise econômica: custo total de propriedade do sistema IoT e economia
operacional anual em relação a cada cenário de referência.

Todos os valores monetários estão em reais (R$) de 2026. Os parâmetros sem fonte
oficial são hipóteses explícitas, submetidas à análise de sensibilidade.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass(frozen=True)
class Costs:
    diesel_rs_l: float = 7.35          # ANP, média nacional do diesel B S10 (27/09-03/10/2026)
    maint_rs_km: float = 1.10          # pneus, freios, manutenção variável (hipótese)
    crew_rs_h: float = 85.0            # equipe (motorista + 2 coletores) com encargos (hipótese)
    shift_h: float = 7.33              # jornada diária paga por turno-equipe (44 h/semana em 6 dias)
    node_hw_rs: float = 250.0          # ESP32 + rádio LoRa + ultrassônico + bateria + caixa IP67 (hipótese)
    node_install_rs: float = 40.0
    gateways: int = 2                  # um principal + um redundante
    gateway_rs: float = 4500.0
    integration_rs: float = 25000.0    # implantação, integração e capacitação (hipótese)
    node_replace_rate: float = 0.07    # fração de nós substituídos ao ano (falha/vandalismo)
    battery_rs_node_yr: float = 15.0
    server_rs_yr: float = 1200.0       # servidor de rede LoRaWAN de código aberto + hospedagem
    support_rs_yr: float = 6000.0      # manutenção técnica
    discount: float = 0.10
    horizon_yr: int = 5

    def to_dict(self):
        return asdict(self)


def capex(c: Costs, n_nodes: int) -> float:
    return n_nodes * (c.node_hw_rs + c.node_install_rs) + c.gateways * c.gateway_rs + c.integration_rs


def opex(c: Costs, n_nodes: int) -> float:
    return (n_nodes * (c.node_hw_rs * c.node_replace_rate + c.battery_rs_node_yr)
            + c.server_rs_yr + c.support_rs_yr)


def annual_savings(c: Costs, base: dict, prop: dict, labor_share: float):
    """Economia anual (R$) da política proposta em relação à de referência.

    `base` e `prop` trazem quantidades anuais: fuel_l, km, shifts (turnos-equipe).
    `labor_share` = fração dos turnos-equipe liberados que vira economia efetiva
    (0 = equipes fixas sem realocação; 1 = turnos integralmente realocados ou evitados)."""
    fuel = (base["fuel_l"] - prop["fuel_l"]) * c.diesel_rs_l
    maint = (base["km"] - prop["km"]) * c.maint_rs_km
    labor = (base["shifts"] - prop["shifts"]) * c.crew_rs_h * c.shift_h * labor_share
    return dict(fuel=fuel, maint=maint, labor=labor, total=fuel + maint + labor)


def npv(c: Costs, n_nodes: int, saving_yr: float):
    k0 = capex(c, n_nodes)
    net = saving_yr - opex(c, n_nodes)
    df = np.array([(1 + c.discount) ** -t for t in range(1, c.horizon_yr + 1)])
    v = -k0 + net * df.sum()
    payback = k0 / net if net > 0 else float("inf")
    return dict(npv=v, capex=k0, opex=opex(c, n_nodes), net_yr=net, payback_yr=payback)


def breakeven_labor_share(c: Costs, n_nodes: int, base: dict, prop: dict):
    """Fração mínima dos turnos-equipe liberados que precisa virar economia para VPL = 0."""
    s0 = annual_savings(c, base, prop, 0.0)["total"]
    s1 = annual_savings(c, base, prop, 1.0)["labor"]
    df = sum((1 + c.discount) ** -t for t in range(1, c.horizon_yr + 1))
    need = capex(c, n_nodes) / df + opex(c, n_nodes) - s0
    if s1 <= 0:
        return float("inf")
    return need / s1
