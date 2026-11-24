"""Calibração e verificação empírica com dados públicos de nível de enchimento.

Base: Wyndham City Council (Victoria, Austrália), "Wyndham Smart Bin Daily Fill level",
data.gov.au, licença CC BY 2.5 AU: 33 lixeiras públicas, uma leitura diária do nível
(0 a 10, em décimos da altura útil), 2018-2021.

1. Calibração: taxa média diária, coeficiente de variação da geração diária (estimado
   pelos ciclos entre coletas, para diluir o arredondamento em décimos) e fator de fim
   de semana.
2. Verificação da regra de decisão: com a taxa e a variância estimadas nos 30 dias
   anteriores, prevê-se o nível g = 1, 2 ou 3 dias à frente, a partir da leitura do dia
   (como P2) ou só do tempo desde a última coleta (como P0), e mede-se a frequência com
   que o nível real supera a previsão com z = 1,645, 2,33 e 3,09.

Uso:  python -m wrsim.calibracao --data results/wyndham_daily.csv --out results/calibracao.json
      (--raw wyndham_smartbin_filllevel.json gera o CSV diário a partir do GeoJSON original)
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd

UNITS = 10.0          # escala do nível (0 a 10)
DROP = 3              # queda mínima (em décimos) que caracteriza uma coleta
STEP = 2.0            # resolução da base: o nível só assume 0, 2, 4, 6, 8 e 10


def load(raw=None, data=None):
    if raw:
        feats = json.load(open(raw, encoding="utf-8"))["features"]
        df = pd.DataFrame([{"bin": f["properties"]["serialNumber"], "desc": f["properties"]["description"],
                            "date": f["properties"]["timestamp"], "fill": f["properties"]["latestFullness"]}
                           for f in feats])
    else:
        df = pd.read_csv(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["bin", "date"]).drop_duplicates(["bin", "date"])


def episodes(s):
    """Divide a série diária de uma lixeira em ciclos entre coletas (lista de séries)."""
    out, cur = [], [s.iloc[0]]
    idx = [s.index[0]]
    for i in range(1, len(s)):
        gap = (s.index[i] - s.index[i - 1]).days
        if gap != 1 or s.iloc[i] <= s.iloc[i - 1] - DROP:
            if len(cur) >= 3:
                out.append(pd.Series(cur, index=idx))
            cur, idx = [], []
        cur.append(s.iloc[i]); idx.append(s.index[i])
    if len(cur) >= 3:
        out.append(pd.Series(cur, index=idx))
    return out


def calibrate(df):
    res, wk = {}, defaultdict(list)
    for b, g in df.groupby("bin"):
        s = g.set_index("date")["fill"].astype(float)
        eps = episodes(s)
        F, K = [], []
        for e in eps:
            # usa o trecho até a primeira saturação (nível 10), que já não mede a geração
            sat = np.where(e.values >= UNITS)[0]
            e2 = e.iloc[: (sat[0] + 1 if len(sat) else len(e))]
            # o sensor marca 0 abaixo de um limiar; o ciclo começa na primeira leitura positiva
            pos = np.where(e2.values > 0)[0]
            if len(pos) == 0:
                continue
            e2 = e2.iloc[pos[0]:]
            if len(e2) < 3:
                continue
            k = len(e2) - 1
            F.append(e2.iloc[-1] - e2.iloc[0]); K.append(k)
            d = e2.diff().dropna()
            for t, v in d.items():
                wk[t.day_of_week].append(v)
        F, K = np.array(F), np.array(K)
        if len(F) < 8 or F.sum() <= 0:
            continue
        mu = F.sum() / K.sum()                                   # décimos por dia
        q = 2 * STEP ** 2 / 12.0                                  # variância do arredondamento (duas leituras)
        s2 = max(np.mean(((F - mu * K) ** 2 - q) / K), 1e-6)
        res[int(b)] = {"desc": g["desc"].iloc[0] if "desc" in g else "", "ciclos": int(len(F)),
                       "taxa_dia": float(mu / UNITS), "cv": float(np.sqrt(s2) / mu),
                       "ciclo_medio_dias": float(K.mean())}
    wkm = np.array([np.mean(wk[d]) for d in range(7)])
    return res, {"media_por_dia_semana": (wkm / wkm.mean()).tolist(),
                 "fator_fim_de_semana": float(wkm[5:].mean() / wkm[:5].mean())}


def backtest(df, hist=30, zs=(1.645, 2.33, 3.09)):
    """Previsão do nível g dias à frente com taxa e variância dos `hist` dias anteriores."""
    hits = {(mode, z): [0, 0] for mode in ("leitura", "sem_leitura") for z in zs}
    err = {"leitura": [], "sem_leitura": []}
    for b, g in df.groupby("bin"):
        s = g.set_index("date")["fill"].astype(float) / UNITS
        eps = episodes(s * UNITS)
        incs = []                    # (data, incremento diário) não saturado
        for e in eps:
            e = e / UNITS
            d = e.diff().dropna()
            ok = e.shift(1).dropna() < 1.0
            incs += [(t, v) for (t, v), o in zip(d.items(), ok.values) if o]
        if len(incs) < hist + 10:
            continue
        inc_s = pd.Series([v for _, v in incs], index=[t for t, _ in incs]).sort_index()
        for e in eps:
            e = e / UNITS
            t0 = e.index[0]
            past = inc_s[(inc_s.index < t0) & (inc_s.index >= t0 - pd.Timedelta(days=hist))]
            if len(past) < hist // 3:
                continue
            r, v = past.mean(), past.var(ddof=1)
            for i in range(len(e)):
                for gd in (1, 2, 3):
                    j = i + gd
                    if j >= len(e) or e.iloc[j] >= 1.0:
                        break
                    actual = e.iloc[j]
                    for mode in ("leitura", "sem_leitura"):
                        if mode == "leitura":
                            base, n_unc = e.iloc[i], gd
                        else:
                            base, n_unc = e.iloc[0] + r * i, i + gd
                        mean = base + r * gd
                        err[mode].append(actual - mean)
                        for z in zs:
                            hits[(mode, z)][0] += actual > mean + z * np.sqrt(max(v, 1e-6) * n_unc)
                            hits[(mode, z)][1] += 1
    return {f"{m}|z={z}": {"excedencia": h[0] / h[1], "n": h[1],
                           "nominal": float(1 - __import__('scipy').stats.norm.cdf(z))}
            for (m, z), h in hits.items()} | {
        f"eam_{m}": float(np.mean(np.abs(e))) for m, e in err.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="")
    ap.add_argument("--data", default="results/wyndham_daily.csv")
    ap.add_argument("--out", default="results/calibracao.json")
    ap.add_argument("--save-csv", default="")
    a = ap.parse_args()
    df = load(a.raw or None, None if a.raw else a.data)
    if a.save_csv:
        df[["bin", "desc", "date", "fill"]].to_csv(a.save_csv, index=False)
    bins, week = calibrate(df)
    cv = np.array([b["cv"] for b in bins.values()]); tx = np.array([b["taxa_dia"] for b in bins.values()])
    geral = np.array([" G " in (" " + b["desc"] + " ") for b in bins.values()])
    out = {"fonte": "Wyndham City Council, Smart Bin Daily Fill level (data.gov.au, CC BY 2.5 AU)",
           "n_lixeiras": len(bins), "n_dias": int(df.date.nunique()),
           "periodo": [str(df.date.min().date()), str(df.date.max().date())],
           "cv": {"mediana": float(np.median(cv)), "p25": float(np.percentile(cv, 25)), "p75": float(np.percentile(cv, 75)),
                  "mediana_geral": float(np.median(cv[geral])) if geral.any() else None,
                  "mediana_reciclavel": float(np.median(cv[~geral])) if (~geral).any() else None},
           "taxa_dia": {"mediana": float(np.median(tx)), "p25": float(np.percentile(tx, 25)), "p75": float(np.percentile(tx, 75))},
           "semana": week, "backtest_30d": backtest(df, 30), "backtest_7d": backtest(df, 7),
           "lixeiras": bins}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "lixeiras"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
