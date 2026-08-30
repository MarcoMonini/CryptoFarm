"""L'audit: la permutazione conserva cio' che deve, e i ranghi si sommano dentro il simbolo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import confluence_audit as audit


def _candele(barre: int = 4000, seme: int = 0) -> pd.DataFrame:
    """Candele **coerenti**: massimo sopra il corpo, minimo sotto, sempre.

    Non si riusa `confluence_lab._finte`: quello genera massimi sotto il corpo nel 6% delle barre,
    cioe' candele che non esistono (su 50.000 barre vere di BTC le violazioni sono zero). Qui la
    proprieta' da verificare e' che la permutazione **conservi** la coerenza, e su un ingresso gia'
    incoerente non si distinguerebbe dal suo contrario.
    """
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2021-01-01", periods=barre, freq="15min", name="Open time")
    apertura = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.004, barre)))
    chiusura = apertura * np.exp(rng.normal(0, 0.003, barre))
    corpo_alto = np.maximum(apertura, chiusura)
    corpo_basso = np.minimum(apertura, chiusura)
    return pd.DataFrame(
        {
            "Open": apertura,
            "High": corpo_alto * np.exp(abs(rng.normal(0, 0.002, barre))),
            "Low": corpo_basso * np.exp(-abs(rng.normal(0, 0.002, barre))),
            "Close": chiusura,
            "Volume": rng.random(barre) * 10,
        },
        index=idx,
    )


def test_la_permutazione_conserva_la_deriva_e_la_coerenza_delle_candele():
    """Il null deve togliere **solo** il tempismo.

    Se togliesse anche la deriva, una strategia lunga sembrerebbe brava per il solo fatto di stare
    esposta a un mercato che sale, e il confronto misurerebbe l'esposizione invece del tempismo.
    """
    candele = _candele()
    finte = audit.permuta(candele, np.random.default_rng(7))

    assert len(finte) == len(candele)
    assert finte.index.equals(candele.index)

    # La deriva totale: identica, perche' e' la somma degli stessi rendimenti in ordine diverso.
    vera = np.log(candele["Close"].iloc[-1] / candele["Open"].iloc[0])
    permutata = np.log(finte["Close"].iloc[-1] / finte["Open"].iloc[0])
    assert permutata == pytest.approx(vera, abs=1e-9)

    # Le candele restano possibili.
    assert (finte["High"] >= finte[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (finte["Low"] <= finte[["Open", "Close"]].min(axis=1) + 1e-9).all()

    # L'insieme dei rendimenti di barra e' lo stesso insieme, riordinato.
    def per_barra(df):
        return np.sort(np.log(df["Close"].to_numpy() / df["Open"].to_numpy()))

    assert per_barra(finte) == pytest.approx(per_barra(candele))


def test_la_permutazione_toglie_davvero_la_correlazione_seriale():
    """Se la permutazione lasciasse i trend al loro posto, il null non sarebbe un null."""
    candele = _candele(barre=20000)
    finte = audit.permuta(candele, np.random.default_rng(3))

    def autocorrelazione(df):
        chiusure = np.log(df["Close"].to_numpy())
        return abs(pd.Series(chiusure).autocorr(lag=1))

    # Il prezzo vero e' quasi una passeggiata: l'autocorrelazione del **livello** resta alta in
    # entrambi (e' un cammino in tutti e due i casi). Cio' che deve sparire e' la memoria della
    # *sequenza*, e si vede sul segno dei rendimenti di barra riordinati.
    def memoria(df):
        r = np.log(df["Close"].to_numpy() / df["Open"].to_numpy())
        return abs(pd.Series(r).autocorr(lag=1))

    assert memoria(finte) < 0.05, "la permutazione ha lasciato memoria fra barre consecutive"
    assert autocorrelazione(finte) > 0.9, "il prezzo permutato non e' piu' un cammino continuo"


def test_il_rango_si_calcola_dentro_il_simbolo_non_fra_simboli():
    """Due asset con rese incomparabili non devono pesare in modo diverso sulla classifica.

    E' la ragione per cui si sommano ranghi e non rendimenti: qui l'asset «grande» ha rese dieci
    volte quelle del «piccolo», ma l'**ordine** delle configurazioni e' lo stesso, quindi la
    classifica deve dirle equivalenti.
    """
    righe = []
    for simbolo, scala in (("GRANDE", 10.0), ("PICCOLO", 1.0)):
        for i, theta in enumerate((0.25, 0.35, 0.45)):
            righe.append(
                {
                    "simbolo": simbolo,
                    "finestra": "tutto",
                    "theta_base": theta,
                    "extra_%": scala * (i + 1),
                    "rendimento_%": scala * (i + 1),
                    "n_trade": 10,
                    "trade_anno": 5.0,
                    "drawdown_%": 5.0,
                    "sharpe": 0.1,
                    "necessarieta_max": 0.5,
                }
            )
    con_rango = audit.con_rango(pd.DataFrame(righe))
    top = audit.classifica(con_rango, "tutto")

    # theta 0,45 e' la migliore in tutti e due i simboli, quindi rango 1,0 in tutti e due.
    assert top["rango_mediano"].iloc[0] == pytest.approx(1.0)
    assert top.index[0].startswith("0.45")
    # E il rango non dipende dalla scala: le tre configurazioni prendono 1/3, 2/3, 1 in entrambi.
    assert sorted(con_rango[con_rango["simbolo"] == "GRANDE"]["rango"]) == pytest.approx(
        sorted(con_rango[con_rango["simbolo"] == "PICCOLO"]["rango"])
    )


def test_il_valore_p_non_restituisce_mai_zero():
    """Con `n` permutazioni il minimo onesto e' `1/(n+1)`: scrivere 0 sarebbe precisione non comprata."""
    mc = pd.DataFrame(
        [{"simbolo": "BTCUSDT", "vero": True, "extra_%": 999.0, "n_trade": 10}]
        + [{"simbolo": "BTCUSDT", "vero": False, "extra_%": float(i), "n_trade": 10} for i in range(99)]
    )
    p = audit.valore_p(mc)
    assert p["meglio_del_vero"].iloc[0] == 0
    assert p["valore_p"].iloc[0] == pytest.approx(1 / 100)
