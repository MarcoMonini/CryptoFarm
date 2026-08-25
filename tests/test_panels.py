"""Il registro di `trading/panels.py` deve corrispondere alla realta', non alle intenzioni.

Un registro scritto a mano si stacca dal codice in silenzio: basta rinominare una colonna, o
aggiungere una voce al menu senza registrarla, e la pagina mostra un indicatore vuoto o ne nasconde
uno che serve. Nessuno se ne accorge guardando, perche' una traccia mancante non e' un errore --
e' solo una linea che non c'e'.

Questi test chiudono le quattro strade per cui puo' succedere: una voce di menu senza registro, un
parametro che in `config` non esiste, una serie che nel frame non c'e', un pannello dichiarato che
poi non produce niente.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import config, indicators, panels, strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions, simulate_trading_with_commisions


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Candele sintetiche con tutte le colonne che `add_technical_indicator` produce."""
    generatore = np.random.default_rng(11)
    chiusure = 30_000 * np.exp(np.cumsum(generatore.normal(0.0003, 0.008, 600)))
    ampiezza = np.abs(generatore.normal(0, 0.004, 600)) * chiusure
    aperture = np.concatenate([[chiusure[0]], chiusure[:-1]])
    grezzo = pd.DataFrame(
        {
            "Open": aperture,
            "High": np.maximum(aperture, chiusure) + ampiezza,
            "Low": np.minimum(aperture, chiusure) - ampiezza,
            "Close": chiusure,
            "Volume": generatore.uniform(10, 900, 600),
        },
        index=pd.date_range("2024-01-01", periods=600, freq="1h", name="Open time"),
    )
    return indicators.add_technical_indicator(grezzo)


@pytest.fixture(scope="module")
def frame_laterale() -> pd.DataFrame:
    """Mercato che oscilla senza andare da nessuna parte: ADX basso.

    Serve alla strategia di ritorno alla media, che entra solo quando l'ADX dice che un trend non
    c'e'. Sulla passeggiata casuale in trend non produce operazioni, e il confronto fra i due
    motori resterebbe non verificato proprio dove serve.
    """
    generatore = np.random.default_rng(3)
    passi = np.arange(1_200)
    # Il rumore non e' decorativo: una sinusoide liscia ha ADX mediano 51, perche' per l'ADX
    # un'oscillazione regolare e' un trend fortissimo. Con il rumore scende a 22 e la strategia
    # trova le barre di intervallo che cerca.
    chiusure = 30_000 + 900 * np.sin(passi / 9.0) + generatore.normal(0, 250, passi.size)
    aperture = np.concatenate([[chiusure[0]], chiusure[:-1]])
    grezzo = pd.DataFrame(
        {
            "Open": aperture,
            "High": np.maximum(aperture, chiusure) + 60,
            "Low": np.minimum(aperture, chiusure) - 60,
            "Close": chiusure,
            "Volume": np.full(passi.size, 500.0),
        },
        index=pd.date_range("2024-01-01", periods=passi.size, freq="1h", name="Open time"),
    )
    return indicators.add_technical_indicator(grezzo)


def test_il_menu_e_il_registro_coincidono() -> None:
    """Ogni voce di `config.STRATEGIES` ha una riga nel registro, e viceversa.

    Le due liste vivono in file diversi: senza questo controllo una voce aggiunta al menu resta
    senza indicatori e senza parametri, e la pagina la mostra spoglia invece di rompersi.
    """
    menu = {voce for voce in config.STRATEGIES if voce != panels.VUOTA}
    assert menu == set(panels.STRATEGIE)


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_ogni_parametro_di_un_indicatore_esiste_in_config(chiave: str) -> None:
    for nome in panels.INDICATORI[chiave].parametri:
        assert isinstance(getattr(config, nome, None), config.Param), f"{chiave}: manca config.{nome}"


@pytest.mark.parametrize("nome", sorted(panels.STRATEGIE))
def test_ogni_strategia_riferisce_indicatori_e_parametri_esistenti(nome: str) -> None:
    strategia = panels.STRATEGIE[nome]
    for chiave in strategia.indicatori:
        assert chiave in panels.INDICATORI, f"{nome}: indicatore sconosciuto {chiave}"
    for parametro in strategia.parametri:
        assert isinstance(getattr(config, parametro, None), config.Param), f"{nome}: manca config.{parametro}"


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_le_serie_dichiarate_esistono_davvero_nel_frame(chiave: str, frame: pd.DataFrame) -> None:
    """E' il controllo che intercetta un nome di colonna sbagliato, come la vecchia `EMA200`."""
    indicatore = panels.INDICATORI[chiave]
    prodotte = indicatore.serie(frame, ExtraCache(frame), panels.valori_predefiniti())
    for traccia in indicatore.tracce:
        assert traccia.serie in prodotte, f"{chiave}: la traccia '{traccia.nome}' non ha la serie {traccia.serie}"
        assert prodotte[traccia.serie].notna().any(), f"{chiave}: la serie {traccia.serie} e' tutta vuota"


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_un_indicatore_con_pannello_disegna_qualcosa(chiave: str) -> None:
    """Un pannello senza tracce sarebbe un riquadro vuoto che occupa spazio nel grafico."""
    indicatore = panels.INDICATORI[chiave]
    if indicatore.pannello is not None:
        assert indicatore.tracce, f"{chiave}: dichiara un pannello ma non disegna nulla"


def test_senza_strategia_si_mostra_tutto() -> None:
    assert panels.indicatori_di(panels.VUOTA) == tuple(panels.INDICATORI)
    assert panels.indicatori_di("Trend Zones") == ("medie_trend",)


def test_i_parametri_non_si_ripetono_e_includono_le_dipendenze() -> None:
    """`Close ATR` non disegna medie, ma le sue bande dipendono da EMA Short via KAMA."""
    parametri = panels.parametri_di("Close ATR")
    assert len(parametri) == len(set(parametri))
    assert "EMA_SHORT" in parametri
    assert "RSI_SHORT" not in parametri


def test_i_colori_degli_indicatori_non_sono_quelli_di_stato() -> None:
    """Verde e rosso dicono rialzo e ribasso: un indicatore che li usa si legge come un segnale."""
    stato = {panels.RIALZO, panels.RIBASSO}
    for chiave, indicatore in panels.INDICATORI.items():
        for traccia in indicatore.tracce:
            assert traccia.colore not in stato, f"{chiave}: '{traccia.nome}' usa un colore di stato"


# -------------------------------------------------------------------------------------------------
# L'adattatore fra i due motori
# -------------------------------------------------------------------------------------------------

NUOVE = ("Donchian Breakout", "Squeeze Breakout", "Trend Pullback", "Ichimoku Trend", "Band Reversion")


@pytest.mark.parametrize("nome", NUOVE)
def test_le_strategie_nuove_nella_pagina_non_vanno_mai_corte(nome: str, frame: pd.DataFrame) -> None:
    """La pagina le chiama con `allow_short=False`: il verso corto e' misurato in perdita."""
    eventi = strategies_ls.STRATEGIES[_funzione(nome)](frame, ExtraCache(frame), allow_short=False)
    assert all(obiettivo >= 0 for _, _, obiettivo in eventi)


@pytest.mark.parametrize("nome", NUOVE)
def test_la_conversione_in_due_liste_conserva_le_operazioni(
    nome: str, frame: pd.DataFrame, frame_laterale: pd.DataFrame
) -> None:
    """Le stesse operazioni, agli stessi prezzi, dei due motori.

    Il formato a due liste accoppia per indice, quindi la conversione regge solo se gli eventi si
    alternano ingresso/uscita -- vero senza il verso corto, falso con. Il controllo e' diretto:
    stesse operazioni di `simulate_positions`, e a commissioni nulle anche lo stesso capitale.
    """
    valori = panels.valori_predefiniti()
    for candele in (frame, frame_laterale):
        cache = ExtraCache(candele)
        eventi = strategies_ls.STRATEGIES[_funzione(nome)](candele, cache, allow_short=False)
        if len(eventi) >= 4:
            break
    else:
        pytest.fail(f"{nome} non produce operazioni in nessuno dei due scenari: il confronto sarebbe vuoto")

    acquisti, uscite = panels.STRATEGIE[nome].esegui(candele, cache, valori)
    a_posizioni = simulate_positions(eventi, wallet=100, fee_percent=0.0, carry_daily_percent=0.0)
    a_segnali = simulate_trading_with_commisions(acquisti, uscite, wallet=100, fee_percent=0.0)

    assert [(o["Buy_Time"], o["Sell_Time"]) for o in a_segnali] == [
        (o["Buy_Time"], o["Sell_Time"]) for o in a_posizioni
    ]
    assert [(o["Buy_Price"], o["Sell_Price"]) for o in a_segnali] == [
        (o["Buy_Price"], o["Sell_Price"]) for o in a_posizioni
    ]
    assert a_segnali[-1]["Wallet_After"] == pytest.approx(a_posizioni[-1]["Wallet_After"], rel=1e-9)


def _funzione(voce_di_menu: str) -> str:
    """Dalla voce del menu al nome in `strategies_ls.STRATEGIES`."""
    return {
        "Donchian Breakout": "donchian_breakout",
        "Squeeze Breakout": "squeeze_breakout",
        "Trend Pullback": "trend_pullback",
        "Ichimoku Trend": "ichimoku_trend",
        "Band Reversion": "band_reversion_gated",
    }[voce_di_menu]
