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

from cryptofarm.trading import config, indicators, panels


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
    prodotte = indicatore.serie(frame)
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
