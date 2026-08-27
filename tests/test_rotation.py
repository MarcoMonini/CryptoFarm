"""La contabilita' della rotazione trasversale.

Cinque proprieta', ognuna il piu' piccolo caso che fallisce se il motore si rompe. La quarta ha
gia' trovato un difetto vero: la prima versione teneva i pesi normalizzati a somma uno, e quando
meno di `top` asset avevano forza positiva la quota che sarebbe dovuta restare in contanti spariva
a ogni ribilanciamento. Su cinque anni faceva -100%.

Le serie sono sintetiche di proposito: il test non deve dipendere dallo store, che nella CI non
esiste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import rotation

INDICE = pd.date_range("2021-01-01", periods=120, freq="1D", name="Open time")


@pytest.fixture
def rampa() -> pd.DataFrame:
    """Un solo asset che raddoppia in linea retta."""
    return pd.DataFrame({"BTCUSDT": np.linspace(100.0, 200.0, 120)}, index=INDICE)


@pytest.fixture
def misto() -> pd.DataFrame:
    """Uno che raddoppia e uno che si dimezza: il caso in cui meta' capitale deve restare fermo."""
    return pd.DataFrame(
        {"BTCUSDT": np.linspace(100.0, 200.0, 120), "ETHUSDT": np.linspace(100.0, 50.0, 120)},
        index=INDICE,
    )


def test_prezzi_fermi_non_muovono_il_capitale_e_non_comprano_niente() -> None:
    """Momento zero non e' forza positiva: non si compra, e il capitale resta dov'e'."""
    piatto = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 100.0}, index=INDICE)
    esito = rotation.backtest(piatto, lookback=10, top=1, every=5, fee=0.0)
    assert esito["rendimento_%"] == pytest.approx(0.0)
    assert esito["ribilanciamenti"] == 0


def test_il_rendimento_parte_dall_acquisto_non_dall_inizio_della_serie(rampa: pd.DataFrame) -> None:
    """Le prime `lookback` barre il momento non e' definito: quel tratto non si guadagna."""
    comprato_a = rampa["BTCUSDT"].iloc[10]
    atteso = (200.0 / comprato_a - 1) * 100
    esito = rotation.backtest(rampa, lookback=10, top=1, every=5, fee=0.0)
    assert esito["rendimento_%"] == pytest.approx(atteso, abs=0.2)


def test_la_commissione_toglie(rampa: pd.DataFrame) -> None:
    gratis = rotation.backtest(rampa, lookback=10, top=1, every=5, fee=0.0)["rendimento_%"]
    pagando = rotation.backtest(rampa, lookback=10, top=1, every=5, fee=0.5)["rendimento_%"]
    assert pagando < gratis


def test_la_quota_in_contanti_non_sparisce(misto: pd.DataFrame) -> None:
    """`top=2` con un solo asset in forza: meta' investita, meta' ferma -- non meta' persa.

    Con la contabilita' a pesi normalizzati questo valeva -100%.
    """
    meta = rotation.backtest(misto, lookback=10, top=2, every=5, fee=0.0)["rendimento_%"]
    pieno = rotation.backtest(misto, lookback=10, top=1, every=5, fee=0.0)["rendimento_%"]
    assert 0 < meta < pieno


def test_nessun_look_ahead(misto: pd.DataFrame) -> None:
    """Troncare la serie non cambia le rotazioni gia' decise."""
    intera = rotation.backtest(misto, lookback=10, top=1, every=5, fee=0.0)["_holdings"]
    troncata = rotation.backtest(misto.iloc[:80], lookback=10, top=1, every=5, fee=0.0)["_holdings"]
    assert intera[: len(troncata)] == troncata


def test_la_forza_negativa_non_si_compra() -> None:
    """Un universo dove tutto scende resta interamente in contanti."""
    giu = pd.DataFrame(
        {"BTCUSDT": np.linspace(200.0, 100.0, 120), "ETHUSDT": np.linspace(200.0, 80.0, 120)},
        index=INDICE,
    )
    esito = rotation.backtest(giu, lookback=10, top=1, every=5, fee=0.0)
    assert esito["rendimento_%"] == pytest.approx(0.0)
    assert esito["_holdings"] == []


def test_il_filtro_di_regime_pretende_btc(misto: pd.DataFrame) -> None:
    """Senza BTC nell'universo l'interruttore non ha su cosa guardare, e deve dirlo."""
    senza_btc = misto.rename(columns={"BTCUSDT": "SOLUSDT"})
    with pytest.raises(ValueError, match="BTCUSDT"):
        rotation.backtest(senza_btc, lookback=10, top=1, every=5, regime="btc")


def test_i_riferimenti_battono_la_rotazione_quando_tutto_sale(rampa: pd.DataFrame) -> None:
    """Comprare e tenere parte dalla prima barra, la rotazione dalla prima con momento definito."""
    riferimenti = rotation.benchmarks(rampa)
    assert riferimenti["BTC comprare e tenere"]["rendimento_%"] == pytest.approx(100.0)
    rotazione = rotation.backtest(rampa, lookback=10, top=1, every=5, fee=0.0)
    assert rotazione["rendimento_%"] < riferimenti["BTC comprare e tenere"]["rendimento_%"]


def test_uno_store_vuoto_da_un_frame_vuoto_invece_di_sollevare(monkeypatch):
    """E' la condizione in cui gira il servizio pubblico: il piano gratuito non ha dischi
    persistenti, quindi `market_data/` e' vuoto.

    Senza l'uscita anticipata `pd.DataFrame({})` nasce con un RangeIndex di interi e il confronto
    con la data solleva `TypeError: Invalid comparison between dtype=int64 and str`. Non e' un
    dettaglio di tipi: faceva cadere l'**intera** vista di rotazione invece di mostrare l'avviso
    che le sta accanto da sempre, e nascondeva anche la raccolta di `test_simulator_page.py`.
    """
    monkeypatch.setattr(rotation, "load_klines", lambda *args, **kwargs: pd.DataFrame())
    vuoto = rotation.load_universe(["MAINONESISTE"], "1d", "2021-01-01")
    assert vuoto.empty
    assert isinstance(vuoto.index, pd.DatetimeIndex), "un indice di interi rompe ogni filtro a valle"
    assert vuoto.shape[1] == 0
