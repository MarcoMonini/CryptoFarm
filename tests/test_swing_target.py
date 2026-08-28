"""Il target di prossimita' agli estremi: forma, saturazione e confine col futuro."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.labeling import swing_target


def _rango_centrato_a_forza_bruta(close: np.ndarray, window: int) -> np.ndarray:
    """Definizione ingenua, O(n x window): il rango medio della barra centrale nella sua finestra."""
    fuori = np.full(len(close), np.nan)
    for i in range(window, len(close) - window):
        finestra = close[i - window : i + window + 1]
        sotto = (finestra < close[i]).sum()
        pari = (finestra == close[i]).sum()
        rango = sotto + (pari + 1) / 2  # rango medio sui pari merito, come pandas
        fuori[i] = (rango - 1.0) / window - 1.0
    return fuori


def test_coincide_con_la_definizione_ingenua():
    close = np.random.default_rng(0).normal(size=400).cumsum() + 100.0
    atteso = _rango_centrato_a_forza_bruta(close, 20)
    ottenuto = swing_target(close, 20).to_numpy()
    valide = ~np.isnan(atteso)
    assert np.allclose(ottenuto[valide], atteso[valide])


def test_massimo_e_minimo_saturano():
    close = np.concatenate([np.arange(30.0), np.arange(28.0, -1.0, -1.0)])  # V rovesciata
    target = swing_target(close, 10).to_numpy()
    assert target[29] == pytest.approx(1.0)  # la punta
    close_v = -close
    assert swing_target(close_v, 10).to_numpy()[29] == pytest.approx(-1.0)


def test_dentro_una_tendenza_regolare_vale_zero():
    """La proprieta' per cui questo target esiste: la salita non e' un massimo."""
    close = np.arange(200.0)
    target = swing_target(close, 20).to_numpy()
    centro = target[50:150]
    assert np.allclose(centro, 0.0), f"in tendenza il target dovrebbe stare a 0, sta a {centro[:5]}"


def test_le_ultime_barre_non_sono_etichettabili():
    close = np.random.default_rng(1).normal(size=100).cumsum() + 50.0
    target = swing_target(close, 12)
    assert target.iloc[-12:].isna().all()
    assert target.iloc[:12].isna().all()
    assert target.iloc[12:-12].notna().all()


def test_ritardare_di_una_barra_non_basta():
    """Il target ritardato di 1 conosce ancora quasi tutto il futuro che il target di oggi conosce.

    E' la trappola che il docstring segnala: solo un ritardo > window scollega le due finestre.
    """
    close = pd.Series(np.random.default_rng(2).normal(size=3000).cumsum() + 100.0)
    target = swing_target(close, 24)
    assert target.corr(target.shift(1)) > 0.75  # misurato 0,80: sa quasi tutto del futuro
    assert abs(target.corr(target.shift(25))) < 0.4  # misurato -0,22: informazione, non copia
