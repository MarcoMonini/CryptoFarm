"""Le 41 colonne del modello a swing: niente look-ahead, e le scale lunghe arrivano in ritardo."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.bar_features import ASSET_COLUMNS, SWING_COLUMNS, build_swing_features


@pytest.fixture
def candele():
    """Sei mesi di barre 5m: bastano a riempire EMA200 su 1d senza pesare nei test."""
    n = 6 * 30 * 288
    idx = pd.date_range("2023-01-01", periods=n, freq="5min", name="Open time")
    rng = np.random.default_rng(7)
    close = pd.Series(rng.normal(scale=2.0, size=n).cumsum() + 20_000.0, index=idx)
    corpo = pd.concat([close.shift(1).bfill(), close], axis=1)
    return pd.DataFrame(
        {
            "Open": corpo.min(axis=1),
            "High": corpo.max(axis=1) + 3.0,
            "Low": corpo.min(axis=1) - 3.0,
            "Close": close,
            "Volume": rng.lognormal(size=n),
        },
        index=idx,
    )


def test_le_colonne_sono_quelle_dichiarate(candele):
    frame = build_swing_features("BTCUSDT", candele)
    assert list(frame.columns) == SWING_COLUMNS
    assert len(SWING_COLUMNS) == 41


def test_troncare_la_serie_non_cambia_il_passato(candele):
    """Se una feature a `t` cambia quando arrivano barre dopo `t`, quella feature legge il futuro.

    E' il controllo che coglie il look-ahead senza doverlo cercare colonna per colonna: si taglia
    **dentro** una barra giornaliera gia' cominciata, non su un confine, perche' un taglio
    allineato ai confini passa anche con l'allineamento sbagliato.
    """
    taglio = candele.index[len(candele) // 2 + 137]  # dentro la barra 1d e dentro quella 1h
    intero = build_swing_features("BTCUSDT", candele)
    troncato = build_swing_features("BTCUSDT", candele[candele.index <= taglio])
    comune = troncato.index
    for colonna in SWING_COLUMNS:
        a = intero.loc[comune, colonna].to_numpy()
        b = troncato[colonna].to_numpy()
        entrambe = ~(np.isnan(a) | np.isnan(b))
        assert np.isnan(a).sum() == np.isnan(b).sum(), f"{colonna}: NaN diversi"
        assert np.allclose(a[entrambe], b[entrambe]), f"{colonna} cambia quando arriva il futuro"


def test_la_scala_lunga_non_e_disponibile_prima_di_chiudere(candele):
    """La barra giornaliera etichettata `d` si puo' leggere solo da `d + 1 giorno`."""
    frame = build_swing_features("BTCUSDT", candele)
    colonna = frame["dist_ema50_atr@1d"]
    primo_noto = colonna.first_valid_index()
    # 50 giorni di EMA piu' il giorno di attesa: mai prima del cinquantunesimo giorno di storico.
    assert primo_noto >= candele.index[0] + pd.Timedelta(days=50)
    # e il valore deve restare costante per l'intera giornata che lo consuma
    giorno = colonna.loc[primo_noto : primo_noto + pd.Timedelta(hours=23, minutes=55)]
    assert giorno.nunique() == 1


def test_senza_store_di_posizionamento_le_due_colonne_sono_nan(candele):
    """Simbolo inesistente: le colonne di posizionamento devono essere NaN dichiarati, non zeri."""
    frame = build_swing_features("NONESISTEUSDT", candele)
    assert frame["affollamento_conti"].isna().all()
    assert frame["affollamento_posizioni"].isna().all()
    # le altre no: dipendono solo dalle candele
    assert frame[ASSET_COLUMNS[0]].notna().any()
