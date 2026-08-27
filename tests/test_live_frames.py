"""La barra in formazione e' l'unico punto dove `cummax` e `max` si confondono.

`groupby.transform("max")` restituisce il massimo **dell'intero periodo**, cioe' anche di barre
che alle 10:00 non sono ancora accadute. E' un errore di una lettera rispetto a `cummax`, non lo
segnala nessun tipo, e trasforma il backtest in una macchina che conosce il futuro. Questi test
esistono per quello.
"""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.data.klines import resample_klines
from cryptofarm.trading.live_frames import _selfcheck, forming_bars, provisional_ema


@pytest.fixture
def candele():
    idx = pd.date_range("2024-06-01", periods=3 * 96, freq="15min", name="Open time")
    prezzo = np.arange(len(idx), dtype=float)  # monotona: rende ovvio quale barra e' stata letta
    return pd.DataFrame(
        {"Open": prezzo, "High": prezzo + 1, "Low": prezzo - 1, "Close": prezzo, "Volume": np.ones(len(idx))},
        index=idx,
    )


def test_selfcheck_del_modulo():
    _selfcheck()


def test_il_massimo_in_formazione_non_conosce_il_resto_del_periodo(candele):
    """Su una serie crescente il massimo del periodo sta alla fine: se la barra delle 00:15 lo
    conoscesse gia', il difetto sarebbe questo."""
    f = forming_bars(candele, "1d")

    # Seconda barra del primo giorno: ha visto due barre, il suo massimo e' quello della seconda.
    assert f.high[1] == candele["High"].iloc[1]
    # ...e non quello dell'intera giornata.
    assert f.high[1] < candele["High"].iloc[:96].max()
    # All'ultima barra del giorno, invece, coincidono.
    assert f.high[95] == candele["High"].iloc[:96].max()


def test_alla_chiusura_la_barra_in_formazione_e_la_barra_aggregata(candele):
    for interval in ("1h", "4h", "1d"):
        f = forming_bars(candele, interval)
        aggregate = resample_klines(candele, interval)
        assert f.closes_here.sum() == len(aggregate), interval
        assert np.allclose(f.high[f.closes_here], aggregate["High"].to_numpy()), interval
        assert np.allclose(f.volume[f.closes_here], aggregate["Volume"].to_numpy()), interval


def test_il_volume_in_formazione_cresce_dentro_il_periodo_e_riparte_dopo(candele):
    f = forming_bars(candele, "1h")
    assert list(f.volume[:4]) == [1.0, 2.0, 3.0, 4.0]
    assert f.volume[4] == 1.0  # nuova ora, si riparte


def test_una_storia_troncata_produce_gli_stessi_valori_gia_emessi(candele):
    intero = forming_bars(candele, "4h")
    troncato = forming_bars(candele.iloc[:100], "4h")
    for a, b in ((troncato.high, intero.high), (troncato.low, intero.low), (troncato.volume, intero.volume)):
        assert np.allclose(a, b[:100])


def test_la_ema_provvisoria_sta_fra_lo_stato_chiuso_e_il_prezzo_corrente(candele):
    aggregate = resample_klines(candele, "4h")
    chiusa = aggregate["Close"].ewm(span=3, adjust=False).mean()

    prov = provisional_ema(chiusa.to_numpy(), aggregate.index, "4h", candele.index, candele["Close"].to_numpy(), 3)

    visti = ~np.isnan(prov)
    assert visti.any()
    # Su una serie crescente la provvisoria sta sempre sopra lo stato dell'ultima chiusura
    # e sotto il prezzo corrente: e' una media dei due, con il peso di uno span 3.
    from cryptofarm.trading.mtf import align_to_lower

    base = align_to_lower(chiusa.to_numpy(), aggregate.index, "4h", candele.index)
    assert (prov[visti] >= base[visti] - 1e-9).all()
    assert (prov[visti] <= candele["Close"].to_numpy()[visti] + 1e-9).all()
