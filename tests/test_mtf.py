"""L'allineamento fra intervalli e' il punto in cui una strategia multi-timeframe imbroglia.

Questi test sono scritti per fallire se qualcuno "semplifica" `align_to_lower` in un `reindex`
diretto: il difetto non si vede leggendo il codice, e produce backtest ottimi e falsi.
"""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading.mtf import _selfcheck, align_to_lower


def test_selfcheck_del_modulo():
    _selfcheck()


def test_la_barra_lunga_e_disponibile_solo_dopo_la_sua_chiusura():
    ore = pd.date_range("2024-05-01", periods=12, freq="4h")
    valori = np.arange(len(ore), dtype=float)
    quarti = pd.date_range("2024-05-01", periods=12 * 16, freq="15min")

    serie = pd.Series(align_to_lower(valori, ore, "4h", quarti), index=quarti)

    # Dentro la barra 4h che va dalle 04:00 alle 08:00 si vede la barra chiusa alle 04:00,
    # cioe' quella etichettata 00:00, e mai la propria.
    assert serie.loc["2024-05-01 05:45"] == 0.0
    assert serie.loc["2024-05-01 07:45"] == 0.0
    assert serie.loc["2024-05-01 08:00"] == 1.0


def test_prima_della_prima_chiusura_lunga_non_si_indovina():
    giorni = pd.date_range("2024-05-01", periods=3, freq="1D")
    quarti = pd.date_range("2024-05-01", periods=96, freq="15min")

    allineato = align_to_lower(np.arange(3.0), giorni, "1d", quarti)

    assert np.isnan(allineato).all(), "un giorno intero senza barre 1d chiuse deve restare NaN"


def test_troncare_le_barre_corte_non_cambia_i_valori_gia_emessi():
    giorni = pd.date_range("2024-05-01", periods=6, freq="1D")
    valori = np.arange(6.0)
    quarti = pd.date_range("2024-05-01", periods=6 * 96, freq="15min")

    intero = align_to_lower(valori, giorni, "1d", quarti)
    troncato = align_to_lower(valori, giorni, "1d", quarti[:200])

    assert np.array_equal(troncato, intero[:200], equal_nan=True)


def test_una_serie_di_lunghezza_sbagliata_e_un_errore_non_un_allineamento_silenzioso():
    giorni = pd.date_range("2024-05-01", periods=4, freq="1D")
    with pytest.raises(ValueError, match="3 valori"):
        align_to_lower(np.arange(3.0), giorni, "1d", giorni)
