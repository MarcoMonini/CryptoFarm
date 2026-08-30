"""Lo store del posizionamento: allineamento senza look-ahead e aggiornamento incrementale.

Niente rete: i due `fetch_*` vengono sostituiti. Quello che va protetto qui non e' il download
ma l'allineamento -- e' l'unico punto del modulo in cui si puo' leggere il futuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.data import positioning


def _metriche(giorno: str) -> pd.DataFrame:
    """Un giorno di istantanee a 5 minuti, con valori che crescono di uno a ogni riga."""
    index = pd.date_range(giorno, periods=288, freq="5min", name="Open time")
    valori = np.arange(288, dtype=float)
    return pd.DataFrame({nome: valori for nome in positioning._METRICS_RENAME.values()}, index=index)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    giorni = {}

    def finto_metrics(symbol, day):
        return giorni.get(day)

    def finto_funding(symbol, month):
        if month != "2024-01":
            return None
        # due applicazioni: alle 08:00 e alle 16:00 del primo giorno
        index = pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 16:00"], name="Open time")
        return pd.Series([0.01, 0.02], index=index, name="funding_rate")

    monkeypatch.setattr(positioning, "fetch_metrics_day", finto_metrics)
    monkeypatch.setattr(positioning, "fetch_funding_month", finto_funding)
    monkeypatch.setattr(positioning, "FIRST_DAY", "2024-01-01")
    return tmp_path, giorni


def test_reindex_non_guarda_avanti(tmp_path):
    """Su un indice piu' lento, ogni riga vede l'istantanea all'inizio della barra, mai dopo."""
    frame = _metriche("2024-01-01")
    frame["funding_rate"] = 0.0
    frame[positioning.COLUMNS].to_parquet(positioning.store_path("TESTUSDT", tmp_path))

    orario = pd.date_range("2024-01-01", periods=24, freq="1h", name="Open time")
    letto = positioning.load_positioning("TESTUSDT", orario, store_dir=tmp_path)

    # la barra oraria che apre alle 01:00 e' la 12esima istantanea da 5m, cioe' il valore 12
    assert letto.loc[pd.Timestamp("2024-01-01 01:00"), "open_interest"] == 12.0
    # e non la 23esima, che sarebbe il valore alla chiusura di quella barra
    assert letto.loc[pd.Timestamp("2024-01-01 01:00"), "open_interest"] != 23.0
    # nessun valore letto puo' venire da dopo il proprio timestamp
    atteso = np.arange(0, 288, 12, dtype=float)
    assert np.array_equal(letto["open_interest"].to_numpy(), atteso)


def test_funding_e_una_funzione_a_gradini(store, monkeypatch):
    """Prima della prima applicazione e' NaN; dopo tiene il valore fino alla successiva."""
    tmp_path, giorni = store
    giorni["2024-01-01"] = _metriche("2024-01-01")
    monkeypatch.setattr(pd.Timestamp, "utcnow", staticmethod(lambda: pd.Timestamp("2024-01-02 12:00")))

    frame = positioning.update_symbol("TESTUSDT", tmp_path, workers=2)

    assert np.isnan(frame.loc[pd.Timestamp("2024-01-01 07:55"), "funding_rate"])
    assert frame.loc[pd.Timestamp("2024-01-01 08:00"), "funding_rate"] == 0.01
    assert frame.loc[pd.Timestamp("2024-01-01 15:55"), "funding_rate"] == 0.01
    assert frame.loc[pd.Timestamp("2024-01-01 23:55"), "funding_rate"] == 0.02


def test_aggiornamento_incrementale_riscarica_solo_la_coda(store, monkeypatch):
    """Il giorno gia' in archivio viene riscaricato (poteva essere parziale), quelli prima no."""
    tmp_path, giorni = store
    giorni["2024-01-01"] = _metriche("2024-01-01")
    monkeypatch.setattr(pd.Timestamp, "utcnow", staticmethod(lambda: pd.Timestamp("2024-01-02 12:00")))
    positioning.update_symbol("TESTUSDT", tmp_path, workers=2)

    chiesti = []
    vero = positioning.fetch_metrics_day

    def registra(symbol, day):
        chiesti.append(day)
        return vero(symbol, day)

    monkeypatch.setattr(positioning, "fetch_metrics_day", registra)
    giorni["2024-01-02"] = _metriche("2024-01-02")
    monkeypatch.setattr(pd.Timestamp, "utcnow", staticmethod(lambda: pd.Timestamp("2024-01-03 12:00")))

    frame = positioning.update_symbol("TESTUSDT", tmp_path, workers=2)

    assert chiesti == ["2024-01-01", "2024-01-02"], chiesti
    assert len(frame) == 576
    assert frame.index.is_monotonic_increasing and not frame.index.has_duplicates


def test_store_assente_non_solleva(tmp_path):
    """La condizione in cui gira il servizio pubblico: nessuno store, nessuna eccezione."""
    assert positioning.load_positioning("MANCAUSDT", store_dir=tmp_path).empty

    orario = pd.date_range("2024-01-01", periods=5, freq="1h", name="Open time")
    letto = positioning.load_positioning("MANCAUSDT", orario, store_dir=tmp_path)
    assert list(letto.columns) == positioning.COLUMNS
    assert len(letto) == 5 and letto.isna().all().all()
