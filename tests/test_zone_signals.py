"""The zone signals: the module's own check, and the strategy on the simulator's chart."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import config, panels, zone_signals
from cryptofarm.trading.simulator import trading_analysis


def test_selfcheck():
    zone_signals._selfcheck()


@pytest.fixture
def candele() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    close = 30_000 * np.exp(np.cumsum(rng.normal(0, 0.004, 600)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.002,
            "Low": np.minimum(open_, close) * 0.998,
            "Close": close,
            "Volume": rng.uniform(1, 2, 600),
        },
        index=pd.date_range("2024-01-01", periods=600, freq="15min", name="Open time"),
    )


def disegna(candele, intervallo, **valori):
    figura, operazioni, _ = trading_analysis(
        asset="TESTUSDC",
        interval=intervallo,
        wallet=100.0,
        valori=valori,
        strategia=config.ZONE_STRATEGY,
        market_data=candele,
    )
    return figura, operazioni


def test_at_15m_every_zone_draws_its_indicator_and_trades_nothing(candele):
    figura, operazioni = disegna(candele, "15m")
    assert operazioni.empty
    assi = {traccia.yaxis or "y" for traccia in figura.data}
    attesi = panels.pannelli_di(config.ZONE_STRATEGY, {**panels.valori_predefiniti(), "INTERVALLO": "15m"})
    assert len(attesi) == 5 and len(assi) == 1 + len(attesi)
    nomi = {traccia.name for traccia in figura.data}
    assert {"Window high", "Fast KAMA", "Slow KAMA", "VWAP", "EMA (slope)"} <= nomi


def test_a_zone_left_out_takes_its_indicator_with_it(candele):
    figura, _ = disegna(candele, "15m", ZONE_SIGNALS=["vwap"])
    nomi = {traccia.name for traccia in figura.data}
    assert "VWAP" in nomi and "Fast KAMA" not in nomi
    assert len({traccia.yaxis or "y" for traccia in figura.data}) == 1, "a price-line zone needs no panel"


def test_without_the_pair_in_the_store_nothing_is_marked(candele):
    figura, _ = disegna(candele, "15m")
    assert not [t for t in figura.data if t.legendgroup == "zona_segnali"]
    assert "VWAP · edge" not in {t.name for t in figura.data}


def test_with_an_edge_each_zone_marks_under_the_low_with_its_own_symbol(candele, monkeypatch):
    # An edge at zero: below VWAP, below KAMA, under the window high... about half the candles, so marks exist.
    monkeypatch.setattr(zone_signals, "soglia", lambda base, chiave, w, coda: 0.0)
    figura, _ = disegna(candele, "15m")
    segni = [t for t in figura.data if t.legendgroup == "zona_segnali"]
    assert len(segni) == 9 and len({t.marker.symbol for t in segni}) == 9
    vwap = next(t for t in segni if t.name == zone_signals.SEGNALI["vwap"].nome)
    y = pd.Series(np.asarray(vwap.y, dtype=float), index=candele.index)
    assert y.notna().any() and (y.dropna() < candele.Low[y.notna()]).all()
    # The edge is drawn too, a price line for a distance and a level in the panel for the others.
    assert {"VWAP · edge", "Bar range · edge"} <= {t.name for t in figura.data}


def test_at_5m_only_the_bollinger_width_and_elsewhere_nothing(candele, monkeypatch):
    monkeypatch.setattr(zone_signals, "soglia", lambda base, chiave, w, coda: 0.0)
    figura, _ = disegna(candele, "5m")
    assert [t.name for t in figura.data if t.legendgroup == "zona_segnali"] == [
        zone_signals.SEGNALI["bb_width_5m"].nome
    ]
    figura, _ = disegna(candele, "1h")
    assert [t.name for t in figura.data] == ["TESTUSDC"], "at 1h only the candles"
