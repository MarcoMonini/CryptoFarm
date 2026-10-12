"""The decomposer page brought over from TradingVision: its own self-checks, and the one part that is
new here, reading CryptoFarm's candle store instead of TradingVision's."""

import runpy

import numpy as np
import pandas as pd

from cryptofarm import paths
from cryptofarm.trading import decomposer, project_features, zones


def test_selfchecks():
    decomposer._selfcheck()
    zones._selfcheck()
    project_features._selfcheck()
    runpy.run_module("cryptofarm.trading.candle_features", run_name="__main__")


def test_reads_the_klines_store(tmp_path, monkeypatch):
    when = pd.date_range("2024-01-01", periods=12, freq="5min", name="Open time")
    close = 100 + np.arange(12.0)
    df = pd.DataFrame(
        {"Open": close - 0.5, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1.0}, index=when
    )
    for name in ("ETHUSDT-5m", "BTCUSDT-5m", "ZZZUSDT-5m", "BTCUSDT-positioning"):
        df.to_parquet(tmp_path / f"{name}.parquet")
    monkeypatch.setattr(paths, "MARKET_DATA_DIR", tmp_path)

    assert decomposer.pairs(decomposer.store()) == ["BTC", "ETH", "ZZZ"]
    out = decomposer.candles("BTC", "15m", 0.0)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"] and len(out) == 4
    assert out.close.iloc[0] == close[2] and out.high.iloc[0] == close[2] + 1 and out.volume.iloc[0] == 3.0


def test_zones_figure_outlines_the_zone():
    """One heatmap per pair and period, every band against every N even where a band is empty, and the outline on
    the zone's own cell: band 0 is the lowest row, N 6 the third column."""
    rows = [(p, b, n, 0.1, 0.0, 0.05, 10) for p in zones.PERIODS for b in range(1, 10) for n in zones.HORIZONS]
    table = pd.DataFrame(rows, columns=["period", "band", "n", "mean", "all", "se", "count"])
    table = table.assign(excess=table["mean"] - table["all"], lo=0.0, hi=1.0)
    table = table.assign(t=table.excess / table.se)
    zone = zones.ZONES[0]
    fig = decomposer.zones_figure({"BTC": table, "ETH": table}, zone)
    assert len(fig.data) == 2 * len(zones.PERIODS)
    assert fig.data[0].z.shape == (len(zones.QS) + 1, len(zones.HORIZONS)) and np.isnan(fig.data[0].z[0]).all()
    box = fig.layout.shapes[0]
    assert (box.x0 + box.x1) / 2 == zones.HORIZONS.index(zone[5]) and (box.y0 + box.y1) / 2 == zone[4]


def test_hover_formats_are_short_and_render():
    """Every number format Plotly gets from the page: no `+` flag, which this Plotly leaves unapplied and prints the
    raw float instead (`0.5114346962411359%`), and three decimals at most."""
    import inspect
    import re

    source = inspect.getsource(decomposer)
    specs = re.findall(r"%\{\{?[\w\[\]]+:([^}]+)\}", source) + re.findall(r"hoverformat=\"([^\"]*)\"", source)
    assert specs, "the scan found no format at all"
    bad = [s for s in specs if s.startswith("+") or re.search(r"\.(?:[4-9]|\d{2,})[fge]", s)]
    assert not bad, bad


def test_page_runs_with_every_turn_choice(tmp_path, monkeypatch):
    """The whole page on a small made-up store, once per choice of turns on the candles. The functions have their
    self-checks; this is the level that sees `main` itself — a local named like a module function, as `top` was for
    a moment, breaks a section further down the page, and only running it shows it."""
    from streamlit.testing.v1 import AppTest

    when = pd.date_range("2020-11-01", "2021-02-15", freq="5min", name="Open time")
    rng = np.random.default_rng(0)
    for symbol in ("BTC", "ETH"):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, len(when))))
        df = pd.DataFrame(
            {"Open": close, "High": close * 1.001, "Low": close * 0.999, "Close": close, "Volume": 1.0}, index=when
        )
        df.to_parquet(tmp_path / f"{symbol}USDT-5m.parquet")
    monkeypatch.setattr(paths, "MARKET_DATA_DIR", tmp_path)

    at = AppTest.from_file(decomposer.__file__, default_timeout=120).run()
    for method in decomposer.TURN_METHODS:
        next(r for r in at.radio if r.label == "Turns on the candles").set_value(method).run()
        assert not at.exception, (method, [e.value for e in at.exception])
        assert [s.value for s in at.subheader][-1] == "The zones of the study", method
