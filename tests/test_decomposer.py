"""The decomposer page brought over from TradingVision: its own self-checks, and the one part that is
new here, reading CryptoFarm's candle store instead of TradingVision's."""

import runpy

import numpy as np
import pandas as pd

from cryptofarm import paths
from cryptofarm.trading import decomposer, zones


def test_selfchecks():
    decomposer._selfcheck()
    zones._selfcheck()
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
