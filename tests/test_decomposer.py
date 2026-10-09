"""The decomposer page brought over from TradingVision: its own self-checks, and the one part that is
new here, reading CryptoFarm's candle store instead of TradingVision's."""

import runpy

import numpy as np
import pandas as pd

from cryptofarm import paths
from cryptofarm.trading import decomposer


def test_selfchecks():
    decomposer._selfcheck()
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
