"""The indicators this project trades on that `candle_features` does not already hold, as columns for the zones study.

`candle_features` is TradingVision's set, 29 columns. CryptoFarm's strategies and models read more
(`indicators.add_technical_indicator`, `indicators_extra.ExtraCache`, `ml/features.py`, `ml/bar_features.py`):
the stochastic, StochRSI, MFI, the OBV's slope, the Bollinger, Keltner and Donchian channels and the squeeze
between the first two, the ATR bands around KAMA, Ichimoku, the EMA crossovers. These are those, in the same
contract as `candle_features.features`: one row per bar, causal, scale free so BTC and ETH compare, every
window derived from one `n` so the study can scan it, finite or NaN and never an infinity.

Left out because `candle_features` already has them under another name: RSI, TSI, ADX, ATR in % of the
close, the distance from PSAR (at `ta`'s 0.2 cap, the project's 0.4 aside), the volume against its median,
the bar's range. Left out because the study cannot read them: the futures positioning (no history before
2020-09, so no discovery period) and the cross-sectional rank (a property of the universe, not of a pair).

Secondary parameters are the project's own defaults, fixed: Bollinger at 2 deviations, Keltner at 2 ATR,
StochRSI and the stochastic smoothed over 3, Ichimoku's tenkan at a third of the kijun (9/26) and its span B
at twice it (52/26), the fast EMA of a crossover at a quarter of the slow one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import KAMAIndicator, StochasticOscillator, StochRSIIndicator
from ta.trend import IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator

LABELS = {
    "stochastic_k": "Stochastic %K",
    "stochastic_k_minus_d": "Stochastic %K − %D",
    "stoch_rsi": "StochRSI",
    "money_flow_index": "Money Flow Index",
    "obv_slope": "OBV Slope",
    "distance_from_ema_atr": "Distance from EMA (ATR)",
    "ema_spread_pct": "EMA Spread, fast − slow",
    "distance_from_kama_atr": "Distance from KAMA (ATR bands)",
    "donchian_position": "Donchian Position",
    "bollinger_position": "Bollinger %B",
    "bollinger_width": "Bollinger Width",
    "keltner_position": "Keltner Position",
    "squeeze_ratio": "Squeeze, Bollinger / Keltner",
    "distance_from_cloud_pct": "Distance from Ichimoku Cloud",
    "tenkan_kijun_spread_pct": "Tenkan − Kijun",
}
COLUMNS = list(LABELS)


def features(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """`COLUMNS` for one OHLCV frame with lowercase columns, every window from `n`. All NaN on a frame too short
    to fill `2n + 1` bars, as `candle_features.features`."""
    if len(df) < 2 * n + 1:
        return pd.DataFrame(np.nan, index=df.index, columns=COLUMNS)
    h, low, c, v = df.high, df.low, df.close, df.volume
    fast = max(n // 4, 2)
    atr = AverageTrueRange(h, low, c, window=n).average_true_range()
    atr = atr.where(atr > 0)  # `ta` writes 0 until the window fills, a plausible and wrong value
    stoch = StochasticOscillator(h, low, c, window=n, smooth_window=3)
    obv = OnBalanceVolumeIndicator(c, v).on_balance_volume()
    ema = c.ewm(span=n, adjust=False).mean()
    ema_fast = c.ewm(span=fast, adjust=False).mean()
    # The channel of the n bars before this one: what is seen at the close, and the only one a close can break.
    donchian_high, donchian_low = h.rolling(n).max().shift(), low.rolling(n).min().shift()
    bb = BollingerBands(c, window=n, window_dev=2)
    bb_up, bb_low, bb_mid = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    kc = KeltnerChannel(h, low, c, window=n, window_atr=n, multiplier=2, original_version=False)
    kc_up, kc_low = kc.keltner_channel_hband(), kc.keltner_channel_lband()
    tenkan_n = max(n // 3, 2)
    # `visual=True` shifts both spans forward by the kijun: the cloud over today's bar was computed n bars ago.
    cloud = IchimokuIndicator(h, low, window1=tenkan_n, window2=n, window3=2 * n, visual=True)
    cloud_mid = (cloud.ichimoku_a() + cloud.ichimoku_b()) / 2
    tenkan = (h.rolling(tenkan_n).max() + low.rolling(tenkan_n).min()) / 2
    kijun = (h.rolling(n).max() + low.rolling(n).min()) / 2
    f = {
        "stochastic_k": stoch.stoch() / 50 - 1,
        "stochastic_k_minus_d": (stoch.stoch() - stoch.stoch_signal()) / 100,
        "stoch_rsi": StochRSIIndicator(c, window=n, smooth1=3, smooth2=3).stochrsi_k(),
        "money_flow_index": MFIIndicator(h, low, c, v, window=n).money_flow_index() / 50 - 1,
        "obv_slope": obv.diff(n) / v.rolling(n).sum().replace(0, np.nan),
        "distance_from_ema_atr": (c - ema) / atr,
        "ema_spread_pct": np.log(ema_fast / ema),
        "distance_from_kama_atr": (c - KAMAIndicator(c, window=n).kama()) / atr,
        "donchian_position": (c - donchian_low) / (donchian_high - donchian_low),
        "bollinger_position": (c - bb_low) / (bb_up - bb_low),
        "bollinger_width": (bb_up - bb_low) / bb_mid,
        "keltner_position": (c - kc_low) / (kc_up - kc_low),
        "squeeze_ratio": (bb_up - bb_low) / (kc_up - kc_low),
        "distance_from_cloud_pct": np.log(c / cloud_mid),
        "tenkan_kijun_spread_pct": np.log(tenkan / kijun),
    }
    return pd.DataFrame(f)[COLUMNS].replace([np.inf, -np.inf], np.nan)


def _selfcheck() -> None:
    """Shape, finiteness past warm-up, the bounded columns' ranges, and causality."""
    rng = np.random.default_rng(0)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, 800))), index=pd.date_range("2024", periods=800, freq="5min")
    )
    df = pd.DataFrame(
        {
            "open": close.shift().fillna(close.iloc[0]),
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": rng.lognormal(0, 1, 800),
        }
    )
    n = 24
    out = features(df, n)
    assert list(out.columns) == COLUMNS and out.index.equals(df.index)
    tail = out.iloc[4 * n :]
    assert np.isfinite(tail.to_numpy()).all(), tail.columns[~np.isfinite(tail.to_numpy()).all(axis=0)].tolist()
    for column in ("stochastic_k", "money_flow_index"):
        assert tail[column].between(-1, 1).all(), column
    assert tail.stoch_rsi.between(0, 1).all() and tail.stochastic_k_minus_d.between(-1, 1).all()
    assert tail.bollinger_width.gt(0).all() and tail.squeeze_ratio.gt(0).all()
    # The Donchian channel is the bars before: a close at a new high reads above 1, which `close_position_in_window`
    # (the window including this bar) never does.
    assert tail.donchian_position.max() > 1
    # Causality: a column must not move when a later bar changes.
    cut = 500
    assert np.allclose(features(df.iloc[:cut], n).iloc[-1], out.iloc[cut - 1], equal_nan=True)
    assert features(df.iloc[: 2 * n], n).isna().all().all()


if __name__ == "__main__":
    _selfcheck()
    print(f"ok — {len(COLUMNS)} columns")
