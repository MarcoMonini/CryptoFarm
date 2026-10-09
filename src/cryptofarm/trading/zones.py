"""The zones of `reports/zones_study.html`, as the decomposer re-reads them on this repository's store.

Ported from TradingVision's `zones` (PR #22), the part that measures one zone, without the search over
every column that chose them: that search was run there once, with the choice on 2017-2020, and running
it again here would be choosing again. What is kept is the reading.

**A zone** is a timeframe, a column of `candle_features` at a window, a band of it and an N (`ZONES`, the
twenty rows of the report's two tables: four families that hold, and the momentum that does not). The
bands are cut at the quantiles `QS` of the column on **discovery**, the store's first bar to 2020-12-31,
and those edges are applied unchanged to **confirm**, 2021-01 to 2025-06, and **recent**, after. For each
band and each N of `HORIZONS`: the mean change in percent from the close to the close N candles later,
less the mean of every candle of the same period at the same N (`excess`), and its standard error over
non-overlapping blocks of N candles on the clock, since neighbouring candles share N − 1 candles of their
future. A candle whose future crosses into the next period counts in neither.

An excess per candle is not a trade: the candles of one crash all sit in the bottom band and share their
future. `trades` takes one trade an episode: the indicator is read at the close of the first candle in the band,
`DELAY` candles go by, the trade opens at the next candle's open and is held N candles, nothing new until it
closes; gross, against `ROUND_TRIP`, the 0.20% the report prices against. `DELAY` is 1 by default, a whole candle
to see the signal and act on it, the conservative reading; at 0 the trade opens at the next candle's open, the
first price anyone could act on, and costs 0.1 to 0.2% a trade less on family 1 at 15m on confirm: the bounce
starts in the first candle. TradingVision entered at the signal candle's own close. Against the next open, on
confirm and recent, that differs by 0.006% a trade at most; on discovery by up to 0.3% (ETH 15m), because in
Binance's first months the next candle's first print could sit far from the close — 48.5% on ETH on 2017-08-22 —
on books too thin to fill at it. The choice of zones is not touched: `cells` reads close to close.

**`PROJECT_ZONES`** are the twenty that `scripts/zones_scan.py` found holding when the same search ran over this
project's own indicators (`project_features`), sorted into the same families plus one: momentum by where the
price sits, which holds where RSI/TSI momentum did not, and under the fee. Every one of them is 0.75 to 0.998
Spearman with a column of the first study (`reports/zones_project_trades.csv`): re-expressions, not new findings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.trading import candle_features, project_features

QS = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)  # the band edges, quantiles of the discovery period
HORIZONS = (1, 3, 6, 12, 24, 48, 96, 192)  # N, in candles
PERIODS = ("discovery", "confirm", "recent")
CONFIRM = (pd.Timestamp("2021-01"), pd.Timestamp("2025-06"))  # the store is tz-naive UTC, as these
ROUND_TRIP = 0.20  # percent: 0.10% a side at OKX, the report's fee
DELAY = 1  # candles between the signal's close and the entry's open: one to see it and act, the default
SYMBOLS = ("BTC", "ETH")  # the report's: a zone passed on both, with one sign
# The columns that read no window: a zone on one of them carries window 0, and any window computes it.
FIXED = (
    "log_return",
    "candle_body_pct",
    "bar_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position_in_bar",
    "distance_from_psar_pct",
)
FAMILIES = {
    "1": "Price far below its average → bounce",
    "2": "Huge bars → rise over the next day",
    "3": "Close at an extreme → the next candle reverts",
    "4": "Slow bounce, solid but small",
    "5": "Price far above its average → keeps rising, under the fee",
    "m": "Momentum: does not hold",
}
# (family, timeframe, column, window, band, N): band b spans the quantiles QS[b-1] to QS[b] of discovery.
ZONES = (
    ("1", "15m", "distance_from_vwap_pct", 12, 0, 6),
    ("1", "15m", "ema_slope", 12, 0, 6),
    ("1", "15m", "cum_log_return_window", 6, 0, 6),
    ("1", "15m", "distance_from_ema_pct", 12, 0, 6),
    ("1", "15m", "distance_from_window_high_pct", 6, 0, 24),
    ("1", "15m", "distance_from_kama_pct", 6, 0, 24),
    ("1", "5m", "distance_from_vwap_pct", 24, 0, 24),
    ("1", "5m", "distance_from_kama_pct", 12, 0, 24),
    ("1", "5m", "distance_from_psar_pct", 0, 0, 48),
    ("2", "15m", "bar_range_pct", 0, 9, 96),
    ("2", "15m", "upper_wick_pct", 0, 9, 96),
    ("2", "5m", "bar_range_pct", 0, 9, 96),
    ("3", "5m", "close_position_in_window", 6, 1, 1),
    ("3", "15m", "close_position_in_bar", 0, 1, 1),
    ("4", "15m", "close_position_in_window", 48, 0, 6),
    ("4", "15m", "distance_from_kama_pct", 6, 1, 12),
    ("m", "15m", "rsi_centered", 24, 8, 48),
    ("m", "15m", "tsi_momentum", 24, 8, 48),
    ("m", "5m", "rsi_centered", 48, 8, 192),
    ("m", "5m", "tsi_momentum", 96, 8, 192),
)
# The same, for `project_features`' columns: what `scripts/zones_scan.py` found holding on 2021-2025.
PROJECT_ZONES = (
    ("1", "15m", "ema_spread_pct", 12, 0, 6),
    ("1", "15m", "distance_from_cloud_pct", 6, 0, 6),
    ("1", "15m", "tenkan_kijun_spread_pct", 6, 0, 24),
    ("1", "5m", "ema_spread_pct", 6, 0, 24),
    ("1", "5m", "distance_from_cloud_pct", 6, 0, 24),
    ("1", "5m", "tenkan_kijun_spread_pct", 6, 0, 96),
    ("2", "15m", "bollinger_width", 6, 9, 96),
    ("2", "5m", "bollinger_width", 6, 9, 96),
    ("3", "5m", "stoch_rsi", 6, 9, 1),
    ("3", "5m", "stoch_rsi", 6, 2, 1),
    ("3", "5m", "distance_from_ema_atr", 6, 0, 1),
    ("3", "15m", "donchian_position", 6, 9, 3),
    ("4", "15m", "donchian_position", 96, 0, 6),
    ("4", "15m", "donchian_position", 24, 1, 6),
    ("4", "15m", "bollinger_position", 24, 1, 6),
    ("4", "5m", "donchian_position", 96, 0, 24),
    ("4", "5m", "distance_from_ema_atr", 24, 1, 24),
    ("5", "15m", "bollinger_position", 96, 9, 48),
    ("5", "5m", "distance_from_ema_atr", 48, 9, 192),
    ("5", "5m", "obv_slope", 96, 8, 192),
)
# Both studies in one list, as the page offers them: by family in `FAMILIES`' order, 15m before 5m, and within that
# the report's zones before this project's. Which study a zone comes from is a matter of provenance, not of reading.
ALL_ZONES = tuple(sorted(ZONES + PROJECT_ZONES, key=lambda zone: (list(FAMILIES).index(zone[0]), zone[1] != "15m")))


def label(zone: tuple) -> str:
    """One line per zone, as the report's tables name it."""
    family, timeframe, column, window, band, n = zone
    name = (candle_features.LABELS | project_features.LABELS)[column].removesuffix(" (%)").removesuffix(" (N)") + (
        f" ({window})" if window else ""
    )
    return f"{family} · {timeframe} · {name} · {quantiles(band)} · N {n}"


def period_of(when: pd.DatetimeIndex) -> np.ndarray:
    """0 discovery, 1 confirm, 2 recent, by the bar's open."""
    return np.searchsorted(np.array(CONFIRM, dtype="datetime64[ns]"), when.to_numpy(), side="right")


def band_edges(x: pd.Series, period: np.ndarray) -> np.ndarray:
    """The inner edges of the bands: `QS` of `x` on discovery. A tie repeats an edge and leaves the band between the
    copies empty rather than merging it, so band b is the quantiles `QS[b-1]` to `QS[b]` on every pair alike."""
    return np.nanquantile(x.to_numpy()[period == 0], QS)


def quantiles(band: int) -> str:
    """Band b's name, the quantiles of discovery it spans."""
    q = (0, *QS, 1)
    return f"{q[band]:.0%}–{q[band + 1]:.0%}"


def bands(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Each candle's band, 0 to len(edges), closed on the left; −1 where `x` is NaN."""
    v = x.to_numpy()
    return np.where(np.isnan(v), -1, np.searchsorted(edges, v, side="right"))


def cells(code: np.ndarray, close: pd.Series, period: np.ndarray, step: pd.Timedelta) -> pd.DataFrame:
    """By period, band and N: the band's mean change to N candles later (`mean`), every candle's (`all`), the
    excess, its blocked standard error and t, and the count. `code` is `bands`' output, `period` `period_of`'s."""
    c = close.to_numpy()
    ns = close.index.asi8
    rows = []
    for n in HORIZONS:
        future = np.full(len(c), np.nan)
        future[:-n] = (c[n:] / c[:-n] - 1) * 100
        same = np.zeros(len(c), bool)
        same[:-n] = period[n:] == period[:-n]
        for p in range(len(PERIODS)):
            have = np.isfinite(future) & same & (period == p)
            if not have.any():
                continue
            every = future[have].mean()
            m = have & (code >= 0)
            k, f = code[m], future[m]
            block = ns[m] // (n * step.value)
            key, inv = np.unique(k * (block.max() + 1) + block, return_inverse=True)
            bmean = np.bincount(inv, f) / np.bincount(inv)
            bband = key // (block.max() + 1)
            nb = np.bincount(bband)
            s1, s2 = np.bincount(bband, bmean), np.bincount(bband, bmean**2)
            with np.errstate(invalid="ignore", divide="ignore"):
                sd = np.sqrt((s2 - s1**2 / nb) / (nb - 1))
                mean = np.bincount(k, f) / np.bincount(k)
            for b in np.flatnonzero(nb):
                rows.append((PERIODS[p], b, n, mean[b], every, sd[b] / np.sqrt(nb[b]), int((k == b).sum())))
    out = pd.DataFrame(rows, columns=["period", "band", "n", "mean", "all", "se", "count"])
    out["excess"] = out["mean"] - out["all"]
    return out.assign(t=out.excess / out.se)


def trades(hit: np.ndarray, open_: pd.Series, close: pd.Series, n: int, sign: float, delay: int = DELAY) -> np.ndarray:
    """The change in percent of one trade an episode. The signal is the first candle of `hit` (a boolean mask), known
    at its close; `delay` candles later the trade opens, at the open of the candle after them (0: the next candle's
    open), is held `n` candles in the direction `sign` and closes at the close of the n-th, and no new entry comes
    until it is closed. A signal whose exit runs past the store's end opens none."""
    o, c = open_.to_numpy(), close.to_numpy()
    out, free = [], -1
    for i in np.flatnonzero(hit):
        entry, exit_ = i + 1 + delay, i + delay + n
        if i > free and exit_ < len(c):
            out.append(sign * (c[exit_] / o[entry] - 1) * 100)
            free = exit_
    return np.array(out)


def _selfcheck() -> None:
    """The zones list against `candle_features`, the bands and the blocked error on hand-made series."""
    for zone in ZONES:
        _, timeframe, column, window, band, n = zone
        assert column in candle_features.COLUMNS and (window == 0) == (column in FIXED), zone
        assert 0 <= band <= len(QS) and n in HORIZONS and timeframe in ("5m", "15m"), zone
    assert len(set(ALL_ZONES)) == len(ZONES) + len(PROJECT_ZONES) and ALL_ZONES[0][0] == "1" and ALL_ZONES[-1][0] == "m"
    for zone in PROJECT_ZONES:
        family, timeframe, column, window, band, n = zone
        assert column in project_features.COLUMNS and window > 0 and family in FAMILIES, zone
        assert 0 <= band <= len(QS) and n in HORIZONS and timeframe in ("5m", "15m"), zone
    assert label(ZONES[0]) == "1 · 15m · Distance from VWAP (12) · 0%–1% · N 6"

    when = pd.date_range("2020-12-01", periods=5000, freq="15min")  # across the cut
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(when)))), index=when)
    df = pd.DataFrame({"open": close.shift().bfill(), "close": close, "volume": rng.uniform(1, 2, len(when))})
    df["high"], df["low"] = df[["open", "close"]].max(axis=1) * 1.001, df[["open", "close"]].min(axis=1) * 0.999
    a, b = candle_features.features(df, 6)[list(FIXED)], candle_features.features(df, 48)[list(FIXED)]
    assert a.iloc[100:].equals(b.iloc[100:]), "a FIXED column reads the window"

    period = period_of(when)
    assert period[0] == 0 and period[-1] == 1 and (np.diff(period) >= 0).all()
    x = pd.Series(np.arange(len(when), dtype=float), index=when).where(lambda s: s % 7 > 0)
    edges = band_edges(x, period)
    code = bands(x, edges)
    assert (code[x.isna().to_numpy()] == -1).all() and code.max() == len(edges)
    # A tie at the bottom: 30% zeros fill the band their quantile reaches and leave the ones below empty, so band 4
    # is 25-50% here as on a pair with no tie.
    ties = pd.Series(np.r_[np.zeros(300), np.arange(1.0, 701)], index=when[:1000])
    c = bands(ties, band_edges(ties, np.zeros(1000, int)))
    assert (c[:300] == 4).all() and set(c[300:]) == set(range(4, len(QS) + 1)) and quantiles(4) == "25%–50%"
    # On a close rising 1% a candle every band's mean is the same, so every excess is zero, and the count of a
    # band is the candles that have a future inside their own period.
    up = pd.Series(100 * 1.01 ** np.arange(len(when)), index=when)
    t = cells(code, up, period, pd.Timedelta("15min"))
    assert np.allclose(t.excess, 0, atol=1e-9) and np.allclose(t["mean"][t.n == 3], (1.01**3 - 1) * 100)
    split = int((period == 0).sum())
    assert t[(t.n == 3) & (t.period == "discovery")]["count"].sum() == int(x.iloc[: split - 3].notna().sum())
    # The blocked error on independent noise is the naive one, std / sqrt(n), within sampling error.
    noise = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200_000))),
        index=pd.date_range("2018-01-01", periods=200_000, freq="5min"),
    )
    one = cells(np.zeros(len(noise), int), noise, period_of(noise.index), pd.Timedelta("5min"))
    r = one[(one.n == 1) & (one.period == "discovery")].iloc[0]
    naive = noise.pct_change().shift(-1)[period_of(noise.index) == 0].std() * 100 / np.sqrt(r["count"])
    assert abs(r.se / naive - 1) < 0.05
    # Trades: one an episode, the next only after the last is closed, none past the end; a short gains on a fall.
    # Each opens at the next candle's open, never at the signal's own close: with every open 1% under the close
    # before it, the trade pays that 1% back on top of the n candles' rise.
    hit = np.zeros(len(up), bool)
    hit[[10, 11, 12, 20, len(up) - 2]] = True
    g = trades(hit, up.shift().fillna(up.iloc[0]), up, 5, 1.0, delay=0)
    assert len(g) == 2 and np.allclose(g, (1.01**5 - 1) * 100)
    assert np.allclose(trades(hit, up.shift().fillna(up.iloc[0]), up, 5, -1.0, delay=0), -g)
    g = trades(hit, up.shift().fillna(up.iloc[0]) * 0.99, up, 5, 1.0, delay=0)
    assert np.allclose(g, (1.01**5 / 0.99 - 1) * 100)
    # A candle late by default: the entry and the exit move one candle on, the episode is held as long. On a close
    # that rises then falls the delay is what decides the sign: the signal at 3 buys the top at delay 1.
    bump = pd.Series([100.0, 100, 100, 100, 110, 99, 99, 99, 99, 99])
    one = np.zeros(len(bump), bool)
    one[3] = True
    assert np.allclose(trades(one, bump.shift().bfill(), bump, 1, 1.0, delay=0), 10.0)
    assert np.allclose(trades(one, bump.shift().bfill(), bump, 1, 1.0), (99 / 110 - 1) * 100)
    assert len(trades(one, bump.shift().bfill(), bump, 6, 1.0)) == 0, "the exit at 10 is past the end"
