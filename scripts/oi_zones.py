"""The zones study of `reports/zones_study.html`, run on the open interest of the Binance perpetual.

Does the open interest, in some band, say something about the price over the next H candles that holds out of the
years it was found on? Same search as `scripts/zones_scan.py`, with the rules fixed before looking, over five columns
built from the open interest (`oi_features`), every window of `WINDOWS`, ten bands at the discovery quantiles
`zones.QS`, every H of `zones.HORIZONS`, BTC and ETH at 5m and 15m.

**The periods are not the first study's.** The store's open interest starts in 2021 (BTC in January, ETH in
December), so 2017-2020 does not exist here. Discovery is 2021-12 → 2023-12 (`PERIODS`), the bear market and its
bottom; confirm 2024-01 → 2025-06; recent after. Two years of discovery against the first study's three and a half:
fewer crashes to choose on, which is the reason for the cross-asset check below.

**The alignment.** `positioning.load_positioning` gives each candle the snapshot taken at its open, never a later
one: the column a candle is banded on is older than its close, so nothing here can read the future.

**After the search**, for each zone that holds: one trade an episode, a candle late (`zones.trades`), gross and net at
three round-trip fees with 0.01% funding every 8 hours held; the Spearman correlation of the column with the return
over the same window, since an open interest that only moves with the price is the price; and how many of the store's
other pairs, each banded on its own discovery, give the same sign on confirm.

**What came out (2026-10-10, store to 2026-08-27): nothing, and the controls say the nothing is real.**

- *Choose then confirm*: 9,405 cells, **0** pass discovery at 4 standard errors on both pairs; at 3, one does and fails
  confirm. Two years of discovery are short, so the same search was run on two price columns over the same years
  (`CONTROLS`, the first studies' return over the window and distance from KAMA): they reach 2.1 standard errors at
  best and pass nothing either. On this split the method cannot decide, for the price or the open interest.
- *Stability by fold* (`stability`), which uses the whole open-interest era: 2021-12 → 2025-06 in seven half-years,
  a cell kept when it beats every candle of the same half-year in **all seven, on BTC and on ETH**. Open interest
  **0 of 4,248** cells; price controls **11 of 1,856**, eight of them the distance from KAMA — family 1 of the
  first study, found again. The method finds the price's effect and none in the open interest.
- *Does it sharpen what works?* The best zones' episodes (15m, a candle late, 2021-12 → 2025-06, BTC and ETH)
  split by whether the open interest fell or rose over the zone's window: distance from KAMA (6) +0.22% ± 0.27
  against +0.53% ± 0.30; from the 6-candle high +0.64% ± 0.34 against +0.35% ± 0.52; from VWAP (12) +0.39% and
  +0.39%. No direction, and every gap inside its errors.

The four textbook readings of open interest against price, the leverage it implies against traded value, its level
and its change: none of them says anything about the next 5 minutes to 2 days on BTC or ETH that holds across
half-years, alone or on top of the price's own zones.

    .venv312/bin/python -m scripts.oi_zones          # ~15 minutes; tables in reports/oi_zones*.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptofarm.data import klines
from cryptofarm.data.positioning import load_positioning
from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import zones
from scripts.zones_scan import T_HOLD, T_PICK, load

WINDOWS = (6, 12, 24, 48, 96, 288)  # the indicators' grid, plus 288: the open interest moves slowly
PERIODS = (pd.Timestamp("2021-12-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-06-01"))
FEES = (0.04, 0.10, 0.20)  # round trip, percent of notional
FUNDING_8H = 0.01  # percent of notional every 8 hours held, Binance's base rate
REPORTS = PROJECT_ROOT / "reports"
FOLDS = 7  # half-years from PERIODS[0] to PERIODS[2], for `stability`
MIN_N = 30  # candles a half-year must hold for its excess to be read
# The first studies' price columns, read the same way as a control: what the method finds where an effect is known.
CONTROLS = {"control_price_return": "cum_log_return_window", "control_distance_kama": "distance_from_kama_pct"}
LABELS = {
    "oi_change": "OI change",
    "oi_zscore": "OI z-score",
    "oi_change_price_down": "OI change while price falls",
    "oi_change_price_up": "OI change while price rises",
    "oi_vs_volume": "OI vs traded value",
}


def oi_features(df: pd.DataFrame, oi: pd.DataFrame, w: int) -> pd.DataFrame:
    """The five columns at window `w`, on candles `df` (lowercase OHLCV) and the open interest aligned to them."""
    q = oi["open_interest"].replace(0, np.nan)
    change = np.log(q / q.shift(w))
    price = np.log(df.close / df.close.shift(w))
    out = pd.DataFrame(
        {
            "oi_change": change,
            "oi_zscore": (q - q.rolling(w).mean()) / q.rolling(w).std(),
            # The four textbook readings, two columns: a band of the first is "while the price fell, the open
            # interest fell this much" -- the bottom tail is longs liquidated -- and of the second the same in a rise.
            "oi_change_price_down": change.where(price < 0),
            "oi_change_price_up": change.where(price > 0),
            # Spot traded value as the activity the open interest is set against: leverage relative to trading.
            "oi_vs_volume": np.log(oi["open_interest_value"] / (df.volume * df.close).rolling(w).sum()),
        },
        index=df.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def period_of(index: pd.DatetimeIndex) -> np.ndarray:
    """0 discovery, 1 confirm, 2 recent; -1 before the open interest starts, which no band reaches."""
    p = np.searchsorted(np.array(PERIODS[1:], dtype="datetime64[ns]"), index.to_numpy(), side="right")
    return np.where(index.to_numpy() < np.datetime64(PERIODS[0]), -1, p)


def frames(symbol: str, timeframe: str) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    df = load(symbol, timeframe)
    df = df[df.index >= PERIODS[0]]
    return df, load_positioning(f"{symbol}USDT", df.index), period_of(df.index)


def scan(symbol: str, timeframe: str) -> pd.DataFrame:
    df, oi, period = frames(symbol, timeframe)
    step = pd.Timedelta(timeframe.replace("m", "min"))
    out = []
    for w in WINDOWS:
        f = oi_features(df, oi, w)
        for column in LABELS:
            x = f[column]
            edges = np.nanquantile(x.to_numpy()[period == 0], zones.QS)
            t = zones.cells(zones.bands(x, edges), df.close, period, step)
            out.append(t.assign(column=column, window=w))
        print(f"  {symbol} {timeframe} window {w}", flush=True)
    return pd.concat(out).assign(symbol=symbol, timeframe=timeframe)


def pick(table: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """`zones_scan.pick`'s criterion: beyond `T_PICK` on discovery on every pair with one sign, the best per
    (timeframe, column, band), held at `T_HOLD` on confirm."""
    key = ["timeframe", "column", "window", "band", "n"]
    wide = table.pivot_table(index=key, columns=["period", "symbol"], values=["excess", "t"])
    t = wide["t"]["discovery"][symbols]
    sign = np.sign(t).nunique(axis=1).eq(1) & np.sign(t).ne(0).all(axis=1)
    weakest = t.abs().min(axis=1).where(sign)
    passed = weakest[weakest >= T_PICK].rename("pick").reset_index()
    best = passed.loc[passed.groupby(["timeframe", "column", "band"]).pick.idxmax()]
    out = wide.loc[best.set_index(key).index]
    flat = pd.DataFrame(index=out.index)
    flat.insert(0, "quantiles", [zones.quantiles(b) for b in flat.index.get_level_values("band")])
    for p in zones.PERIODS:
        for s in symbols:
            flat[f"{p} {s} %"] = out[("excess", p, s)] if ("excess", p, s) in out else np.nan
            flat[f"{p} {s} t"] = out[("t", p, s)] if ("t", p, s) in out else np.nan
    d = np.sign(flat[f"discovery {symbols[0]} t"])
    flat["dir"] = np.where(d > 0, "long", "short")
    flat["holds"] = (flat[[f"confirm {s} t" for s in symbols]].mul(d, axis=0) >= T_HOLD).all(axis=1)
    return flat.join(best.set_index(key).pick).sort_values("pick", ascending=False)


def detail(z: pd.DataFrame, symbols: list[str], others: list[str]) -> pd.DataFrame:
    """For every zone that holds: trades per episode on confirm and recent, pooled over `symbols`, gross and net at
    `FEES`; the column's Spearman with the return over its window; the other pairs agreeing in sign on confirm."""
    cache: dict = {}

    def get(s, tf, w):
        if (s, tf) not in cache:
            cache[(s, tf)] = frames(s, tf)
        df, oi, period = cache[(s, tf)]
        if (s, tf, w) not in cache:
            cache[(s, tf, w)] = oi_features(df, oi, w)
        return df, period, cache[(s, tf, w)]

    rows = []
    for (tf, column, w, band, n), r in z[z.holds].iterrows():
        sign = 1.0 if r.dir == "long" else -1.0
        hours = n * pd.Timedelta(tf.replace("m", "min")) / pd.Timedelta(hours=1)
        row = {
            "timeframe": tf,
            "column": column,
            "window": w,
            "quantiles": zones.quantiles(band),
            "n": n,
            "dir": r.dir,
            "hours": hours,
        }
        for p, name in ((1, "confirm"), (2, "recent")):
            g = []
            for s in symbols:
                df, period, f = get(s, tf, w)
                x = f[column]
                code = zones.bands(x, np.nanquantile(x.to_numpy()[period == 0], zones.QS))
                g.append(zones.trades((code == band) & (period == p), df.open, df.close, n, sign))
            g = np.concatenate(g)
            row[f"{name} trades"] = len(g)
            row[f"{name} gross"] = g.mean() if len(g) else np.nan
            row[f"{name} se"] = g.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan
            row[f"{name} win"] = (g > 0).mean() if len(g) else np.nan
        cost = FUNDING_8H * hours / 8
        for fee in FEES:
            row[f"net {fee}"] = row["confirm gross"] - fee - cost
            row[f"t {fee}"] = row[f"net {fee}"] / row["confirm se"]
        df, period, f = get(symbols[0], tf, w)
        price = np.log(df.close / df.close.shift(w))
        row["rho_price"] = f[column].corr(price, method="spearman")
        agree = 0
        for s in others:
            df, period, f = get(s, tf, w)
            x = f[column]
            if (period == 0).sum() < 1000 or x[period == 0].notna().sum() < 1000:
                continue
            code = zones.bands(x, np.nanquantile(x.to_numpy()[period == 0], zones.QS))
            c = zones.cells(code, df.close, period, pd.Timedelta(tf.replace("m", "min")))
            c = c[(c.period == "confirm") & (c.band == band) & (c.n == n)]
            agree += bool(len(c) and np.sign(c.excess.iloc[0]) == sign)
            row["others"] = row.get("others", 0) + 1
        row["others agree"] = agree
        rows.append(row)
    return pd.DataFrame(rows)


def stability(symbols: list[str], timeframes: list[str]) -> pd.DataFrame:
    """Every cell of the open interest's columns and of `CONTROLS`, read in `FOLDS` half-years: the mean return a
    candle late and held H, less every candle's of the same half-year. A cell is stable when that excess has one
    sign in every half-year on every pair; the half-years must hold `MIN_N` candles each."""
    from cryptofarm.trading import candle_features
    from scripts.winrate_folds import forward

    edges = pd.date_range(PERIODS[0], PERIODS[2], periods=FOLDS + 1)[1:-1].to_numpy()
    rows = []
    for tf in timeframes:
        for s in symbols:
            df, oi, _ = frames(s, tf)
            fold = np.searchsorted(edges, df.index.to_numpy(), side="right")
            fold = np.where(df.index.to_numpy() >= np.datetime64(PERIODS[2]), FOLDS, fold)
            fwd = {h: forward(df.open, df.close, fold, h) for h in zones.HORIZONS}
            for w in WINDOWS:
                f = oi_features(df, oi, w)
                cf = candle_features.features(df, w)
                for name, column in CONTROLS.items():
                    f[name] = cf[column]
                for col in f:
                    x = f[col]
                    code = zones.bands(x, np.nanquantile(x.to_numpy()[fold < FOLDS], zones.QS))
                    for h in zones.HORIZONS:
                        r = fwd[h]
                        ok = np.isfinite(r) & (fold < FOLDS)
                        base = pd.Series(r[ok]).groupby(fold[ok]).mean()
                        m = ok & (code >= 0)
                        g = pd.DataFrame({"fold": fold[m], "band": code[m], "r": r[m]})
                        g = g.groupby(["band", "fold"]).r.agg(["mean", "count"]).reset_index()
                        g["excess"] = g["mean"] - g.fold.map(base)
                        rows.append(g.assign(symbol=s, timeframe=tf, column=col, window=w, h=h))
    t = pd.concat(rows)
    key = ["timeframe", "column", "window", "band", "h"]
    per = t.groupby([*key, "symbol"]).agg(
        up=("excess", lambda e: int((e > 0).sum())),
        down=("excess", lambda e: int((e < 0).sum())),
        folds=("excess", "size"),
        min_n=("count", "min"),
        excess=("excess", "mean"),
    )
    per = per[(per.folds == FOLDS) & (per.min_n >= MIN_N)].reset_index()
    wide = per.pivot_table(index=key, columns="symbol", values=["up", "down", "excess"])
    out = pd.DataFrame(index=wide.index)
    out["stable"] = np.where(
        (wide["up"][symbols] == FOLDS).all(axis=1),
        "long",
        np.where((wide["down"][symbols] == FOLDS).all(axis=1), "short", ""),
    )
    for sym in symbols:
        out[f"{sym} excess"] = wide[("excess", sym)]
    out["family"] = np.where(
        out.index.get_level_values("column").isin(list(CONTROLS)), "price control", "open interest"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=list(zones.SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m"])
    args = parser.parse_args()
    table = pd.concat([scan(s, tf) for tf in args.timeframes for s in args.symbols])
    z = pick(table, args.symbols)
    others = [s.removesuffix("USDT") for s in klines.DEFAULT_SYMBOLS if s.removesuffix("USDT") not in args.symbols]
    held = detail(z, args.symbols, others)
    z.round(4).to_csv(REPORTS / "oi_zones.csv")
    held.round(4).to_csv(REPORTS / "oi_zones_trades.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 200)
    print(
        f"{len(table) // len(zones.PERIODS):,} cells; {len(z)} zones pass discovery, {int(z.holds.sum())} hold "
        f"({int((z.holds & (z.dir == 'long')).sum())} long). Tables in reports/oi_zones*.csv"
    )
    print(z.round(3).to_string())
    print(held.round(3).to_string())
    st = stability(args.symbols, args.timeframes)
    st.round(4).to_csv(REPORTS / "oi_zones_folds.csv")
    print("\nStable in every half-year on every pair, by family and column:")
    print(st.groupby(["family", "stable"]).size().to_string())
    print(st[st.stable != ""].round(3).to_string())


def _selfcheck() -> None:
    """The periods, and the columns on hand-made candles and open interest."""
    when = pd.date_range("2021-11-30", "2025-07-01", freq="1D")
    p = period_of(when)
    assert p[0] == -1 and p[1] == 0 and p[-1] == 2 and when[p == 1][0] == PERIODS[1]
    df = pd.DataFrame({"close": [100.0, 99, 98, 99, 100, 101], "volume": 1.0}, index=when[:6])
    oi = pd.DataFrame({"open_interest": [10.0, 11, 12, 11, 10, 10], "open_interest_value": 1000.0}, index=df.index)
    f = oi_features(df, oi, 2)
    assert np.isclose(f.oi_change.iloc[2], np.log(12 / 10))
    # At row 2 the price fell over the window and the open interest rose: only the "falls" column is set.
    assert np.isclose(f.oi_change_price_down.iloc[2], np.log(12 / 10)) and np.isnan(f.oi_change_price_up.iloc[2])
    assert np.isclose(f.oi_change_price_up.iloc[5], np.log(10 / 11)) and np.isnan(f.oi_change_price_down.iloc[5])
    assert np.isclose(f.oi_vs_volume.iloc[5], np.log(1000 / (100 + 101)))


if __name__ == "__main__":
    main()
