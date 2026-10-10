"""Which indicator zones win most often, and in every year: the win rate fold by fold.

The zones study ranked cells by how much they return. This ranks them by how often they win, and asks that it hold
in every one of `K` time folds rather than on average. Same grid: the 29 columns of `candle_features` and the 13 of
`project_features`, every window of `WINDOWS`, ten bands at the quantiles `zones.QS`, every horizon H of
`zones.HORIZONS`, on BTC and ETH, 5m and 15m.

**The folds.** The store from its first bar to `HOLDOUT` is cut in `K` folds of equal time, about a year each. What
comes after `HOLDOUT` is not a fold: nothing is chosen on it, it is read once at the end (`holdout` columns). Without
it, "the cell whose worst fold is best" would be the best of tens of thousands of cells on all the data there is.

**The return.** For every candle, the change from the open `zones.DELAY + 1` candles after it to the close H candles
after that: a candle late, held H, as `zones.trades` prices it. A candle whose exit falls in another fold counts in
neither. **The win rate** is the share of candles in the band whose return is above zero, gross; it is set against the
share of every candle of the same fold (`base`), since in a bull year almost any long wins. A cell is read in one
direction for all folds: long, or short when the long share is low in every fold.

**The bands** are cut at the quantiles of the column over the folds together (the holdout excluded), so a band means
the same threshold in every fold. A tail of 2017's volatility fires less in 2024: the per-fold count shows it.

A per-candle win rate counts the candles of one crash once each, and they win or lose together. Read a cell's count
with that in mind; the shortlist is the place to price it per episode.

    .venv312/bin/python -m scripts.winrate_folds                 # ~15 minutes; tables in analysis_cache/ and reports/
    .venv312/bin/python -m scripts.winrate_folds --timeframes 15m
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import candle_features, project_features, zones
from scripts.zones_scan import load

K = 8  # folds of equal time before the holdout
HOLDOUT = pd.Timestamp("2025-06")  # `zones.CONFIRM[1]`: what the first study called "recent", read once here
WINDOWS = (6, 12, 24, 48, 96)
MIN_N = 30  # candles a fold must hold for its win rate to be read
CACHE = PROJECT_ROOT / "analysis_cache" / "winrate_folds.parquet"
REPORTS = PROJECT_ROOT / "reports"


def fold_of(index: pd.DatetimeIndex, k: int = K, end: pd.Timestamp = HOLDOUT) -> np.ndarray:
    """0 to k-1 for the k equal-time folds from the first bar to `end`, k for the holdout after it."""
    edges = pd.date_range(index[0], end, periods=k + 1)[1:-1]
    fold = np.searchsorted(edges.to_numpy(), index.to_numpy(), side="right")
    return np.where(index.to_numpy() >= np.datetime64(end), k, fold)


def forward(open_: pd.Series, close: pd.Series, fold: np.ndarray, h: int, delay: int = zones.DELAY) -> np.ndarray:
    """Each candle's return in percent, entered at the open `delay + 1` candles later and held `h` candles; NaN when
    the exit is past the end or in another fold."""
    o, c = open_.to_numpy(), close.to_numpy()
    entry, exit_ = 1 + delay, delay + h
    out = np.full(len(c), np.nan)
    out[: len(c) - exit_] = (c[exit_:] / o[entry : len(o) - exit_ + entry] - 1) * 100
    same = np.zeros(len(c), bool)
    same[: len(c) - exit_] = fold[exit_:] == fold[: len(c) - exit_]
    return np.where(same, out, np.nan)


def counts(code: np.ndarray, fold: np.ndarray, fwd: np.ndarray, bands: int) -> pd.DataFrame:
    """By fold and band (-1: every candle of the fold): candles, longs that won, shorts that won, summed return."""
    ok = np.isfinite(fwd)
    rows = []
    for b, mask in [(-1, ok)] + [(b, ok & (code == b)) for b in range(bands)]:
        f, r = fold[mask], fwd[mask]
        n = np.bincount(f, minlength=K + 1)
        rows.append(
            pd.DataFrame(
                {
                    "fold": np.arange(K + 1),
                    "band": b,
                    "n": n,
                    "up": np.bincount(f, r > 0, minlength=K + 1),
                    "down": np.bincount(f, r < 0, minlength=K + 1),
                    "sum": np.bincount(f, r, minlength=K + 1),
                }
            )
        )
    return pd.concat(rows)


def scan(symbol: str, timeframe: str) -> pd.DataFrame:
    """`counts` for every column at every window and every H on one pair and timeframe."""
    df = load(symbol, timeframe)
    fold = fold_of(df.index)
    fwd = {h: forward(df.open, df.close, fold, h) for h in zones.HORIZONS}
    out = []
    for w in WINDOWS:
        for module in (candle_features, project_features):
            f = module.features(df, w)
            for column in module.COLUMNS:
                if column in zones.FIXED and w != WINDOWS[0]:
                    continue
                x = f[column]
                code = zones.bands(x, np.nanquantile(x.to_numpy()[fold < K], zones.QS))
                for h in zones.HORIZONS:
                    t = counts(code, fold, fwd[h], len(zones.QS) + 1)
                    out.append(t.assign(column=column, window=0 if column in zones.FIXED else w, h=h))
        print(f"  {symbol} {timeframe} window {w}", flush=True)
    return pd.concat(out).assign(symbol=symbol, timeframe=timeframe)


def cells(table: pd.DataFrame) -> pd.DataFrame:
    """One row per (timeframe, column, window, band, H, pair): the direction, the worst and median fold's win rate,
    the worst fold's lift over every candle of that fold, the thinnest fold, the mean return, and the holdout's."""
    key = ["timeframe", "symbol", "column", "window", "h"]
    base = table[table.band == -1].set_index([*key, "fold"])[["n", "up", "down"]]
    t = table[table.band >= 0].join(base, on=[*key, "fold"], rsuffix="_all")
    sel, hold = t[t.fold < K], t[t.fold == K]
    up, down = sel.up / sel.n, sel.down / sel.n
    sel = sel.assign(
        wr_up=up, wr_down=down, lift_up=up - sel.up_all / sel.n_all, lift_down=down - sel.down_all / sel.n_all
    )
    g = sel.groupby([*key, "band"])
    out = pd.DataFrame(
        {
            "min_n": g.n.min(),
            "up_min": g.wr_up.min(),
            "down_min": g.wr_down.min(),
            "up_med": g.wr_up.median(),
            "down_med": g.wr_down.median(),
            "up_lift": g.lift_up.min(),
            "down_lift": g.lift_down.min(),
            "mean": g["sum"].sum() / g.n.sum(),
        }
    )
    short = out.down_min > out.up_min
    res = pd.DataFrame(
        {
            "dir": np.where(short, "short", "long"),
            "wr_min": np.where(short, out.down_min, out.up_min),
            "wr_med": np.where(short, out.down_med, out.up_med),
            "lift_min": np.where(short, out.down_lift, out.up_lift),
            "min_n": out.min_n,
            "mean": np.where(short, -out["mean"], out["mean"]),
        },
        index=out.index,
    )
    h = hold.set_index([*key, "band"]).reindex(res.index)
    res["hold_n"] = h.n
    with np.errstate(invalid="ignore", divide="ignore"):
        res["hold_wr"] = np.where(short, h.down / h.n, h.up / h.n)
    return res


def shortlist(c: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Cells read on every pair in one direction, each fold holding `MIN_N` candles, ranked by the worse pair's worst
    fold."""
    c = c[c.min_n >= MIN_N].reset_index()
    wide = c.pivot_table(
        index=["timeframe", "column", "window", "band", "h"],
        columns="symbol",
        values=["wr_min", "wr_med", "lift_min", "min_n", "mean", "hold_n", "hold_wr"],
    )
    dirs = c.pivot_table(
        index=["timeframe", "column", "window", "band", "h"], columns="symbol", values="dir", aggfunc="first"
    )
    both = dirs[symbols].notna().all(axis=1) & dirs[symbols].nunique(axis=1).eq(1)
    wide = wide[both]
    flat = pd.DataFrame(index=wide.index)
    flat["dir"] = dirs.loc[wide.index, symbols[0]]
    flat.insert(0, "quantiles", [zones.quantiles(b) for b in wide.index.get_level_values("band")])
    for s in symbols:
        for v in ("wr_min", "wr_med", "lift_min", "min_n", "mean", "hold_wr", "hold_n"):
            flat[f"{s} {v}"] = wide[(v, s)]
    flat["wr_min"] = flat[[f"{s} wr_min" for s in symbols]].min(axis=1)
    return flat.sort_values("wr_min", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=list(zones.SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m"])
    args = parser.parse_args()
    table = pd.concat([scan(s, tf) for tf in args.timeframes for s in args.symbols])
    CACHE.parent.mkdir(exist_ok=True)
    table.to_parquet(CACHE)
    s = shortlist(cells(table), args.symbols)
    s.round(4).to_csv(REPORTS / "winrate_folds.csv")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100)
    print(f"{len(s):,} cells read on both pairs with {MIN_N}+ candles a fold; the 40 whose worst fold wins most often:")
    print(s.head(40).round(3).to_string())


def _selfcheck() -> None:
    """The folds, the forward return and the counts on hand-made candles."""
    when = pd.date_range("2017-08-01", "2026-01-01", freq="1D")
    fold = fold_of(when)
    assert fold[0] == 0 and fold[-1] == K and set(fold) == set(range(K + 1))
    assert (np.diff(fold) >= 0).all() and when[fold == K][0] == HOLDOUT
    sizes = np.bincount(fold)[:K]
    assert sizes.max() - sizes.min() <= 1, "folds of equal time"
    # A close rising 1% a candle with every open 1% under the close before: a candle late and held 2, the trade runs
    # from the open of candle i+2 (0.99 c[i+1]) to the close of i+3, +1.01^2 / 0.99 - 1 every time.
    close = pd.Series(100 * 1.01 ** np.arange(len(when)), index=when)
    open_ = close.shift().bfill() * 0.99
    fwd = forward(open_, close, fold, 2, delay=1)
    inside = np.isfinite(fwd)
    assert np.allclose(fwd[inside], (1.01**2 / 0.99 - 1) * 100)
    # No candle's exit is in another fold, and the last three of each fold have none.
    last = np.r_[np.flatnonzero(np.diff(fold)), len(fold) - 1]
    assert all(np.isnan(fwd[i - 2 : i + 1]).all() and np.isfinite(fwd[i - 3]) for i in last)
    # Counts: band -1 is every candle, a band only its own; ups and downs add to the candles that moved.
    code = np.arange(len(when)) % 3
    t = counts(code, fold, np.where(inside, np.where(code == 2, -1.0, 1.0), np.nan), 3)
    assert (t[t.band == -1].n.to_numpy() == t[t.band >= 0].groupby("fold").n.sum().to_numpy()).all()
    assert (t[t.band == 2].up == 0).all() and (t[t.band == 0].down == 0).all()
    # A short reads in the same direction in every fold, and its lift is over every candle's share of downs.
    table = t.assign(column="x", window=6, h=2, symbol="BTC", timeframe="1d")
    c = cells(table)
    assert c.loc[("1d", "BTC", "x", 6, 2, 2), "dir"] == "short" and c.loc[("1d", "BTC", "x", 6, 2, 2), "wr_min"] == 1
    assert np.isclose(c.loc[("1d", "BTC", "x", 6, 2, 0), "lift_min"], 1 - 2 / 3, atol=0.01)


if __name__ == "__main__":
    main()
