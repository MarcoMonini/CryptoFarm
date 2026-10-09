"""The zones study of `reports/zones_study.html`, run again over the indicators this project reads.

TradingVision's search (`tradingvision.zones`, PR #22) covered the 29 columns of `candle_features`. This is the
same search, with the same rules fixed before looking, over the thirteen of `trading/project_features.py`:
every column at every window of `WINDOWS`, on BTC and ETH, 5m and 15m; bands at the discovery quantiles
`zones.QS`; N in `zones.HORIZONS`; a zone passes discovery (to 2020-12) beyond `T_PICK` standard errors on both
pairs with one sign, the best (window, N) per (timeframe, column, band) kept; it holds when confirm
(2021-01 to 2025-06) gives the same sign on both at `T_HOLD`. Every zone that holds is then priced one trade
an episode, gross, against `zones.ROUND_TRIP`.

One thing differs from the first run and has to be said: confirm has been read once already, for the first 29
columns, and the reading is known — a price far under its average bounces. A new column that measures the same
thing (Bollinger %B, the distance from KAMA in ATR) will hold, and is not a second finding. The output marks each
zone's correlation with the first study's columns for that reason.

    .venv312/bin/python -m scripts.zones_scan                # ~10 minutes; tables in reports/
    .venv312/bin/python -m scripts.zones_scan --symbols BTC ETH --timeframes 15m
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptofarm.data import klines
from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import candle_features, project_features, zones

WINDOWS = (6, 12, 24, 48, 96)
T_PICK, T_HOLD = 4.0, 2.0  # the first study's: about Bonferroni over ~10,400 cells a pair here, ~17,700 there
REPORTS = PROJECT_ROOT / "reports"


def load(symbol: str, timeframe: str) -> pd.DataFrame:
    return klines.load_klines(f"{symbol}USDT", timeframe, clip=False).rename(columns=str.lower)


def scan(symbol: str, timeframe: str) -> pd.DataFrame:
    """`zones.cells` for every column at every window on one pair and timeframe, with each band's edges."""
    df = load(symbol, timeframe)
    period, step = zones.period_of(df.index), pd.Timedelta(timeframe.replace("m", "min"))
    out = []
    for w in WINDOWS:
        f = project_features.features(df, w)
        for column in project_features.COLUMNS:
            edges = zones.band_edges(f[column], period)
            names = np.r_[-np.inf, edges, np.inf]
            t = zones.cells(zones.bands(f[column], edges), df.close, period, step)
            t["lo"], t["hi"] = names[t.band], names[t.band + 1]
            out.append(t.assign(column=column, window=w))
        print(f"  {symbol} {timeframe} window {w}", flush=True)
    return pd.concat(out).assign(symbol=symbol, timeframe=timeframe)


def pick(table: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """The criterion of the docstring: discovery cells beyond `T_PICK` on every pair with one sign, the best per
    (timeframe, column, band), with every period's excess and t per pair beside them and the verdict."""
    key = ["timeframe", "column", "window", "band", "n"]
    wide = table.pivot_table(index=key, columns=["period", "symbol"], values=["excess", "t"])
    t = wide["t"]["discovery"][symbols]
    sign = np.sign(t).nunique(axis=1).eq(1) & np.sign(t).ne(0).all(axis=1)
    weakest = t.abs().min(axis=1).where(sign)
    passed = weakest[weakest >= T_PICK].rename("pick").reset_index()
    best = passed.loc[passed.groupby(["timeframe", "column", "band"]).pick.idxmax()]
    out = wide.loc[best.set_index(key).index]
    flat = pd.DataFrame(index=out.index)
    for p in zones.PERIODS:
        for s in symbols:
            flat[f"{p} {s} %"] = out[("excess", p, s)] if ("excess", p, s) in out else np.nan
            flat[f"{p} {s} t"] = out[("t", p, s)] if ("t", p, s) in out else np.nan
    flat.insert(0, "quantiles", [zones.quantiles(b) for b in flat.index.get_level_values("band")])
    d = np.sign(flat[f"discovery {symbols[0]} t"])
    flat["holds"] = (flat[[f"confirm {s} t" for s in symbols]].mul(d, axis=0) >= T_HOLD).all(axis=1)
    return flat.join(best.set_index(key).pick).sort_values("pick", ascending=False)


def detail(z: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """For every zone that holds: one trade an episode per period and pair, gross, in the zone's direction, and the
    column it is closest to among `candle_features` at the same window (Spearman, on the first pair's candles), so a
    re-expression of the first study's zones reads as one."""
    frames, rows = {}, []
    for (tf, column, w, band, n), r in z[z.holds].iterrows():
        sign = np.sign(r[f"discovery {symbols[0]} t"])
        for s in symbols:
            if (s, tf) not in frames:
                frames[(s, tf)] = load(s, tf)
            df = frames[(s, tf)]
            if (s, tf, w) not in frames:
                frames[(s, tf, w)] = (project_features.features(df, w), candle_features.features(df, w))
            mine, theirs = frames[(s, tf, w)]
            x, period = mine[column], zones.period_of(df.index)
            code = zones.bands(x, zones.band_edges(x, period))
            row = {"timeframe": tf, "column": column, "window": w, "quantiles": zones.quantiles(band), "n": n}
            row["symbol"] = s
            if s == symbols[0]:
                rho = theirs.iloc[::7].corrwith(x.iloc[::7], method="spearman")  # every 7th bar: fast, same answer
                row["closest"], row["rho"] = rho.abs().idxmax(), rho[rho.abs().idxmax()]
            for p, name in enumerate(zones.PERIODS):
                g = zones.trades((code == band) & (period == p), df.open, df.close, n, sign)
                row |= {
                    f"{name} trades": len(g),
                    f"{name} mean": g.mean() if len(g) else np.nan,
                    f"{name} se": g.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                    f"{name} win": (g > 0).mean() if len(g) else np.nan,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--symbols", nargs="+", default=list(zones.SYMBOLS))
    parser.add_argument("--timeframes", nargs="+", default=["5m", "15m"])
    args = parser.parse_args()
    table = pd.concat([scan(s, tf) for tf in args.timeframes for s in args.symbols])
    z = pick(table, args.symbols)
    trades = detail(z, args.symbols)
    REPORTS.mkdir(exist_ok=True)
    z.round(4).to_csv(REPORTS / "zones_project.csv")
    trades.round(4).to_csv(REPORTS / "zones_project_trades.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    print(
        f"{len(table) // len(zones.PERIODS):,} cells; {len(z)} zones pass discovery, {int(z.holds.sum())} hold on "
        f"confirm. Tables in {REPORTS.relative_to(PROJECT_ROOT)}/zones_project*.csv"
    )
    print(z.round(3).to_string())
    print("\nOne trade an episode, gross, in the zone's direction:")
    print(trades.round(3).to_string())


if __name__ == "__main__":
    main()
