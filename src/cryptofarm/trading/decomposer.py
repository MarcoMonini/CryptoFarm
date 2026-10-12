"""Streamlit page: what the candles of one pair look like, read from the store and nothing else.

Descriptive statistics of the market itself, before any label or model. It runs locally on the
Parquet store under `market_data/` (`data.klines`: 5m candles, aggregated to the timeframe picked with
`klines.load_klines`, wicks left raw) and downloads nothing: a pair is offered only if its file is there. It is not
deployed — the image carries no store — and it has no sidebar: the controls sit above what they
move.

A candle's change is open to close, `close / open - 1`, in percent. The dumps trade around the
clock, so a candle opens where the one before closed on almost every bar, and close to close would
give the same numbers.

**The mean is the statistic near zero, not the standard deviation.** Over all candles the mean
change is the drift, a few thousandths of a percent; the standard deviation is the size of a typical
candle — a 50% annual volatility is 0.27% a 15m candle — and is never near zero. The up and down
candles each get their own pair: the mean of the up candles is how far an up candle goes on average,
their standard deviation how much one up candle differs from another. Split at zero, a symmetric
distribution gives the two sides mirror values, and for a normal one the standard deviation of a
side is sqrt(1 - 2/pi) = 0.60 of the whole (asserted in `_selfcheck`). Candles with open equal to
close count in "all" and in neither side.

**The histogram counts every candle loaded.** Its axis is symmetric around zero and spans the larger
of the 0.1th and 99.9th percentiles in absolute value, so a handful of crashes cannot flatten the
bell into a spike; the candles beyond it are counted in its caption, never dropped silently. One bin
is centred on zero. Expect it centred on zero but not normal: crypto returns have a sharper peak and
heavier tails than a normal of the same deviation. The page draws no normal over it. The same buttons as
the candles' take the same statistic off every candle, over the same bins.

Under the candles, in a chart of its own over the same period, the same candles each started from
zero: open and close as changes from the candle's own open, in percent. The price level, the path
between candles and the wicks go, and what is left is each candle's body, the change the histogram
counts. A control right above that chart, and moving nothing else, subtracts a statistic from each of
its candles (`offset`), toward zero: the standard deviation of all candles from every candle, each
side's standard deviation from its candles, or each side's mean from its candles (`less`). The open
stays at zero and the body shrinks by it: a 1% candle less a 0.2% deviation closes at 0.8%. A candle
smaller than the statistic crosses zero, and keeps the colour of the side it came from: an up candle
is green whichever way it now points. The statistics are the ones the page shows, over the whole
history and not over the period. The two charts are one figure on one time axis, so a zoom on either
moves both. The period is read by these two alone, and its default, the last 30 days, is there to keep
the drawing light: the numbers and the histogram read the whole history.

Beside the histogram, in the same row, one section over every candle loaded, numbers measured on BTC
15m to 2026-10-06 (319,870 candles): **Memory** (`acf`): the autocorrelation of the change and of its
size, lags 1 to 10,000 candles (`ACF_LAGS`), against ±2/√n. The change sits at −0.006 at lag 1 (5m
−0.029, the bid-ask bounce) and near zero after; its size starts at 0.39, is 0.22 a day later and
still 0.12 at lag 2,000 (21 days), with a bump every day of lags from the volatility's daily cycle. So
the global standard deviation the buttons subtract is a long-run average over periods whose typical
candle is several times apart. The ±2/√n band holds for independent candles only; with clustered sizes
the true band is wider, the reason the project takes its errors on non-overlapping blocks
(TradingVision's `metrics.blocked`).

**Time of day and week** (`clock`): the mean and the standard deviation of the change by weekday and
UTC hour of the open, two heatmaps side by side, 1,905 candles a cell on BTC 15m. The standard
deviation is the cycle behind the autocorrelation's daily bumps: its median over the week is 0.48% at
14:00 UTC, the US open, and Saturday and Sunday sit at 0.32-0.33% against 0.39-0.41% on weekdays
(ETH 15m the same shape). A cell's standard deviation is one heavy-tailed sample, though: the
largest, Friday 02:00 at 0.73%, and ETH's Tuesday 04:00 at 2.59%, against a typical 0.5%, are a
handful of crashes rather than an hour. The mean is almost all noise: 12 of 168 cells lie beyond two
standard errors of zero, against 8 by chance, and the standard error is understated because sizes
cluster. The one that stands out is Saturday 00:00 UTC, positive at 3.6 standard errors on BTC 15m,
3.8 on BTC 1h and 4.4 on ETH 15m: past a Bonferroni cut of about 3.4 for 168 cells, measured on the
whole history with nothing held out, so a lead and not a finding.

**The future of a candle** (`ahead`): the candles grouped by their change in standard deviations of
all candles, bands of 0.5σ from −3σ to +3σ and one open band past each end (`SIGMA_STEP`,
`SIGMA_REACH`; a flat candle falls in 0 to +0.5σ), and for each band and each N from 1 to N max the
mean cumulative change from the candle's close to the close N candles later — what a buyer at the
close holds after N candles. The candle's own body is not in it: it is known when the future starts.
Two heatmaps, band against N: the mean, and its excess over the mean of every candle at the same N
in standard errors. The excess, because the drift grows with N and would colour every band alike;
standard errors over non-overlapping blocks of N candles on the clock (`metrics.blocked`'s rule),
because adjacent candles share N − 1 of the candles their futures are made of and in a band of large
candles they arrive together. On BTC 15m to 2026-10-06, N to 96 (a day): one candle later the small
up candles, 0 to +1σ, give back a little, −3.3 and −3.1 standard errors; a day later both tails sit
above the drift, below −3σ +1.36% against +0.15% at 9.5, above +3σ +0.78% at 4.9. Both sides rising
reads as the tails' periods, not their sign — large candles crowd into the volatile bull runs — and
1,344 cells measured on the whole history with nothing held out, adjacent N being one test and not
many, make it a lead.

**The future after an RSI**: the same two heatmaps with the candles banded by Wilder's RSI at their
close (`rsi_bands`, 14 candles as `features` reads it, 10 points a band), and the same N. On BTC 15m
the RSI does not read as the textbook does: above 70 is not overbought but the start of a run. From
80 to 90 (2,041 candles) the excess over every candle's is +0.12% at N 10, +0.56% at 48 and +0.51%
at 96, at 2.7, 5.5 and 3.8 standard errors; ETH 15m +0.60% at 48, 5.1. Below 30 there is a short
bounce, 20 to 30 at +0.07% and 3.8 five candles later on BTC, that turns into continuation by half a
day: ETH 0 to 10 at −2.2% at 48, −4.9. The bands past 10 and 90 hold about a hundred candles each, a
handful of episodes. The momentum above 70 is the one size worth a fee, 0.20% a round trip at OKX,
and it is measured on the whole history with nothing held out: a lead.

**The future after an indicator**: the same again for any of the 27 columns (`COLUMNS`) of `features` and
`project_features` — this project's own indicators, listed after them — that held in at least one study, with
their window as an input (`EXTREMA_WINDOW` by default), banded in deciles (`quantile_bands`) because
no two share a scale; a decile dilutes a tail, so RSI's top decile reads weaker here than its 80 to
90 band above. The extreme deciles on BTC and ETH 15m, N 10 / 48 / 96, fall in three families. A
price far below its own average bounces within a few hours: the bottom decile of the distance from
VWAP, KAMA, EMA or PSAR, of the log return, of the close's place in the window, +0.06 to +0.11% at N
10 at 4 to 6.6 standard errors — below the 0.20% fee, the candle bodies' reversal again. Large bars
come before rises at every N: the top decile of the bar's range is +0.32% at 96 on BTC (5.5) and
+0.58% on ETH (8.3), the upper wick +0.33% and +0.50% (6.0, 7.4), the volatility windows the same
on ETH; the volatile periods are the bull runs, as for both tails of the bodies. And momentum over
half a day: the RSI and the TSI's top decile +0.19 to +0.24% at 48 (4.4 to 5.3), their bottom
decile −0.11 to −0.23% (−2.4 to −4.0). The exception is the PSAR, whose bottom decile — a price
furthest under it — keeps rising, +0.16% and +0.25% at 48 (3.8, 5.0). 58 deciles at 96 N each on the
whole history: a map of where to look, not a result. The menu held all 42 columns when these were measured. The
momentum did not hold out of sample in the zones study, and RSI and TSI left it with thirteen columns no study found
anything in (`DROPPED`): the log return, the lower wick, the candle body, the window range, the volatility expansion,
four of volume, the money flow, the ADX, the squeeze and the stochastic.

**What is left of the leg** (`pivots`, `legs_of`, `to_pivot`): TradingVision's pivots, the ones its `legsweep` swept — a
close strictly beyond the `window` closes on both sides, two of a kind in a row merged into the more extreme — at
windows 6, 12, 24 and 48 candles of the timeframe picked (`LEG_WINDOWS`). One histogram per window, two by two, in bins
of 0.01% opening on 0 to 5% (`LEG_BIN`, `LEG_VIEW`; a zoom out shows the rest), counting one of two things on buttons
(`LEG_COUNTS`). **Pivots**, the default: every leg once, its change from pivot to pivot, a down leg's sign flipped — the
measure the next section slices on. **Candles**: every candle, its change from its close to the close of the next pivot,
folded the same way, so a leg of 40 candles counts 40 times, from its whole size down to nearly nothing. It is
hindsight, the most a reader who knew the pivots could take. On BTC 15m to 2026-08-19 the median leg is 0.92%, 1.43%,
2.21% and 3.48% at the four windows (ETH 1.26 to 4.73%), and 2.8%, 0.7%, 0.2% and 0.1% of legs are under the 0.20% round
trip. The median candle has 0.60%, 0.89%, 1.35% and 2.14% left at the four windows (ETH 0.83 to 2.93%), and 17.3%,
10.3%, 5.6% and 2.8% of the candles have less than the 0.20% round trip; under 0.5% are negative, candles beyond the
pivot their leg ends on.

**Pivots of a size** (`legs_of`, `excess`): the legs between two pivots whose change falls in a slice, 1% to 1.2% by
default (`LEG_SLICE`), up legs (from a low) and down legs (from a high) apart, and where every indicator of the menus
stands on the pivot they start on, in deciles over every candle: one heatmap of the indicators by decile per side,
coloured by the share of the slice's pivots in each from 0, none, and marked ▲ ▼ where it is beyond 3 standard errors of
the reference's share — every candle, or every pivot that starts a leg of the same side whatever its size; and for one
indicator, decile against the 96 candles before the pivot. On BTC 15m at window 24 (237 up, 261 down; ETH 143 up),
against every candle the indicators say what a pivot is — a low sits in the bottom decile of the Donchian position at
+37 se. Against every same-kind pivot what is left is the size: the 1% legs start from shallower extremes, fewer in the
bottom decile of the distance from PSAR, EMA, VWAP and KAMA (−6.6 to −7.2 se) and of the large bars (−6.4), on ETH the
same at up to −10.2. A leg of 1% is a small one, and small legs are the quiet periods'; that is the one thing the slice
has in common that every pivot has not. A mean correlation of their paths, along the leg and over the candles before it,
was measured and dropped: the slice was no more alike than any leg of its side (0.620 along the up legs against 0.670
for all of them).

**The zones of the study** (`zones`): the twenty zones of `reports/zones_study.html` — four families that
held from 2017-2020 to 2021-2025, and the momentum that did not — each at its own timeframe, window, band
and N, re-read on this store; and the twenty that `scripts/zones_scan.py` found holding among this project's
own indicators (`zones.PROJECT_ZONES`), in the same list, by family (`zones.ALL_ZONES`). One zone at a time: a
grid of small heatmaps, BTC and ETH down, discovery, confirm and recent across, every band of the column's 2017-2020
quantiles against every N of the study, in standard errors of the excess, with the zone's cell outlined; under
it the zone's cell in numbers and one trade an episode against the 0.20% fee. The bands' edges are
discovery's on every period, so the recent column is the one where the 2017-2020 tails fire rarely.

    streamlit run src/cryptofarm/trading/decomposer.py

Brought over from TradingVision (PRs #21 and #22) with `candle_features` as its one dependency; the
numbers above were measured there, on its store, and the study they led to is `reports/zones_study.html`.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator

from cryptofarm import paths
from cryptofarm.data import klines
from cryptofarm.trading import candle_features as features
from cryptofarm.trading import project_features, zones
from cryptofarm.trading.candle_features import EXTREMA_WINDOW

TIMEFRAMES = ("5m", "15m", "1h", "4h", "1d")
TIMEFRAME = "15m"  # v2's, the timeframe the project studies
PERIOD = pd.Timedelta(days=30)  # the candle charts' default window
BINS = 601  # odd, so one bin is centred on zero
TAIL = 0.001  # the histogram's axis ends at this quantile on either side, whichever is further out
ACF_LAGS = (
    10_000  # the autocorrelation's last lag, in candles: 35 days of 5m, 104 of 15m; a quarter of the data at most
)
SIGMA_STEP, SIGMA_REACH = 0.5, 3.0  # the future's bands: their width and the last closed edge, in σ of all candles
RSI_WINDOW, RSI_STEP = 14, 10  # Wilder's window, the default of `ta` and of every charting tool; the bands' width
QUANTILES = 10  # an indicator's bands: deciles, as many candles in each, since no two indicators share a scale
HORIZON, HORIZON_MAX = 96, 500  # the future's default N max (a day of 15m) and the slider's end: 1.6s per 96 on BTC 15m
LEG_WINDOWS = (6, 12, 24, 48)  # the pivot windows of the legs' histograms, candles on each side; 24 is TradingVision's
LEG_BIN, LEG_VIEW = 0.01, 5.0  # the legs' bin width and the end of the axis they open on, in %
LEG_COUNTS = {"pivots": "Pivots: every leg once", "candles": "Candles: what each had left"}  # the histograms' buttons
LEG_SLICE = (1.0, 1.2)  # the default slice of the pivots' study: legs whose size, in %, falls in it
LEG_LOOKBACK = 96  # the candles before the pivot the indicator-by-candle chart reaches back to
SIDES_OF_LEG = {"up": "Up legs, from a low", "down": "Down legs, from a high"}
BASES = {"candles": "every candle", "pivots": "every pivot of the same kind, any size"}
# What the slice is compared with, per side, as the hover names it.
REFERENCES = {
    "candles": {"up": "every candle", "down": "every candle"},
    "pivots": {"up": "every low that starts a leg up, any size", "down": "every high that starts a leg down, any size"},
}
EXCESS_MARK = 3.0  # the standard errors past which a share is marked above or below its reference
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")  # pandas' dayofweek, Monday 0
SIDES = {"all": "All candles", "up": "Up candles", "down": "Down candles"}
# What the lower candle chart can take off each candle, named as the statistics at the top of the page
# are, so a choice points at the numbers it subtracts without repeating them.
# The columns of TradingVision's 29 and this project's 13 that no study found anything in: no zone of `zones` that holds
# (families 1 to 5; RSI and TSI only in the momentum one, which does not) and no filter the pivot study chose. Their
# columns are still computed, by `features` and `project_features`, and the zones that read them still read them.
DROPPED = (
    "rsi_centered",
    "tsi_momentum",
    "money_flow_index",
    "adx_trend_strength",
    "squeeze_ratio",
    "log_return",
    "volatility_expansion",
    "lower_wick_pct",
    "log_volume_vs_median",
    "log_dollar_volume",
    "volume_trend",
    "on_balance_volume_zscore",
    "stochastic_k_minus_d",
    "candle_body_pct",
    "window_range_pct",
)
# The indicator menu and the pivot study's rows: TradingVision's columns, then this project's, less `DROPPED`.
COLUMNS = [c for c in features.COLUMNS + project_features.COLUMNS if c not in DROPPED]
LABELS = features.LABELS | project_features.LABELS
OFFSETS = {
    "none": "Nothing",
    "std": "Std dev of all candles",
    "side std": "Std dev of up / down candles",
    "side mean": "Mean of up / down candles",
}


def store() -> Path:
    """The store, looked up at call time so a test can point `paths.MARKET_DATA_DIR` elsewhere."""
    return paths.MARKET_DATA_DIR


def pairs(where: Path) -> list[str]:
    """The symbols with a 5m file in the store: the training universe's order first, then the rest
    alphabetically. The positioning files (`*USDT-positioning.parquet`) are not candles and stay out."""
    held = {p.name.removesuffix("USDT-5m.parquet") for p in where.glob("*USDT-5m.parquet")}
    order = [s.removesuffix("USDT") for s in klines.DEFAULT_SYMBOLS]
    return [s for s in order if s in held] + sorted(held - set(order))


@st.cache_data(show_spinner="Reading the store…", max_entries=8)
def candles(symbol: str, timeframe: str, mtime: float) -> pd.DataFrame:
    """One pair's OHLC and volume at `timeframe`. `mtime` is the file's and is unused in the body: it is in the
    cache key so that updating the store re-reads it. Not `_mtime` — Streamlit leaves a parameter
    whose name starts with an underscore out of the key."""
    df = klines.load_klines(f"{symbol}USDT", timeframe, store_dir=store(), clip=False)
    return df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]


def change(df: pd.DataFrame) -> pd.Series:
    """Each candle's change from open to close, in percent."""
    return (df.close / df.open - 1) * 100


def from_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Each candle on its own open: open, high, low and close as percent changes from the open, so every
    candle starts at zero rather than where the one before closed. Its close is `change`."""
    return df[["open", "high", "low", "close"]].div(df.open, axis=0).sub(1).mul(100)


def sides(pct: pd.Series) -> pd.DataFrame:
    """Mean and standard deviation of the change over all candles, the up ones and the down ones."""
    groups = {"all": pct, "up": pct[pct > 0], "down": pct[pct < 0]}
    return pd.DataFrame({k: {"mean": v.mean(), "std": v.std()} for k, v in groups.items()}).T


def offset(pct: pd.Series, stats: pd.DataFrame, how: str) -> pd.Series:
    """What `how` takes off each candle, signed so that subtracting it moves the candle toward zero:
    positive on an up candle, negative on a down one, zero on a flat one. `stats` is `sides`' table;
    the mean of the down candles is negative already, their standard deviation is not."""
    side = np.sign(pct)
    up, down = {
        "none": (0.0, 0.0),
        "std": (stats.loc["all", "std"], -stats.loc["all", "std"]),
        "side std": (stats.loc["up", "std"], -stats.loc["down", "std"]),
        "side mean": (stats.loc["up", "mean"], stats.loc["down", "mean"]),
    }[how]
    return pd.Series(np.select([side > 0, side < 0], [up, down], 0.0), index=pct.index)


def less(zero: pd.DataFrame, minus: pd.Series) -> pd.DataFrame:
    """The candles of `from_zero` with `minus` (an `offset`) taken off their body, and their wicks dropped:
    the open stays at zero, the close moves by it, high and low are the body's ends."""
    close = zero.close - minus.reindex(zero.index)
    return zero.assign(close=close, high=close.clip(lower=0), low=close.clip(upper=0))


def histogram(pct: pd.Series, bins: int = BINS, tail: float = TAIL) -> tuple[np.ndarray, np.ndarray, int]:
    """The counts of `pct` in `bins` equal bins over ±R, their edges, and how many candles lie beyond
    ±R. R is the larger of the `tail` and `1 - tail` quantiles in absolute value."""
    reach = float(np.abs(pct.quantile([tail, 1 - tail])).max()) or float(np.abs(pct).max()) or 1.0
    edges = np.linspace(-reach, reach, bins + 1)
    counts, _ = np.histogram(pct, edges)
    return counts, edges, int(len(pct) - counts.sum())


def candles_figure(df: pd.DataFrame, uirevision: str, minus: dict[str, pd.Series]) -> go.Figure:
    """The candles of `df` over the same candles each from zero (`from_zero`), less what each choice of
    `OFFSETS` takes off (`minus`, its `offset`). One figure on one time axis, so a zoom or a pan on either chart
    moves both: two Streamlit charts cannot share a zoom, the page never hears of one. The choice is a row of
    buttons inside the figure, right above the lower chart it moves, because a Streamlit widget cannot sit
    between the rows of one figure; every choice is drawn and the buttons show one, in the browser."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Candlestick(x=df.index, open=df.open, high=df.high, low=df.low, close=df.close), row=1, col=1)
    zero, up = from_zero(df), df.close > df.open
    for how in OFFSETS:
        cut = less(zero, minus[how])
        # One trace per side of the candle as it was, painted that side's colour whichever way it points now: a
        # candle smaller than what `less` took off crosses zero and would otherwise change colour.
        for mask, colour in ((up, "#3D9970"), (~up, "#FF4136")):  # Plotly's own increasing and decreasing colours
            frame = cut[mask]
            side = dict(line_color=colour, fillcolor=colour)
            fig.add_trace(
                go.Candlestick(
                    x=frame.index,
                    open=frame.open,
                    high=frame.high,
                    low=frame.low,
                    close=frame.close,
                    yhoverformat="+.3f",
                    increasing=side,
                    decreasing=side,
                    visible=how == "none",  # the chart opens on the candles as they are, whatever the order above
                ),
                row=2,
                col=1,
            )
    choices(fig, [[True] + [h == how for h in OFFSETS for _ in "ud"] for how in OFFSETS], fig.layout.yaxis2.domain[1])
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#888", row=2, col=1)
    fig.update_layout(
        height=880,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        # Keeps zoom and pan across reruns until the pair, the timeframe or the period changes.
        uirevision=uirevision,
    )
    return fig


def choices(fig: go.Figure, shown: list[list[bool]], y: float) -> None:
    """A row of buttons, one per choice of `OFFSETS`, at height `y` of `fig`'s paper: each shows the traces
    its row of `shown` marks. It opens on Nothing, the figure as drawn."""
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=list(OFFSETS).index("none"),
                buttons=[
                    dict(label=OFFSETS[how], method="restyle", args=[{"visible": v}]) for how, v in zip(OFFSETS, shown)
                ],
                x=0,
                xanchor="left",
                y=y,
                yanchor="bottom",
                pad=dict(b=4),
            )
        ]
    )


def histogram_figure(pct: pd.Series, stats: pd.DataFrame, edges: np.ndarray) -> go.Figure:
    """The counts of `pct` in the bins `edges`, less what each choice of `OFFSETS` takes off each candle
    (`offset`), one at a time on buttons above the chart. The bins are the ones of the candles as they are:
    subtracting toward zero never takes a candle further out unless the statistic is beyond the axis."""
    fig = go.Figure()
    for how in OFFSETS:
        counts, _ = np.histogram(pct - offset(pct, stats, how), edges)
        fig.add_trace(
            go.Bar(
                x=(edges[:-1] + edges[1:]) / 2,
                y=counts,
                width=np.diff(edges),
                marker=dict(color="#3498db", line=dict(width=0)),
                customdata=np.c_[edges[:-1], edges[1:]],
                hovertemplate="%{customdata[0]:+.3f}% to %{customdata[1]:+.3f}%<br>%{y:,} candles<extra></extra>",
                visible=how == "none",
            )
        )
    choices(fig, [[h == how for h in OFFSETS] for how in OFFSETS], 1.0)
    fig.update_layout(
        height=460,
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0,  # contiguous bins: a gap would read as a range no candle fell in
        xaxis_title="change from open to close, %",
        yaxis_title="candles",
    )
    return fig


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    """The autocorrelation of `x` at lags 1 to `lags`: sum of x_t x_(t+k) over sum of x_t², both on
    the demeaned series, through the FFT so a million candles take a fraction of a second."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    f = np.fft.rfft(x, 1 << (2 * len(x) - 1).bit_length())  # padded past 2n: no lag wraps onto another
    cov = np.fft.irfft(f * np.conj(f))[: lags + 1]
    return cov[1:] / cov[0]


def clock(pct: pd.Series) -> pd.DataFrame:
    """Mean, standard deviation and count of the change by weekday and UTC hour of the candle's open."""
    return pct.groupby([pct.index.dayofweek, pct.index.hour]).agg(["mean", "std", "count"])


def clock_figure(table: pd.DataFrame, stat: str) -> go.Figure:
    """`clock`'s `stat` as a heatmap, weekdays down and UTC hours across. The mean is drawn on a scale
    centred on zero, and its hover carries the standard error, std / √n, it has to be read against."""
    grid, n = table[stat].unstack(), table["count"].unstack()
    se = (table["std"] / np.sqrt(table["count"])).unstack()
    mean = stat == "mean"
    fig = go.Figure(
        go.Heatmap(
            x=[f"{h:02d}" for h in grid.columns],
            y=[DAYS[d] for d in grid.index],
            z=grid.to_numpy(),
            customdata=np.dstack([n.to_numpy(), se.to_numpy()]),
            colorscale="RdBu" if mean else "Viridis",
            zmid=0 if mean else None,
            colorbar=dict(ticksuffix="%"),
            hovertemplate="%{y} %{x}:00 UTC<br>"
            + ("mean %{z:+.4f}% ± %{customdata[1]:.4f}" if mean else "std %{z:.4f}%")
            + "<br>%{customdata[0]:,} candles<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")  # Monday on top, as a calendar reads
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="hour of the open, UTC")
    return fig


def acf_figure(pct: pd.Series, step: pd.Timedelta) -> go.Figure:
    """The autocorrelation of the change and of its size, lag by lag, over the band ±2/√n that
    independent candles would stay inside; `step` is one candle, for the hover."""
    lags = min(ACF_LAGS, len(pct) // 4)
    k = np.arange(1, lags + 1)
    days = k * (step / pd.Timedelta(days=1))
    fig = go.Figure()
    for name, colour, x in (("change", "#3498db", pct), ("size of the change", "#e67e22", pct.abs())):
        fig.add_trace(
            go.Scatter(
                x=k,
                y=acf(x.to_numpy(), lags),
                name=name,
                line=dict(color=colour, width=1),
                customdata=days,
                hovertemplate=f"{name}<br>lag %{{x:,}} candles, %{{customdata:.2f}} days<br>"
                "autocorrelation %{y:+.4f}<extra></extra>",
            )
        )
    band = 2 / math.sqrt(len(pct))
    for y in (band, -band):
        fig.add_hline(y=y, line=dict(color="#888", dash="dot"))
    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.update_xaxes(type="log", title="lag, candles")
    fig.update_yaxes(title="autocorrelation")
    # The histogram's height and top margin, which holds its buttons: side by side, the two plot areas line up.
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0), legend=dict(x=0.99, xanchor="right", y=0.99))
    return fig


def bands(pct: pd.Series, sigma: float) -> pd.Series:
    """Each candle's band of `SIGMA_STEP` standard deviations `sigma`, closed on the left, from −`SIGMA_REACH`
    to +`SIGMA_REACH` and one open band past each end; a flat candle falls in the one starting at zero."""
    edges = np.arange(-SIGMA_REACH, SIGMA_REACH + SIGMA_STEP / 2, SIGMA_STEP)
    labels = (
        [f"< {edges[0]:+.1f}σ"]
        + [f"{a:+.1f}σ to {b:+.1f}σ" for a, b in zip(edges[:-1], edges[1:])]
        + [f"≥ {edges[-1]:+.1f}σ"]
    )
    return pd.cut(pct / sigma, [-np.inf, *edges, np.inf], right=False, labels=labels)


@st.cache_data(show_spinner="Following every candle N candles ahead…", max_entries=16)
def rsi_bands(close: pd.Series, window: int) -> pd.Series:
    """Each candle's band of `RSI_STEP` points of Wilder's RSI over `window` candles, read at its close, `ta`'s as in
    `features`; NaN until the window fills. Closed on the left, the last band closed on both sides: an RSI of 100,
    a window with no down candle, falls in it."""
    edges = np.arange(0, 100, RSI_STEP)
    labels = [f"{a} to {a + RSI_STEP}" for a in edges]
    return pd.cut(RSIIndicator(close, window=window).rsi(), [*edges, np.inf], right=False, labels=labels)


def quantile_bands(x: pd.Series, q: int = QUANTILES) -> pd.Series:
    """Each candle's `q`-quantile of `x`, named by its edges; NaN where `x` is. An indicator with ties at an edge,
    the age of a window's extreme or a close at the bar's high, has fewer bands: a value is never split."""
    band = pd.qcut(x, q, duplicates="drop")
    return band.cat.rename_categories([f"{i.left:.4g} to {i.right:.4g}" for i in band.cat.categories])


@st.cache_data(show_spinner="Computing the indicators…", max_entries=16)
def indicators(symbol: str, timeframe: str, mtime: float, window: int, project: bool = False) -> pd.DataFrame:
    """`features`' columns on one pair's candles, or `project_features`' with `project`, with `window` as their N;
    `mtime` as in `candles`. Two frames and not one, so a column of either costs only its own module."""
    module = project_features if project else features
    return module.features(candles(symbol, timeframe, mtime), window)


def indicator(symbol: str, timeframe: str, mtime: float, window: int, column: str) -> pd.Series:
    """One column of `COLUMNS`, from whichever module holds it."""
    return indicators(symbol, timeframe, mtime, window, column in project_features.LABELS)[column]


def ahead(band: pd.Series, close: pd.Series, horizons: int, step: pd.Timedelta) -> pd.DataFrame:
    """By band (`bands` or `rsi_bands`; a candle in none counts nowhere) and N from 1 to `horizons`: the mean, over the
    band's candles, of the change in percent from the close to the close N candles later, how many candles have
    one, the mean over every candle (`all`), and the standard error of the band's mean over non-overlapping
    blocks of N candles of `step` on the clock (`se`) with the excess over `all` in it (`t`). A candle whose
    future runs past the store's last bar has none and counts nowhere."""
    rows = []
    for n in range(1, horizons + 1):
        future = (close.shift(-n) / close - 1) * 100
        by = future.groupby(band, observed=True)
        blocks = future.groupby([band, close.index.floor(n * step)], observed=True).mean()
        block = blocks.groupby(level=0, observed=True)
        rows.append(
            pd.DataFrame({"mean": by.mean(), "count": by.count(), "se": block.std() / np.sqrt(block.count())})
            .assign(all=future.mean(), n=n)
            .set_index("n", append=True)
        )
    table = pd.concat(rows)
    return table.assign(t=(table["mean"] - table["all"]) / table["se"])


def ahead_figure(table: pd.DataFrame, stat: str, step: pd.Timedelta) -> go.Figure:
    """`ahead`'s `stat`, `mean` or `t`, as a heatmap: bands up the side, the highest on top, N across.
    Both on a scale centred on zero; the t's is fixed at ±4, so noise reads pale whatever the largest cell."""
    grid = table[stat].unstack()
    n = grid.columns.to_numpy()
    hours = n * (step / pd.Timedelta(hours=1))
    custom = np.dstack(
        [table[c].unstack().to_numpy() for c in ("mean", "se", "all", "t", "count")]
        + [np.broadcast_to(hours, grid.shape)]
    )
    t = stat == "t"
    fig = go.Figure(
        go.Heatmap(
            x=n,
            y=grid.index.astype(str),
            z=grid.to_numpy(),
            customdata=custom,
            colorscale="RdBu",
            zmid=0,
            zmin=-4 if t else None,
            zmax=4 if t else None,
            colorbar=dict(ticksuffix="" if t else "%"),
            hovertemplate="%{y}, N %{x} candles (%{customdata[5]:.2f}h)<br>"
            "mean %{customdata[0]:+.4f}% ± %{customdata[1]:.4f}<br>"
            "every candle %{customdata[2]:+.4f}%, excess %{customdata[3]:+.2f} se<br>"
            "%{customdata[4]:,} candles<extra></extra>",
        )
    )
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="N, candles after the close")
    return fig


@st.cache_data(show_spinner="Reading the zone…", max_entries=16)
def zone_tables(symbol: str, timeframe: str, mtime: float, zone: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`zones.cells` for every band and N of `zone`'s column on one pair, with each band's edges in the column's
    units (`lo`, `hi`), and `zones.trades` at the zone's band and N by period, in the direction its discovery
    excess on this pair points, entered `zones.DELAY` candles late (`mean`, `se`, `win`) and at the next open
    (`now`); `mtime` as in `candles`."""
    _, _, column, window, band, n = zone
    df = candles(symbol, timeframe, mtime)
    x = indicator(symbol, timeframe, mtime, window or EXTREMA_WINDOW, column)
    period = zones.period_of(df.index)
    edges = zones.band_edges(x, period)
    code = zones.bands(x, edges)
    table = zones.cells(code, df.close, period, pd.Timedelta(timeframe.replace("m", "min")))
    names = np.r_[-np.inf, edges, np.inf]
    table["lo"], table["hi"] = names[table.band], names[table.band + 1]
    found = table[(table.period == "discovery") & (table.band == band) & (table.n == n)].t
    sign = float(np.sign(found.iloc[0])) if len(found) and np.isfinite(found.iloc[0]) else 1.0
    rows = []
    for p, name in enumerate(zones.PERIODS):
        hit = (code == band) & (period == p)
        g = zones.trades(hit, df.open, df.close, n, sign)
        now = zones.trades(hit, df.open, df.close, n, sign, delay=0)
        rows.append(
            {
                "period": name,
                "trades": len(g),
                "mean": g.mean() if len(g) else np.nan,
                "se": g.std() / np.sqrt(len(g)) if len(g) > 1 else np.nan,
                "win": (g > 0).mean() if len(g) else np.nan,
                "now": now.mean() if len(now) else np.nan,
            }
        )
    return table, pd.DataFrame(rows)


def zones_figure(tables: dict[str, pd.DataFrame], zone: tuple) -> go.Figure:
    """One small heatmap per pair and period, pairs down and periods across: `zones.cells`' t by band (the
    column's discovery quantiles, the highest on top) and N, on one scale fixed at ±4, the zone's cell outlined."""
    band, n = zone[4], zone[5]
    bands, horizons = range(len(zones.QS) + 1), list(zones.HORIZONS)
    fig = make_subplots(
        rows=len(tables),
        cols=len(zones.PERIODS),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
        column_titles=list(zones.PERIODS),
        row_titles=list(tables),
    )
    for row, (symbol, table) in enumerate(tables.items(), start=1):
        for col, p in enumerate(zones.PERIODS, start=1):
            at = table[table.period == p]
            grid = {
                c: at.pivot(index="band", columns="n", values=c).reindex(index=bands, columns=horizons)
                for c in ("t", "excess", "se", "count", "lo", "hi")
            }
            fig.add_trace(
                go.Heatmap(
                    x=[str(h) for h in horizons],
                    y=[zones.quantiles(b) for b in bands],
                    z=grid["t"].to_numpy(),
                    customdata=np.dstack([grid[c].to_numpy() for c in ("excess", "se", "count", "lo", "hi")]),
                    colorscale="RdBu",
                    zmid=0,
                    zmin=-4,
                    zmax=4,
                    showscale=row == col == 1,
                    colorbar=dict(title="se", len=0.6, x=1.06),
                    hovertemplate=f"{symbol} · {p}<br>band %{{y}}: %{{customdata[3]:.4g}} to %{{customdata[4]:.4g}}"
                    "<br>N %{x}: excess %{customdata[0]:+.4f}% ± %{customdata[1]:.4f}, %{z:+.2f} se"
                    "<br>%{customdata[2]:,} candles<extra></extra>",
                ),
                row=row,
                col=col,
            )
            x, y = horizons.index(n), band
            fig.add_shape(
                type="rect",
                x0=x - 0.5,
                x1=x + 0.5,
                y0=y - 0.5,
                y1=y + 0.5,
                line=dict(color="#f5c518", width=2.5),
                row=row,
                col=col,
            )
    # The N are labels, not a scale: 1 to 192 on a linear axis would squeeze the short ones into a sliver.
    fig.update_xaxes(type="category")
    fig.update_xaxes(title_text="N, candles after the close", row=len(tables))
    fig.update_layout(height=230 * len(tables) + 60, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def zone_summary(tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]], zone: tuple) -> pd.DataFrame:
    """The zone's own cell and its trades, one row per pair and period, as the report's table reads them: gross, won
    and net a candle late, the default, and beside them the gross at the next open, to see what the candle costs."""
    band, n = zone[4], zone[5]
    rows = []
    for symbol, (table, trades) in tables.items():
        for p, t in zip(zones.PERIODS, trades.itertuples()):
            c = table[(table.period == p) & (table.band == band) & (table.n == n)]
            cell = c.iloc[0] if len(c) else None
            gross = f"{t.mean:+.3f}% ± {t.se:.3f}" if t.trades > 1 else "—"
            rows.append(
                {
                    "pair · period": f"{symbol} · {p}",
                    "excess a candle": f"{cell.excess:+.4f}% ({cell.t:+.1f} se)" if cell is not None else "—",
                    "candles": int(cell["count"]) if cell is not None else 0,
                    "trades": t.trades,
                    "gross, a candle late": gross,
                    "won": f"{t.win:.0%}" if t.trades else "—",
                    "net of 0.20%": f"{t.mean - zones.ROUND_TRIP:+.3f}%" if t.trades else "—",
                    "gross at the next open": f"{t.now:+.3f}%" if t.trades else "—",
                }
            )
    return pd.DataFrame(rows).set_index("pair · period")


def pivots(close: pd.Series, window: int) -> pd.Series:
    """TradingVision's `data.pivots.find_pivots` without the amplitude, the pivots `legsweep` swept: +1 on a pivot
    high, −1 on a pivot low, indexed by the pivot's candle. A close strictly above (below) the `window` closes on
    either side of it is a pivot high (low); two of a kind in a row are one leg, and the more extreme of the run is
    kept, so highs and lows alternate. Centred, so hindsight: a pivot is known `window` candles after it."""

    def side(s: pd.Series, how: str) -> pd.Series:  # the extreme of the `window` closes before each, itself left out
        return s.rolling(window).agg(how).shift()

    back = close.iloc[::-1]
    high = (close > side(close, "max")) & (close > side(back, "max").iloc[::-1])
    low = (close < side(close, "min")) & (close < side(back, "min").iloc[::-1])
    kind = pd.Series(np.select([high, low], [1, -1], 0), index=close.index)
    piv = kind[kind != 0]
    if piv.empty:
        return piv
    run = (piv != piv.shift()).cumsum()
    return piv.loc[(close.loc[piv.index] * piv).groupby(run).idxmax()]  # the highest high, the lowest low


def to_pivot(close: pd.Series, piv: pd.Series) -> pd.Series:
    """Each candle's change in percent from its close to the close of the first pivot after it (`pivots`), folded:
    as a long when that pivot is a high, as a short when it is a low, so on either side it is what the leg had
    left to give. A pivot's own candle runs to the next pivot; the candles past the last one have none and are out."""
    at = close.index.get_indexer(piv.index)
    if not len(at):
        return pd.Series(dtype=float)
    i = np.arange(at[-1])
    k = np.searchsorted(at, i, side="right")
    c = close.to_numpy()
    return pd.Series((c[at[k]] / c[i] - 1) * 100 * piv.to_numpy()[k], index=close.index[i])


@st.cache_data(show_spinner="Finding the pivots…", max_entries=8)
def leg_returns(symbol: str, timeframe: str, mtime: float) -> dict[int, dict[str, pd.Series]]:
    """At each of `LEG_WINDOWS`, the two counts of `LEG_COUNTS`: every leg once, its change from pivot to pivot with a
    down leg's sign flipped (`legs_of`), and every candle, what was left to its next pivot (`to_pivot`). The pivots
    alternate, so a flipped down leg is its size. `mtime` as in `candles`."""
    close = candles(symbol, timeframe, mtime).close
    out = {}
    for w in LEG_WINDOWS:
        p = pivots(close, w)
        out[w] = {"pivots": legs_of(close, p).change.abs() if len(p) else pd.Series(dtype=float)}
        out[w]["candles"] = to_pivot(close, p)
    return out


def legs_figure(
    legs: dict[int, dict[str, pd.Series]], width: float = LEG_BIN, view: float = LEG_VIEW, tail: float = TAIL
) -> go.Figure:
    """One histogram per window of `leg_returns`, two by two, each on its own axes, in bins `width` wide with an edge
    on zero, one count of `LEG_COUNTS` at a time on buttons above them: the pivots first. The bins run from zero, or
    below it if some candles sit beyond the pivot their leg ends on, to the `1 - tail` quantile or `view`, whichever
    is further; the axis opens on 0 to `view`, and a zoom out or a double click shows the rest. The titles say what
    that opening view leaves out on either side, and change with the button."""
    unit = {"pivots": "legs", "candles": "candles"}
    titles = {
        how: [
            (
                f"<b>{w} candles</b>: {len(r):,} {unit[how]}, median {r.median():.2f}%, mean {r.mean():.2f}%"
                f"<br>{(r > view).mean():.1%} beyond {view:g}%, {(r < 0).mean():.2%} below 0%"
                if len(r)
                else f"{w} candles: no pivot"
            )
            for w, r in ((w, c[how]) for w, c in legs.items())
        ]
        for how in LEG_COUNTS
    }
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles["pivots"], vertical_spacing=0.14, horizontal_spacing=0.06)
    drawn = []  # the count of each trace, in the order they are added: what a button shows
    for how in LEG_COUNTS:
        for j, r in enumerate(c[how] for c in legs.values()):
            if not len(r):
                continue
            lo, hi = min(0.0, r.quantile(tail)), max(view, r.quantile(1 - tail))
            edges = np.arange(np.floor(lo / width), np.ceil(hi / width) + 1) * width
            counts, _ = np.histogram(r, edges)
            fig.add_trace(
                go.Bar(
                    x=(edges[:-1] + edges[1:]) / 2,
                    y=counts,
                    width=np.diff(edges),
                    marker=dict(color="#3498db", line=dict(width=0)),
                    customdata=np.c_[edges[:-1], edges[1:]],
                    hovertemplate="%{customdata[0]:+.2f}% to %{customdata[1]:+.2f}%<br>%{y:,} "
                    + unit[how]
                    + "<extra></extra>",
                    visible=how == "pivots",
                ),
                row=j // 2 + 1,
                col=j % 2 + 1,
            )
            drawn.append(how)
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                active=0,
                buttons=[
                    dict(
                        label=label,
                        method="update",
                        args=[
                            {"visible": [d == how for d in drawn]},
                            {f"annotations[{i}].text": t for i, t in enumerate(titles[how])}
                            | {"yaxis.title.text": unit[how], "yaxis3.title.text": unit[how]},
                        ],
                    )
                    for how, label in LEG_COUNTS.items()
                ],
                x=0,
                xanchor="left",
                y=1.0,
                yanchor="bottom",
                pad=dict(b=36),
            )
        ]
    )
    fig.update_xaxes(ticksuffix="%", range=[0, view])
    fig.update_yaxes(title_text="legs", col=1)
    fig.update_layout(height=820, margin=dict(l=0, r=0, t=100, b=0), bargap=0, showlegend=False)
    fig.update_annotations(font_size=13)
    return fig


def legs_of(close: pd.Series, piv: pd.Series) -> pd.DataFrame:
    """One row per leg between two `pivots`: where it starts and ends (positions in `close`), and its change from
    the first pivot's close to the second's in percent, signed: positive from a low, negative from a high."""
    at = close.index.get_indexer(piv.index)
    c = close.to_numpy()
    return pd.DataFrame({"start": at[:-1], "end": at[1:], "change": (c[at[1:]] / c[at[:-1]] - 1) * 100})


@st.cache_data(show_spinner="Banding the indicators…", max_entries=8)
def decile_codes(symbol: str, timeframe: str, mtime: float, window: int) -> pd.DataFrame:
    """Every column of `COLUMNS` at `window` in `QUANTILES` quantiles over every candle, as codes from 0 (the lowest)
    up; fewer on a column with ties at an edge, NaN where the column is; `mtime` as in `candles`."""
    x = pd.concat([indicators(symbol, timeframe, mtime, window, p) for p in (False, True)], axis=1)[list(COLUMNS)]
    return x.apply(lambda s: pd.qcut(s, QUANTILES, labels=False, duplicates="drop")).astype("float32")


def excess(group: np.ndarray, base: np.ndarray, q: int = QUANTILES) -> dict[str, np.ndarray]:
    """Per column of `group` and `base` (rows of band codes, NaN where none) and per band 0 to `q - 1`: the share of
    the group's rows in the band, the base's, how many group rows have a code, and the excess in standard errors of
    a share drawn from the base, √(b(1 − b)/n). A band nobody falls in has no standard error and stays NaN."""
    n = (~np.isnan(group)).sum(0)
    share = np.stack([(group == k).sum(0) for k in range(q)], -1) / np.maximum(n, 1)[:, None]
    b = np.stack([(base == k).sum(0) for k in range(q)], -1) / np.maximum((~np.isnan(base)).sum(0), 1)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (share - b) / np.sqrt(b * (1 - b) / n[:, None])
    full = np.broadcast_to
    return {
        "share": share,
        "base": full(b, share.shape),
        "z": np.where(b > 0, z, np.nan),
        "n": full(n[:, None], share.shape),
    }


def top(*cells: dict[str, np.ndarray]) -> float:
    """The colour scale's end for `excess_figure`, shared by the charts it is given: their largest share, in %."""
    return max(float(np.max(c["share"], initial=0)) for c in cells) * 100 or 100.0


def excess_figure(
    cells: dict[str, np.ndarray],
    x: list,
    y: list,
    xtitle: str,
    reference: str,
    zmax: float = 100.0,
    text: bool = False,
    xnames: list[str] | None = None,
) -> go.Figure:
    """`excess`' share of the pivots in each band as a heatmap from 0, none, to `zmax` %. With `text` the share is
    written in every cell, marked ▲ or ▼ when it is more than `EXCESS_MARK` standard errors above or below the share
    of `reference` (`excess`' base, named as the hover names it) in the same band. A band nobody falls in is 0, the
    ones an indicator does not have (ties at an edge) too. The hover is written here, in words, with `xnames` naming
    the columns where `x` alone would not say what they are. `cells` rows are `y`."""
    share, base, z, n = cells["share"] * 100, cells["base"] * 100, cells["z"], cells["n"]
    mark = np.select([z > EXCESS_MARK, z < -EXCESS_MARK], ["▲", "▼"], "")
    names = xnames or [str(v) for v in x]
    hover = [
        [
            f"<b>{y[i]}, {names[j]}</b><br>"
            f"slice: {round(share[i, j] * n[i, j] / 100):,} of its {n[i, j]:,} pivots are here, {share[i, j]:.1f}%<br>"
            f"{reference}: {base[i, j]:.1f}% are here<br>"
            + (
                f"difference {share[i, j] - base[i, j]:+.1f} points, {z[i, j]:+.1f} standard errors"
                if np.isfinite(z[i, j])
                else "no standard error: the reference has none here"
            )
            for j in range(share.shape[1])
        ]
        for i in range(share.shape[0])
    ]
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=share,
            colorscale="Blues",
            zmin=0,
            zmax=zmax,
            colorbar=dict(ticksuffix="%"),
            text=np.char.add(np.vectorize("{:.0f}".format)(share), mark) if text else None,
            texttemplate="%{text}" if text else None,
            textfont=dict(size=10),
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(height=max(360, 18 * len(y)), margin=dict(l=0, r=0, t=10, b=0), xaxis_title=xtitle)
    return fig


def main() -> None:
    st.set_page_config(page_title="Decomposer", layout="wide", initial_sidebar_state="collapsed")
    st.title("Decomposer")

    where = store()
    held = pairs(where)
    if not held:
        st.info(f"No 5m candles under `{where}`. Fill the store first: `python -m cryptofarm.data.klines --update`.")
        return
    left, right = st.columns(2)
    symbol = left.selectbox("Asset", held, help="the pairs with a file in the store; nothing is downloaded")
    timeframe = right.selectbox(
        "Timeframe",
        TIMEFRAMES,
        index=TIMEFRAMES.index(TIMEFRAME),
        help="the store holds 5m candles; the longer ones are aggregated from them",
    )
    mtime = (where / f"{symbol}USDT-5m.parquet").stat().st_mtime
    df = candles(symbol, timeframe, mtime)
    pct = change(df)
    st.caption(
        f"{len(df):,} {timeframe} candles of {symbol}USDT on Binance, {df.index[0]:%Y-%m-%d} to "
        f"{df.index[-1]:%Y-%m-%d}. The numbers and the histogram read all of them; the candle charts read "
        "the period picked above them."
    )

    stats = sides(pct)
    for column, (side, title) in zip(st.columns(len(SIDES)), SIDES.items()):
        column.metric(f"{title}: mean", f"{stats.loc[side, 'mean']:+.4f}%")
        column.metric(f"{title}: standard deviation", f"{stats.loc[side, 'std']:.4f}%")
    flat = int((pct == 0).sum())
    st.caption(
        "Change of a candle: close / open − 1. The mean of all candles is the one near zero; their standard "
        f"deviation is the size of a typical candle. {flat:,} candles ({flat / len(pct):.1%}) closed where they "
        "opened: they count in all candles and in neither side."
    )

    st.subheader("Candles")
    first, last = df.index[0].date(), df.index[-1].date()
    period = st.date_input(
        "Period",
        value=(max(first, last - PERIOD.to_pytimedelta()), last),
        min_value=first,
        max_value=last,
        help="only the two candle charts read it, to keep the drawing light: the numbers and the histogram read every "
        "candle",
    )
    # Mid-pick the widget holds the start alone; the charts run to the last candle until the end is picked.
    start, end = period if len(period) == 2 else (period[0], last)
    window = df.loc[str(start) : str(end)]
    view = f"{symbol}-{timeframe}-{start}-{end}"
    st.markdown("**Under the candles, the same candles each from its own open**")
    st.caption(
        "The body alone, as the % change from the candle's own open: every candle starts at zero and its body is "
        "its change. The buttons above it subtract a statistic from "
        "each candle's body: the open stays at zero, a 1% candle less 0.2% closes at 0.8%. Up / down: an up candle "
        "loses the up candles' statistic, a down candle the down candles'. A candle smaller than it crosses zero and "
        "keeps its colour. The statistics are the ones at the top, over every candle loaded."
    )
    minus = {how: offset(pct, stats, how).loc[window.index] for how in OFFSETS}
    st.plotly_chart(candles_figure(window, view, minus), use_container_width=True, key="candles")

    left, right = st.columns(2)
    left.subheader("Distribution of the candles' change")
    counts, edges, beyond = histogram(pct)
    left.plotly_chart(histogram_figure(pct, stats, edges), use_container_width=True, key="histogram")
    left.caption(
        f"Every candle loaded, in {BINS} bins of {edges[1] - edges[0]:.4f}% over ±{edges[-1]:.3f}%, the 0.1th or "
        f"99.9th percentile, whichever is further from zero. {beyond:,} candles ({beyond / len(pct):.2%}) lie "
        "beyond it and are not drawn. The buttons subtract from each candle what they subtract in the chart above."
    )

    right.subheader("Memory")
    step = pd.Timedelta(timeframe.replace("m", "min"))
    right.plotly_chart(acf_figure(pct, step), use_container_width=True, key="acf")
    right.caption(
        "The correlation of each candle with the one k candles later, for the change (its sign) and for its size "
        "(its absolute value), every candle loaded. Dotted, ±2/√n, where independent candles would stay 95% of the "
        "time; candles whose size clusters are not independent and the real band is wider, so a change "
        "autocorrelation just outside it is no evidence. Near zero for the change, positive and slow to decay for "
        "the size: the sign of the next candle is not in this one, its size is. A bump every day of lags is the "
        "volatility's daily cycle."
    )

    st.subheader("Time of day and week")
    table = clock(pct)
    left, right = st.columns(2)
    left.markdown("**Mean of the change**")
    left.plotly_chart(clock_figure(table, "mean"), use_container_width=True, key="clock-mean")
    right.markdown("**Standard deviation of the change**")
    right.plotly_chart(clock_figure(table, "std"), use_container_width=True, key="clock-std")
    se = float((table["std"] / np.sqrt(table["count"])).median())
    st.caption(
        f"Every candle loaded, by the weekday and the UTC hour it opened in: {int(table['count'].median()):,} "
        f"candles a cell. A cell's mean has a standard error of about {se:.4f}%, std / √n, and more than that "
        "because sizes cluster: a mean within two of it of zero is no evidence, and among 168 cells a few lie "
        "beyond by chance. The standard deviation is the cycle the autocorrelation's bumps come from."
    )

    st.subheader("The future of a candle")
    horizons = st.slider(
        "N max, candles",
        1,
        HORIZON_MAX,
        HORIZON,
        help="each N from 1 to this one is a column; the first run of a pair and N max takes a few seconds",
    )
    table = ahead(bands(pct, stats.loc["all", "std"]), df.close, horizons, step)
    left, right = st.columns(2)
    left.markdown("**Mean change from the close to N candles later**")
    left.plotly_chart(ahead_figure(table, "mean", step), use_container_width=True, key="ahead-mean")
    right.markdown("**Its excess over every candle's, in standard errors**")
    right.plotly_chart(ahead_figure(table, "t", step), use_container_width=True, key="ahead-t")
    count = table["count"].xs(1, level="n")
    st.caption(
        "Every candle loaded, banded by its change in standard deviations of all candles "
        f"({stats.loc['all', 'std']:.4f}%), "
        f"from {count.min():,} candles in the thinnest band to {count.max():,} in the fullest. The mean grows with N "
        "because the drift does; the right chart takes off the mean of every candle at the same N, so a band "
        "coloured there is one that did differently. Its standard error is over non-overlapping blocks of N "
        "candles, since neighbouring candles share most of their future; among this many cells a few pass ±3 by "
        "chance, and a run of N in a row past it is one result, not many."
    )

    st.subheader("The future after an RSI")
    window = st.number_input("RSI window, candles", 2, 500, RSI_WINDOW, help="Wilder's RSI, as `features` reads it")
    table = ahead(rsi_bands(df.close, window), df.close, horizons, step)
    left, right = st.columns(2)
    left.markdown("**Mean change from the close to N candles later**")
    left.plotly_chart(ahead_figure(table, "mean", step), use_container_width=True, key="rsi-mean")
    right.markdown("**Its excess over every candle's, in standard errors**")
    right.plotly_chart(ahead_figure(table, "t", step), use_container_width=True, key="rsi-t")
    count = table["count"].xs(1, level="n")
    st.caption(
        f"The same as above with the candles banded by their RSI at the close, {RSI_STEP} points a band, and the "
        f"same N max: from {count.min():,} candles in the thinnest band to {count.max():,} in the fullest. Overbought "
        "and oversold, 70 and 30, are band edges."
    )

    st.subheader("The future after an indicator")
    left, right = st.columns(2)
    column = left.selectbox(
        "Indicator",
        COLUMNS,
        format_func=LABELS.get,
        help="the 27 columns of `features` and `project_features` that held in at least one study: TradingVision's, "
        "then this project's",
    )
    n = right.number_input(
        "Indicator window, candles", 2, 500, EXTREMA_WINDOW, help="`features`' N: every window of every column"
    )
    table = ahead(quantile_bands(indicator(symbol, timeframe, mtime, n, column)), df.close, horizons, step)
    left, right = st.columns(2)
    left.markdown("**Mean change from the close to N candles later**")
    left.plotly_chart(ahead_figure(table, "mean", step), use_container_width=True, key="indicator-mean")
    right.markdown("**Its excess over every candle's, in standard errors**")
    right.plotly_chart(ahead_figure(table, "t", step), use_container_width=True, key="indicator-t")
    st.caption(
        f"The same again with the candles banded by the indicator at their close, in {QUANTILES} quantiles: as many "
        "candles in each band, named by its edges in the indicator's own units. An indicator with ties at an edge "
        "has fewer bands. A decile is wide and dilutes a tail: RSI's top decile reads weaker than its 80 to 90 band "
        "above."
    )

    st.subheader("What is left of the leg")
    st.plotly_chart(legs_figure(leg_returns(symbol, timeframe, mtime)), use_container_width=True, key="legs")
    st.caption(
        f"TradingVision's pivots at the windows {', '.join(map(str, LEG_WINDOWS))} in {timeframe} candles (a close "
        "beyond the window's on both sides, the more extreme of two of a kind in a row), over every candle loaded. "
        "Two counts on the buttons. Pivots: every leg once, its change from one pivot to the next, the size of the "
        "slice below. Candles: every candle, its change from its close to the next pivot's — what the leg had left — "
        "so a long leg counts once a candle, and the mass near zero is the candles just before a pivot. Both long "
        "toward a high and short toward a low with the sign flipped, in one histogram. Hindsight, a pivot is known a "
        "window after it: the most a perfect reader could take, before the 0.20% round trip."
    )

    st.subheader("Pivots of a size")
    cols = st.columns(4)
    w = cols[0].selectbox("Pivot window, candles", LEG_WINDOWS, index=LEG_WINDOWS.index(EXTREMA_WINDOW))
    lo = cols[1].number_input("Leg from, %", 0.0, 100.0, LEG_SLICE[0], 0.1, help="in size: up and down legs alike")
    hi = cols[2].number_input("Leg to, %", 0.0, 100.0, LEG_SLICE[1], 0.1, help="excluded")
    base = cols[3].radio(
        "Compare the slice with",
        list(BASES),
        format_func=BASES.get,
        horizontal=True,
        help="what ▲ and ▼ measure the slice's share against: every candle loaded, or every pivot that starts a leg "
        "of the same side whatever its size, so that what is left is what this size has of its own",
    )
    legs = legs_of(df.close, pivots(df.close, w))
    codes = decile_codes(symbol, timeframe, mtime, w)
    v = codes.to_numpy()
    deciles = [f"D{k + 1}" for k in range(QUANTILES)]
    same = {s: legs[np.sign(legs.change) == (1 if s == "up" else -1)] for s in SIDES_OF_LEG}
    cut = {s: g[g.change.abs().between(lo, hi, inclusive="left")].start.to_numpy() for s, g in same.items()}
    st.markdown("**Where the indicators stand on the pivot**")
    cells = {s: excess(v[cut[s]], v if base == "candles" else v[same[s].start.to_numpy()]) for s in SIDES_OF_LEG}
    zmax = top(*cells.values())
    for column, (s, title) in zip(st.columns(2), SIDES_OF_LEG.items()):
        fig = excess_figure(
            cells[s], deciles, [LABELS[c] for c in COLUMNS], "decile over every candle", REFERENCES[base][s], zmax, True
        )
        column.markdown(f"{title}: {len(cut[s]):,} pivots")
        column.plotly_chart(fig.update_yaxes(autorange="reversed"), use_container_width=True, key=f"pivot-{s}")
    st.caption(
        f"Every indicator of the menus at window {w}, read at the close of the pivot the leg starts on and banded in "
        "deciles over every candle loaded, D1 the lowest. The number in a cell is the % of the slice's pivots in that "
        "decile, and its colour: white is none, a row adds up to 100, and the scale ends at the largest share of the "
        f"two charts, {zmax:.0f}%. ▲ and ▼ mark a share more than {EXCESS_MARK:g} standard errors above or below the "
        f"share of {BASES[base]} in that decile, √(b(1 − b)/n); the hover has both, in numbers. A 0 can also be a "
        "decile the indicator does not have (ties at an edge). Compared with every candle a low sits low on any "
        "oscillator by construction; compared with every pivot of the same kind what is left "
        "is what this size has of its own. Hindsight: the pivot is known a window after it. Among "
        f"{2 * len(COLUMNS) * QUANTILES} cells a few are marked by chance."
    )

    column = st.selectbox("Indicator, candle by candle before the pivot", COLUMNS, format_func=LABELS.get)
    x = v[:, COLUMNS.index(column)]
    lags = np.arange(LEG_LOOKBACK + 1)

    def back(
        at: np.ndarray,
    ) -> np.ndarray:  # the column at each of `at` and the candles before it, NaN before the store
        i = at[:, None] - lags
        return np.where(i >= 0, x[np.maximum(i, 0)], np.nan)

    cells = {
        s: {
            k: a.T
            for k, a in excess(back(cut[s]), x[:, None] if base == "candles" else back(g.start.to_numpy())).items()
        }
        for s, g in same.items()
    }
    zmax = top(*cells.values())
    for col, (s, title) in zip(st.columns(2), SIDES_OF_LEG.items()):
        col.markdown(title)
        col.plotly_chart(
            excess_figure(
                cells[s],
                -lags,
                deciles,
                "candles from the pivot",
                REFERENCES[base][s],
                zmax,
                xnames=["the pivot"] + [f"{k} candles before the pivot" for k in lags[1:]],
            ),
            use_container_width=True,
        )
    st.caption(
        "The same for one indicator, decile against the candles before the pivot: at 0 the column of the chart above, "
        f"to the left how the condition builds. The scale ends at {zmax:.0f}%, the largest share of the two charts."
    )

    st.subheader("The zones of the study")
    picked = st.selectbox(
        "Zone",
        range(len(zones.ALL_ZONES)),
        format_func=lambda i: zones.label(zones.ALL_ZONES[i]),
        help="family · timeframe · column (window) · band in 2017-2020 quantiles · N, from both studies: "
        "TradingVision's 29 columns (`reports/zones_study.html`) and this project's 13 "
        "(`reports/zones_study_project.html`); " + "; ".join(f"{k} {v}" for k, v in zones.FAMILIES.items()),
    )
    zone = zones.ALL_ZONES[picked]
    have = [s for s in zones.SYMBOLS if s in held]
    if not have:
        st.info(f"Neither {' nor '.join(zones.SYMBOLS)} is in the store.")
        return
    read = {s: zone_tables(s, zone[1], (where / f"{s}USDT-5m.parquet").stat().st_mtime, zone) for s in have}
    st.plotly_chart(zones_figure({s: r[0] for s, r in read.items()}, zone), use_container_width=True, key="zones")
    st.dataframe(zone_summary(read, zone), use_container_width=True)
    st.caption(
        f"{zone[1]} candles of {' and '.join(have)}, whatever the timeframe picked at the top. Each band is the "
        "column's quantiles on discovery (to 2020-12), the same edges applied to confirm (2021-01 to 2025-06) and "
        "recent; each cell the excess of the mean change N candles after the close over every candle's of the same "
        "period, in standard errors over blocks of N candles. Outlined, the zone: chosen beyond 4 se on both pairs on "
        "discovery, and held at 2 on confirm unless its family says it does not. Under it one trade an episode: read "
        "at the close of the first candle in the band, opened a candle late — at the open of the second candle "
        "after it — held N, gross, then less the 0.20% round trip; the last column enters at the next open "
        "instead, without the candle of delay."
    )


def _selfcheck() -> None:
    """The arithmetic on hand-made candles, and the property the docstring quotes."""
    df = pd.DataFrame({"open": [100.0, 100.0, 100.0, 100.0, 100.0], "close": [101.0, 99.0, 102.0, 98.0, 100.0]})
    pct = change(df)
    assert np.allclose(pct, [1.0, -1.0, 2.0, -2.0, 0.0])
    s = sides(pct)
    assert np.isclose(s.loc["all", "mean"], 0.0) and np.isclose(s.loc["all", "std"], np.sqrt(2.5))
    # A flat candle is in neither side: two up, two down.
    assert np.isclose(s.loc["up", "mean"], 1.5) and np.isclose(s.loc["down", "mean"], -1.5)
    assert np.isclose(s.loc["up", "std"], np.sqrt(0.5)) and np.isclose(s.loc["down", "std"], np.sqrt(0.5))
    # From zero: every candle opens at 0, closes at its change, and keeps its wicks in percent.
    ohlc = df.assign(high=[103.0, 100.5, 102.0, 100.0, 101.0], low=[99.5, 97.0, 100.0, 96.0, 99.0])
    zero = from_zero(ohlc)
    assert (zero.open == 0).all() and np.allclose(zero.close, pct)
    assert np.allclose(zero.high, [3.0, 0.5, 2.0, 0.0, 1.0]) and np.allclose(zero.low, [-0.5, -3.0, 0.0, -4.0, -1.0])
    # The offsets: toward zero on either side, nothing on a flat candle.
    s = sides(pct)
    assert np.allclose(offset(pct, s, "none"), 0.0)
    assert np.allclose(offset(pct, s, "std"), np.array([1, -1, 1, -1, 0]) * np.sqrt(2.5))
    assert np.allclose(offset(pct, s, "side std"), np.array([1, -1, 1, -1, 0]) * np.sqrt(0.5))
    assert np.allclose(pct - offset(pct, s, "side mean"), [-0.5, 0.5, 0.5, -0.5, 0.0]), "beyond its side's mean"
    # Less: the body shrinks by the offset, the open stays, and the wicks go: high and low are the body's ends.
    cut = less(zero, pd.Series([0.2, -0.2, 2.5, -0.2, 0.0], index=zero.index))
    assert (cut.open == 0).all() and np.allclose(cut.close, [0.8, -0.8, -0.5, -1.8, 0.0])
    assert np.allclose(cut.high, [0.8, 0.0, 0.0, 0.0, 0.0]) and np.allclose(cut.low, [0.0, -0.8, -0.5, -1.8, 0.0])

    # Normal changes: the mean of all is near zero and the standard deviation is not, and each side's
    # standard deviation is sqrt(1 - 2/pi) of the whole.
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(0.0, 0.3, 400_000))
    s = sides(normal)
    assert abs(s.loc["all", "mean"]) < 0.002 and abs(s.loc["all", "std"] - 0.3) < 0.002
    for side in ("up", "down"):
        assert abs(s.loc[side, "std"] / s.loc["all", "std"] - np.sqrt(1 - 2 / np.pi)) < 0.005
    assert np.isclose(s.loc["up", "mean"], -s.loc["down", "mean"], rtol=0.01)

    counts, edges, beyond = histogram(normal)
    assert len(counts) == BINS and np.isclose(edges[0], -edges[-1]), "symmetric around zero"
    assert np.isclose(edges[BINS // 2] + edges[BINS // 2 + 1], 0.0), "one bin centred on zero"
    assert counts.sum() + beyond == len(normal) and beyond <= 2 * TAIL * len(normal) + 1
    assert counts.argmax() in range(BINS // 2 - 5, BINS // 2 + 6), "the bell peaks at zero"
    # All candles flat: a degenerate axis is widened rather than handed to numpy with equal edges.
    counts, edges, beyond = histogram(pd.Series(np.zeros(10)))
    assert counts.sum() == 10 and beyond == 0

    # Clock: one cell per weekday and hour; 2024-01-01 is a Monday, so Monday 14:00 holds 14 twice.
    hours = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
    table = clock(pd.Series(hours.hour.astype(float), index=hours))
    assert len(table) == 7 * 24 and (table["count"] == 2).all() and (table["std"] == 0).all()
    assert table.loc[(0, 14), "mean"] == 14 and clock_figure(table, "mean").data[0].z.shape == (7, 24)
    # Autocorrelation: equal to the sum it stands for, near zero on independent draws and φ^k on an AR(1).
    x = rng.normal(size=5_000)
    d = x - x.mean()
    assert np.allclose(acf(x, 3), [(d[:-k] * d[k:]).sum() / (d * d).sum() for k in (1, 2, 3)])
    assert np.abs(acf(normal.to_numpy(), 50)).max() < 4 / np.sqrt(len(normal))
    ar = np.zeros(200_000)
    for t, e in enumerate(rng.normal(size=len(ar) - 1), start=1):
        ar[t] = 0.5 * ar[t - 1] + e
    assert np.allclose(acf(ar, 3), [0.5, 0.25, 0.125], atol=0.01)

    # Bands: closed on the left, open past ±3σ, a flat candle in the one starting at zero.
    b = bands(pd.Series([-9.0, -3.0, -0.1, 0.0, 0.49, 0.5, 3.0]), 1.0)
    assert list(b) == ["< -3.0σ", "-3.0σ to -2.5σ", "-0.5σ to +0.0σ", "+0.0σ to +0.5σ", "+0.0σ to +0.5σ"] + [
        "+0.5σ to +1.0σ",
        "≥ +3.0σ",
    ]
    assert len(b.cat.categories) == 2 * SIGMA_REACH / SIGMA_STEP + 2
    # Ahead: a close rising 10% a candle is 1.1^N − 1 ahead from every candle that has a future, and none past the end.
    when = pd.date_range("2024-01-01", periods=40, freq="15min", tz="UTC")
    close = pd.Series(100 * 1.1 ** np.arange(40), index=when)
    pct = pd.Series(rng.normal(size=40), index=when)
    table = ahead(bands(pct, pct.std()), close, 3, pd.Timedelta("15min"))
    for n in (1, 2, 3):
        at = table.xs(n, level="n")
        assert np.allclose(at["mean"], (1.1**n - 1) * 100) and at["count"].sum() == 40 - n
        assert np.isclose(at["all"].iloc[0], (1.1**n - 1) * 100)
    assert ahead_figure(table, "t", pd.Timedelta("15min")).data[0].z.shape[1] == 3
    # RSI bands: NaN until the window fills, 100 on a close that only rose, 0 on one that only fell.
    r = rsi_bands(close, 14)
    assert r.iloc[:13].isna().all() and (r.iloc[13:] == "90 to 100").all()
    assert (rsi_bands(close[::-1].set_axis(when), 14).iloc[13:] == "0 to 10").all()
    assert len(r.cat.categories) == 100 // RSI_STEP
    # The menu: 27 columns, none dropped, every column of a zone that holds still in it (the momentum's need not be).
    assert len(COLUMNS) == 27 and not set(DROPPED) & set(COLUMNS)
    assert set(DROPPED) <= set(features.COLUMNS + project_features.COLUMNS), "a dropped name that is no column"
    assert {z[2] for z in zones.ALL_ZONES if z[0] != "m"} <= set(COLUMNS)
    # Quantile bands: as many in each, NaN kept, and ties never split across an edge.
    q = quantile_bands(pd.Series([np.nan, *range(100)]))
    assert q.isna().iloc[0] and (q.value_counts() == 10).all() and len(q.cat.categories) == QUANTILES
    q = quantile_bands(pd.Series([0.0] * 50 + list(range(1, 51))))
    assert q.iloc[:50].nunique() == 1 and len(q.cat.categories) < QUANTILES

    # Pivots, on TradingVision's triangle wave: peaks at 110 every 20 candles, troughs at 100 halfway between.
    wave = pd.Series(np.abs(np.arange(200) % 20 - 10.0) + 100.0)
    p = pivots(wave, 5)
    assert p.diff().dropna().abs().eq(2).all() and p.index.to_series().diff().dropna().eq(10).all()
    assert pivots(wave.iloc[:3], 5).empty and to_pivot(wave.iloc[:3], pivots(wave.iloc[:3], 5)).empty
    # To the pivot: a trough runs +10% to the peak, a peak 1/11 short to the trough, the candle before a peak 1/109.
    r = to_pivot(wave, p)
    assert len(r) == p.index[-1] and (r >= 0).all()
    assert np.allclose(r.loc[p.index[:-1]], np.where(p.iloc[:-1] < 0, 10.0, 100 / 11))
    assert np.isclose(r.loc[p.index[p > 0][1] - 1], 100 / 109)
    sizes = legs_of(wave, p).change.abs()
    empty = pd.Series(dtype=float)
    fig = legs_figure(
        {w: {"pivots": sizes, "candles": r} for w in LEG_WINDOWS[:3]} | {48: {"pivots": empty, "candles": empty}}
    )
    assert len(fig.data) == 6 and "no pivot" in fig.layout.annotations[3].text
    # The pivots open the figure, each leg once; the button swaps the traces and the titles with them.
    assert [t.visible for t in fig.data] == [True] * 3 + [False] * 3 and fig.data[0].y.sum() == len(sizes)
    pivots_button, candles_button = fig.layout.updatemenus[0].buttons
    assert candles_button.args[0]["visible"] == [False] * 3 + [True] * 3 and fig.data[3].y.sum() == len(r)
    assert "legs" in fig.layout.annotations[0].text and "candles," in candles_button.args[1]["annotations[0].text"]
    # Bins of LEG_BIN with an edge on zero, out to the larger of the view and the data; the axis opens on 0 to LEG_VIEW.
    bar = fig.data[3]
    assert np.allclose(bar.width, LEG_BIN) and np.isclose(bar.customdata[:, 0], 0).any()
    assert bar.customdata[-1, 1] >= LEG_VIEW and tuple(fig.layout.xaxis.range) == (0, LEG_VIEW)

    # Legs: on the wave every leg runs a trough to a peak, +10%, or a peak to a trough, −1/11.
    legs = legs_of(wave, p)
    assert np.allclose(legs.change, np.where(p.iloc[:-1] < 0, 10.0, -100 / 11)) and (legs.end - legs.start == 10).all()
    # Excess: 2 of 4 in band 0 against a base with a quarter there is +0.25 over √(0.25·0.75/4) = 1.155 se.
    e = excess(np.array([[0.0], [0.0], [1.0], [np.nan], [2.0]]), np.array([[0.0], [1.0], [2.0], [3.0]]), q=4)
    assert np.allclose(e["share"][0], [0.5, 0.25, 0.25, 0.0]) and e["n"][0, 0] == 4
    assert np.isclose(e["z"][0, 0], 0.25 / np.sqrt(0.25 * 0.75 / 4)) and np.isclose(
        e["z"][0, 3], -0.25 / np.sqrt(0.25 * 0.75 / 4)
    )
    # One base column against many group columns, as the candle-by-candle chart reads it.
    assert excess(np.zeros((5, 3)), np.array([[0.0], [1.0]]), q=2)["base"].shape == (3, 2)
    # The scale: the largest |z| of the charts given, never under 4, NaN cells ignored.
    assert np.isclose(top(e), 50.0) and top({"share": np.zeros((1, 2))}) == 100.0
    # The figure: the share coloured from 0 with no gap, marked against the reference, the hover in words.
    t = excess_figure(e, ["a", "b", "c", "d"], ["x"], "", "every candle", text=True).data[0]
    assert list(t.text[0]) == ["50▲" if e["z"][0, 0] > EXCESS_MARK else "50", "25", "25", "0"] and t.zmin == 0
    assert np.isclose(t.z[0][0], 50) and not np.isnan(np.asarray(t.z, float)).any()
    assert "2 of its 4 pivots are here, 50.0%" in t.hovertext[0][0] and "every candle: 25.0%" in t.hovertext[0][0]
    t = excess_figure(
        {k: v[:, :2] for k, v in e.items()} | {"z": np.array([[3.5, np.nan]]), "base": np.array([[0.1, 0.0]])},
        ["a", "b"],
        ["x"],
        "",
        "every low",
        text=True,
    ).data[0]
    assert list(t.text[0]) == ["50▲", "25"] and "+3.5 standard errors" in t.hovertext[0][0]
    assert "no standard error" in t.hovertext[0][1]


if __name__ == "__main__":
    main()
