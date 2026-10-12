"""The swing structure of the close with no window, read live, and Wyckoff's last point of support on it.

**The hierarchy.** Level 0 is every turn of the close: each bar where it changes direction, a run of
equal closes counting once. A high climbs to level k+1 when it is higher than the highs on either side
of it at level k, a low when it is lower than the lows: Dow's higher highs and lower lows applied to
themselves, until nothing climbs. Two highs left in a row at a level get the lowest low between them
from the level below, so every level alternates. No window and no threshold: the levels are the scales
the market itself made, each one's legs about 3.2 times the length of the one below. Measured on the
fifteen `klines.DEFAULT_SYMBOLS` from 2017, the median leg is 5 / 18 / 58 / 185 bars at levels 1-4 on
15m and 5 / 18 / 57 / 175 on 1h, so `HORIZON` is one table for both. It is the same hierarchy as
`decomposer.swing_levels`, dated live.

**Live, by construction.** A turn reaches level k+1 once the next turn of its kind at level k is
decided at level k, so `climb` dates every level of every turn from the bar it became final on, and
nothing after that bar ever changes it: rerunning the hierarchy on every prefix of a series agrees with
those dates on every decision (`tests/test_swing_structure.py`, and 252,880 decisions on 1,500 bars of
BTC 15m). The price is the lag: at levels 2-4 the confirmation comes 19 / 63 / 199 bars after the
turn, longer than the leg itself, because Dow's rule confirms a high only with the lower high that
follows it. A threshold zigzag on the same legs confirms in a quarter of the bars at the same cost in
price: about 60% of the next leg is gone by the confirmation either way.

**The rule (`posizioni`).** Wyckoff's last point of support: at level k, the bar a higher low is
confirmed in an up structure (the last two highs and the last two lows rising) goes long; the trade
closes at the first close under that low, or after `HORIZON[k]` bars. A lower high in a down structure
goes short. `choch` keeps only the first such low after a change of character, the structure before
the higher high being down; `aligned` keeps only the ones where level k+1 points the same way, the
method's rule against trading against the larger structure. An opposite signal reverses; a repeated one
is ignored. An entry whose close is already past its pivot is skipped: the confirmation is late, and
by then the support can be broken, which made one entry in seven a trade stopped on the next bar.

The stop is read on the close and filled at the close, unlike `strategies_ls`, whose stops fill inside
the bar at their level: it is how the rule was measured.

**What it is worth.** The Wyckoff rules were translated onto the levels and measured with every pivot
used only from its confirmation, on the fifteen pairs, discovery 2018-2022 against confirmation
2023-2026, per event and before the fee. Of 192 cells (15m and 1h, levels 1-4, four entry rules, two
exits, four contexts) the rank in discovery says nothing about the rank in confirmation (Spearman
-0.06), and only five cells were above 1.5 standard errors in both periods, all of them this rule with
its stop. The spring and the upthrust lose once the break restarts, the backup to the creek loses at
every small level, and the length of a range does not predict the move after it. `main` prices
`posizioni` itself, one position at a time, at 0.1% a side; its run of 2026-10-12, gross a trade with
one standard error over blocks of `HORIZON` bars, 2018-2022 / 2023-2026:

    15m, level 4, choch and aligned    +129.1 ± 60.0 bp (201 trades, 10/15 pairs up) / +74.5 ± 43.1 (214, 8/15)
    15m, level 4, choch only            +52.5 ± 30.1 / +24.4 ± 21.4
    15m, level 4, aligned only          +37.0 ± 32.8 /  +9.1 ± 25.1
    1h,  level 4, choch and aligned    +244.9 ± 203.5 / -231.2 ± 201.1

The first line is the page's default: 3.7 trades a year per pair, +109 / +55 bp net of the round trip,
about two standard errors in each period and chosen out of 192 cells: a candidate to watch, not a
result. At 1h no variant of level 4 kept on 2023-2026 what it made on 2018-2022.

    python -m cryptofarm.trading.swing_structure     # the table, over the local store
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

LEVEL = 4
# The median leg at each level, in bars: the trade's time exit.
HORIZON = {1: 5, 2: 18, 3: 58, 4: 185}
SPLIT = pd.Timestamp("2023-01-01")
NEVER = np.iinfo(np.int64).max


def turns(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every turn of `close`: its bar, its kind (+1 a high, -1 a low, alternating) and the bar it is known
    on, the next close that moves the other way. The first and last bar are never turns."""
    step = np.sign(np.diff(close))
    moved = np.flatnonzero(step)
    flip = np.flatnonzero(step[moved][1:] != step[moved][:-1])
    return moved[flip] + 1, step[moved][flip].astype(int), moved[flip + 1] + 1


def hierarchy(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The level of every turn in hindsight: `(bar, kind, level)`. `climb` is the same thing dated live."""
    at, kind, known = turns(close)
    return at, kind, _ladder(close[at] * kind, kind, known)[0]


def climb(close: np.ndarray) -> list[pd.DataFrame]:
    """The pivots of every level, each dated from the bar its level became final on.

    One frame per level, 0 being every turn, with columns `bar`, `kind`, `price` and `confirmed`, sorted by
    `confirmed` then `bar`: the order a live reader learns them in. A pivot not yet final at the last bar
    is left out.
    """
    at, kind, known = turns(close)
    _, conf = _ladder(close[at] * kind, kind, known)
    out = []
    for m in conf:
        i = np.array([j for j in m if m[j] != NEVER], dtype=np.int64)
        frame = pd.DataFrame({"bar": at[i], "kind": kind[i], "price": close[at[i]], "confirmed": [m[j] for j in i]})
        out.append(frame.sort_values(["confirmed", "bar"], ignore_index=True))
    return out


def _ladder(v: np.ndarray, kind: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, list[dict[int, int]]]:
    """The climb itself, on `v` = price x kind so that larger is more extreme for either kind. Returns the
    final level of every turn and, per level, the bar each turn's place there was decided on."""
    level = np.zeros(len(v), np.int64)
    conf = [dict(enumerate(known.tolist()))]
    idx = np.arange(len(v))
    while len(idx) > 2:
        x, prev = v[idx], conf[-1]
        side = np.r_[-np.inf, -np.inf, x, -np.inf, -np.inf]
        keep = np.flatnonzero((x > side[:-4]) & (x > side[4:]))  # the same kind sits two places away

        def decided(a: int) -> int:  # once it and the next of its kind at this level are decided
            return max(prev[idx[a]], prev[idx[a + 2]] if a + 2 < len(idx) else NEVER)

        out, new = [], {}
        if len(keep):
            out.append(keep[0])
            new[idx[keep[0]]] = decided(keep[0])
        for a, b in zip(keep[:-1], keep[1:]):
            if kind[idx[a]] == kind[idx[b]]:  # two of a kind: the most extreme of the other kind between them
                between = np.arange(a + 1, b)
                between = between[kind[idx[between]] != kind[idx[a]]]
                g = between[np.argmax(x[between])]
                out.append(g)
                new[idx[g]] = decided(b)
            out.append(b)
            new[idx[b]] = decided(b)
        if len(out) == len(idx):
            break
        idx = idx[out]
        level[idx] += 1
        conf.append(new)
    return level, conf


def structure(piv: pd.DataFrame, n: int) -> np.ndarray:
    """+1 up, -1 down, 0 neither at every one of `n` bars, from the last two highs and lows confirmed by it."""
    state = np.zeros(n, np.int8)
    for t, s in _walk(piv)[0]:
        state[t:] = s
    return state


def lps(piv: pd.DataFrame) -> pd.DataFrame:
    """Every last point of support (+1) and of supply (-1): the bar a higher low is confirmed in an up
    structure, or a lower high in a down one. `stop` is that pivot's price; `choch` says the structure
    before the extreme that made it was the opposite one."""
    return pd.DataFrame(_walk(piv)[1], columns=["bar", "side", "stop", "choch"])


def _walk(piv: pd.DataFrame) -> tuple[list, list]:
    changes, events, highs, lows = [], [], [], []
    for r in piv.itertuples():
        mine = lows if r.kind < 0 else highs
        mine.append((r.bar, r.price))
        mine.sort()  # a gap-filled pivot can arrive after a later one
        if len(lows) < 2 or len(highs) < 2:
            continue
        s = _trend(highs[-2:], lows[-2:])
        changes.append((r.confirmed, s))
        if r.bar == mine[-1][0] and s == -r.kind:
            before = _trend(highs[-3:-1], lows[-3:-1]) if len(highs) > 2 and len(lows) > 2 else 0
            events.append((r.confirmed, s, r.price, before == -s))
    return changes, events


def _trend(highs: list, lows: list) -> int:
    (_, h1), (_, h2), (_, l1), (_, l2) = *highs, *lows
    return 1 if h2 > h1 and l2 > l1 else -1 if h2 < h1 and l2 < l1 else 0


def signals(
    close: np.ndarray, level: int = LEVEL, choch: bool = True, aligned: bool = True, ladder: list | None = None
) -> pd.DataFrame:
    """The rule's entries on one series: `lps` at `level` whose close is still on the right side of the
    pivot, kept by `choch` and by the structure of the level above (`aligned`), with that structure as a
    column. `ladder` is `climb(close)`, when the caller has it already."""
    ladder = climb(close) if ladder is None else ladder
    if level + 1 >= len(ladder):
        return pd.DataFrame(columns=["bar", "side", "stop", "choch", "above"])
    ev = lps(ladder[level])
    ev["above"] = structure(ladder[level + 1], len(close))[ev.bar.to_numpy(dtype=np.int64)] * ev.side
    holding = (close[ev.bar.to_numpy(dtype=np.int64)] - ev.stop) * ev.side > 0
    keep = holding & (ev.choch | (not choch)) & ((ev.above > 0) | (not aligned))
    return ev[keep].reset_index(drop=True)


def posizioni(
    candles: pd.DataFrame,
    level: int = LEVEL,
    choch: bool = True,
    aligned: bool = True,
    allow_short: bool = True,
    ladder: list | None = None,
) -> list[tuple]:
    """The rule as position changes `(time, price, target)`, target in {+1, 0, -1}, the format of
    `strategies_ls` that `pnl.simulate_positions` executes.

    The entry is the close of the bar the pivot is confirmed on. A trade ends at the first close past its
    stop or after `HORIZON[level]` bars (target 0, at that close), or on a signal of the other side, which
    reverses at the same close. Without `allow_short` a short signal only closes a long.
    """
    close = candles["Close"].to_numpy(dtype=float)
    when = candles.index
    ev = signals(close, level, choch, aligned, ladder)
    at = {int(b): (int(s), float(p)) for b, s, p in zip(ev.bar, ev.side, ev.stop)}
    h, side, entry, stop, out = HORIZON[level], 0, -1, np.nan, []
    for i in range(len(close)):
        if side and (close[i] * side < stop * side or i - entry >= h):
            side = 0
            out.append((when[i], close[i], 0))
        if i in at and at[i][0] != side:
            new, level_at = at[i]
            if new < 0 and not allow_short:
                if side:
                    side = 0
                    out.append((when[i], close[i], 0))
                continue
            side, entry, stop = new, i, level_at
            out.append((when[i], close[i], side))
    return out


def pivots_of(candles: pd.DataFrame, level: int = LEVEL) -> dict[str, pd.Series]:
    """The highs and lows of `level` confirmed by the last bar, at the bar they happened on: for the chart,
    which shows the structure in hindsight while the rule reads it only from each confirmation. A kind with
    no pivot yet, as on a short history, has no series rather than an empty one."""
    close = candles["Close"].to_numpy(dtype=float)
    ladder = climb(close)
    piv = ladder[level] if level < len(ladder) else pd.DataFrame(columns=["bar", "kind", "price"])
    out = {}
    for name, kind in (("struttura_massimi", 1), ("struttura_minimi", -1)):
        mine = piv[piv.kind == kind]
        if mine.empty:
            continue
        s = pd.Series(np.nan, index=candles.index)
        s.iloc[mine.bar.to_numpy(dtype=np.int64)] = mine.price.to_numpy(dtype=float)
        out[name] = s
    return out


def trades(events: list) -> pd.DataFrame:
    """One row per hold of a list of position changes: entry time, side, gross log return."""
    rows, side, t0, p0 = [], 0, None, np.nan
    for t, p, target in events:
        if side:
            rows.append((t0, side, side * np.log(p / p0)))
        side, t0, p0 = target, t, p
    return pd.DataFrame(rows, columns=["entry", "side", "gross"])


def main() -> None:
    from cryptofarm import paths
    from cryptofarm.data import klines

    symbols = [a for a in sys.argv[1:] if not a.endswith(("m", "h"))] or klines.DEFAULT_SYMBOLS
    timeframes = [a for a in sys.argv[1:] if a.endswith(("m", "h"))] or ["15m", "1h"]
    fee_bp = 2 * 0.1 * 100
    rows = []
    for tf in timeframes:
        bars = {s: klines.load_klines(s, tf, store_dir=paths.MARKET_DATA_DIR, clip=False) for s in symbols}
        ladders = {s: climb(b["Close"].to_numpy(dtype=float)) for s, b in bars.items()}
        years = sum((b.index[-1] - b.index[0]) / pd.Timedelta("365D") for b in bars.values())
        for level in (3, 4):
            for choch in (False, True):
                for aligned in (False, True):
                    held = pd.concat(
                        [
                            trades(posizioni(b, level, choch, aligned, ladder=ladders[s])).assign(symbol=s)
                            for s, b in bars.items()
                        ],
                        ignore_index=True,
                    )
                    for late, part in held.groupby(held.entry >= SPLIT):
                        block = part.groupby(part.entry.dt.floor(pd.Timedelta(tf) * HORIZON[level])).gross.mean()
                        per = part.groupby("symbol").gross.mean()
                        rows.append(
                            {
                                "tf": tf, "level": level, "choch": choch, "aligned": aligned,
                                "period": "2023-" if late else "-2022", "trades": len(part),
                                "a_year": round(len(held) / years, 2), "bp": part.gross.mean() * 1e4,
                                "se_bp": block.std() / np.sqrt(len(block)) * 1e4,
                                "net_bp": part.gross.mean() * 1e4 - fee_bp,
                                "pairs_up": f"{(per > 0).sum()}/{len(per)}",
                            }
                        )  # fmt: skip
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
