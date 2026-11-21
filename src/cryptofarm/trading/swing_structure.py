"""The swing structure of the close with no window, read live, and Wyckoff's last point of support on it.

**The hierarchy.** Level 0 is every turn of the close: each bar where it changes direction, a run of
equal closes counting once. A high climbs to level k+1 when it is higher than the highs on either side
of it at level k, a low when it is lower than the lows: Dow's higher highs and lower lows applied to
themselves, until nothing climbs. Two highs left in a row at a level get the lowest low between them
from the level below, so every level alternates. No window and no threshold: the levels are the scales
the market itself made, each one's legs about 3.2 times the length of the one below. Measured on the
fifteen `klines.DEFAULT_SYMBOLS` from 2017, the median leg is 5 / 18 / 58 / 185 bars at levels 1-4 on
15m and 5 / 18 / 57 / 175 on 1h, so `HORIZON` is one table for both. It is the same hierarchy as
`decomposer.swing_levels`, which reads it from here, dated live.

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
goes short. `choch` keeps only the first such low after a change of character, the last structure that
was not neutral having been down; `aligned` keeps only the ones where level k+1 points the same way, the
method's rule against trading against the larger structure. An opposite signal reverses; a repeated one
is ignored. An entry whose close is already past its pivot is skipped: the confirmation is late, and
by then the support can be broken, which made one entry in seven a trade stopped on the next bar.

**The entry is not at the low.** At level 4 on 15m a pivot is confirmed a median 245 bars after it
happened, longer than the level's median leg, so the trade opens well after the higher low it is named
for, and the stop on that low sits a median 5-6% from the entry (6.4% on 2018-2022, 4.8% on 2023-2026,
fifteen pairs). A faster confirmation of the same pivots (an adaptive zigzag at a fraction of the level's
median leg) was measured and did worse: at a quarter of the leg it enters ten bars after the pivot,
trades 23 times a year per pair and makes nothing gross.

The stop is read on the close and filled at the close, unlike `strategies_ls`, whose stops fill inside
the bar at their level: it is how the rule was measured.

**What it is worth.** The Wyckoff rules were translated onto the levels and measured with every pivot
used only from its confirmation, on the fifteen pairs, discovery 2018-2022 against confirmation
2023-2026, per event and before the fee. Of 192 cells (15m and 1h, levels 1-4, four entry rules, two
exits, four contexts) the rank in discovery says nothing about the rank in confirmation (Spearman
-0.06), and only five cells were above 1.5 standard errors in both periods, all of them this rule with
its stop. The spring and the upthrust lose once the break restarts, the backup to the creek loses at
every small level, and the length of a range does not predict the move after it. `main` prices
`posizioni` itself, one position at a time, at 0.1% a side; its run of 2026-10-12, after a bar of
confirmations is read whole, gross a trade with one standard error over blocks of `HORIZON` bars,
2018-2022 / 2023-2026:

    15m, level 4, choch and aligned    +126.6 ± 56.3 bp (254 trades, 10/15 pairs up) / +62.5 ± 40.0 (249, 8/15)
    15m, level 4, choch only            +65.5 ± 30.4 / +19.0 ± 18.9
    15m, level 4, aligned only          +47.2 ± 35.2 /  +6.5 ± 25.1
    1h,  level 4, choch and aligned    +281.9 ± 200.0 / -52.7 ± 173.1

The first line is the page's default: 4.5 trades a year per pair, +107 / +43 bp net of the round trip,
2.2 and 1.6 standard errors and chosen out of 192 cells: a candidate to watch, not a result. At 1h no
variant of level 4 kept on 2023-2026 what it made on 2018-2022.

**Against chance at matched exposure** (`control`, run of 2026-11-21). Every hold of the rule keeps its
pair, side and length and is moved, all holds of a period by one common random fraction of it, so the
exposure, the long/short mix (51% long at the default) and the timing across pairs, which move together,
are kept and only the *when* is random. Percentile of the rule's mean among 1,000 such placements,
2018-2022 / 2023-2026; the placements make +0 to +3 bp a hold at 15m:

    15m, level 4, choch and aligned    98.2 / 94.3   (seeds 1-3 at 2,000 draws: 98.2-98.8 / 94.8-95.2)
    15m, level 4, choch only           99.1 / 82.4
    15m, level 4, aligned only         92.3 / 59.4
    15m, level 4, neither              92.7 / 50.3
    1h,  level 4, every variant        84.5-97.6 / 26.8-38.8

The default sits on the 95th percentile on 2023-2026, not past it; the entry model this project wires in
sits at the 100th. The edge is real money against a placement that makes nothing (+62.5 bp gross a hold
against +0.9), but it is one cell picked out of 192 on both periods, so the 2023-2026 percentile is not
an out-of-sample one, and each filter taken away halves what is left. Still a candidate, not a result.

    python -m cryptofarm.trading.swing_structure             # the table, over the local store
    python -m cryptofarm.trading.swing_structure --control   # the random control, about 15 s
"""

from __future__ import annotations

import bisect
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
    """Every last point of support (+1) and of supply (-1): a bar on which a higher low is confirmed and is
    the latest low of an up structure, or a lower high the latest high of a down one. `stop` is that pivot's
    price; `choch` says the last structure that was not neutral before this bar was the opposite one."""
    return pd.DataFrame(_walk(piv)[1], columns=["bar", "side", "stop", "choch"])


def _walk(piv: pd.DataFrame) -> tuple[list, list]:
    """The structure and the events, a confirmation bar at a time.

    Every pivot confirmed on a bar is added before anything is decided on it. 43% of the pivots of a level
    share their confirmation bar with another one (a gap-filled pivot is decided with the pivot that closes
    its gap), and reading them one at a time decided on a state that was already stale at that bar: on BTC
    15m level 4, 116 events of 546 existed only that way and 91 were missed.
    """
    changes, events, highs, lows = [], [], [], []
    last = 0  # the last structure that was not neutral, before the bar being read
    for t, bar in piv.groupby("confirmed", sort=True):
        for r in bar.itertuples():
            bisect.insort(lows if r.kind < 0 else highs, (r.bar, r.price))  # a pivot can predate a known one
        if len(lows) < 2 or len(highs) < 2:
            continue
        s = _trend(highs[-2:], lows[-2:])
        changes.append((t, s))
        latest = (lows if s > 0 else highs)[-1] if s else None
        if latest is not None and latest[0] in set(bar.bar):
            events.append((t, s, latest[1], last == -s))
        last = s or last
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


def holds(events: list, index: pd.DatetimeIndex) -> np.ndarray:
    """One row per hold of a list of position changes: entry bar, exit bar, side."""
    rows, side, a = [], 0, 0
    for b, (_, _, target) in zip(index.get_indexer([e[0] for e in events]), events):
        if side:
            rows.append((a, b, side))
        side, a = target, b
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def shifted(close: np.ndarray, held: np.ndarray, lo: int, hi: int, offset: int) -> np.ndarray:
    """The log return of every hold moved `offset` bars later, its entry wrapping inside [lo, hi): same side,
    same length, so the same exposure. At offset 0 they are the holds themselves."""
    a, b, s = held.T
    d = b - a
    start = np.minimum(lo + (a - lo + offset) % (hi - lo), len(close) - 1 - d)
    return s * np.log(close[start + d] / close[start])


def control(close: dict, held: dict, late: bool, draws: int = 1000, seed: int = 0) -> dict:
    """The rule's mean gross log return a hold against `draws` placements of the same holds at random, in one
    period: every pair's holds keep their side and length and move by one common fraction of the period, so
    the exposure, the long/short mix and the timing across pairs (crypto moves together) are all kept and
    only the *when* is random. `close` and `held` are by symbol; `held` is `holds` of the rule."""
    u = np.random.default_rng(seed).random(draws)
    real, null, n = [], np.zeros(draws), 0
    for s, c in close.items():
        split = int(np.searchsorted(held[s][1], SPLIT))
        lo, hi = (split, len(c)) if late else (0, split)
        h = held[s][0][(held[s][0][:, 0] >= lo) & (held[s][0][:, 0] < hi)]
        if not len(h):
            continue
        real.append(shifted(c, h, lo, hi, 0))
        null += [shifted(c, h, lo, hi, int(x * (hi - lo))).sum() for x in u]
        n += len(h)
    mean, null = float(np.concatenate(real).mean()), null / n
    return {"bp": mean * 1e4, "null_bp": float(null.mean()) * 1e4, "null_p95_bp": float(np.percentile(null, 95))
            * 1e4, "pct": 100 * float((null < mean).mean())}  # fmt: skip


def main() -> None:
    from cryptofarm import paths
    from cryptofarm.data import klines

    if "--control" in sys.argv:
        return main_control()
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


def main_control() -> None:
    """`control` on the level-4 variants, 15m and 1h, both periods: `python -m ... --control`."""
    from cryptofarm import paths
    from cryptofarm.data import klines

    rows = []
    for tf in ("15m", "1h"):
        bars = {
            s: klines.load_klines(s, tf, store_dir=paths.MARKET_DATA_DIR, clip=False) for s in klines.DEFAULT_SYMBOLS
        }
        close = {s: b["Close"].to_numpy(dtype=float) for s, b in bars.items()}
        ladders = {s: climb(c) for s, c in close.items()}
        for choch in (False, True):
            for aligned in (False, True):
                held = {
                    s: (holds(posizioni(b, LEVEL, choch, aligned, ladder=ladders[s]), b.index), b.index)
                    for s, b in bars.items()
                }
                for late in (False, True):
                    rows.append({"tf": tf, "choch": choch, "aligned": aligned, "period": "2023-" if late else "-2022",
                                 **control(close, held, late)})  # fmt: skip
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
