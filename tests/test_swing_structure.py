"""`trading/swing_structure`: the hierarchy dated live never revises a decision, and the rule reads no future."""

import numpy as np
import pandas as pd

from cryptofarm.trading import swing_structure as ss


def walk(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.standard_t(3, n) * 0.004))
    close[200:203] = close[199]  # a run of equal closes is one turn
    return close


def candles(close):
    idx = pd.date_range("2024-01-01", periods=len(close), freq="15min")
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1.0}, index=idx)


def test_turns_alternate_and_are_known_later():
    close = walk()
    at, kind, known = ss.turns(close)
    assert (np.diff(kind) != 0).all() and (known > at).all()
    assert (np.sign(close[known] - close[known - 1]) == -kind).all()


def test_the_live_levels_are_the_hindsight_ones_and_never_revised():
    """The whole hierarchy rerun on every prefix agrees with `climb` from each pivot's `confirmed` on."""
    close = walk()
    ladder = ss.climb(close)
    h_at, _, h_level = ss.hierarchy(close)
    final = dict(zip(h_at.tolist(), h_level.tolist()))
    for k, piv in enumerate(ladder):
        assert all(final[b] >= k for b in piv.bar), "a pivot dated at a level it does not reach"
    checked = 0
    for e in range(300, len(close), 7):
        p_at, _, p_level = ss.hierarchy(close[: e + 1])
        live = dict(zip(p_at.tolist(), p_level.tolist()))
        for k in range(1, len(ladder)):
            for b in ladder[k].bar[ladder[k].confirmed <= e]:
                assert live[b] >= k, f"level {k} of the turn at {b} revised at bar {e}"
                checked += 1
    assert checked > 1000, checked


def test_cutting_the_future_moves_no_signal():
    close = walk()
    cut = 900
    a, b = ss.signals(close, 2, False, False), ss.signals(close[:cut], 2, False, False)
    assert a[a.bar < cut - 50].equals(b[b.bar < cut - 50])


def test_a_hand_built_change_of_character():
    """A down structure with its last point of supply, a higher high, then two higher lows: the first of
    them is the one after the change of character."""
    kinds, prices = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1], [10, 5, 9, 4, 8, 3, 12, 6, 14, 8]
    piv = pd.DataFrame({"bar": range(10), "kind": kinds, "price": prices, "confirmed": range(1, 11)})
    ev = ss.lps(piv)
    assert ev.side.tolist() == [-1, 1, 1] and ev.choch.tolist() == [False, True, False]
    assert ev.stop.tolist() == [8, 6, 8] and ev.bar.tolist() == [5, 8, 10]
    assert ss.structure(piv, 12).tolist() == [0] * 4 + [-1] * 3 + [0] + [1] * 4


def test_positions_enter_on_the_right_side_and_exit_by_stop_or_time():
    close = walk()
    ev = ss.signals(close, 1, False, False)
    assert len(ev) and ((close[ev.bar] - ev.stop) * ev.side > 0).all(), "an entry with its stop already passed"
    df = candles(close)
    events = ss.posizioni(df, 1, False, False)
    targets = [t for _, _, t in events]
    assert targets and set(targets) <= {-1, 0, 1}
    assert all(a != b for a, b in zip(targets, targets[1:])), "a repeated state is not a change"
    held = ss.trades(events)
    assert len(held) == sum(1 for t in targets[:-1] if t != 0)
    long_only = [t for _, _, t in ss.posizioni(df, 1, False, False, allow_short=False)]
    assert -1 not in long_only and 1 in long_only


def test_no_hold_outlasts_its_level():
    df = candles(walk(3000, 3))
    events = ss.posizioni(df, 2, False, False)
    bars = df.index.get_indexer([t for t, _, _ in events])
    targets = [t for _, _, t in events]
    for (b0, t0), b1 in zip(zip(bars, targets), bars[1:]):
        if t0 != 0:
            assert b1 - b0 <= ss.HORIZON[2]


def test_the_chart_series_sit_on_the_pivots():
    df = candles(walk())
    series = ss.pivots_of(df, 2)
    piv = ss.climb(df["Close"].to_numpy())[2]
    assert set(series) == {"struttura_massimi", "struttura_minimi"}
    assert series["struttura_massimi"].notna().sum() == (piv.kind == 1).sum()
    assert series["struttura_minimi"].notna().sum() == (piv.kind == -1).sum()


def test_a_bar_is_read_whole():
    """A higher low and a higher high confirmed on the same bar: read one at a time, the low was judged
    against the old highs, the structure looked neutral and the last point of support was lost."""
    piv = pd.DataFrame(
        {
            "bar": range(7),
            "kind": [1, -1, 1, -1, 1, -1, 1],
            "price": [10, 5, 9, 4, 8, 6, 12],
            "confirmed": [1, 2, 3, 4, 5, 7, 7],
        }
    )
    ev = ss.lps(piv)
    assert ev.bar.tolist() == [5, 7] and ev.side.tolist() == [-1, 1]
    assert ev.stop.tolist() == [8, 6] and ev.choch.tolist() == [False, True]


def test_the_order_inside_a_confirmation_bar_does_not_matter():
    piv = ss.climb(walk(3000, 3))[2]
    shuffled = piv.sample(frac=1.0, random_state=1).sort_values("confirmed", kind="stable", ignore_index=True)
    assert ss.lps(piv).equals(ss.lps(shuffled))
    assert (ss.structure(piv, 3000) == ss.structure(shuffled, 3000)).all()
