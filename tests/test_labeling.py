"""Test del labeling. Serie costruite a mano perche' l'esito atteso sia verificabile a occhio."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.labeling import (
    BUY,
    HOLD,
    SELL,
    barrier_widths,
    extrema_labels,
    format_distribution,
    label_distribution,
    triple_barrier_labels,
)


def _frame(close, high=None, low=None, atr_percent=1.0):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float) if high is not None else close
    low = np.asarray(low, dtype=float) if low is not None else close
    return pd.DataFrame(
        {
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "ATR": np.full(len(close), atr_percent, dtype=float),
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="5min"),
    )


def test_barrier_widths_scale_with_volatility():
    atr_percent = pd.Series([2.0, 4.0])  # 2% and 4% of price

    take_profit, stop_loss = barrier_widths(atr_percent, tp_multiple=1.5, sl_multiple=1.5)

    assert take_profit.tolist() == pytest.approx([0.03, 0.06])
    assert stop_loss.tolist() == pytest.approx([0.03, 0.06])


def test_barrier_widths_never_fall_below_the_fee_floor():
    # A 0.01% ATR would put the barriers well inside the round-trip fee, so a "winning" label
    # would describe a losing trade.
    atr_percent = pd.Series([0.01])

    take_profit, stop_loss = barrier_widths(
        atr_percent, tp_multiple=1.5, sl_multiple=1.5, round_trip_fee=0.002, fee_floor_multiple=3.0
    )

    assert take_profit.tolist() == pytest.approx([0.006])
    assert stop_loss.tolist() == pytest.approx([0.006])


def test_triple_barrier_marks_buy_when_the_upside_is_reached_first():
    # Barriers are +/-1.5% (ATR 1% x 1.5). Price rises 3% two bars later, never dips.
    close = [100.0, 100.0, 103.0, 103.0, 103.0, 103.0]
    labels = triple_barrier_labels(_frame(close), horizon=3)

    assert labels.iloc[0] == BUY


def test_triple_barrier_marks_sell_when_the_downside_is_reached_first():
    close = [100.0, 100.0, 97.0, 97.0, 97.0, 97.0]
    labels = triple_barrier_labels(_frame(close), horizon=3)

    assert labels.iloc[0] == SELL


def test_triple_barrier_marks_hold_when_neither_barrier_is_reached():
    close = [100.0, 100.1, 99.9, 100.05, 100.0, 100.0]
    labels = triple_barrier_labels(_frame(close), horizon=3)

    assert labels.iloc[0] == HOLD


def test_triple_barrier_resolves_a_same_bar_touch_pessimistically():
    # One bar reaches both barriers. OHLC cannot say which came first inside the bar, so the
    # label must assume the stop - assuming the profit builds a model that expects more than
    # it will get in live execution.
    close = [100.0, 100.0, 100.0, 100.0]
    high = [100.0, 102.0, 100.0, 100.0]
    low = [100.0, 98.0, 100.0, 100.0]
    labels = triple_barrier_labels(_frame(close, high, low), horizon=2)

    assert labels.iloc[0] == SELL


def test_triple_barrier_never_reads_the_entry_bar_itself():
    # The entry bar's own high clears the upper barrier, but the trade opens at its close, so
    # that move is already gone. Only later bars may resolve the label.
    close = [100.0, 100.0, 100.0, 100.0]
    high = [110.0, 100.0, 100.0, 100.0]
    low = [100.0, 100.0, 100.0, 100.0]
    labels = triple_barrier_labels(_frame(close, high, low), horizon=2)

    assert labels.iloc[0] == HOLD


def test_triple_barrier_leaves_the_unobservable_tail_as_hold():
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    labels = triple_barrier_labels(_frame(close), horizon=3)

    # The last `horizon` bars have no future to resolve against.
    assert (labels.iloc[-3:] == HOLD).all()


def test_triple_barrier_labels_every_bar_it_can():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 3000)))
    frame = _frame(close, close * 1.002, close * 0.998, atr_percent=1.0)

    labels = triple_barrier_labels(frame, horizon=48)

    # Unlike extrema labelling, which leaves 97% of bars as hold, this must resolve most bars.
    resolved = (labels != HOLD).mean()
    assert resolved > 0.5
    # And both directions must be represented - a labelling that only ever fires one way would
    # teach a directional bias that is an artefact of the method.
    assert (labels == BUY).sum() > 0
    assert (labels == SELL).sum() > 0


def test_triple_barrier_crosses_the_chunk_boundary_consistently(monkeypatch):
    # The look-ahead matrix is built in chunks to bound memory; the chunk edges must not change
    # any label.
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 2000)))
    frame = _frame(close, close * 1.003, close * 0.997)

    reference = triple_barrier_labels(frame, horizon=24)
    monkeypatch.setattr("cryptofarm.ml.labeling.CHUNK_ROWS", 97)
    chunked = triple_barrier_labels(frame, horizon=24)

    pd.testing.assert_series_equal(reference, chunked)


def test_triple_barrier_requires_the_normalized_atr_column():
    frame = _frame([100.0, 101.0, 102.0]).drop(columns=["ATR"])

    with pytest.raises(KeyError):
        triple_barrier_labels(frame, horizon=1)


def test_label_distribution_reports_counts_and_shares():
    labels = np.array([HOLD, HOLD, BUY, SELL])

    distribution = label_distribution(labels)

    assert distribution["hold"] == 2
    assert distribution["buy"] == 1
    assert distribution["hold_pct"] == pytest.approx(0.5)
    assert "buy=1" in format_distribution(labels, "stadio")


def test_extrema_labels_still_work_for_comparison():
    high = [1, 1, 1, 2, 3, 4, 10, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    low = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 0, 3, 4, 5, 5, 5, 5]
    frame = pd.DataFrame(
        {"Open": low, "High": high, "Low": low, "Close": high},
        index=pd.date_range("2024-01-01", periods=len(high), freq="h"),
    )

    labels = extrema_labels(frame, window_pivot=6, min_return=0, cooldown=0)

    assert labels.iloc[6] == SELL
    assert labels.iloc[14] == BUY
