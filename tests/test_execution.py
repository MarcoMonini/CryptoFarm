"""Test del modello di esecuzione degli ordini limite."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.execution import (
    adverse_selection_report,
    limit_fills,
    round_trip_cost,
)


def _market(close, low=None, atr_percent=1.0):
    close = np.asarray(close, dtype=float)
    low = np.asarray(low, dtype=float) if low is not None else close * 0.9999
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.0001,
            "Low": low,
            "Close": close,
            "ATR": np.full(len(close), atr_percent),
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="5min"),
    )


def test_a_limit_below_the_market_fills_only_if_price_comes_down_to_it():
    # ATR 1%, offset 0.25 ATR -> limit at 0.25% below the close, i.e. 99.75.
    rising = _market([100.0, 100.5, 101.0, 101.5])
    falling = _market([100.0, 99.9, 99.5, 99.4])

    assert not limit_fills(rising, [0], patience=3)["filled"].iloc[0]
    assert limit_fills(falling, [0], patience=3)["filled"].iloc[0]


def test_an_unfilled_order_is_recorded_rather_than_dropped():
    # An entry the model saw and execution failed to convert is an opportunity cost, not a
    # non-event: dropping it would inflate the expectancy per trade.
    rising = _market([100.0, 101.0, 102.0, 103.0])

    fills = limit_fills(rising, [0], patience=3)

    assert len(fills) == 1
    assert not fills["filled"].iloc[0]
    assert np.isnan(fills["fill_price"].iloc[0])


def test_patience_bounds_how_long_the_order_waits():
    close = np.concatenate([np.full(5, 100.0), np.full(5, 99.0)])
    market = _market(close)

    impatient = limit_fills(market, [0], patience=2)
    patient = limit_fills(market, [0], patience=8)

    assert not impatient["filled"].iloc[0]
    assert patient["filled"].iloc[0]


def test_fill_requires_crossing_not_merely_touching_by_default():
    # Being conservative about queue position - which this model does not simulate - is the
    # cheapest defence against overstating maker execution.
    close = np.array([100.0, 100.0, 100.0])
    exactly_at_limit = np.array([100.0, 99.75, 99.75])
    market = _market(close, low=exactly_at_limit)

    assert not limit_fills(market, [0], patience=2, require_cross=True)["filled"].iloc[0]
    assert limit_fills(market, [0], patience=2, require_cross=False)["filled"].iloc[0]


def test_adverse_selection_is_detected_when_fills_cluster_on_falling_markets():
    # Build a market that alternates: some entries are followed by a fall (and fill), others by
    # a rise (and no fill). The report must show the gap.
    segments = []
    for direction in [-1, 1] * 30:
        base = 100.0
        segments.append(base * (1 + direction * 0.004 * np.arange(6) / 5))
    close = np.concatenate(segments)
    market = _market(close)
    entries = np.arange(0, len(close) - 10, 6)

    report = adverse_selection_report(limit_fills(market, entries, patience=5))

    assert 0.0 < report["fill_rate"] < 1.0
    # Orders fill when the market is going down and miss when it is going up: that difference
    # is exactly what a certain-fill simulation would silently pocket.
    assert report["market_return_when_filled"] < report["market_return_when_missed"]
    assert report["adverse_selection"] > 0


def test_round_trip_cost_defaults_to_maker_entry_and_taker_exit():
    # A stop-loss cannot wait on the book, so assuming maker on both sides is the most common
    # way these simulations flatter themselves.
    assert round_trip_cost() == pytest.approx(0.0002 + 0.0010)
    assert round_trip_cost("maker", "maker") == pytest.approx(0.0004)
    assert round_trip_cost("taker", "taker") == pytest.approx(0.0020)


def test_round_trip_cost_rejects_unknown_modes():
    with pytest.raises(ValueError):
        round_trip_cost("vip9", "taker")
