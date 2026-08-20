"""Test dell'etichettatura per directional change."""

import numpy as np
import pytest

from cryptofarm.ml.directional_change import (
    BUY,
    HOLD,
    SELL,
    capturable_fraction,
    directional_change_pivots,
    label_distribution,
    leg_table,
    soft_labels,
    tune_threshold,
)


def _zigzag(levels, bars_per_leg=10):
    """Serie che sale e scende fra i livelli dati, in modo lineare."""
    parts = []
    for start, end in zip(levels[:-1], levels[1:]):
        parts.append(np.linspace(start, end, bars_per_leg, endpoint=False))
    parts.append(np.array([levels[-1]]))
    return np.concatenate(parts)


def test_pivot_is_dated_at_confirmation_not_at_the_extreme():
    # Price rises to 110, then falls. With a 5% threshold the peak becomes knowable only once
    # price has come back down to 104.5 - several bars after the extreme itself.
    close = _zigzag([100.0, 110.0, 100.0], bars_per_leg=20)
    pivots = directional_change_pivots(close, close, threshold=0.05)

    assert len(pivots) >= 1
    peak = pivots[pivots["kind"] == 1].iloc[0]
    assert peak["confirm_bar"] > peak["extreme_bar"]
    # The confirmation must be the first bar that actually crosses the threshold.
    assert close[int(peak["confirm_bar"])] <= peak["price"] * 0.95


def test_a_smaller_threshold_finds_more_pivots():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 5000)))

    fine = directional_change_pivots(close, close, 0.003)
    coarse = directional_change_pivots(close, close, 0.02)

    assert len(fine) > len(coarse)


def test_pivots_alternate_between_highs_and_lows():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 3000)))

    kinds = directional_change_pivots(close, close, 0.01)["kind"].to_numpy()

    assert len(kinds) > 4
    # A peak must be followed by a trough and vice versa, otherwise the legs are meaningless.
    assert (np.abs(np.diff(kinds)) == 2).all()


def test_confirmation_delay_is_never_negative():
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, 4000)))

    pivots = directional_change_pivots(close, close, 0.005)

    assert (pivots["confirm_bar"] >= pivots["extreme_bar"]).all()


def test_soft_labels_cover_more_than_the_exact_extreme():
    close = _zigzag([100.0, 110.0, 100.0, 110.0], bars_per_leg=20)
    pivots = directional_change_pivots(close, close, 0.03)

    labels = soft_labels(close, pivots, capture=0.60)

    # Strict labelling would mark one bar per leg; the soft version must mark a zone.
    assert (labels == BUY).sum() > 1
    assert (labels == SELL).sum() > 1


def test_soft_labels_mark_buy_where_most_of_the_up_leg_remains():
    # Down to 100, up to 112, down again - the final peak is needed for the up-leg to close.
    close = _zigzag([110.0, 100.0, 112.0, 100.0], bars_per_leg=20)
    pivots = directional_change_pivots(close, close, 0.03)

    labels = soft_labels(close, pivots, capture=0.60)

    trough = int(pivots[pivots["kind"] == -1].iloc[0]["extreme_bar"])
    assert labels[trough] == BUY
    # Near the top of the up-leg almost nothing is left to capture, so it cannot be a buy.
    assert labels[trough + 18] != BUY


def test_a_higher_capture_requirement_marks_fewer_bars():
    close = _zigzag([100.0, 110.0, 100.0, 110.0], bars_per_leg=25)
    pivots = directional_change_pivots(close, close, 0.03)

    lenient = soft_labels(close, pivots, capture=0.30)
    strict = soft_labels(close, pivots, capture=0.90)

    assert (lenient != HOLD).sum() > (strict != HOLD).sum()


def test_leg_table_measures_size_and_direction():
    close = _zigzag([100.0, 110.0, 99.0], bars_per_leg=20)
    pivots = directional_change_pivots(close, close, 0.03)

    legs = leg_table(pivots)

    assert not legs.empty
    assert (legs["size"] > 0).all()
    assert set(legs["direction"].unique()) <= {1, -1}


def test_capturable_fraction_is_below_one_because_confirmation_arrives_late():
    # This is the central claim of the strategy: by the time a trough is confirmed the price has
    # already moved back by the threshold, so part of the leg is gone.
    # A leg only exists between two confirmed pivots, so the series must close the up-leg.
    close = _zigzag([110.0, 100.0, 112.0, 100.0], bars_per_leg=40)
    pivots = directional_change_pivots(close, close, 0.03)

    legs = capturable_fraction(leg_table(pivots), close)
    up_legs = legs[legs["direction"] == 1]

    assert not up_legs.empty
    assert (up_legs["capturable_at_confirm"] < 1.0).all()
    assert (up_legs["capturable_at_confirm"] > 0.0).all()


def test_tune_threshold_lands_inside_the_requested_band_when_possible():
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 288 * 200)))  # 200 giorni di barre 5m

    threshold, rate = tune_threshold(close, close, days=200, target_per_day=(8.0, 12.0))

    assert 8.0 <= rate <= 12.0
    assert threshold > 0


def test_label_distribution_sums_to_one():
    close = _zigzag([100.0, 110.0, 100.0], bars_per_leg=20)
    labels = soft_labels(close, directional_change_pivots(close, close, 0.03))

    distribution = label_distribution(labels)

    assert sum(distribution.values()) == pytest.approx(1.0)
