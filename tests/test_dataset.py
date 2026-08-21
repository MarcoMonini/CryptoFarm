"""Test della costruzione del dataset."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.dataset import build_design_matrix, build_samples, create_sequences, time_split


def _features(n=200, freq="5min", start="2024-01-01", gap_after=None):
    index = pd.date_range(start, periods=n, freq=freq)
    if gap_after is not None:
        shifted = index.to_series()
        shifted.iloc[gap_after:] += pd.Timedelta(hours=6)
        index = pd.DatetimeIndex(shifted)
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "RSI": rng.uniform(-1, 1, n),
            "STOCH": rng.uniform(-1, 1, n),
            "STOCH_S": rng.uniform(-1, 1, n),
            "ATR": rng.uniform(0.1, 1.0, n),
            "TSI": rng.uniform(-1, 1, n),
            "VOLUME": rng.uniform(-1, 1, n),
            "TIMEFRAME": np.zeros(n),
        },
        index=index,
    )


def test_design_matrix_encodes_prices_only_as_returns():
    features = _features()

    matrix = build_design_matrix(features, lags=(1, 3))

    # No column may carry an absolute price level: those are incomparable across assets and eras.
    assert not any(column in matrix.columns for column in ("Open", "High", "Low", "Close"))
    assert "RET_1" in matrix.columns and "RET_3" in matrix.columns
    assert matrix.dtypes.unique().tolist() == [np.dtype("float32")]


def test_design_matrix_returns_match_the_log_change_in_close():
    features = _features(n=20)

    matrix = build_design_matrix(features, lags=(1,))

    expected = np.log(features["Close"].iloc[5] / features["Close"].iloc[4]) * 100
    assert matrix["RET_1"].iloc[5] == pytest.approx(expected, rel=1e-5)


def test_design_matrix_position_feature_places_close_within_the_recent_range():
    index = pd.date_range("2024-01-01", periods=4, freq="5min")
    features = pd.DataFrame(
        {
            "Open": [10.0, 10.0, 10.0, 10.0],
            "High": [12.0, 12.0, 12.0, 12.0],
            "Low": [8.0, 8.0, 8.0, 8.0],
            "Close": [10.0, 10.0, 10.0, 12.0],
            "RSI": 0.0,
            "STOCH": 0.0,
            "STOCH_S": 0.0,
            "ATR": 1.0,
            "TSI": 0.0,
            "VOLUME": 0.0,
            "TIMEFRAME": 0.0,
        },
        index=index,
    )

    matrix = build_design_matrix(features, lags=(3,))

    assert matrix["POS_3"].iloc[2] == pytest.approx(0.5)  # close halfway up the 8..12 range
    assert matrix["POS_3"].iloc[3] == pytest.approx(1.0)  # close at the top of the range


def test_build_samples_drops_rows_whose_future_is_not_observable():
    features = _features(n=200)
    labels = pd.Series(1, index=features.index)

    matrix, selected = build_samples(features, labels, expected_minutes=5, horizon=10, lags=(1, 2), stride=1)

    # The last `horizon` rows are labelled hold only because their future is missing.
    assert matrix.index.max() <= features.index[-11]


def test_build_samples_drops_rows_spanning_a_time_gap():
    features = _features(n=200, gap_after=100)
    labels = pd.Series(1, index=features.index)

    matrix, _ = build_samples(features, labels, expected_minutes=5, horizon=5, lags=(1, 5), stride=1)

    # Rows whose lag window or label horizon crosses the six-hour hole must be gone.
    positions = features.index.get_indexer(matrix.index)
    assert not np.any((positions > 94) & (positions < 101))


def test_build_samples_applies_the_stride():
    features = _features(n=300)
    labels = pd.Series(1, index=features.index)

    dense, _ = build_samples(features, labels, expected_minutes=5, horizon=5, lags=(1, 2), stride=1)
    sparse, _ = build_samples(features, labels, expected_minutes=5, horizon=5, lags=(1, 2), stride=10)

    assert len(sparse) == pytest.approx(len(dense) / 10, abs=2)


def test_build_samples_keeps_matrix_and_labels_aligned():
    features = _features(n=200)
    labels = pd.Series(np.arange(200), index=features.index)

    matrix, selected = build_samples(features, labels, expected_minutes=5, horizon=5, lags=(1, 2), stride=7)

    assert matrix.index.equals(selected.index)


def test_time_split_cuts_on_the_date_and_leaves_an_embargo():
    index = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=100, freq="D"))

    train, validation = time_split(index, train_fraction=0.8, embargo=pd.Timedelta(days=3))

    assert train.sum() > 0 and validation.sum() > 0
    # Nothing may sit in both, and the embargo must leave a gap between them.
    assert not (train & validation).any()
    last_train = index[train].max()
    first_validation = index[validation].min()
    assert first_validation - last_train > pd.Timedelta(days=5)


def test_time_split_is_global_so_symbols_cannot_leak_into_each_other():
    # Two symbols sharing a calendar: the split must fall on the same date for both, otherwise
    # one asset's future is another asset's training data.
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    index = pd.DatetimeIndex(list(dates) + list(dates))

    train, validation = time_split(index, train_fraction=0.8, embargo=pd.Timedelta(0))

    assert index[train].max() < index[validation].min()


def test_create_sequences_normalises_each_window_to_its_own_opening():
    features = _features(n=80)
    labels = pd.Series(np.arange(80), index=features.index)

    windows, y = create_sequences(features, labels, window_size=10)

    assert windows.dtype == np.float32
    assert windows.shape == (70, 10, 11)
    # Every window starts at zero on the Open channel: prices are relative to the window itself.
    assert np.allclose(windows[:, 0, 0], 0.0, atol=1e-4)
    assert y[0] == 10


def test_create_sequences_returns_empty_when_the_window_does_not_fit():
    features = _features(n=5)
    labels = pd.Series(np.arange(5), index=features.index)

    windows, y = create_sequences(features, labels, window_size=10)

    assert windows.shape[0] == 0
    assert y.shape[0] == 0


def test_cusum_fires_on_accumulated_moves_not_on_the_clock():
    from cryptofarm.ml.dataset import cusum_events

    # A long quiet stretch then a sustained move: the filter must stay silent through the quiet
    # part and fire on the move, which is the whole point of sampling on events.
    quiet = 100 * (1 + 0.00002 * np.sin(np.arange(600)))
    move = 100 * np.exp(np.cumsum(np.full(100, 0.002)))
    close = pd.Series(np.concatenate([quiet, move]))

    events = cusum_events(close, threshold_sigma=3.0, volatility_window=288)

    assert len(events) > 0
    assert (events >= 600).mean() > 0.8


def test_cusum_fires_less_often_as_the_threshold_rises():
    from cryptofarm.ml.dataset import cusum_events

    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 5000))))

    assert len(cusum_events(close, 2.0)) > len(cusum_events(close, 5.0))


def test_cusum_returns_nothing_when_volatility_cannot_be_estimated():
    from cryptofarm.ml.dataset import cusum_events

    close = pd.Series(100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.002, 50))))

    assert len(cusum_events(close, 3.0, volatility_window=288)) == 0
