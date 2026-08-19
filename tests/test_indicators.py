import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.trainer import (
    WINDOW_SIZE,
    add_technical_indicator,
    apply_label_cooldown,
    build_sequence_valid_mask,
    calculate_percentage_changes,
    calculate_relative_extrema,
    downsample_holds,
    filter_labels_by_future_return,
    get_model_predictions,
    normalize_scale_dependent_features,
    split_train_val,
)


def _index(n, freq="h"):
    return pd.date_range("2024-01-01", periods=n, freq=freq)


def test_add_technical_indicator_computes_ta_columns_within_expected_ranges():
    n = 40
    rng = np.random.default_rng(seed=42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=_index(n))

    result = add_technical_indicator(df, rsi_window=6, atr_window=6)

    for column in ("RSI", "ATR", "STOCH", "STOCH_S", "TSI"):
        assert column in result.columns

    # add_technical_indicator fillna(0)s all indicator NaNs, so the result must be NaN-free.
    assert not result[["RSI", "ATR", "STOCH", "STOCH_S", "TSI"]].isna().any().any()

    # RSI and the stochastic oscillator are bounded [0, 100] by definition once warmed up.
    warmed_up = result.iloc[15:]
    assert warmed_up["RSI"].between(0, 100).all()
    assert warmed_up["STOCH"].between(0, 100).all()

    # The original DataFrame must not be mutated (function works on a copy).
    assert "RSI" not in df.columns


def _indicator_frame(close, atr):
    return pd.DataFrame(
        {
            "Close": [close, close],
            "ATR": [atr, atr],
            "RSI": [75.0, 25.0],
            "STOCH": [100.0, 0.0],
            "STOCH_S": [50.0, 50.0],
            "TSI": [40.0, -40.0],
        }
    )


def test_normalize_scale_dependent_features_makes_atr_comparable_across_assets():
    # Same relative volatility (1% of price), wildly different price scale: after normalisation
    # the two assets must produce the same feature value, otherwise multi-asset training is
    # dominated by whichever asset happens to have the largest absolute price.
    expensive = _indicator_frame(close=60000.0, atr=600.0)
    cheap = _indicator_frame(close=150.0, atr=1.5)

    expensive_norm = normalize_scale_dependent_features(expensive)
    cheap_norm = normalize_scale_dependent_features(cheap)

    assert expensive_norm["ATR"].tolist() == pytest.approx([1.0, 1.0])
    assert cheap_norm["ATR"].tolist() == pytest.approx([1.0, 1.0])
    # The input frame must not be mutated (function works on a copy).
    assert expensive["ATR"].tolist() == [600.0, 600.0]


def test_normalize_scale_dependent_features_rescales_bounded_oscillators():
    # The bounded oscillators must land in [-1, 1] so they carry the same weight as the
    # percentage-point price features instead of dominating the gradients.
    normalized = normalize_scale_dependent_features(_indicator_frame(close=100.0, atr=1.0))

    assert normalized["RSI"].tolist() == pytest.approx([0.5, -0.5])
    assert normalized["STOCH"].tolist() == pytest.approx([1.0, -1.0])
    assert normalized["STOCH_S"].tolist() == pytest.approx([0.0, 0.0])
    assert normalized["TSI"].tolist() == pytest.approx([0.4, -0.4])


def test_calculate_relative_extrema_labels_known_peak_and_trough_when_filters_disabled():
    # A single, well-isolated local max in High at index 6 and local min in Low at index 14
    # (order=window_pivot/2=3 neighbours on each side). With min_return/cooldown disabled this
    # is the original purely-local labelling.
    high = [1, 1, 1, 2, 3, 4, 10, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    low = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 0, 3, 4, 5, 5, 5, 5]
    df = pd.DataFrame({"Open": low, "High": high, "Low": low, "Close": high}, index=_index(len(high)))

    labeled = calculate_relative_extrema(df.copy(), window_pivot=6, min_return=0, cooldown=0)

    assert labeled["Label"].iloc[6] == 2  # relative max (High)
    assert labeled["Label"].iloc[14] == 1  # relative min (Low)
    other_rows = labeled.drop(index=labeled.index[[6, 14]])
    assert (other_rows["Label"] == 0).all()


def test_calculate_relative_extrema_drops_noise_sized_extrema():
    # Same shape as above but flat prices around the extrema: no tradable move follows either
    # pivot, so with the default forward-return filter every label must be dropped.
    n = 21
    high = [100.0] * n
    low = [100.0] * n
    high[6] = 100.2  # +0.2% blip: a local maximum, but not a tradable one
    low[14] = 99.8  # -0.2% blip
    df = pd.DataFrame({"Open": low, "High": high, "Low": low, "Close": [100.0] * n}, index=_index(n))

    labeled = calculate_relative_extrema(df.copy(), window_pivot=6, min_return=0.015, return_horizon=5, cooldown=0)

    assert (labeled["Label"] == 0).all()


def test_filter_labels_by_future_return_keeps_only_tradable_signals():
    df = pd.DataFrame(
        {
            "High": [100.0, 100.0, 100.0, 105.0, 100.0, 100.0],
            "Low": [95.0, 95.0, 95.0, 95.0, 90.0, 95.0],
            "Close": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        },
        index=_index(6),
    )
    labels = pd.Series([1, 2, 0, 0, 1, 2], index=df.index, dtype="int64")

    filtered = filter_labels_by_future_return(df, labels, min_return=0.02, horizon=3)

    assert filtered.iloc[0] == 1  # +5% within the next 3 candles -> tradable buy
    assert filtered.iloc[1] == 2  # -10% within the next 3 candles -> tradable sell
    assert filtered.iloc[4] == 0  # nothing moves afterwards -> dropped
    # The last row has no observable future at all: NaN must drop the label, not keep it.
    assert filtered.iloc[5] == 0


def test_apply_label_cooldown_keeps_first_signal_of_a_cluster():
    labels = pd.Series([1, 0, 2, 0, 0, 0, 0, 1], index=_index(8), dtype="int64")

    filtered = apply_label_cooldown(labels, cooldown=3)

    assert filtered.tolist() == [1, 0, 0, 0, 0, 0, 0, 1]
    # The input series must not be mutated in place.
    assert labels.tolist() == [1, 0, 2, 0, 0, 0, 0, 1]


def test_build_sequence_valid_mask_rejects_windows_crossing_a_time_gap():
    # 15-minute candles with a two-hour hole between rows 4 and 5.
    minutes = [0, 15, 30, 45, 60, 180, 195, 210]
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=m) for m in minutes])

    mask = build_sequence_valid_mask(index, expected_minutes=15, window_size=3)

    # Sequence i covers rows i..i+3 (window plus the row its label comes from).
    assert mask.tolist() == [True, True, False, False, False]


def test_split_train_val_drops_embargo_window_around_the_split():
    X = np.arange(100).reshape(100, 1, 1)
    y = np.arange(100)

    X_train, y_train, X_val, y_val = split_train_val(X, y, train_split=0.8, embargo_steps=10)

    assert len(y_train) == 70
    assert len(y_val) == 10
    assert y_train[-1] == 69
    assert y_val[0] == 90


def test_downsample_holds_reaches_the_target_hold_to_signal_ratio():
    y = np.array([0] * 100 + [1] * 10 + [2] * 10)
    X = np.arange(len(y) * 6).reshape(len(y), 2, 3)
    rng = np.random.default_rng(0)

    X_out, y_out = downsample_holds(X, y, hold_to_signal_ratio=2.0, rng=rng)

    assert (y_out == 0).sum() == 40  # 2 x the 20 signal samples
    assert (y_out == 1).sum() == 10  # every signal sample is preserved
    assert (y_out == 2).sum() == 10
    assert len(X_out) == len(y_out)


def test_calculate_percentage_changes_is_cumulative_relative_to_previous_close():
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 108.0],
            "High": [105.0, 110.0, 112.0],
            "Low": [95.0, 100.0, 104.0],
            "Close": [102.0, 108.0, 106.0],
        },
        index=_index(3),
    )

    result = calculate_percentage_changes(df)

    assert result["Open"].tolist() == pytest.approx([0.0, 2.0, 7.882353], abs=1e-4)
    assert result["High"].tolist() == pytest.approx([5.0, 9.843137, 11.586057], abs=1e-4)
    assert result["Low"].tolist() == pytest.approx([-5.0, 0.039216, 4.178649], abs=1e-4)
    assert result["Close"].tolist() == pytest.approx([2.0, 7.882353, 6.030501], abs=1e-4)


class _RecordingModel:
    """Modello finto che memorizza il tensore ricevuto e predice sempre "hold"."""

    def __init__(self):
        self.seen = None

    def predict(self, X, verbose=0):
        self.seen = X
        probabilities = np.zeros((len(X), 3))
        probabilities[:, 0] = 1.0
        return probabilities


def _market_frame(atr_column):
    n = WINDOW_SIZE + 10
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, n),
            "High": close + rng.uniform(0.1, 1.0, n),
            "Low": close - rng.uniform(0.1, 1.0, n),
            "Close": close,
            # Colonne indicatore fornite dal chiamante, deliberatamente incoerenti tra le due
            # chiamate: la dashboard le calcola con i periodi scelti dagli slider.
            "RSI": np.full(n, 30.0),
            "STOCH": np.full(n, 30.0),
            "STOCH_S": np.full(n, 30.0),
            "ATR": np.full(n, atr_column),
            "TSI": np.full(n, 30.0),
        },
        index=_index(n),
    )


def test_get_model_predictions_ignores_caller_supplied_indicator_columns():
    # The model is only valid for features computed the way it was trained. Whatever indicator
    # columns the caller already has in the frame must not reach it, or a dashboard slider
    # silently changes the model's input and it degrades with no visible error.
    first, second = _RecordingModel(), _RecordingModel()

    get_model_predictions(_market_frame(atr_column=2.0), first)
    get_model_predictions(_market_frame(atr_column=99.0), second)

    assert np.array_equal(first.seen, second.seen)


def test_get_model_predictions_aligns_predictions_with_the_original_index():
    df = _market_frame(atr_column=2.0)

    result = get_model_predictions(df, _RecordingModel())

    assert len(result) == len(df) - WINDOW_SIZE
    assert result.index[0] == df.index[WINDOW_SIZE]
    # The stub is fully confident about "hold", so every prediction must be 0.
    assert (result["Prediction"] == 0).all()
