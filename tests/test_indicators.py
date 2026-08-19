import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.trainer import (
    add_technical_indicator,
    calculate_percentage_changes,
    calculate_relative_extrema,
)


def test_add_technical_indicator_computes_ta_columns_within_expected_ranges():
    n = 40
    rng = np.random.default_rng(seed=42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )

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


def test_calculate_relative_extrema_labels_known_peak_and_trough():
    # A single, well-isolated local max in High at index 6 and local min in Low at index 14
    # (order=window_pivot/2=3 neighbours on each side).
    high = [1, 1, 1, 2, 3, 4, 10, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    low = [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 0, 3, 4, 5, 5, 5, 5]
    df = pd.DataFrame(
        {"Open": low, "High": high, "Low": low, "Close": high},
        index=pd.date_range("2024-01-01", periods=len(high), freq="h"),
    )

    labeled = calculate_relative_extrema(df.copy(), window_pivot=6)

    assert labeled["Label"].iloc[6] == 2  # relative max (High)
    assert labeled["Label"].iloc[14] == 1  # relative min (Low)
    other_rows = labeled.drop(index=labeled.index[[6, 14]])
    assert (other_rows["Label"] == 0).all()


def test_calculate_percentage_changes_is_cumulative_relative_to_previous_close():
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 108.0],
            "High": [105.0, 110.0, 112.0],
            "Low": [95.0, 100.0, 104.0],
            "Close": [102.0, 108.0, 106.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="h"),
    )

    result = calculate_percentage_changes(df)

    assert result["Open"].tolist() == pytest.approx([0.0, 2.0, 7.882353], abs=1e-4)
    assert result["High"].tolist() == pytest.approx([5.0, 9.843137, 11.586057], abs=1e-4)
    assert result["Low"].tolist() == pytest.approx([-5.0, 0.039216, 4.178649], abs=1e-4)
    assert result["Close"].tolist() == pytest.approx([2.0, 7.882353, 6.030501], abs=1e-4)
