"""Test delle feature. La proprieta' centrale da difendere e' che siano scale-free."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.features import (
    FEATURES,
    add_technical_indicators,
    build_feature_frame,
    normalize_indicators,
)


def _candles(n=200, price=100.0, volume=1000.0, seed=5):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.0005, n)),
            "High": close * (1 + rng.uniform(0.0005, 0.004, n)),
            "Low": close * (1 - rng.uniform(0.0005, 0.004, n)),
            "Close": close,
            "Volume": volume * rng.uniform(0.5, 2.0, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


def test_indicators_are_added_without_touching_the_input():
    candles = _candles()

    result = add_technical_indicators(candles)

    for column in ("RSI", "ATR", "STOCH", "STOCH_S", "TSI"):
        assert column in result.columns
    assert "RSI" not in candles.columns


def test_the_same_market_at_two_price_scales_produces_identical_features():
    # This is the property that makes one model across many assets possible at all. BTC near
    # 100,000 and DOGE near 0.2 must look the same to the model when they move the same way;
    # an absolute ATR alone would make the model learn the asset's identity instead.
    cheap = _candles(price=0.2, volume=1e9, seed=11)
    expensive = cheap.copy()
    expensive[["Open", "High", "Low", "Close"]] *= 500_000
    expensive["Volume"] /= 500_000

    cheap_features = build_feature_frame(cheap, "5m")
    expensive_features = build_feature_frame(expensive, "5m")

    for column in ("RSI", "ATR", "STOCH", "STOCH_S", "TSI", "VOLUME"):
        np.testing.assert_allclose(
            cheap_features[column].to_numpy(),
            expensive_features[column].to_numpy(),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"la feature {column} dipende dalla scala del prezzo",
        )


def test_atr_is_expressed_as_a_share_of_price():
    frame = pd.DataFrame(
        {
            "Close": [200.0, 200.0],
            "ATR": [4.0, 4.0],
            "RSI": [50.0, 50.0],
            "STOCH": [50.0, 50.0],
            "STOCH_S": [50.0, 50.0],
            "TSI": [0.0, 0.0],
            "Volume": [1.0, 1.0],
        }
    )

    result = normalize_indicators(frame, "5m")

    assert result["ATR"].tolist() == pytest.approx([2.0, 2.0])


def test_bounded_oscillators_are_recentred_on_zero():
    frame = pd.DataFrame(
        {
            "Close": [100.0, 100.0],
            "ATR": [1.0, 1.0],
            "RSI": [100.0, 0.0],
            "STOCH": [75.0, 25.0],
            "STOCH_S": [50.0, 50.0],
            "TSI": [100.0, -100.0],
            "Volume": [1.0, 1.0],
        }
    )

    result = normalize_indicators(frame, "5m")

    assert result["RSI"].tolist() == pytest.approx([1.0, -1.0])
    assert result["STOCH"].tolist() == pytest.approx([0.5, -0.5])
    assert result["TSI"].tolist() == pytest.approx([1.0, -1.0])


def test_volume_is_measured_against_its_own_recent_normal():
    # A volume spike must read the same whether the asset normally trades in thousands or in
    # billions of units.
    quiet = pd.DataFrame(
        {
            "Close": [100.0] * 10,
            "ATR": [1.0] * 10,
            "RSI": [50.0] * 10,
            "STOCH": [50.0] * 10,
            "STOCH_S": [50.0] * 10,
            "TSI": [0.0] * 10,
            "Volume": [10.0] * 9 + [40.0],
        }
    )
    loud = quiet.copy()
    loud["Volume"] *= 1e9

    assert normalize_indicators(quiet, "5m")["VOLUME"].iloc[-1] == pytest.approx(
        normalize_indicators(loud, "5m")["VOLUME"].iloc[-1]
    )
    # And a four-fold spike must read higher than the baseline.
    assert normalize_indicators(quiet, "5m")["VOLUME"].iloc[-1] > normalize_indicators(quiet, "5m")["VOLUME"].iloc[0]


def test_timeframe_is_carried_as_an_explicit_feature():
    candles = _candles(n=60)

    five = build_feature_frame(candles, "5m")["TIMEFRAME"].iloc[0]
    hour = build_feature_frame(candles, "1h")["TIMEFRAME"].iloc[0]

    # A single model spans several timeframes, so it must be able to tell them apart.
    assert five == pytest.approx(0.0)
    assert hour > five


def test_build_feature_frame_drops_the_warm_up_instead_of_filling_it_with_zeros():
    candles = _candles(n=200)

    result = build_feature_frame(candles, "5m")

    # A zero RSI is a plausible and wrong value that the model cannot tell from a genuinely low
    # one, so warm-up rows must be gone rather than filled.
    assert not result[[column for column in FEATURES if column in result.columns]].isna().any().any()
    assert len(result) < len(candles)
