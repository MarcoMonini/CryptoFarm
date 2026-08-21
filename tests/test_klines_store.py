"""Test dello store di candele. Nessuna rete: i dump sono costruiti in memoria."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from cryptofarm.data.klines import (
    _open_times_to_datetime,
    _parse_dump,
    clip_wicks,
    interval_to_minutes,
    resample_klines,
    wick_outliers,
)


def _make_dump(open_times, unit_divisor=1, header=False):
    """Costruisce lo zip di un dump Binance con i timestamp nell'unita' richiesta."""
    rows = []
    if header:
        rows.append("open_time,open,high,low,close,volume,close_time,qav,trades,tbb,tbq,ignore")
    for position, stamp in enumerate(open_times):
        price = 100 + position
        rows.append(
            f"{int(stamp * unit_divisor)},{price},{price + 2},{price - 1},{price + 1}," f"{10 + position},0,0,0,0,0,0"
        )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("dump.csv", "\n".join(rows))
    return buffer.getvalue()


def test_interval_to_minutes_handles_the_supported_units():
    assert interval_to_minutes("5m") == 5
    assert interval_to_minutes("1h") == 60
    assert interval_to_minutes("1d") == 1440
    with pytest.raises(ValueError):
        interval_to_minutes("1y")


def test_open_times_are_read_as_milliseconds_before_2025():
    milliseconds = pd.Series([1717200000000, 1717200300000])

    result = _open_times_to_datetime(milliseconds)

    assert result.iloc[0] == pd.Timestamp("2024-06-01 00:00:00")
    assert result.iloc[1] == pd.Timestamp("2024-06-01 00:05:00")


def test_open_times_are_read_as_microseconds_from_2025():
    # Binance switched the dump timestamps from ms to us in January 2025 with nothing in the
    # file format to signal it. Read as ms these land in the year 57000, and the failure used to
    # be swallowed as "month not available" - silently truncating 18 months of history.
    microseconds = pd.Series([1735689600000000, 1735689900000000])

    result = _open_times_to_datetime(microseconds)

    assert result.iloc[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert result.iloc[1] == pd.Timestamp("2025-01-01 00:05:00")


@pytest.mark.parametrize("header", [False, True])
def test_parse_dump_reads_both_the_headerless_and_headed_layouts(header):
    open_times = [1717200000000, 1717200300000, 1717200600000]

    frame = _parse_dump(_make_dump(open_times, header=header))

    assert list(frame.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(frame) == 3
    assert frame.index[0] == pd.Timestamp("2024-06-01 00:00:00")
    assert frame["Open"].iloc[0] == 100.0


def test_parse_dump_reads_microsecond_dumps():
    open_times = [1735689600000, 1735689900000]

    frame = _parse_dump(_make_dump(open_times, unit_divisor=1000))

    assert frame.index[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert frame.index[1] == pd.Timestamp("2025-01-01 00:05:00")


def test_resample_aggregates_open_high_low_close_volume_correctly():
    index = pd.date_range("2024-01-01", periods=6, freq="5min")
    base = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0],
            "High": [15.0, 11.5, 12.5, 25.0, 21.5, 22.5],
            "Low": [9.0, 10.5, 11.5, 19.0, 20.5, 21.5],
            "Close": [11.0, 12.0, 13.0, 21.0, 22.0, 23.0],
            "Volume": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        },
        index=index,
    )

    result = resample_klines(base, "15m")

    assert len(result) == 2
    # Open is the first of the group, Close the last, High/Low the extremes, Volume the sum.
    assert result["Open"].tolist() == [10.0, 20.0]
    assert result["High"].tolist() == [15.0, 25.0]
    assert result["Low"].tolist() == [9.0, 19.0]
    assert result["Close"].tolist() == [13.0, 23.0]
    assert result["Volume"].tolist() == [6.0, 15.0]
    # Open time labels the start of the bar, matching Binance's own convention.
    assert result.index[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_resample_returns_the_base_interval_untouched():
    index = pd.date_range("2024-01-01", periods=3, freq="5min")
    base = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0],
            "High": [1.0, 2.0, 3.0],
            "Low": [1.0, 2.0, 3.0],
            "Close": [1.0, 2.0, 3.0],
            "Volume": [1.0, 1.0, 1.0],
        },
        index=index,
    )

    assert resample_klines(base, "5m") is base


def test_resample_refuses_intervals_that_are_not_a_multiple_of_the_base():
    index = pd.date_range("2024-01-01", periods=3, freq="5min")
    base = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [1.0]}, index=index[:1])

    with pytest.raises(ValueError):
        resample_klines(base, "7m")


def test_resample_drops_incomplete_groups_rather_than_inventing_bars():
    # A hole in the base series must not become a bar of NaNs in the derived interval.
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 00:05"),
            pd.Timestamp("2024-01-01 00:10"),
            pd.Timestamp("2024-01-01 02:00"),
        ]
    )
    base = pd.DataFrame(
        {
            "Open": [1.0, 2.0, 3.0, 4.0],
            "High": [1.0, 2.0, 3.0, 4.0],
            "Low": [1.0, 2.0, 3.0, 4.0],
            "Close": [1.0, 2.0, 3.0, 4.0],
            "Volume": [1.0, 1.0, 1.0, 1.0],
        },
        index=index,
    )

    result = resample_klines(base, "15m")

    assert not result.isna().any().any()
    assert len(result) == 2  # the empty 15-minute buckets inside the hole are dropped
    assert not np.isin(pd.Timestamp("2024-01-01 00:30"), result.index)


def test_clip_wicks_leaves_normal_bars_alone():
    """Su una serie regolare non deve toccare nulla: il filtro non e' un livellatore."""
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, 600))
    df = pd.DataFrame(
        {
            "Open": close,
            "Close": close,
            "High": close + 0.3,
            "Low": close - 0.3,
            "Volume": 1.0,
        },
        index=pd.date_range("2024-01-01", periods=600, freq="5min", name="Open time"),
    )
    assert not wick_outliers(df).any()
    pd.testing.assert_frame_equal(clip_wicks(df), df)


def test_clip_wicks_compresses_a_liquidation_wick():
    """Il caso ATOM del 2025-10-10: minimo a 0,001 da 1,86, che va compresso ma non cancellato."""
    index = pd.date_range("2024-01-01", periods=600, freq="5min", name="Open time")
    df = pd.DataFrame({"Open": 1.86, "Close": 1.85, "High": 1.87, "Low": 1.84, "Volume": 1.0}, index=index)
    df.iloc[500, df.columns.get_loc("Low")] = 0.001

    clipped = clip_wicks(df)
    assert wick_outliers(df).sum() == 1
    assert clipped["Low"].iloc[500] > 1.0  # compresso
    assert clipped["Low"].iloc[500] < df["Open"].iloc[500]  # ma resta un minimo
    # Il corpo non si tocca: sono prezzi a cui si e' davvero scambiato.
    pd.testing.assert_series_equal(clipped["Open"], df["Open"])
    pd.testing.assert_series_equal(clipped["Close"], df["Close"])
