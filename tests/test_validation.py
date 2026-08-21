"""Test della cross-validation con purging, embargo e pesi di unicita'."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    purge_train_indices,
    sample_uniqueness,
)


def _any_life_overlaps(start, exit_, train, test):
    """Esiste una riga di training la cui vita interseca quella di una riga di test?"""
    train_start = start.iloc[train].to_numpy()[:, None]
    train_exit = exit_.iloc[train].to_numpy()[:, None]
    test_start = start.iloc[test].to_numpy()[None, :]
    test_exit = exit_.iloc[test].to_numpy()[None, :]
    return bool(((train_exit >= test_start) & (train_start <= test_exit)).any())


def _lives(n, span_hours=4, freq="1h"):
    start = pd.Series(pd.date_range("2024-01-01", periods=n, freq=freq))
    return start, start + pd.Timedelta(hours=span_hours)


def test_purging_removes_training_rows_whose_life_overlaps_the_test():
    start, exit_ = _lives(20, span_hours=3)
    starts, exits = start.to_numpy(), exit_.to_numpy()
    test = np.arange(10, 13)
    train = np.setdiff1d(np.arange(20), test)

    kept = purge_train_indices(train, test, starts, exits, np.timedelta64(0, "h"))

    # Rows 7-9 end inside the test window and rows 13-15 start inside it: all must go.
    assert set(kept).isdisjoint({7, 8, 9, 13, 14, 15})
    assert 0 in kept and 19 in kept


def test_embargo_removes_additional_rows_after_the_test_window():
    start, exit_ = _lives(20, span_hours=1)
    starts, exits = start.to_numpy(), exit_.to_numpy()
    test = np.arange(10, 12)
    train = np.setdiff1d(np.arange(20), test)

    without = purge_train_indices(train, test, starts, exits, np.timedelta64(0, "h"))
    with_embargo = purge_train_indices(train, test, starts, exits, np.timedelta64(5, "h"))

    assert len(with_embargo) < len(without)
    # Serial autocorrelation just after the test window is what the embargo is for.
    assert set(with_embargo).isdisjoint({13, 14, 15})


def test_purged_kfold_never_leaks_and_covers_every_row_once_in_test():
    start, exit_ = _lives(200, span_hours=6)
    splitter = PurgedKFold(n_splits=5, embargo=pd.Timedelta(hours=2))

    seen = []
    for train, test in splitter.split(start, exit_):
        seen.append(test)
        assert set(train).isdisjoint(set(test))
        # No surviving training row may overlap the test window.
        assert not _any_life_overlaps(start, exit_, train, test)

    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(200))


def test_purged_kfold_test_blocks_are_contiguous_in_time():
    start, exit_ = _lives(100, span_hours=1)

    for _, test in PurgedKFold(n_splits=4).split(start, exit_):
        # Shuffling rows of a time series destroys the structure being validated.
        assert np.array_equal(test, np.arange(test.min(), test.max() + 1))


def test_cpcv_produces_the_combinatorial_number_of_splits():
    start, exit_ = _lives(240, span_hours=2)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=pd.Timedelta(hours=1))

    splits = list(cv.split(start, exit_))

    assert cv.get_n_splits() == 15
    assert len(splits) == 15
    for train, test in splits:
        assert set(train).isdisjoint(set(test))
        # Purging runs per test block, so rows in the gap between two non-contiguous blocks
        # legitimately survive; what must never happen is a training row whose life overlaps
        # the life of an actual test row.
        assert not _any_life_overlaps(start, exit_, train, test)


def test_cpcv_purges_each_test_block_separately():
    # Two non-contiguous test blocks cover disjoint intervals; treating them as one span would
    # throw away everything in between.
    start, exit_ = _lives(120, span_hours=1)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)

    trains = [len(train) for train, _ in cv.split(start, exit_)]

    # Something in the middle must survive, otherwise the purging is too coarse.
    assert max(trains) > 60


def test_cpcv_rejects_impossible_configurations():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_groups=3, n_test_groups=3)


def test_uniqueness_is_one_when_lives_do_not_overlap():
    start = pd.Series(pd.date_range("2024-01-01", periods=5, freq="10h"))
    exit_ = start + pd.Timedelta(hours=1)

    weights = sample_uniqueness(start, exit_)

    assert np.allclose(weights, 1.0, atol=1e-9)


def test_uniqueness_falls_as_lives_overlap():
    # Ten observations all alive at the same time carry one observation's worth of information.
    start = pd.Series([pd.Timestamp("2024-01-01")] * 10)
    exit_ = start + pd.Timedelta(hours=5)

    weights = sample_uniqueness(start, exit_)

    assert np.allclose(weights, 0.1, atol=1e-6)


def test_uniqueness_is_between_zero_and_one_on_realistic_overlap():
    start, exit_ = _lives(500, span_hours=7)

    weights = sample_uniqueness(start, exit_)

    assert (weights > 0).all()
    assert (weights <= 1.0 + 1e-9).all()
    assert weights.mean() < 0.5  # seven-hour lives on hourly starts overlap heavily


def test_pbo_is_one_when_the_best_in_sample_is_always_the_worst_out_of_sample():
    in_sample = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    out_of_sample = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])

    assert probability_of_backtest_overfitting(in_sample, out_of_sample) == 1.0


def test_pbo_is_zero_when_the_best_in_sample_stays_best_out_of_sample():
    in_sample = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    out_of_sample = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])

    assert probability_of_backtest_overfitting(in_sample, out_of_sample) == 0.0


def test_deflated_sharpe_falls_as_more_configurations_are_tried():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.02, 1.0, 2000)

    few = deflated_sharpe_ratio(returns, trials=1)
    many = deflated_sharpe_ratio(returns, trials=500)

    # The same track record is less convincing when it is the best of five hundred attempts.
    assert few > many


def test_deflated_sharpe_is_undefined_on_a_track_record_too_short_to_judge():
    assert np.isnan(deflated_sharpe_ratio(np.array([0.01, 0.02]), trials=10))


def test_test_windows_keeps_non_adjacent_groups_apart():
    """I gruppi di test non adiacenti devono restare intervalli distinti.

    Regressione: `_cpcv` riduceva il test a `[min, max]` dei suoi `t_start`. Per una combinazione
    come (0, 5) quell'intervallo copre anche i gruppi 1-4, che stanno in training, e la politica
    finiva rimisurata sui propri dati di addestramento.
    """
    from cryptofarm.ml.policy_trainer import _test_windows

    starts = pd.date_range("2024-01-01", periods=60, freq="D").to_numpy()
    # primi dieci e ultimi dieci: due gruppi agli estremi, con 40 righe di training in mezzo
    test_index = np.concatenate([np.arange(10), np.arange(50, 60)])

    windows = _test_windows(starts, test_index)

    assert len(windows) == 2, "due blocchi non adiacenti devono dare due finestre"
    assert windows[0] == (starts[0], starts[9])
    assert windows[1] == (starts[50], starts[59])
    # la copertura non deve toccare il training
    covered = sum(int(((starts >= a) & (starts <= b)).sum()) for a, b in windows)
    assert covered == 20, f"le finestre coprono {covered} righe invece di 20"


def test_test_windows_merges_adjacent_groups():
    """Due gruppi adiacenti sono un intervallo solo: la copertura non deve spezzarsi."""
    from cryptofarm.ml.policy_trainer import _test_windows

    starts = pd.date_range("2024-01-01", periods=60, freq="D").to_numpy()
    windows = _test_windows(starts, np.arange(20, 40))

    assert len(windows) == 1
    assert windows[0] == (starts[20], starts[39])


def test_test_windows_follows_time_order_not_row_order():
    """Le finestre si ricavano dall'ordine temporale, non dall'ordine delle righe."""
    from cryptofarm.ml.policy_trainer import _test_windows

    # righe mescolate: la riga 0 e' l'ultima nel tempo
    starts = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-01-02", "2024-02-01"]).to_numpy()
    windows = _test_windows(starts, np.array([1, 2]))

    assert windows == [(starts[1], starts[2])]
