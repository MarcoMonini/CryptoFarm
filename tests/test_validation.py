"""Test della cross-validation con purging, embargo e pesi di unicita'."""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.validation import (
    CombinatorialPurgedCV,
    PurgedKFold,
    deflated_sharpe_ratio,
    expected_max_sharpe,
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


def test_deflated_sharpe_uses_the_observed_spread_between_trials_when_it_is_given():
    """Una griglia annidata disperde meno di prove indipendenti, e la soglia deve seguirla.

    Senza `trial_variance` si assume la dispersione di prove indipendenti (1/(n-1)). Le
    configurazioni vicine di una griglia sono quasi la stessa strategia, quindi disperdono meno:
    dichiararlo abbassa la soglia, e il verso di quella disuguaglianza e' cio' che il test fissa.
    """
    rng = np.random.default_rng(1)
    returns = rng.normal(0.03, 1.0, 2000)

    stretta = deflated_sharpe_ratio(returns, trials=200, trial_variance=1 / 20_000)
    implicita = deflated_sharpe_ratio(returns, trials=200)
    larga = deflated_sharpe_ratio(returns, trials=200, trial_variance=1 / 200)

    assert stretta > implicita > larga


def test_expected_max_sharpe_grows_with_trials_and_with_their_spread():
    assert expected_max_sharpe(1000, 0.01) > expected_max_sharpe(10, 0.01)
    assert expected_max_sharpe(100, 0.04) > expected_max_sharpe(100, 0.01)
    # Una prova sola, o nessuna dispersione, non regala niente al caso.
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(100, 0.0) == 0.0


def test_deflated_sharpe_is_undefined_on_a_track_record_too_short_to_judge():
    assert np.isnan(deflated_sharpe_ratio(np.array([0.01, 0.02]), trials=10))
