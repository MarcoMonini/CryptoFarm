"""Test delle metriche economiche."""

import numpy as np
import pytest

from cryptofarm.ml.evaluate import (
    best_threshold,
    break_even_precision,
    format_precisione,
    lift_over_base_rate,
    precisione_estremi,
    threshold_sweep,
    trade_expectancy,
)
from cryptofarm.ml.labeling import BUY, HOLD, SELL


def test_symmetric_barriers_demand_a_precision_far_above_a_coin_flip():
    # 0.6% either way with 0.2% round-trip fees: the fee is paid on both outcomes, so break-even
    # sits at two thirds, not a half. This is the arithmetic that rules out symmetric barriers.
    assert break_even_precision(0.006, 0.006, 0.002) == pytest.approx(2 / 3)


def test_doubling_the_profit_barrier_brings_break_even_within_reach():
    assert break_even_precision(0.012, 0.006, 0.002) == pytest.approx(0.4444, abs=1e-4)


def test_break_even_is_unreachable_when_the_barrier_cannot_cover_the_fee():
    assert break_even_precision(0.001, 0.006, 0.002) == float("inf")


def test_expectancy_is_zero_exactly_at_the_break_even_precision():
    take_profit, stop_loss, fee = 0.012, 0.006, 0.002
    precision = break_even_precision(take_profit, stop_loss, fee)

    assert trade_expectancy(precision, take_profit, stop_loss, fee) == pytest.approx(0.0, abs=1e-12)


def test_expectancy_turns_positive_above_break_even_and_negative_below():
    take_profit, stop_loss, fee = 0.012, 0.006, 0.002

    assert trade_expectancy(0.55, take_profit, stop_loss, fee) > 0
    assert trade_expectancy(0.35, take_profit, stop_loss, fee) < 0


def _probabilities(buy_probability):
    buy_probability = np.asarray(buy_probability, dtype=float)
    result = np.zeros((len(buy_probability), 3))
    result[:, BUY] = buy_probability
    result[:, HOLD] = 1.0 - buy_probability
    return result


def test_threshold_sweep_counts_only_the_candles_above_each_threshold():
    y_true = np.array([BUY, BUY, SELL, SELL])
    probabilities = _probabilities([0.9, 0.7, 0.3, 0.1])

    sweep = threshold_sweep(y_true, probabilities, 0.012, 0.006, 0.002, thresholds=(0.5, 0.8))

    assert sweep.loc[sweep["soglia"] == 0.5, "operazioni"].item() == 2
    assert sweep.loc[sweep["soglia"] == 0.5, "win_rate"].item() == pytest.approx(1.0)
    assert sweep.loc[sweep["soglia"] == 0.8, "operazioni"].item() == 1


def test_threshold_sweep_reports_a_losing_selection_as_negative_expectancy():
    # A selection that is right only a quarter of the time must show a loss, however confident
    # the model was.
    y_true = np.array([BUY, SELL, SELL, SELL])
    probabilities = _probabilities([0.9, 0.9, 0.9, 0.9])

    sweep = threshold_sweep(y_true, probabilities, 0.012, 0.006, 0.002, thresholds=(0.5,))

    assert sweep["win_rate"].item() == pytest.approx(0.25)
    assert sweep["atteso_per_trade"].item() < 0


def test_threshold_sweep_handles_a_threshold_nothing_reaches():
    y_true = np.array([BUY, SELL])
    probabilities = _probabilities([0.2, 0.1])

    sweep = threshold_sweep(y_true, probabilities, 0.012, 0.006, 0.002, thresholds=(0.9,))

    assert sweep["operazioni"].item() == 0
    assert np.isnan(sweep["win_rate"].item())


def test_threshold_sweep_uses_the_barriers_of_the_selected_candles():
    # Barriers scale with volatility, so selecting the more volatile candles changes the return
    # per trade, not only the win rate.
    y_true = np.array([BUY, BUY])
    probabilities = _probabilities([0.9, 0.4])
    take_profit = np.array([0.10, 0.01])
    stop_loss = np.array([0.05, 0.005])

    sweep = threshold_sweep(y_true, probabilities, take_profit, stop_loss, 0.002, thresholds=(0.5,))

    # Only the first candle is selected, so the expectancy must reflect its 10% barrier.
    assert sweep["atteso_per_trade"].item() == pytest.approx(0.10 - 0.002)


def test_best_threshold_ignores_selections_too_small_to_mean_anything():
    y_true = np.array([BUY] * 5 + [SELL] * 995)
    probabilities = _probabilities([0.99] * 5 + [0.4] * 995)

    sweep = threshold_sweep(y_true, probabilities, 0.012, 0.006, 0.002, thresholds=(0.3, 0.9))

    # The 0.9 threshold looks perfect but rests on five trades.
    assert best_threshold(sweep, min_trades=100)["soglia"] == 0.3
    assert best_threshold(sweep, min_trades=10_000) is None


def test_lift_measures_the_improvement_over_not_selecting_at_all():
    y_true = np.array([BUY, BUY, SELL, SELL, SELL, SELL, SELL, SELL, SELL, SELL])
    probabilities = _probabilities([0.9, 0.9] + [0.1] * 8)

    # Base rate is 20%; the selection is 100% buy, so the lift is 5x.
    assert lift_over_base_rate(y_true, probabilities, 0.5) == pytest.approx(5.0)


def test_extreme_precision_reads_a_coin_flip_as_the_base_rate():
    rng = np.random.default_rng(0)
    label = rng.normal(size=20_000)

    # A prediction independent of the label picks the bottom decile at random: 10% of it is
    # genuinely a low. This is the number the swing model has to beat, and it beat it by 3x
    # while still losing money — hence the return columns below.
    esito = precisione_estremi(rng.normal(size=20_000), label)

    assert esito["precisione"] == pytest.approx(0.1, abs=0.02)
    assert esito["vantaggio"] == pytest.approx(1.0, abs=0.2)


def test_extreme_precision_separates_being_right_from_being_paid():
    label = np.linspace(-1.0, 1.0, 10_000)
    # Perfect on the ranking, so precision is 1: every flagged bar is a real low.
    esito = precisione_estremi(label.copy(), label, rendimento=-label * 0.01)

    assert esito["precisione"] == pytest.approx(1.0)
    assert esito["rendimento_segnalato"] == pytest.approx(esito["rendimento_vero"])

    # Same precision computed on a payoff that dies before the bar is tradable: the ranking is
    # untouched and the money is gone. The two columns cannot be collapsed into one.
    povero = precisione_estremi(label.copy(), label, rendimento=np.full(10_000, 0.0001))
    assert povero["precisione"] == pytest.approx(1.0)
    assert povero["rendimento_segnalato"] == pytest.approx(0.0001)


def test_extreme_precision_ignores_bars_whose_future_is_not_known_yet():
    label = np.linspace(-1.0, 1.0, 1_000)
    rendimento = -label * 0.01
    rendimento[-200:] = np.nan  # la coda: il futuro non e' ancora arrivato

    esito = precisione_estremi(label.copy(), label, rendimento=rendimento)

    assert np.isfinite(esito["rendimento_segnalato"])
    assert "precisione" in format_precisione(esito)
