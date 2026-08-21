"""Test della generazione dei segnali."""

import numpy as np
import pandas as pd

from cryptofarm.ml.signals import barrier_signals, interval_from_index


class _ScoreModel:
    """Modello finto: assegna P(buy) alto alle candele indicate, basso a tutte le altre."""

    def __init__(self, hot_timestamps=(), probability=0.99):
        self.hot = set(hot_timestamps)
        self.probability = probability
        self.seen_rows = 0

    def predict_proba(self, X):
        self.seen_rows = len(X)
        result = np.zeros((len(X), 3))
        result[:, 1] = self._scores
        result[:, 2] = 1 - result[:, 1]
        return result

    @property
    def classes_(self):
        return np.array([0, 1, 2])


class _AlwaysBuy(_ScoreModel):
    def predict_proba(self, X):
        result = np.zeros((len(X), 3))
        result[:, 1] = self.probability
        return result


class _NeverBuy(_ScoreModel):
    def predict_proba(self, X):
        result = np.zeros((len(X), 3))
        result[:, 1] = 0.01
        result[:, 2] = 0.99
        return result


def _candles(close, high=None, low=None, freq="15min"):
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float) if high is not None else close * 1.001
    low = np.asarray(low, dtype=float) if low is not None else close * 0.999
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": low, "Close": close, "Volume": np.full(len(close), 1000.0)},
        index=pd.date_range("2024-01-01", periods=len(close), freq=freq),
    )


def _trending(n=400, drift=0.0015):
    steps = np.full(n, drift)
    return 100 * np.exp(np.cumsum(steps))


# Candele necessarie prima che il modello possa assegnare un punteggio: gli indicatori devono
# scaldarsi (~37 barre) e la matrice di progetto guarda indietro fino a 55 barre di ritardo.
WARM_UP = 120


def _wobbling(n, level=100.0):
    """Serie quasi piatta ma non costante: una serie costante rende RSI e stocastico indefiniti."""
    return level * (1 + 0.0004 * np.sin(np.arange(n)))


def test_interval_is_inferred_from_the_candle_spacing():
    assert interval_from_index(pd.date_range("2024-01-01", periods=5, freq="5min")) == "5m"
    assert interval_from_index(pd.date_range("2024-01-01", periods=5, freq="1h")) == "1h"


def test_signals_alternate_so_index_pairing_produces_real_trades():
    # simulate_trading_with_commisions pairs buys and sells by position, which is only meaningful
    # when they alternate. The old code emitted a sell on ~60% of candles and buys on a handful.
    candles = _candles(_trending())

    buys, sells = barrier_signals(candles, _AlwaysBuy(), threshold=0.5, horizon=20)

    assert len(buys) > 0
    assert len(sells) == len(buys)
    for (buy_time, _), (sell_time, _) in zip(buys, sells):
        assert sell_time > buy_time
    # No position may open before the previous one closed.
    for previous_sell, next_buy in zip(sells[:-1], buys[1:]):
        assert next_buy[0] > previous_sell[0]


def test_a_model_that_never_signals_produces_no_trades():
    candles = _candles(_trending())

    buys, sells = barrier_signals(candles, _NeverBuy(), threshold=0.5, horizon=20)

    assert buys == []
    assert sells == []


def test_exit_takes_the_profit_barrier_when_price_rises_through_it():
    # Barely moving until the indicators are warm and the first position is open, then a jump far
    # above any plausible take-profit.
    close = np.concatenate([_wobbling(WARM_UP), _wobbling(60, 130.0)])
    candles = _candles(close)

    buys, sells = barrier_signals(candles, _AlwaysBuy(), threshold=0.5, horizon=40)

    assert buys and sells
    entry_price, exit_price = buys[0][1], sells[0][1]
    # The exit is the barrier level, not the candle close, so the gain is capped by the barrier
    # rather than picking up the whole 30% jump.
    assert exit_price > entry_price
    assert exit_price < entry_price * 1.05


def test_exit_takes_the_stop_barrier_when_price_falls_through_it():
    close = np.concatenate([_wobbling(WARM_UP), _wobbling(60, 70.0)])
    candles = _candles(close)

    buys, sells = barrier_signals(candles, _AlwaysBuy(), threshold=0.5, horizon=40)

    assert buys and sells
    entry_price, exit_price = buys[0][1], sells[0][1]
    assert exit_price < entry_price
    # The stop level, not the crashed close: the loss is capped by the barrier.
    assert exit_price > entry_price * 0.95


def test_a_bar_touching_both_barriers_is_resolved_as_a_stop():
    # OHLC cannot say which barrier came first inside a bar. Assuming the profit would make the
    # simulated P&L better than anything achievable live.
    close = _wobbling(WARM_UP + 60)
    high = close * 1.001
    low = close * 0.999
    high[WARM_UP:] = 200.0
    low[WARM_UP:] = 50.0
    candles = _candles(close, high, low)

    buys, sells = barrier_signals(candles, _AlwaysBuy(), threshold=0.5, horizon=40)

    assert buys and sells
    assert sells[0][1] < buys[0][1]


def test_position_is_closed_at_the_time_limit_when_no_barrier_is_touched():
    candles = _candles(_wobbling(400))

    buys, sells = barrier_signals(candles, _AlwaysBuy(), threshold=0.5, horizon=10)

    assert buys and sells
    held_bars = candles.index.get_loc(sells[0][0]) - candles.index.get_loc(buys[0][0])
    assert held_bars == 10


def test_threshold_gates_the_entries():
    candles = _candles(_trending())

    permissive, _ = barrier_signals(candles, _AlwaysBuy(probability=0.6), threshold=0.5, horizon=20)
    strict, _ = barrier_signals(candles, _AlwaysBuy(probability=0.6), threshold=0.9, horizon=20)

    assert len(permissive) > 0
    assert len(strict) == 0


def test_empty_or_too_short_input_yields_no_signals():
    # Too few candles to compute the indicators at all: this must return nothing, not raise an
    # IndexError from inside the indicator library.
    assert barrier_signals(_candles(_wobbling(5)), _AlwaysBuy(), threshold=0.5) == ([], [])


def test_the_model_cannot_score_until_it_has_enough_history():
    # Indicators need to warm up and the design matrix looks back up to its longest lag, so the
    # first scorable candle sits well inside the series. A caller passing a short window gets
    # nothing rather than signals built on incomplete history.
    short = _candles(_wobbling(80))
    long = _candles(_trending(400))

    assert barrier_signals(short, _AlwaysBuy(), threshold=0.5, horizon=20) == ([], [])
    buys, _ = barrier_signals(long, _AlwaysBuy(), threshold=0.5, horizon=20)
    assert buys
    assert long.index.get_loc(buys[0][0]) >= 90


def test_policy_signals_alternate_and_pair_up():
    """Il simulatore accoppia acquisti e vendite per indice: devono essere alternati e pari."""
    from cryptofarm.ml.signals import policy_signals

    class _BuyThenSell:
        """Compra da flat, vende da long: una politica degenere ma perfettamente alternata."""

        def predict_proba(self, X):
            in_position = X[:, -3] > 0.5
            return np.where(in_position[:, None], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])

    index = pd.date_range("2024-01-01", periods=400, freq="15min", name="Open time")
    close = pd.Series(np.linspace(100, 130, 400) + np.sin(np.arange(400) / 7), index=index)
    df = pd.DataFrame({"Open": close, "High": close * 1.002, "Low": close * 0.998, "Close": close, "Volume": 1_000.0})

    buys, sells = policy_signals(df, _BuyThenSell())

    assert len(buys) == len(sells) > 0
    # Ogni vendita segue il proprio acquisto, e ogni acquisto segue la vendita precedente.
    assert all(sell[0] > buy[0] for buy, sell in zip(buys, sells))
    assert all(buy[0] > sell[0] for buy, sell in zip(buys[1:], sells[:-1]))
