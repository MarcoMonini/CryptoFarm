"""Contratto del rollout DAgger: deve visitare stati veri e coerenti con le proprie azioni."""

import numpy as np

from cryptofarm.ml.dagger import episode_starts, rollout, state_coverage
from cryptofarm.ml.directional_change import BUY, HOLD, SELL
from cryptofarm.ml.policy import FLAT, LONG


class _AlwaysBuy:
    """Politica degenere: vuole sempre comprare. Serve a verificare il mascheramento nel rollout."""

    def predict_proba(self, X):
        return np.tile([0.0, 1.0, 0.0], (len(X), 1))


class _Alternating:
    """Compra se flat, vende se long: costringe la traiettoria a cambiare stato ogni barra."""

    def predict_proba(self, X):
        # STATE_IN e' l'ultima colonna meno due (STATE_IN, STATE_PNL, STATE_BARS).
        in_position = X[:, -3] > 0.5
        return np.where(in_position[:, None], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])


def _market(rows=200, features=4):
    return np.zeros((rows, features), dtype=np.float64)


def test_a_policy_that_only_wants_to_buy_cannot_buy_twice():
    """Senza mascheramento resterebbe a comprare a vuoto; con la maschera entra e poi tiene."""
    close = np.linspace(100, 120, 200)
    signals = np.zeros(200, dtype=np.int8)

    visited = rollout(_AlwaysBuy(), _market(), close, signals, episode_bars=50)

    first = visited[visited["row"] < 50].sort_values("row")
    assert first["action"].iloc[0] == BUY
    assert (first["action"].iloc[1:] == HOLD).all()
    assert (first["state"].iloc[1:] == LONG).all()


def test_the_visited_state_follows_from_the_actions_taken():
    close = np.linspace(100, 120, 200)
    signals = np.zeros(200, dtype=np.int8)

    visited = rollout(_Alternating(), _market(), close, signals, episode_bars=50)
    episode = visited[visited["row"] < 50].sort_values("row")

    # Compra, vende, compra, vende: lo stato deve alternarsi di conseguenza.
    assert episode["state"].iloc[:6].tolist() == [FLAT, LONG, FLAT, LONG, FLAT, LONG]
    assert episode["action"].iloc[:4].tolist() == [BUY, SELL, BUY, SELL]


def test_the_entry_price_is_the_close_at_which_the_policy_entered():
    close = np.linspace(100, 120, 200)
    signals = np.zeros(200, dtype=np.int8)

    visited = rollout(_AlwaysBuy(), _market(), close, signals, episode_bars=50).sort_values("row")
    held = visited[(visited["row"] < 50) & (visited["state"] == LONG)]

    assert (held["entry_price"] == close[0]).all()
    # La durata cresce di una barra alla volta a partire dall'ingresso.
    assert held["bars_in_position"].tolist() == list(range(len(held)))


def test_disagreement_is_measured_against_the_expert_not_the_label():
    """Il modello che compra sempre deve risultare in disaccordo dove l'esperto direbbe di uscire."""
    close = np.linspace(100, 120, 200)
    signals = np.zeros(200, dtype=np.int8)
    signals[10:20] = SELL

    visited = rollout(_AlwaysBuy(), _market(), close, signals, episode_bars=50)
    coverage = state_coverage(visited)

    assert coverage["disaccordo"] > 0
    assert coverage["uscite_mancate"] > 0
    assert coverage["quota_long"] > 0.9


def test_episodes_tile_the_series_without_overlapping():
    starts = episode_starts(1000, episode_bars=250)

    assert starts.tolist() == [0, 250, 500]
