"""Contratto della politica condizionata sullo stato."""

import numpy as np
import pytest

from cryptofarm.ml.directional_change import BUY, HOLD, SELL
from cryptofarm.ml.policy import (
    FLAT,
    LONG,
    action_mask,
    expert_actions,
    mask_probabilities,
    position_features,
    randomised_states,
    simulate_expert_states,
)


def test_buying_is_impossible_while_long_and_selling_while_flat():
    mask = action_mask(np.array([FLAT, LONG]))

    assert mask[0].tolist() == [True, True, False]  # flat: hold, buy
    assert mask[1].tolist() == [True, False, True]  # long: hold, sell


def test_masking_moves_the_decision_to_a_valid_action():
    """Il caso che rende utile il mascheramento: il modello vuole comprare, ma e' gia' dentro."""
    probabilities = np.array([[0.1, 0.8, 0.1]])

    masked = mask_probabilities(probabilities, np.array([LONG]))

    assert masked[0, BUY] == 0.0
    assert np.argmax(masked[0]) == HOLD
    assert masked.sum() == pytest.approx(1.0)


def test_masking_leaves_valid_probabilities_ranked_as_they_were():
    probabilities = np.array([[0.2, 0.7, 0.1]])

    masked = mask_probabilities(probabilities, np.array([FLAT]))

    assert np.argmax(masked[0]) == BUY
    assert masked[0, HOLD] / masked[0, BUY] == pytest.approx(0.2 / 0.7)


def test_the_expert_holds_through_a_buy_zone_while_already_long():
    """Restare dentro durante una zona di acquisto e' la condotta giusta, non un'occasione persa."""
    signals = np.array([BUY, BUY, SELL, SELL])
    state = np.array([FLAT, LONG, LONG, FLAT])

    actions = expert_actions(signals, state)

    assert actions.tolist() == [BUY, HOLD, SELL, HOLD]


def test_position_features_are_zero_when_flat():
    close = np.array([100.0, 110.0])
    frame = position_features(close, np.array([FLAT, LONG]), np.array([0.0, 100.0]), np.array([0, 5]))

    assert frame.loc[0].tolist() == [0.0, 0.0, 0.0]
    assert frame.loc[1, "STATE_IN"] == 1.0
    assert frame.loc[1, "STATE_PNL"] == pytest.approx(0.10)
    assert 0.0 < frame.loc[1, "STATE_BARS"] < 1.0


def test_randomised_entries_come_from_prices_the_market_actually_printed():
    close = np.linspace(100, 200, 500)
    rows = np.arange(300, 500)

    state, entry, bars = randomised_states(close, rows, np.random.default_rng(0))

    assert set(np.unique(state)) <= {FLAT, LONG}
    assert 0.3 < (state == LONG).mean() < 0.7
    # Ogni prezzo di ingresso e' una chiusura realmente osservata, alla distanza dichiarata.
    assert np.allclose(entry, close[rows - bars.astype(int)])
    assert (bars <= rows).all()


def test_randomisation_visits_states_the_expert_never_reaches():
    """La ragione per cui esiste: la traiettoria dell'esperto e' una fetta sottile degli stati."""
    signals = np.zeros(400, dtype=np.int8)
    signals[10:20] = BUY
    signals[200:210] = SELL

    expert = simulate_expert_states(signals)
    random_state, _, _ = randomised_states(np.linspace(100, 120, 400), np.arange(400), np.random.default_rng(1))

    # L'esperto entra una volta sola e resta dentro: nessun esempio di uscita in perdita precoce.
    assert (expert == LONG).mean() == pytest.approx(0.475, abs=0.02)
    assert abs((random_state == LONG).mean() - 0.5) < 0.1


def test_expert_states_follow_the_signals():
    signals = np.array([HOLD, BUY, HOLD, SELL, HOLD, BUY])

    state = simulate_expert_states(signals)

    assert state.tolist() == [FLAT, FLAT, LONG, LONG, FLAT, FLAT]
