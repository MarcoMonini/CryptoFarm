"""Il percorso di servizio della politica RL: `ml/signals.rl_exposure` e la voce «AI Model».

I due difetti che questo copre non si vedono leggendo il codice e non fanno sollevare niente:
decidere a ogni barra invece che alla cadenza con cui la politica e' stata addestrata, e perdere
la posizione precedente fra una decisione e l'altra -- che trasformerebbe l'agente in un
classificatore per barra, cioe' esattamente cio' che il costo dentro la ricompensa deve evitare.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml import signals, trainer


class QFinto:
    """`predict` di una funzione dello stato. L'ultima colonna e' la posizione precedente."""

    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(np.asarray(X, dtype=float))


def _q_segue(colonna: int = 0):
    """Lungo quando la colonna e' positiva, senza nessuna preferenza per restare fermo."""
    return [QFinto(lambda X: np.zeros(len(X))), QFinto(lambda X: X[:, colonna])]


def _q_appiccicoso():
    """Preferisce sempre la posizione in cui si trova gia'. Isola la memoria dello stato."""
    return [QFinto(lambda X: 1.0 - X[:, -1]), QFinto(lambda X: X[:, -1])]


@pytest.fixture()
def candele():
    n = 24 * 40
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", name="Open time")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.004, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.003,
            "Low": close * 0.997,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=idx,
    )


def test_la_decisione_resta_ferma_fra_una_cadenza_e_l_altra(candele):
    """La politica e' addestrata a una decisione al giorno: se la pagina ne prendesse una per barra
    pagherebbe ventiquattro volte i costi che il suo obiettivo aveva messo in conto."""
    dentro = signals.rl_exposure(_q_segue(), candele)
    cadenza = signals.swing_cadenza(candele.index)
    assert cadenza == 24
    cambi = np.flatnonzero(np.diff(dentro.astype(np.int8)))
    assert cambi.size, "con questo Q finto qualche cambio ci deve essere"
    assert all((i + 1) % cadenza == 0 for i in cambi), f"cambio fuori cadenza: {cambi.tolist()}"


def test_la_posizione_precedente_arriva_alla_decisione_dopo(candele):
    """Con un Q che preferisce restare dove sta, l'esposizione non deve muoversi mai.

    Se lo stato perdesse la posizione precedente questo Q diventerebbe indifferente e la politica
    oscillerebbe: e' l'unico modo di vedere quel difetto senza guardare i rendimenti.
    """
    assert not signals.rl_exposure(_q_appiccicoso(), candele).any()


def test_il_riscaldamento_resta_fuori_dal_mercato(candele):
    """Sulle prime barre `atr_rel` non esiste: decidere li' vorrebbe dire decidere su soli NaN."""
    dentro = signals.rl_exposure(_q_segue(), candele)
    frame = signals.swing_features(candele)
    caldo = frame["atr_rel"].notna().to_numpy()
    assert not dentro[: np.flatnonzero(caldo)[0]].any()


def test_i_segnali_si_alternano(candele):
    acquisti, vendite = signals.rl_signals(_q_segue(), candele)
    assert acquisti
    assert len(acquisti) - len(vendite) in (0, 1), "acquisti e vendite devono alternarsi"
    for a, v in zip(acquisti, vendite):
        assert a[0] < v[0]


def test_una_serie_corta_non_fa_cadere_la_pagina():
    """Il caso della pagina ai valori di partenza: 240 barre orarie, cioe' dieci barre giornaliere.

    E' lo stesso punto in cui la voce «AI Model» cadeva con `IndexError` da dentro `ta`.
    """
    n = 240
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", name="Open time")
    close = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.004, n)))
    corte = pd.DataFrame(
        {"Open": close, "High": close * 1.002, "Low": close * 0.998, "Close": close, "Volume": 1000.0},
        index=idx,
    )
    dentro = signals.rl_exposure(_q_segue(), corte)
    assert dentro.shape == (n,)


def test_la_politica_guida_la_voce_ai_model(monkeypatch, candele):
    """`active_model_name()` e' l'unica fonte di verita': se dice `rl_model`, `ai_model_simulation`
    deve passare da `rl_signals` e non dalle barriere."""
    from cryptofarm.trading import strategies

    monkeypatch.setattr(strategies, "active_model_name", lambda: "rl_model")
    acquisti, vendite = strategies.ai_model_simulation(candele, _q_segue())
    attesi = signals.rl_signals(_q_segue(), candele)
    assert (acquisti, vendite) == attesi


def test_la_politica_e_in_testa_alla_catena(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    (tmp_path / "swing_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() == "swing_model"
    (tmp_path / "rl_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() == "rl_model"


def test_i_metadata_della_politica_non_dirottano_la_soglia(tmp_path, monkeypatch):
    """`rl_model.json` non ha `decision_threshold` -- non ne usa nessuna. Deve lasciare la ricerca
    proseguire lungo la catena invece di restituire il default e zittire il modello dopo."""
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    (tmp_path / "rl_model.json").write_text(json.dumps({"costo": 0.012, "giri": 3}))
    (tmp_path / "meta_model.json").write_text(json.dumps({"decision_threshold": 0.61}))
    assert trainer.stored_decision_threshold() == pytest.approx(0.61)
