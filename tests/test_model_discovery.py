"""Scoperta degli artefatti addestrati.

La pagina Streamlit ne dipende per una ragione operativa: gli artefatti sono gitignorati, per
cui un clone del repository e l'immagine che va in produzione non ne hanno nessuno. Se
`active_model_name` sollevasse invece di restituire `None`, il simulatore non si aprirebbe.
"""

import json

import pytest

from cryptofarm.ml import trainer
from cryptofarm.trading import config


@pytest.fixture()
def models_dir(tmp_path, monkeypatch):
    """Reindirizza la cartella dei modelli su una vuota, per questo test soltanto."""
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    return tmp_path


def test_no_model_is_reported_as_none_rather_than_raising(models_dir):
    assert trainer.active_model_name() is None


def test_load_signal_model_explains_what_is_missing(models_dir):
    with pytest.raises(FileNotFoundError, match="Nessun modello"):
        trainer.load_signal_model()


def test_the_most_recent_strategy_wins_when_several_are_trained(models_dir):
    (models_dir / "signal_model.joblib").write_bytes(b"")
    (models_dir / "meta_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() == "meta_model"

    (models_dir / "meta_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() == "meta_model"


def test_the_three_action_policy_is_out_of_the_chain(models_dir):
    """`policy_model` non deve piu' vincere per il solo fatto di esistere su disco.

    Era il primo della precedenza, quindi bastava che l'artefatto fosse in `models/` perche' la
    voce «AI Model» eseguisse la politica a tre azioni -- il disegno chiuso in negativo da
    `strategy.md` §12-13. L'artefatto puo' restare sul disco senza governare la pagina.
    """
    (models_dir / "policy_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() is None

    (models_dir / "signal_model.joblib").write_bytes(b"")
    assert trainer.active_model_name() == "signal_model"
    assert "policy_model" not in trainer.MODEL_PRECEDENCE
    # e nemmeno `leg_model`, misurato in perdita a ogni soglia
    assert "leg_model" not in trainer.MODEL_PRECEDENCE


def test_a_keras_artifact_counts_as_a_trained_model(models_dir):
    (models_dir / "signal_model.keras").write_bytes(b"")
    assert trainer.active_model_name() == "signal_model"


def test_the_decision_threshold_falls_back_when_no_metadata_exists(models_dir):
    assert trainer.stored_decision_threshold() == trainer.DEFAULT_DECISION_THRESHOLD

    (models_dir / "signal_model.json").write_text(json.dumps({"decision_threshold": 0.62}))
    assert trainer.stored_decision_threshold() == pytest.approx(0.62)


# --- il menu del simulatore ------------------------------------------------------------


def test_the_ai_strategy_is_not_offered_without_a_model():
    from cryptofarm.trading import config, simulator

    offered = simulator.available_strategies(model=None)
    assert config.AI_STRATEGY not in offered
    # Le altre restano tutte: senza modello il backtest classico funziona lo stesso.
    assert offered == [name for name in config.STRATEGIES if name != config.AI_STRATEGY]


def test_the_ai_strategy_is_offered_once_a_model_is_loaded():
    from cryptofarm.trading import config, simulator

    assert simulator.available_strategies(model=object()) == list(config.STRATEGIES)


def test_la_pagina_parte_anche_senza_nessun_modello(models_dir):
    """La riga di avvio della pagina, non i suoi pezzi.

    `active_model_name` e `available_strategies` erano gia' coperti, ma nessun test toccava il
    punto in cui la pagina li mette insieme -- ed e' quello che ha mandato in errore il servizio:
    `load_signal_model()` senza condizione solleva, e la pagina non si apriva affatto.
    """
    from cryptofarm.trading.simulator import available_strategies, modello_di_sessione

    modello = modello_di_sessione()
    assert modello is None
    assert config.AI_STRATEGY not in available_strategies(modello)


def test_un_regressore_si_salva_e_si_rilegge(tmp_path):
    """`save_model` sceglieva il formato da `predict_proba`: il primo regressore moriva li'.

    Il difetto scattava **dopo** l'addestramento, cioe' nel punto piu' caro possibile.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    from cryptofarm.ml.models import load_model, save_model

    X = np.random.default_rng(0).normal(size=(200, 4))
    modello = HistGradientBoostingRegressor(max_iter=5).fit(X, X[:, 0])
    percorso = tmp_path / "regressore.joblib"
    save_model(modello, percorso)
    assert np.allclose(load_model(percorso).predict(X), modello.predict(X))
