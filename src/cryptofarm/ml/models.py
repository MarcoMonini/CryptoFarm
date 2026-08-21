"""Modelli disponibili, dietro un'unica interfaccia.

Il default e' il **gradient boosting**, non una rete neurale, e la ragione e' misurata: sullo
stesso dataset e le stesse etichette un `HistGradientBoostingClassifier` ha impiegato 3,9
secondi contro i ~25 minuti di un LSTM a tre strati da 747.267 parametri, ottenendo un risultato
migliore. Su una macchina locale la differenza non e' di comodita': quattro secondi per
iterazione permettono di tarare labeling e feature decine di volte al giorno, venticinque minuti
no. E finche' il target non e' solido, iterare sul target vale piu' di qualunque architettura.

I modelli sequenziali restano disponibili dietro la stessa interfaccia per il giorno in cui
avra' senso verificare se una rete aggiunge qualcosa. Le loro dipendenze (TensorFlow) vengono
importate solo se richieste: importarle sempre costa svariati secondi a ogni avvio.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

MODEL_KINDS = ("gbdt", "gru", "cnn", "lstm")


def build_model(kind: str = "gbdt", input_shape: tuple[int, ...] | None = None, **overrides):
    """Costruisce un modello non addestrato del tipo richiesto."""
    if kind == "gbdt":
        parameters = {
            "max_iter": 400,
            "learning_rate": 0.06,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 25,
            "random_state": 42,
        }
        parameters.update(overrides)
        return HistGradientBoostingClassifier(**parameters)

    if kind in ("gru", "cnn", "lstm"):
        if input_shape is None:
            raise ValueError(f"il modello '{kind}' richiede input_shape")
        return _build_sequence_model(kind, input_shape, **overrides)

    raise ValueError(f"Modello sconosciuto: {kind!r}. Disponibili: {MODEL_KINDS}")


def _build_sequence_model(kind: str, input_shape: tuple[int, ...], **overrides):
    from tensorflow.keras.layers import (
        GRU,
        LSTM,
        BatchNormalization,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        Input,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    learning_rate = overrides.get("learning_rate", 1e-3)

    if kind == "gru":
        layers = [Input(shape=input_shape), GRU(64), Dropout(0.2), Dense(32, activation="relu")]
    elif kind == "cnn":
        # Convoluzioni dilatate: coprono la stessa finestra temporale di una ricorrente ma sono
        # parallele sui timestep, il che su CPU vale molto piu' della capacita' in piu'.
        layers = [Input(shape=input_shape)]
        for dilation in (1, 2, 4, 8):
            layers.append(Conv1D(48, kernel_size=3, dilation_rate=dilation, padding="causal", activation="relu"))
            layers.append(BatchNormalization())
        layers.append(GlobalAveragePooling1D())
        layers.append(Dropout(0.2))
    else:
        layers = [Input(shape=input_shape), Bidirectional(LSTM(64)), Dropout(0.2), Dense(32, activation="relu")]

    layers.append(Dense(3, activation="softmax"))
    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def _is_probabilistic(model) -> bool:
    """Il modello espone `predict_proba`, cioe' segue l'interfaccia scikit-learn.

    Il controllo e' sulla capacita' e non su una classe concreta: legare lo smistamento a
    `HistGradientBoostingClassifier` farebbe finire nel ramo Keras qualunque altro classificatore
    compatibile con scikit-learn, con un errore che compare solo a runtime.
    """
    return hasattr(model, "predict_proba")


def fit_model(model, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None, **kwargs):
    """Addestra il modello, qualunque sia la famiglia."""
    if _is_probabilistic(model):
        model.fit(X, y, sample_weight=sample_weight)
        return model
    model.fit(X, y, sample_weight=sample_weight, **kwargs)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Probabilita' per classe, nell'ordine 0=hold, 1=buy, 2=sell.

    Le probabilita' sono il prodotto che conta, non la classe predetta: la decisione di operare
    dipende da una soglia scelta sull'aspettativa economica, non dall'argmax.
    """
    if _is_probabilistic(model):
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.shape[1] == 3:
            return probabilities
        # `predict_proba` restituisce le colonne nell'ordine di `classes_`, che puo' non
        # contenere tutte e tre le classi se una manca dal training set.
        full = np.zeros((len(X), 3), dtype=float)
        for position, label in enumerate(model.classes_):
            full[:, int(label)] = probabilities[:, position]
        return full
    return np.asarray(model.predict(X, verbose=0))


def save_model(model, path) -> None:
    if _is_probabilistic(model):
        import joblib

        joblib.dump(model, path)
    else:
        model.save(path)


def load_model(path):
    """Carica un modello riconoscendone il formato dall'estensione."""
    path = str(path)
    if path.endswith(".joblib"):
        import joblib

        return joblib.load(path)
    from tensorflow.keras.models import load_model as keras_load_model

    return keras_load_model(path)
