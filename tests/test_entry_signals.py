"""La regola d'ingresso servita, e i due numeri che non si possono inventare.

Il modello d'ingresso prevede il rendimento delle prossime H barre, e il suo vantaggio non sta
nell'accuratezza ma nella **selettivita'**: segnalando il 10% delle barre il netto medio e' sotto
la commissione, segnalando lo 0,5% e' dieci volte sopra (`ml/entry_trainer`). Ne segue che soglia e
tenuta *sono* il modello: servirlo con altri due numeri e' servire un'altra strategia, e nessun
risultato misurato la descrive.

Il modello e' finto. Cio' che si verifica e' il codice del servizio, non un artefatto addestrato
che nel repository non c'e'.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml import signals


class ModelloFinto:
    def __init__(self, previsioni):
        self.previsioni = np.asarray(previsioni, dtype=float)

    def predict(self, X):
        return np.resize(self.previsioni, len(X))


@pytest.fixture()
def candele():
    n = 400
    index = pd.date_range("2024-01-01", periods=n, freq="1h", name="Open time")
    close = 100 * np.exp(np.linspace(0, 0.3, n))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close, "Volume": np.full(n, 1000.0)},
        index=index,
    )


def test_due_segnali_dentro_una_posizione_sono_una_posizione_sola():
    """E' la regola con cui il modello e' stato misurato: contarli tutti misura un capitale che
    non si ha, e qui produrrebbe una posizione che non finisce mai."""
    dentro = signals.entry_exposure(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), soglia=0.5, tenuta=3)

    assert list(dentro) == [True, True, True, False, False, False]


def test_dopo_la_tenuta_si_puo_rientrare():
    dentro = signals.entry_exposure(np.array([1.0, 0.0, 1.0, 0.0]), soglia=0.5, tenuta=2)

    assert list(dentro) == [True, True, True, True]


def test_le_barre_di_riscaldamento_non_entrano():
    """Sul riscaldamento la previsione e' NaN, e un confronto con NaN e' falso: va scritto, perche'
    `nan >= soglia` e `nan < soglia` sono entrambi falsi e l'uno o l'altro verso cambia tutto."""
    dentro = signals.entry_exposure(np.array([np.nan, np.nan, 1.0]), soglia=0.5, tenuta=1)

    assert list(dentro) == [False, False, True]


def test_la_tenuta_e_un_tempo_non_un_numero_di_candele():
    """150 barre a 5m sono dodici ore e mezza: sul grafico a 1h devono restare dodici ore e mezza.

    Servire «150 candele» a 1h vorrebbe dire tenere la posizione sei giorni invece di mezzo, con
    una soglia calibrata su tutt'altro orizzonte.
    """
    servizio = {"tenuta": 150}

    assert signals.entry_tenuta(pd.date_range("2024-01-01", periods=5, freq="5min"), servizio) == 150
    assert signals.entry_tenuta(pd.date_range("2024-01-01", periods=5, freq="1h"), servizio) == 13
    assert signals.entry_tenuta(pd.date_range("2024-01-01", periods=5, freq="1D"), servizio) == 1


def test_senza_metadata_non_si_opera(candele, tmp_path, monkeypatch):
    """Soglia e tenuta stanno nell'artefatto. Senza, l'unica alternativa sarebbe inventarle -- e
    una soglia inventata seleziona un'altra popolazione di barre, in silenzio."""
    monkeypatch.setattr(signals, "MODELS_DIR", tmp_path)

    assert signals.entry_signals(candele, ModelloFinto([1.0])) == ([], [])


def test_i_segnali_sono_alternati_e_seguono_lesposizione(candele, tmp_path, monkeypatch):
    monkeypatch.setattr(signals, "MODELS_DIR", tmp_path)
    (tmp_path / "entry_model.json").write_text(json.dumps({"servizio": {"soglia": 0.5, "tenuta": 12}}))
    modello = ModelloFinto([1.0] + [0.0] * 47)

    acquisti, vendite = signals.entry_signals(candele, modello)

    assert acquisti, "con la soglia superata ogni 48 barre le operazioni ci sono"
    assert len(acquisti) == len(vendite) or len(acquisti) == len(vendite) + 1
    tempi = [t for coppia in zip(acquisti, vendite) for t, _ in coppia]
    assert tempi == sorted(tempi), "un acquisto e la sua vendita devono essere in ordine"
