"""Le due soglie del modello delle gambe, e il difetto che le confondeva.

Il difetto, visto in pagina: **tutte le operazioni lunghe una candela**, ingressi buoni e uscite
immediate. La causa era usare la stessa soglia sulle due teste. `P(su)` e `P(giu)` hanno
distribuzioni diverse -- misurato su BTC a 4h, 0,55 seleziona l'8% delle barre sulla prima e l'80%
sulla seconda -- quindi la condizione d'uscita era vera quasi sempre e ogni posizione si chiudeva
alla barra dopo essere stata aperta.

Il test usa un modello finto a probabilita' fisse: cosi' la proprieta' verificata e' quella del
codice dei segnali, non quella di un artefatto addestrato che nel repository non c'e'.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml import trainer
from cryptofarm.ml.signals import leg_signals


class ModelloFinto:
    """Probabilita' costanti nell'ordine (fermo, su, giu)."""

    def __init__(self, p_su: float, p_giu: float):
        self.p = np.array([1.0 - p_su - p_giu, p_su, p_giu])
        self.classes_ = np.array([0, 1, 2])

    def predict_proba(self, X):
        return np.repeat(self.p[None, :], len(X), axis=0)


@pytest.fixture()
def candele():
    """Una salita lenta e regolare: nessuno stop viene toccato, l'uscita puo' solo essere il
    modello o l'orizzonte. Isola la condizione che si vuole misurare."""
    n = 600
    index = pd.date_range("2024-01-01", periods=n, freq="4h", name="Open time")
    close = 100 * np.exp(np.linspace(0, 0.5, n))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=index,
    )


def _durate(candele, acquisti, vendite):
    posizione = pd.Series(np.arange(len(candele)), index=candele.index)
    return np.array([posizione[v[0]] - posizione[a[0]] for a, v in zip(acquisti, vendite)])


def test_una_soglia_alta_sulluscita_non_chiude_subito(candele):
    """Con `P(giu)` sotto la soglia d'uscita, le posizioni devono durare piu' di una barra."""
    modello = ModelloFinto(p_su=0.60, p_giu=0.35)
    acquisti, vendite = leg_signals(candele, modello, threshold=0.55, soglia_uscita=0.75)

    assert acquisti, "con P(su) sopra soglia ci devono essere ingressi"
    durate = _durate(candele, acquisti, vendite)
    assert durate.min() > 1, f"nessuna operazione deve durare una barra sola: {durate[:10]}"
    assert not any("P(giu)" in v[2] for v in vendite), "l'uscita a modello non doveva scattare"


def test_riusare_la_soglia_dingresso_sulluscita_chiude_ogni_barra(candele):
    """La regressione, scritta come tale: e' **questo** che faceva la pagina.

    Stesso modello, stessa serie: passando la soglia d'ingresso anche all'uscita, ogni posizione
    si chiude alla barra successiva. Il test non chiede di non poterlo fare -- chiede che la
    differenza fra i due casi resti visibile, perche' e' la firma del difetto.
    """
    modello = ModelloFinto(p_su=0.60, p_giu=0.35)
    acquisti, vendite = leg_signals(candele, modello, threshold=0.30, soglia_uscita=0.30)

    durate = _durate(candele, acquisti, vendite)
    # L'ultima operazione puo' durare zero barre, perche' la scadenza viene troncata a fine serie.
    assert (durate[:-1] == 1).all(), f"con la soglia bassa su entrambe le teste si esce sempre subito: {durate[:5]}"
    assert np.median(durate) == 1


def test_lo_stop_resta_attivo_anche_senza_uscita_a_modello(candele):
    """Lo stop e' una regola di rischio e non dipende dal parere del modello."""
    n = len(candele)
    discesa = candele.copy()
    crollo = np.linspace(1.0, 0.6, n)
    for colonna in ("Open", "High", "Low", "Close"):
        discesa[colonna] = discesa[colonna] * crollo

    modello = ModelloFinto(p_su=0.60, p_giu=0.35)
    acquisti, vendite = leg_signals(discesa, modello, threshold=0.55, soglia_uscita=0.99)
    assert acquisti
    assert all(v[2] in ("stop", "horizon") for v in vendite), [v[2] for v in vendite]
    assert any(v[2] == "stop" for v in vendite), "su una discesa lo stop deve scattare"


def test_uscita_su_modello_disattivabile(candele):
    """L'ablazione: senza la testa `P(giu)` restano solo stop e orizzonte."""
    # Le tre classi sono una softmax: le probabilita' devono sommare a uno, e un modello finto
    # che non lo rispetta verifica una condizione che non puo' presentarsi.
    modello = ModelloFinto(p_su=0.56, p_giu=0.44)
    _, con = leg_signals(candele, modello, threshold=0.55, soglia_uscita=0.40)
    _, senza = leg_signals(candele, modello, threshold=0.55, soglia_uscita=0.40, uscita_su_modello=False)

    assert con and senza
    # Tutte tranne l'ultima, che puo' cadere sulla scadenza troncata a fine serie.
    assert all("P(giu)" in v[2] for v in con[:-1]), [v[2] for v in con[-3:]]
    assert not any("P(giu)" in v[2] for v in senza)


def test_le_due_soglie_arrivano_dai_metadata_e_sono_distinte(tmp_path, monkeypatch):
    """`stored_exit_threshold` legge la propria chiave, non quella d'ingresso."""
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    assert trainer.stored_exit_threshold() == trainer.DEFAULT_EXIT_THRESHOLD

    (tmp_path / "leg_model.json").write_text(json.dumps({"decision_threshold": 0.55, "exit_threshold": 0.74}))
    assert trainer.stored_decision_threshold() == pytest.approx(0.55)
    assert trainer.stored_exit_threshold() == pytest.approx(0.74)


def test_il_default_duscita_e_alto(tmp_path, monkeypatch):
    """Senza calibrazione l'uscita a modello deve quasi non scattare: l'ablazione la dice dannosa."""
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    assert trainer.stored_exit_threshold() > trainer.DEFAULT_DECISION_THRESHOLD
