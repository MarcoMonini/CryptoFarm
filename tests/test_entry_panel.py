"""La scelta fra i due modelli d'ingresso, e il riquadro che li mette accanto alla loro domanda.

I due artefatti sono la stessa famiglia su due orizzonti e in servizio lavorano insieme -- il
veloce opera, il lento fa da cancello. Sulla pagina si possono invece guardare **uno alla volta**:
sono due strategie diverse, e la differenza si vede solo separandole.

Il modello e' finto. Cio' che si verifica e' l'instradamento della scelta e le unita' del riquadro,
non un artefatto addestrato che nel repository non c'e'.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml import signals
from cryptofarm.trading import panels
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.strategies import ai_model_simulation


class ModelloFinto:
    def __init__(self, previsione: float):
        self.previsione = previsione

    def predict(self, X):
        return np.full(len(X), self.previsione, dtype=float)


class ModelloAImpulsi:
    """Sopra soglia una barra ogni `periodo`, sotto altrove.

    Serve a rendere visibile la tenuta: con una previsione costante sopra soglia si rientra alla
    barra dopo l'uscita, l'esposizione diventa un blocco unico e i due orizzonti producono la
    stessa identica operazione -- cioe' il test passerebbe anche se la scelta non arrivasse.
    """

    def __init__(self, periodo: int):
        self.periodo = periodo

    def predict(self, X):
        return np.where(np.arange(len(X)) % self.periodo == 0, 1.0, 0.0)


@pytest.fixture()
def candele() -> pd.DataFrame:
    n = 900
    index = pd.date_range("2024-01-01", periods=n, freq="5min", name="Open time")
    close = 100 * np.exp(np.linspace(0, 0.2, n))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close, "Volume": np.full(n, 1000.0)},
        index=index,
    )


@pytest.fixture()
def artefatti(tmp_path, monkeypatch):
    """Metadata dei due modelli su disco, con gli orizzonti veri e soglie che lasciano passare."""
    import joblib

    monkeypatch.setattr(signals, "MODELS_DIR", tmp_path)
    for nome, h in ((signals.ENTRY_VELOCE, 20), (signals.ENTRY_LENTO, 150)):
        # Artefatti veri e non file vuoti: il cancello carica il lento da disco per conto suo, e
        # un `.joblib` finto lo farebbe fallire nel deserializzarlo invece che nel servire.
        joblib.dump(ModelloFinto(1.0), tmp_path / f"{nome}.joblib")
        (tmp_path / f"{nome}.json").write_text(
            json.dumps(
                {
                    "servizio": {"soglia": 0.5, "tenuta": h, "cancello": -1.0},
                    "labeling": {"method": "rendimento_futuro", "h": h, "base_interval": "5m"},
                }
            )
        )
    signals.entry_model.cache_clear()
    yield tmp_path
    signals.entry_model.cache_clear()


def test_la_scelta_arriva_al_modello_giusto(candele, artefatti):
    """`famiglia` sceglie l'artefatto: senza, la pagina servirebbe sempre quello in testa.

    Si guarda la **tenuta**, che e' l'unica cosa che distingue i due a parita' di previsione: 20
    barre il veloce, 150 il lento. Con un segnale ogni cento barre il veloce li prende tutti,
    mentre il lento e' ancora dentro a uno su due.
    """
    modello = ModelloAImpulsi(100)

    veloce = ai_model_simulation(candele, modello, famiglia=signals.ENTRY_VELOCE)
    lento = ai_model_simulation(candele, modello, famiglia=signals.ENTRY_LENTO)

    assert len(veloce[0]) == 8, "otto impulsi dopo il riscaldamento, tenuta piu' corta dell'attesa"
    assert len(lento[0]) == 4, "la tenuta da 150 barre ne copre uno su due"


def test_senza_famiglia_si_serve_quello_in_testa(candele, artefatti, monkeypatch):
    """La firma resta compatibile: chi non sceglie ottiene `MODEL_PRECEDENCE`, come prima."""
    from cryptofarm.trading import strategies

    monkeypatch.setattr(strategies, "active_model_name", lambda: signals.ENTRY_LENTO)
    impulsi = ModelloAImpulsi(100)

    senza = ai_model_simulation(candele, impulsi)

    assert senza == ai_model_simulation(candele, impulsi, famiglia=signals.ENTRY_LENTO)
    assert senza != ai_model_simulation(candele, impulsi, famiglia=signals.ENTRY_VELOCE)


def test_il_riquadro_confronta_due_serie_nella_stessa_unita(candele, artefatti):
    """Previsione e bersaglio stanno sullo stesso asse solo perche' sono la stessa quantita'.

    Il bersaglio e' il rendimento logaritmico delle prossime `h` barre e guarda avanti, quindi la
    coda esce vuota di esattamente `h` valori: e' cio' che rende il confronto leggibile e la serie
    non operabile. Un'unita' diversa -- l'etichetta a swing, che vive in [-1, 1] -- schiaccerebbe
    la previsione contro lo zero.
    """
    valori = {**panels.valori_predefiniti(), "FAMIGLIA": signals.ENTRY_VELOCE, "MODELLO": ModelloFinto(0.01)}

    serie = panels.INDICATORI["previsione_ingresso"].serie(candele, ExtraCache(candele), valori)

    assert serie["entry_bersaglio"].isna().sum() == 20, "la coda vuota e' lunga quanto l'orizzonte"
    assert serie["entry_previsto"].dropna().eq(0.01).all()
    assert serie["entry_soglia"].eq(0.5).all(), "la soglia disegnata e' quella dei metadata"
    atteso = np.log(candele["Close"].iloc[20] / candele["Close"].iloc[0])
    assert serie["entry_bersaglio"].iloc[0] == pytest.approx(atteso)


def test_il_bersaglio_copre_lo_stesso_tempo_a_ogni_intervallo(candele, artefatti):
    """`h` e' in barre da 5 minuti: a 15m sono sette barre, non venti.

    E' la stessa regola della tenuta. Disegnare venti barre da quindici minuti confronterebbe la
    previsione con cinque ore di futuro invece delle cento minuti su cui e' addestrata.
    """
    lente = candele.resample("15min").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    valori = {**panels.valori_predefiniti(), "FAMIGLIA": signals.ENTRY_VELOCE, "MODELLO": ModelloFinto(0.01)}

    serie = panels.INDICATORI["previsione_ingresso"].serie(lente, ExtraCache(lente), valori)

    assert serie["entry_bersaglio"].isna().sum() == 7


def test_senza_modello_il_riquadro_non_disegna(candele, artefatti):
    """E' la condizione della produzione, dove `models/` e' vuoto per costruzione."""
    valori = {**panels.valori_predefiniti(), "FAMIGLIA": "", "MODELLO": None}

    assert panels.INDICATORI["previsione_ingresso"].serie(candele, ExtraCache(candele), valori) == {}


def test_il_menu_offre_solo_gli_artefatti_che_ci_sono(artefatti, monkeypatch):
    """Un'etichetta che non ha un file dietro e' un errore che si scopre solo scegliendola."""
    from cryptofarm.trading import simulator

    assert set(simulator.modelli_dingresso().values()) == {signals.ENTRY_VELOCE, signals.ENTRY_LENTO}

    (artefatti / f"{signals.ENTRY_LENTO}.joblib").unlink()
    signals.entry_model.cache_clear()
    assert set(simulator.modelli_dingresso().values()) == {signals.ENTRY_VELOCE}

    (artefatti / f"{signals.ENTRY_VELOCE}.joblib").unlink()
    signals.entry_model.cache_clear()
    assert simulator.modelli_dingresso() == {}
