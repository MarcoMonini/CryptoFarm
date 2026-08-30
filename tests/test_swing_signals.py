"""La regola a esposizione del modello a swing, e cosa **non** e'.

Il modello prevede la prossimita' agli estremi locali, e la tentazione naturale -- comprare i
minimi previsti, vendere i massimi -- e' misurata in perdita a tutte le soglie
(`.claude/docs/modello-swing.md` §5.1): la forma del segnale e' a U, quindi il segno della
previsione non dice il verso. Cio' che resta e' `|previsione|` come interruttore di esposizione.
Questi test fissano quella lettura, perche' e' l'unica differenza che conta e non si vede
guardando i tipi.

Il modello e' finto e restituisce previsioni scritte a mano: la proprieta' verificata e' quella
del codice dei segnali, non quella di un artefatto addestrato che nel repository non c'e'.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml import signals


class ModelloFinto:
    """Restituisce la sequenza data, riciclata sulla lunghezza della matrice."""

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
        {
            "Open": close,
            "High": close * 1.004,
            "Low": close * 0.996,
            "Close": close,
            "Volume": np.full(n, 1000.0),
        },
        index=index,
    )


def test_i_due_poli_entrano_entrambi():
    """Il polo -1 e il polo +1 danno la **stessa** decisione: dentro. E' la forma a U."""
    dentro = signals.swing_exposure(np.array([-0.9, 0.0, 0.9, 0.0]), entra=0.5, esci=0.4, cadenza=1)
    assert list(dentro) == [True, False, True, False]


def test_listeresi_tiene_la_posizione_fra_le_due_soglie():
    """A 0,45 si resta dentro se ci si era, e fuori se non ci si era: e' cio' che evita il
    fruscio di commissioni attorno a una soglia sola."""
    dentro = signals.swing_exposure(np.array([0.6, 0.45, 0.1, 0.45]), entra=0.5, esci=0.4, cadenza=1)
    assert list(dentro) == [True, True, False, False]


def test_la_decisione_resta_ferma_fra_una_cadenza_e_laltra():
    """Fra due punti di decisione lo stato si tiene, anche se la previsione cambia."""
    previsto = np.array([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert list(signals.swing_exposure(previsto, entra=0.5, esci=0.4, cadenza=3)) == [True] * 3 + [False] * 3


def test_le_barre_senza_previsione_non_decidono():
    """Il riscaldamento e' NaN, e un NaN non deve chiudere una posizione aperta."""
    previsto = np.array([np.nan, 0.9, np.nan, np.nan])
    assert list(signals.swing_exposure(previsto, entra=0.5, esci=0.4, cadenza=1)) == [False, True, True, True]


def test_i_segnali_sono_alternati_e_seguono_lesposizione(candele):
    modello = ModelloFinto([0.9] * 24 + [0.0] * 24)
    acquisti, vendite = signals.swing_signals(candele, modello, entra=0.5, esci=0.4)
    assert len(acquisti) == len(vendite) or len(acquisti) == len(vendite) + 1
    tempi = [t for coppia in zip(acquisti, vendite) for t, _ in coppia]
    assert tempi == sorted(tempi), "un acquisto e la sua vendita devono essere in ordine"


def test_le_scale_lunghe_sono_solo_quelle_piu_lunghe_della_base(candele):
    """A base 1h non si aggrega a un'ora: sarebbe ricampionare all'insu', cioe' inventare barre.

    Le colonne che restano fuori diventano NaN, non spariscono: la matrice deve avere sempre le
    41 colonne su cui il modello e' stato addestrato, altrimenti gli alberi leggono la colonna
    sbagliata senza dare nessun segno.
    """
    from cryptofarm.ml.bar_features import SWING_COLUMNS

    viste = {}

    class Registra(ModelloFinto):
        def predict(self, X):
            viste["forma"] = X.shape
            viste["tutte_nan"] = np.isnan(X).all(axis=0)
            return super().predict(X)

    signals.swing_predictions(candele, Registra([0.0]))
    assert viste["forma"] == (len(candele), len(SWING_COLUMNS))
    orarie = [i for i, c in enumerate(SWING_COLUMNS) if c.endswith("@1h")]
    assert viste["tutte_nan"][orarie].all(), "a base 1h le colonne @1h non sono ricavabili"


def test_una_cadenza_e_un_giorno_a_qualunque_intervallo():
    for freq, atteso in (("5min", 288), ("1h", 24), ("1D", 1)):
        index = pd.date_range("2024-01-01", periods=50, freq=freq)
        assert signals.swing_cadenza(index) == atteso


def test_una_scala_troppo_corta_non_viene_nemmeno_provata():
    """Con la finestra di default della pagina -- 240 ore -- la scala giornaliera ha dieci barre.

    `ExtraCache.adx(14)` passa da `ta`, che sotto due finestre solleva `IndexError` invece di
    restituire NaN: senza il taglio, la voce «AI Model» faceva cadere la pagina appena selezionata
    ai valori di partenza. In addestramento non si vedeva, perche' le serie sono di centinaia di
    migliaia di barre.
    """
    n = 240
    index = pd.date_range("2024-01-01", periods=n, freq="1h", name="Open time")
    close = 100 + np.arange(n, dtype=float)
    corte = pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": np.full(n, 1e3)},
        index=index,
    )
    previsto = signals.swing_predictions(corte, ModelloFinto([0.0]))
    assert len(previsto) == n
