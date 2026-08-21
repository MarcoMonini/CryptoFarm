"""Politica a tre azioni condizionata sullo stato della posizione.

Il modello precedente prevedeva una proprieta' del mercato ("questa candela e' un buon acquisto")
e lasciava a un livello sopra il compito di trasformarla in condotta. Qui prevede direttamente
**l'azione**, e per farlo deve sapere in che stato si trova: comprare quando si e' gia' dentro non
e' un errore di previsione, e' un'azione inesistente.

Tre conseguenze, che sono le tre parti di questo modulo.

**Le azioni non valide vengono mascherate, non punite.** Da flat esistono solo HOLD e BUY; da long
solo HOLD e SELL. Mascherare a inferenza invece di sperare che il modello impari a non emetterle
libera capacita' del modello per la sola domanda che conta davvero in quello stato.

**Lo stato entra come feature.** Senza, il modello vede due volte la stessa candela con due azioni
corrette diverse e non ha modo di distinguerle: impara la media delle due, che non e' nessuna
delle due.

**Lo stato di addestramento va randomizzato.** Se lo stato si genera solo seguendo le decisioni
dell'esperto, il modello vede solo gli stati in cui l'esperto passa, e non ha mai visto uno stato
raggiunto per errore -- che e' l'unico tipo di stato in cui si trovera' quando sbagliera'. E'
l'errore di composizione che il DAgger di `dagger.py` corregge iterativamente; randomizzare
l'ingresso e' la sua versione a costo zero, e va fatta comunque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.ml.directional_change import BUY, HOLD, SELL

FLAT, LONG = 0, 1

# Su spot non si vende allo scoperto: SELL chiude una posizione, non ne apre una al contrario.
VALID_ACTIONS = {FLAT: (HOLD, BUY), LONG: (HOLD, SELL)}

POSITION_FEATURES = ("STATE_IN", "STATE_PNL", "STATE_BARS")

# Scala su cui si comprime la durata della posizione. 288 barre sono 24 ore su 5m: oltre quel
# punto la differenza fra "aperta da due giorni" e "da tre" non cambia la decisione.
STATE_BARS_SCALE = 288.0


def action_mask(state: np.ndarray) -> np.ndarray:
    """Matrice (righe x 3) vera dove l'azione e' eseguibile nello stato dato."""
    state = np.asarray(state)
    mask = np.zeros((len(state), 3), dtype=bool)
    mask[:, HOLD] = True
    mask[:, BUY] = state == FLAT
    mask[:, SELL] = state == LONG
    return mask


def mask_probabilities(probabilities: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Azzera le azioni non eseguibili e rinormalizza.

    Rinormalizzare invece di prendere l'argmax fra le sole valide serve a mantenere confrontabili
    le probabilita' fra stati diversi: la soglia di decisione si tara una volta sola.
    """
    masked = np.where(action_mask(state), probabilities, 0.0)
    total = masked.sum(axis=1, keepdims=True)
    # Uno stato con probabilita' nulla su tutte le valide non esiste (HOLD e' sempre valida), ma
    # se il modello emette esattamente zero la divisione va comunque protetta.
    return np.divide(masked, total, out=np.zeros_like(masked), where=total > 0)


def expert_actions(signals: np.ndarray, state: np.ndarray) -> np.ndarray:
    """L'azione corretta dato il segnale dell'etichetta e lo stato in cui ci si trova.

    L'esperto non e' un oracolo sul futuro oltre l'etichetta: e' l'etichetta stessa **letta
    attraverso lo stato**. Da flat un segnale BUY diventa un ingresso e tutto il resto e' attesa;
    da long un segnale SELL diventa un'uscita e tutto il resto e' mantenere -- restare dentro
    durante una zona BUY e' esattamente la condotta giusta, non un'occasione mancata.
    """
    signals, state = np.asarray(signals), np.asarray(state)
    actions = np.full(len(signals), HOLD, dtype=np.int8)
    actions[(state == FLAT) & (signals == BUY)] = BUY
    actions[(state == LONG) & (signals == SELL)] = SELL
    return actions


def position_features(
    close: np.ndarray,
    state: np.ndarray,
    entry_price: np.ndarray,
    bars_in_position: np.ndarray,
) -> pd.DataFrame:
    """Le tre colonne che descrivono la posizione, tutte scale-free come il resto delle feature.

    `STATE_PNL` e' il rendimento non realizzato: e' la variabile che decide un'uscita, perche' la
    stessa candela va tenuta se si e' in guadagno di mezzo punto e mollata se si e' sotto di due.
    `STATE_BARS` da' al modello il tempo trascorso, senza il quale non puo' distinguere una
    posizione appena aperta da una che sta marcendo.
    """
    state = np.asarray(state)
    in_position = (state == LONG).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        pnl = np.where(state == LONG, close / entry_price - 1.0, 0.0)
    return pd.DataFrame(
        {
            "STATE_IN": in_position,
            "STATE_PNL": np.nan_to_num(pnl).astype(np.float32),
            "STATE_BARS": (in_position * np.minimum(bars_in_position, STATE_BARS_SCALE) / STATE_BARS_SCALE).astype(
                np.float32
            ),
        }
    )


def randomised_states(
    close: np.ndarray,
    rows: np.ndarray,
    rng: np.random.Generator,
    long_fraction: float = 0.5,
    max_bars: int = int(STATE_BARS_SCALE),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stato di posizione campionato invece che dedotto, per le righe indicate.

    L'ingresso non e' inventato: si campiona **quante barre fa** e si prende il prezzo di chiusura
    di allora. Il P&L che ne risulta e' quindi un P&L che quel mercato ha davvero prodotto, con la
    sua distribuzione vera -- un P&L estratto da una gaussiana insegnerebbe al modello relazioni
    fra guadagno e contesto che nei dati non esistono.

    `long_fraction` a 0,5 e' deliberato: e' l'unico modo di avere altrettanti esempi di uscita
    quanti di ingresso, mentre una politica ragionevole passa la maggior parte del tempo flat.
    """
    rows = np.asarray(rows)
    state = np.where(rng.random(len(rows)) < long_fraction, LONG, FLAT).astype(np.int8)
    # Distribuzione geometrica: le posizioni giovani sono le piu' frequenti, come nella realta',
    # ma la coda lunga garantisce anche esempi di posizioni vecchie.
    bars = np.minimum(rng.geometric(1.0 / 24.0, len(rows)), max_bars)
    bars = np.minimum(bars, rows)  # non si puo' essere entrati prima dell'inizio della serie
    entry = close[rows - bars]
    return state, entry.astype(np.float64), bars.astype(np.float64)


def simulate_expert_states(signals: np.ndarray) -> np.ndarray:
    """Lo stato in cui si trova chi segue l'esperto alla lettera, barra per barra.

    Serve come termine di paragone della randomizzazione: e' la distribuzione di stati che il
    modello vedrebbe **senza** randomizzare, ed e' visibilmente piu' povera.
    """
    state = np.zeros(len(signals), dtype=np.int8)
    current = FLAT
    for bar, signal in enumerate(signals):
        state[bar] = current
        if current == FLAT and signal == BUY:
            current = LONG
        elif current == LONG and signal == SELL:
            current = FLAT
    return state
