"""DAgger: addestrare sugli stati che la politica visita davvero, non su quelli dell'esperto.

Il problema che risolve e' l'unico specifico dell'imitazione, e non si vede in nessuna metrica di
classificazione. Un modello addestrato sulle scelte dell'esperto vede solo gli stati **in cui
l'esperto passa**: posizioni aperte al momento giusto, mai una posizione aperta per errore due
barre prima di un crollo. Alla prima decisione sbagliata la politica finisce in uno stato che non
ha mai visto, sbaglia di nuovo con piu' margine, e l'errore si compone lungo la traiettoria. La
precision fuori campione resta ottima mentre il P&L affonda, perche' misurano cose diverse:
l'accuratezza e' misurata sulla distribuzione dell'esperto, il P&L su quella della politica.

DAgger chiude l'anello: si fa girare la politica corrente, si raccolgono gli stati in cui
*lei* finisce, l'esperto etichetta quegli stati, e li si aggiunge al dataset. La randomizzazione
dello stato in `policy.py` copre lo stesso buco a costo zero ma alla cieca, campionando stati
plausibili invece di quelli effettivamente raggiunti; le due cose si sommano.

**Il rollout e' sequenziale per costruzione** -- l'azione di adesso decide lo stato di dopo -- e
questo lo renderebbe insostenibile: mezzo milione di chiamate a `predict` per simbolo. Qui si
batchano gli **episodi**: la serie si spezza in tratti indipendenti che avanzano in parallelo,
quindi il numero di chiamate scende alla lunghezza di un tratto e ogni chiamata ne serve
qualche centinaio. Stesso risultato, due ordini di grandezza in meno di attese.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.ml.directional_change import BUY, SELL
from cryptofarm.ml.models import predict_proba
from cryptofarm.ml.policy import (
    FLAT,
    LONG,
    STATE_BARS_SCALE,
    expert_actions,
    mask_probabilities,
    position_features,
)

DEFAULT_EPISODE_BARS = 2000


def episode_starts(rows: int, episode_bars: int = DEFAULT_EPISODE_BARS) -> np.ndarray:
    """Inizi dei tratti in cui spezzare la serie perche' avanzino in parallelo."""
    return np.arange(0, max(rows - episode_bars, 1), episode_bars)


def rollout(
    model,
    market: np.ndarray,
    close: np.ndarray,
    signals: np.ndarray,
    episode_bars: int = DEFAULT_EPISODE_BARS,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    """Fa girare la politica e restituisce gli stati visitati con l'azione dell'esperto.

    `market` sono le feature di mercato gia' costruite, senza le colonne di posizione: quelle
    vengono riempite qui a ogni passo, ed e' l'unica parte che cambia lungo la traiettoria.

    La colonna `action` e' cio' che la politica ha fatto, `expert` cio' che avrebbe dovuto fare.
    Le righe dove le due divergono sono il valore aggiunto dell'iterazione; quelle dove coincidono
    non sono inutili, tengono il dataset rappresentativo.
    """
    starts = episode_starts(len(close), episode_bars)
    episodes = len(starts)
    state = np.full(episodes, FLAT, dtype=np.int8)
    entry = close[starts].astype(np.float64)
    bars_in = np.zeros(episodes, dtype=np.float64)

    visited_rows, visited_state, visited_entry, visited_bars, visited_action = [], [], [], [], []

    for step in range(episode_bars):
        rows = starts + step
        alive = rows < len(close)
        if not alive.any():
            break
        rows = rows[alive]
        current_state, current_entry, current_bars = state[alive], entry[alive], bars_in[alive]

        block = position_features(close[rows], current_state, current_entry, current_bars)
        features = np.hstack([market[rows], block.to_numpy()])
        probabilities = mask_probabilities(predict_proba(model, features), current_state)

        # Si agisce solo con convinzione: sotto la soglia si resta fermi. E' la stessa regola che
        # vale in produzione, e simularne una diversa produrrebbe stati che non si verificheranno.
        actions = np.where(probabilities.max(axis=1) >= decision_threshold, np.argmax(probabilities, axis=1), 0)

        visited_rows.append(rows)
        visited_state.append(current_state.copy())
        visited_entry.append(current_entry.copy())
        visited_bars.append(current_bars.copy())
        visited_action.append(actions)

        entering = (current_state == FLAT) & (actions == BUY)
        exiting = (current_state == LONG) & (actions == SELL)
        next_state = np.where(entering, LONG, np.where(exiting, FLAT, current_state)).astype(np.int8)
        next_entry = np.where(entering, close[rows], current_entry)
        next_bars = np.where(
            entering, 0.0, np.where(next_state == LONG, np.minimum(current_bars + 1, STATE_BARS_SCALE), 0.0)
        )

        state[alive], entry[alive], bars_in[alive] = next_state, next_entry, next_bars

    rows = np.concatenate(visited_rows)
    visited = pd.DataFrame(
        {
            "row": rows,
            "state": np.concatenate(visited_state),
            "entry_price": np.concatenate(visited_entry),
            "bars_in_position": np.concatenate(visited_bars),
            "action": np.concatenate(visited_action).astype(np.int8),
        }
    )
    visited["expert"] = expert_actions(signals[visited["row"].to_numpy()], visited["state"].to_numpy())
    return visited


def state_coverage(visited: pd.DataFrame) -> dict[str, float]:
    """Quanto la traiettoria della politica si discosta da quella dell'esperto.

    `disaccordo` e' il numero che dice se l'iterazione e' servita: quando smette di scendere il
    DAgger ha finito, e continuare aggiunge righe senza aggiungere informazione.
    """
    return {
        "righe": float(len(visited)),
        "quota_long": float((visited["state"] == LONG).mean()),
        "disaccordo": float((visited["action"] != visited["expert"]).mean()),
        "ingressi_mancati": float(((visited["expert"] == BUY) & (visited["action"] != BUY)).mean()),
        "uscite_mancate": float(((visited["expert"] == SELL) & (visited["action"] != SELL)).mean()),
    }
