"""Meta-labeling: il modello non predice il mercato, predice se un trade vale la pena.

Struttura in due stadi, come indicato in `strategy.md`:

- **Primario**: il filtro CUSUM. E' deliberatamente permissivo -- segnala ogni volta che il
  prezzo ha accumulato un movimento di dimensione rilevante, circa 30 volte al giorno per
  simbolo, senza pretendere di sapere in che direzione andra'. Recall alto, precision bassa.
- **Secondario**: un classificatore binario addestrato **solo sui candidati del primario**, che
  risponde a "questo ingresso chiude in profitto netto?".

Il guadagno non e' architetturale, e' nella definizione dell'etichetta. Quella del secondario e'
gia' **al netto di commissioni e riempimento**: incorpora il costo maker in ingresso, quello
taker in uscita, e il fatto che un ordine limite possa non riempirsi affatto. Quindi la
precision del secondario *e'* il win rate netto della strategia, senza traduzioni intermedie --
ed e' il motivo per cui il vincolo economico smette di essere un filtro applicato a valle di un
modello che lo ignora.

Un secondo effetto, non secondario: le classi risultano quasi bilanciate invece che al 3% come
nel labeling per estremi, quindi le probabilita' restano calibrate e una soglia su di esse ha
significato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.ml.execution import apply_execution, limit_fills
from cryptofarm.ml.labeling import triple_barrier_events


def build_meta_labels(
    features: pd.DataFrame,
    event_positions: np.ndarray,
    horizon_bars: int,
    tp_multiple: float,
    sl_multiple: float,
    round_trip_fee: float,
    fee_floor_multiple: float,
    offset_atr: float,
    patience: int,
    entry_mode: str = "maker",
    exit_mode: str = "taker",
) -> pd.DataFrame:
    """Etichette del secondario sui soli eventi del primario.

    Colonne restituite (indicizzate sui timestamp degli eventi):

    - `meta_label`  1 se il trade chiude in profitto netto, 0 altrimenti
    - `net_return`  rendimento netto realizzato (0 se l'ordine non si e' riempito)
    - `traded`      se l'ordine limite si e' riempito
    - `t_start` / `t_exit`  vita dell'osservazione, indispensabile al purging
    - `outcome`     esito della barriera: 0 timeout, 1 take-profit, 2 stop-loss
    """
    events = triple_barrier_events(
        features,
        horizon=horizon_bars,
        tp_multiple=tp_multiple,
        sl_multiple=sl_multiple,
        round_trip_fee=round_trip_fee,
        fee_floor_multiple=fee_floor_multiple,
    )
    events["Close"] = features["Close"].to_numpy()

    selected = events.iloc[event_positions]
    fills = limit_fills(features, event_positions, offset_atr=offset_atr, patience=patience)
    executed = apply_execution(selected, fills, entry_mode=entry_mode, exit_mode=exit_mode)

    return pd.DataFrame(
        {
            "meta_label": (executed["net_return"].to_numpy() > 0).astype(np.int8),
            "net_return": executed["net_return"].to_numpy(),
            "traded": executed["traded"].to_numpy(),
            "outcome": selected["Label"].to_numpy(),
            "t_start": selected.index,
            "t_exit": selected["t_exit"].to_numpy(),
            "tp_width": selected["tp_width"].to_numpy(),
            "sl_width": selected["sl_width"].to_numpy(),
        },
        index=selected.index,
    )


def expectancy_by_quantile(
    scores: np.ndarray,
    net_return: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.10, 0.13, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    """Aspettativa netta operando solo sulla frazione di eventi con punteggio piu' alto.

    E' la forma operativa della domanda: un bot non sceglie una probabilita', sceglie **quanto
    essere selettivo**. La quota 0,13 corrisponde ai 4 trade/giorno/simbolo del target, dati
    ~30 eventi CUSUM al giorno.
    """
    order = np.argsort(-scores)
    rows = []
    for quantile in quantiles:
        count = max(1, int(len(scores) * quantile))
        chosen = order[:count]
        returns = net_return[chosen]
        rows.append(
            {
                "quota": quantile,
                "operazioni": count,
                "soglia": float(scores[chosen].min()),
                "win_rate": float((returns > 0).mean()),
                "atteso_netto": float(returns.mean()),
                "totale": float(returns.sum()),
            }
        )
    return pd.DataFrame(rows)
