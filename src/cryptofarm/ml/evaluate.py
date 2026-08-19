"""Valutazione del modello, in termini economici prima che statistici.

Una macro F1 non dice se una strategia guadagna. Con il triple-barrier la traduzione e' diretta:
l'etichetta e' l'esito di un trade, quindi la precision sulle candele in cui il modello dice
"compra" **e'** il win rate, e da quella e dall'ampiezza delle barriere si ricava l'aspettativa
per operazione al netto delle commissioni.

Il numero che decide tutto e' la **precision di break-even**:

    p* = (u_sl + f) / ((u_tp - f) + (u_sl + f))

con `u_tp` e `u_sl` le ampiezze delle barriere e `f` le commissioni di andata e ritorno. Un
modello con precision sotto p* perde denaro anche se le sue metriche sembrano buone; sopra p*
guadagna anche se l'accuracy complessiva e' modesta. E' l'unica soglia che conta.

Da qui discende anche come si usa il modello: non con l'argmax, ma con una **soglia sulla
probabilita'**, scelta perche' massimizza l'aspettativa. Ed e' il motivo per cui il dataset non
va ribilanciato -- un modello ribilanciato produce probabilita' che non corrispondono piu' alle
frequenze reali, e la soglia perde significato.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from cryptofarm.ml.labeling import BUY, HOLD, SELL


def break_even_precision(take_profit: float, stop_loss: float, round_trip_fee: float) -> float:
    """Precision minima perche' operare non sia in perdita."""
    gain = take_profit - round_trip_fee
    loss = stop_loss + round_trip_fee
    if gain <= 0:
        return float("inf")
    return loss / (gain + loss)


def trade_expectancy(precision: float, take_profit: float, stop_loss: float, round_trip_fee: float) -> float:
    """Rendimento atteso per operazione, in frazione del capitale impegnato."""
    return precision * (take_profit - round_trip_fee) - (1.0 - precision) * (stop_loss + round_trip_fee)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    report = classification_report(
        y_true, y_pred, labels=[HOLD, BUY, SELL], target_names=["hold", "buy", "sell"], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[HOLD, BUY, SELL])
    return f"{report}\nConfusion matrix (righe=reale, colonne=predetto), ordine hold/buy/sell:\n{matrix}"


def threshold_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    take_profit: np.ndarray | float,
    stop_loss: np.ndarray | float,
    round_trip_fee: float,
    thresholds: tuple[float, ...] = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70),
) -> pd.DataFrame:
    """Per ogni soglia su P(buy): quante operazioni, con che win rate e con che aspettativa.

    L'aspettativa e' calcolata sulle barriere **effettive** delle candele selezionate, non su un
    valore mediano: le barriere sono proporzionali alla volatilita', quindi selezionare candele
    piu' o meno volatili cambia il rendimento per operazione oltre al win rate.
    """
    probability_buy = probabilities[:, BUY]
    take_profit = np.broadcast_to(np.asarray(take_profit, dtype=float), probability_buy.shape)
    stop_loss = np.broadcast_to(np.asarray(stop_loss, dtype=float), probability_buy.shape)

    rows = []
    for threshold in thresholds:
        selected = probability_buy >= threshold
        count = int(selected.sum())
        if count == 0:
            rows.append(
                {
                    "soglia": threshold,
                    "operazioni": 0,
                    "quota": 0.0,
                    "win_rate": np.nan,
                    "break_even": np.nan,
                    "atteso_per_trade": np.nan,
                    "atteso_totale": np.nan,
                }
            )
            continue

        wins = y_true[selected] == BUY
        win_rate = float(wins.mean())
        # Il rendimento realizzato di ogni operazione: la sua barriera di profitto se vinta, la
        # sua barriera di perdita se persa, sempre al netto delle commissioni.
        outcome = np.where(wins, take_profit[selected] - round_trip_fee, -(stop_loss[selected] + round_trip_fee))
        rows.append(
            {
                "soglia": threshold,
                "operazioni": count,
                "quota": count / len(probability_buy),
                "win_rate": win_rate,
                "break_even": break_even_precision(
                    float(take_profit[selected].mean()), float(stop_loss[selected].mean()), round_trip_fee
                ),
                "atteso_per_trade": float(outcome.mean()),
                "atteso_totale": float(outcome.sum()),
            }
        )
    return pd.DataFrame(rows)


def format_sweep(sweep: pd.DataFrame) -> str:
    """Rende leggibile lo sweep, con le percentuali gia' convertite."""
    display = sweep.copy()
    for column in ("quota", "win_rate", "break_even", "atteso_per_trade"):
        display[column] = display[column].map(lambda value: "-" if pd.isna(value) else f"{value:.2%}")
    display["atteso_totale"] = display["atteso_totale"].map(lambda value: "-" if pd.isna(value) else f"{value:+.1f}x")
    display["soglia"] = display["soglia"].map(lambda value: f"{value:.2f}")
    return display.to_string(index=False)


def best_threshold(sweep: pd.DataFrame, min_trades: int = 200) -> dict | None:
    """La soglia con l'aspettativa totale piu' alta, tra quelle con abbastanza operazioni.

    Il vincolo sul numero minimo esiste perche' le soglie molto alte selezionano pochissime
    candele: la loro aspettativa e' rumore, e presa alla lettera porterebbe a un modello che
    opera tre volte l'anno su una statistica priva di significato.
    """
    eligible = sweep[(sweep["operazioni"] >= min_trades) & sweep["atteso_per_trade"].notna()]
    if eligible.empty:
        return None
    return eligible.loc[eligible["atteso_totale"].idxmax()].to_dict()


def lift_over_base_rate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> float:
    """Quanto la selezione del modello migliora la frequenza dei buy rispetto al non selezionare."""
    base_rate = float((y_true == BUY).mean())
    selected = probabilities[:, BUY] >= threshold
    if selected.sum() == 0 or base_rate == 0:
        return float("nan")
    return float((y_true[selected] == BUY).mean() / base_rate)


def signal_summary(probabilities: np.ndarray, threshold: float) -> str:
    """Quante candele superano la soglia e come sono distribuite le probabilita'."""
    probability_buy = probabilities[:, BUY]
    quantiles = np.percentile(probability_buy, [50, 90, 99])
    return (
        f"P(buy): mediana {quantiles[0]:.3f}, p90 {quantiles[1]:.3f}, p99 {quantiles[2]:.3f} | "
        f"sopra soglia {threshold:.2f}: {int((probability_buy >= threshold).sum())} su {len(probability_buy)}"
    )


__all__ = [
    "BUY",
    "HOLD",
    "SELL",
    "break_even_precision",
    "trade_expectancy",
    "classification_summary",
    "threshold_sweep",
    "format_sweep",
    "best_threshold",
    "lift_over_base_rate",
    "signal_summary",
]
