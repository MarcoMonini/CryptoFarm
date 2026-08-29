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


def ranking_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """AUC di P(buy) contro l'esito reale: c'e' segnale, indipendentemente dalla soglia?

    E' la domanda che va posta per prima. Precision e aspettativa dipendono da dove si mette la
    soglia e da quanto sono larghe le barriere; l'AUC no -- misura solo se il modello sa
    **ordinare** le candele meglio del caso. A 0,50 non c'e' nulla da estrarre e nessuna soglia
    potra' renderlo redditizio; sopra 0,55 c'e' segnale, e resta da vedere se basta a coprire le
    commissioni.
    """
    from sklearn.metrics import roc_auc_score

    positives = (y_true == BUY).astype(int)
    if positives.sum() == 0 or positives.sum() == len(positives):
        return float("nan")
    return float(roc_auc_score(positives, probabilities[:, BUY]))


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


# Quote di candele su cui si opera, dalla piu' selettiva alla piu' larga. Sono quantili del
# punteggio e non probabilita' assolute: una soglia come 0,40 significa cose opposte quando il
# base rate e' l'11% o il 37%, e su un modello poco calibrato puo' non essere raggiunta mai.
# "Opero sul mio miglior mezzo percento di occasioni" invece e' confrontabile ovunque.
DEFAULT_QUANTILES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def quantile_sweep(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    take_profit: np.ndarray | float,
    stop_loss: np.ndarray | float,
    round_trip_fee: float,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Per ogni quota di candele selezionate: win rate, break-even e aspettativa.

    Seleziona la frazione richiesta di candele con P(buy) piu' alta. E' la forma giusta della
    domanda operativa -- un bot non sceglie una probabilita', sceglie quanto essere selettivo --
    ed e' invariante al base rate e alla calibrazione del modello.
    """
    probability_buy = probabilities[:, BUY]
    take_profit = np.broadcast_to(np.asarray(take_profit, dtype=float), probability_buy.shape)
    stop_loss = np.broadcast_to(np.asarray(stop_loss, dtype=float), probability_buy.shape)
    order = np.argsort(-probability_buy)

    rows = []
    for quantile in quantiles:
        count = int(len(probability_buy) * quantile)
        if count < 1:
            continue
        selected = order[:count]
        wins = y_true[selected] == BUY
        outcome = np.where(wins, take_profit[selected] - round_trip_fee, -(stop_loss[selected] + round_trip_fee))
        # Aspettativa lorda: lo stesso conto senza commissioni. Separare le due dice se il
        # problema e' che il modello non prevede nulla o che l'edge c'e' ma non copre i costi --
        # due diagnosi opposte, con rimedi opposti.
        gross = np.where(wins, take_profit[selected], -stop_loss[selected])
        rows.append(
            {
                "soglia": float(probability_buy[selected].min()),
                "operazioni": count,
                "quota": quantile,
                "win_rate": float(wins.mean()),
                "break_even": break_even_precision(
                    float(take_profit[selected].mean()), float(stop_loss[selected].mean()), round_trip_fee
                ),
                "barriera": float(take_profit[selected].mean()),
                "atteso_lordo": float(gross.mean()),
                "atteso_per_trade": float(outcome.mean()),
                "atteso_totale": float(outcome.sum()),
            }
        )
    return pd.DataFrame(rows)


# Costi di esecuzione realistici su Binance spot, per lato.
FEE_SCENARIOS = {
    "market, standard (0,10%/lato)": 0.0020,
    "market, sconto BNB (0,075%/lato)": 0.0015,
    "maker/limit, standard (0,02%/lato)": 0.0004,
    "esecuzione a costo nullo": 0.0,
}


def fee_sensitivity(gross_per_trade: float, trades: int) -> pd.DataFrame:
    """Cosa resta dell'edge lordo sotto diversi regimi di commissioni.

    E' la tabella che conta quando l'edge esiste ma e' dello stesso ordine dei costi: la
    differenza fra una strategia che perde e una che guadagna non sta nel modello ma nel modo in
    cui gli ordini vengono eseguiti. Ordini a mercato pagano il taker, ordini limite il maker --
    su Binance un fattore cinque.
    """
    rows = []
    for name, fee in FEE_SCENARIOS.items():
        net = gross_per_trade - fee
        rows.append(
            {
                "esecuzione": name,
                "costo_andata_ritorno": fee,
                "netto_per_trade": net,
                "netto_totale": net * trades,
            }
        )
    return pd.DataFrame(rows)


def format_fee_sensitivity(table: pd.DataFrame) -> str:
    display = table.copy()
    display["costo_andata_ritorno"] = display["costo_andata_ritorno"].map(lambda value: f"{value:.3%}")
    display["netto_per_trade"] = display["netto_per_trade"].map(lambda value: f"{value:+.3%}")
    display["netto_totale"] = display["netto_totale"].map(lambda value: f"{value:+.1%}")
    return display.to_string(index=False)


def format_sweep(sweep: pd.DataFrame) -> str:
    """Rende leggibile lo sweep, con le percentuali gia' convertite."""
    display = sweep.copy()
    percent_columns = ("quota", "win_rate", "break_even", "barriera", "atteso_lordo", "atteso_per_trade")
    for column in (name for name in percent_columns if name in display.columns):
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
    "ranking_auc",
    "HOLD",
    "SELL",
    "break_even_precision",
    "trade_expectancy",
    "classification_summary",
    "threshold_sweep",
    "quantile_sweep",
    "fee_sensitivity",
    "format_fee_sensitivity",
    "FEE_SCENARIOS",
    "DEFAULT_QUANTILES",
    "format_sweep",
    "best_threshold",
    "lift_over_base_rate",
    "signal_summary",
]


# --- Il metro del modello a swing ---------------------------------------------------------------
# L'IC contro l'etichetta non dice se il modello serve: misurato il 2026-08-29, un modello con IC
# +0,50 contro l'etichetta a gambe segnalava minimi che rendevano +0,026% su 1,7 ore, contro lo
# 0,765% dei minimi veri e lo 0,2% di commissione. Il difetto non si vedeva perche' l'IC medio
# sulle barre premia l'interpolazione a meta' gamba, che e' l'80% del campione e non si opera.
#
# Qui il metro e' la **precisione sugli estremi**: delle barre che il modello segnala come minimi,
# quante lo sono davvero, e quanto rendono. Con quota 0,1 il caso vale 10%; il modello del
# 2026-08-29 valeva 30%, e serve circa il 50% perche' il segnalato copra le commissioni.


def precisione_estremi(
    previsto: np.ndarray,
    etichetta: np.ndarray,
    quota: float = 0.1,
    rendimento: np.ndarray | None = None,
) -> dict[str, float]:
    """Quante delle barre segnalate come minimi lo sono davvero, e quanto rendono.

    Segnalate e vere sono entrambe la coda bassa di ampiezza `quota`: i minimi, dove la previsione
    e l'etichetta valgono -1. Per i massimi si passano i due vettori cambiati di segno.

    `rendimento` sono i rendimenti futuri sull'orizzonte della gamba, e servono a tradurre la
    precisione in denaro: `rendimento_segnalato` e' cio' che incasserebbe chi opera sul modello,
    `rendimento_vero` il tetto dell'oracolo.
    """
    previsto, etichetta = np.asarray(previsto, float), np.asarray(etichetta, float)
    if not 0.0 < quota < 1.0:
        raise ValueError("quota deve stare fra 0 e 1")
    if len(previsto) != len(etichetta):
        raise ValueError("previsto ed etichetta devono avere la stessa lunghezza")
    segnalate = previsto <= np.quantile(previsto, quota)
    vere = etichetta <= np.quantile(etichetta, quota)
    precisione = float((segnalate & vere).sum() / max(segnalate.sum(), 1))
    esito = {
        "segnalate": int(segnalate.sum()),
        "precisione": precisione,
        "caso": quota,
        # Quante volte meglio del caso. E' la cifra confrontabile fra quote diverse.
        "vantaggio": precisione / quota,
    }
    if rendimento is not None:
        rendimento = np.asarray(rendimento, float)
        buoni = np.isfinite(rendimento)
        esito["rendimento_segnalato"] = float(np.mean(rendimento[segnalate & buoni]))
        esito["rendimento_vero"] = float(np.mean(rendimento[vere & buoni]))
    return esito


def format_precisione(esito: dict[str, float]) -> str:
    righe = [
        f"  segnalate                {esito['segnalate']:>9,} barre",
        f"  precisione               {100 * esito['precisione']:>8.1f}%   (caso {100 * esito['caso']:.0f}%, "
        f"vantaggio {esito['vantaggio']:.1f}x)",
    ]
    if "rendimento_segnalato" in esito:
        righe += [
            f"  rendimento del segnalato {100 * esito['rendimento_segnalato']:>+8.3f}%   <- il numero che conta",
            f"  rendimento dei minimi veri{100 * esito['rendimento_vero']:>+8.3f}%   (il tetto)",
        ]
    return "\n".join(righe)
