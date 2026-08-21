"""Esecuzione: cosa succede davvero quando si piazza un ordine limite.

Tutta la strategia vive in modalita' maker -- con commissioni taker il divario da colmare e' di
12-23 punti percentuali di win rate, con commissioni maker di 3-5. Ma una simulazione maker che
assume riempimento certo **sovrastima proprio la variabile su cui la strategia poggia**, e lo fa
per un motivo che non e' un dettaglio implementativo:

**la selezione avversa.** Un ordine limite di acquisto sotto il prezzo si riempie quando il
mercato scende fino a li'. Quindi si riempie *sistematicamente* nei momenti in cui il prezzo sta
scendendo, e non si riempie quando sale -- cioe' proprio quando il trade sarebbe stato buono. Il
riempimento non e' un evento casuale indipendente dall'esito: e' correlato negativamente con esso.
Ignorarlo significa attribuirsi i trade vincenti che in realta' non sarebbero mai partiti.

Il modello qui e' deliberatamente semplice e conservativo:

- l'ordine si riempie se il prezzo **attraversa** il livello entro la finestra di attesa;
- se non si riempie, il trade non avviene: e' un costo opportunita', non un evento neutro;
- il riempimento e' valutato sui minimi e massimi delle candele successive, mai su quella corrente.

Non modella la coda del book (la posizione in coda, il fatto che il prezzo debba *superare* il
livello e non solo toccarlo per garantire l'esecuzione). Per questo `require_cross` esiste ed e'
attivo di default: richiedere l'attraversamento stretto invece del semplice tocco e' il modo
piu' economico di essere conservativi su cio' che non si modella.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Costi per lato su Binance spot.
MAKER_FEE = 0.0002
TAKER_FEE = 0.0010
BNB_TAKER_FEE = 0.00075

FEE_MODES = {"maker": MAKER_FEE, "taker": TAKER_FEE, "bnb": BNB_TAKER_FEE}

# Quanto sotto il prezzo corrente si piazza il limite, in frazione dell'ATR.
DEFAULT_OFFSET_ATR = 0.25
# Per quante candele si lascia l'ordine sul book prima di rinunciare.
DEFAULT_PATIENCE_BARS = 6


def limit_fills(
    df: pd.DataFrame,
    entry_positions: np.ndarray,
    offset_atr: float = DEFAULT_OFFSET_ATR,
    patience: int = DEFAULT_PATIENCE_BARS,
    require_cross: bool = True,
) -> pd.DataFrame:
    """Simula un ordine limite di acquisto per ogni posizione di ingresso richiesta.

    Il limite e' posto a `Close * (1 - offset_atr * ATR%)`, e resta sul book `patience` candele.

    Restituisce, per ogni ingresso: se si e' riempito, a quale prezzo, dopo quante candele, e --
    dato chiave per misurare la selezione avversa -- quale sarebbe stato il rendimento del
    mercato nella finestra di attesa.
    """
    close = df["Close"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    atr = df["ATR"].to_numpy(dtype=float) / 100.0

    entry_positions = np.asarray(entry_positions, dtype=np.int64)
    limit_price = close[entry_positions] * (1.0 - offset_atr * atr[entry_positions])

    filled = np.zeros(len(entry_positions), dtype=bool)
    fill_bar = np.full(len(entry_positions), -1, dtype=np.int64)
    forward_return = np.full(len(entry_positions), np.nan)

    last = len(close) - 1
    for i, position in enumerate(entry_positions):
        stop = min(position + patience, last)
        if stop <= position:
            continue
        window = low[position + 1 : stop + 1]
        touched = window < limit_price[i] if require_cross else window <= limit_price[i]
        if touched.any():
            filled[i] = True
            fill_bar[i] = position + 1 + int(touched.argmax())
        forward_return[i] = close[stop] / close[position] - 1.0

    return pd.DataFrame(
        {
            "entry_bar": entry_positions,
            "limit_price": limit_price,
            "filled": filled,
            "fill_bar": fill_bar,
            "fill_price": np.where(filled, limit_price, np.nan),
            "market_return_during_wait": forward_return,
        }
    )


def adverse_selection_report(fills: pd.DataFrame) -> dict:
    """Quantifica la selezione avversa: il mercato si muove diversamente quando l'ordine si riempie?

    Se il rendimento medio del mercato durante l'attesa e' **negativo sui riempiti e positivo sui
    non riempiti**, la selezione avversa e' presente e misurata. E' il numero che dice quanto una
    simulazione a riempimento certo starebbe mentendo.
    """
    filled = fills["filled"].to_numpy()
    market = fills["market_return_during_wait"].to_numpy()
    valid = np.isfinite(market)
    if valid.sum() == 0:
        return {}
    on_fill = market[valid & filled]
    on_miss = market[valid & ~filled]
    return {
        "fill_rate": float(filled[valid].mean()),
        "market_return_when_filled": float(on_fill.mean()) if len(on_fill) else float("nan"),
        "market_return_when_missed": float(on_miss.mean()) if len(on_miss) else float("nan"),
        "adverse_selection": (
            float(on_miss.mean() - on_fill.mean()) if len(on_fill) and len(on_miss) else float("nan")
        ),
        "n_filled": int(len(on_fill)),
        "n_missed": int(len(on_miss)),
    }


def round_trip_cost(entry_mode: str = "maker", exit_mode: str = "taker") -> float:
    """Commissioni di andata e ritorno per una combinazione di modalita' di esecuzione.

    Il default riflette la realta' operativa: l'ingresso puo' attendere sul book (maker), l'uscita
    su stop-loss no -- deve avvenire subito, quindi paga il taker. Assumere maker su entrambi i
    lati e' l'ottimismo piu' comune in questo tipo di simulazione.
    """
    for mode in (entry_mode, exit_mode):
        if mode not in FEE_MODES:
            raise ValueError(f"modalita' sconosciuta: {mode!r}. Disponibili: {sorted(FEE_MODES)}")
    return FEE_MODES[entry_mode] + FEE_MODES[exit_mode]


def apply_execution(
    events: pd.DataFrame,
    fills: pd.DataFrame,
    entry_mode: str = "maker",
    exit_mode: str = "taker",
) -> pd.DataFrame:
    """Applica esito del riempimento e costi reali agli eventi triple-barrier.

    Il costo di uscita **dipende da quale barriera e' stata toccata**, e trattarlo come una
    costante e' un errore sistematico in entrambe le direzioni:

    - uscita in **take-profit**: e' un ordine limite gia' a riposo sul book, quindi paga maker;
    - uscita in **stop-loss** o su barriera temporale: deve eseguire subito, quindi paga taker.

    Assumere taker ovunque penalizza il terzo di uscite che sono in profitto; assumere maker
    ovunque e' l'ottimismo piu' comune in questo tipo di simulazione, perche' uno stop non puo'
    aspettare sul book.

    Gli ingressi non riempiti restano nel risultato con `traded = False` e rendimento nullo: non
    sono trade perduti da ignorare, sono occasioni che il modello ha visto e che l'esecuzione non
    ha convertito. Escluderli dal conteggio gonfierebbe l'aspettativa per operazione.
    """
    from cryptofarm.ml.labeling import BUY

    entry_fee = FEE_MODES[entry_mode]
    exit_fee_default = FEE_MODES[exit_mode]

    result = events.copy()
    filled = fills["filled"].to_numpy()
    result["traded"] = filled

    # Il take-profit si esegue come maker; tutto il resto paga la modalita' richiesta.
    took_profit = events["Label"].to_numpy() == BUY
    exit_fee = np.where(took_profit, MAKER_FEE, exit_fee_default)
    fee = entry_fee + exit_fee

    # Riempirsi sotto il prezzo di riferimento e' un vantaggio: il rendimento si misura dal
    # prezzo effettivo di ingresso, non da quello che si sperava.
    improvement = np.where(filled, (events["Close"].to_numpy() / fills["fill_price"].to_numpy()) - 1.0, 0.0)
    result["net_return"] = np.where(filled, events["exit_return"].to_numpy() + improvement - fee, 0.0)
    result["fee_paid"] = np.where(filled, fee, 0.0)
    return result
