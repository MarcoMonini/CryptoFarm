"""Definizione delle etichette: quando comprare, quando vendere, quando stare fermi.

Il metodo di riferimento e' il **triple-barrier**: per ogni candela si fissano tre barriere --
un take-profit sopra, uno stop-loss sotto, e un limite temporale -- e l'etichetta e' quale
viene toccata per prima.

Perche' questo e non i minimi/massimi locali (il metodo precedente, tenuto in fondo al modulo
per confronto):

- E' definito su **ogni** candela, non solo sulle poche che sono estremi. Su 350.000 candele il
  metodo per estremi ne etichettava ~12.000 (3,5%); qui sono 350.000.
- La distribuzione risulta naturalmente equilibrata, quindi non serve nessun downsampling --
  e senza downsampling il modello resta calibrato sulla frequenza reale degli eventi.
- L'etichetta non e' una proprieta' geometrica della curva ma **l'esito di un trade**: "comprando
  qui con questo TP e questo SL, il TP arriva prima dello SL". La precision del modello e' quindi
  direttamente il win rate della strategia, senza bisogno di interpretazioni.
- Non e' una lama di coltello: il metodo per estremi chiede di indovinare la candela esatta, e
  un segnale una candela in anticipo conta come errore totale pur valendo quasi lo stesso.

Le barriere sono **proporzionali alla volatilita' (ATR)** e non percentuali fisse: una soglia
dell'1% e' un movimento raro su BTC in un'ora e rumore su un'altcoin in un giorno. Con un
pavimento legato alle commissioni, perche' un'etichetta "vincente" il cui movimento non copre
0,2% di andata e ritorno insegna al modello a perdere soldi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import argrelextrema

# Etichette. Coincidono con la convenzione gia' usata da `trading/simulator.py`.
HOLD, BUY, SELL = 0, 1, 2
LABEL_NAMES = {HOLD: "hold", BUY: "buy", SELL: "sell"}

# Parametri di riferimento del triple-barrier.
#
# Le barriere sono **asimmetriche**, e la ragione e' aritmetica. Con commissioni f di andata e
# ritorno, barriera di profitto u_tp e barriera di perdita u_sl, la precision necessaria solo per
# andare in pari e'  p = (u_sl + f) / ((u_tp - f) + (u_sl + f)).  Con barriere simmetriche allo
# 0,6% e f = 0,2% servirebbe il **66,7%** di precision -- un'asticella che nessun modello su dati
# di mercato tiene stabilmente. Con il take-profit al doppio dello stop-loss la stessa soglia
# scende al **44,4%**, che e' un obiettivo realistico.
#
# Il prezzo di questa scelta e' che "profitto prima della perdita" diventa piu' raro (circa un
# terzo delle candele invece di meta'): meno esempi positivi, ma ognuno vale molto di piu'.
TP_ATR_MULTIPLE = 1.5  # take-profit in multipli dell'ATR corrente
SL_ATR_MULTIPLE = 1.0  # stop-loss: meta' del take-profit
HORIZON_BARS = 96  # limite temporale in barre
ROUND_TRIP_FEE = 0.002  # 0,1% per lato su spot Binance
FEE_FLOOR_MULTIPLE = 3.0  # nessuna barriera sotto 3x le commissioni di andata e ritorno

# Le finestre di look-ahead vengono elaborate a blocchi: la matrice completa dei futuri di 11
# milioni di candele x 96 barre non entrerebbe in memoria, un blocco alla volta si'.
CHUNK_ROWS = 200_000


def barrier_widths(
    atr_percent: pd.Series,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
    round_trip_fee: float = ROUND_TRIP_FEE,
    fee_floor_multiple: float = FEE_FLOOR_MULTIPLE,
) -> tuple[np.ndarray, np.ndarray]:
    """Ampiezza delle barriere per ogni candela, in frazione del prezzo.

    `atr_percent` e' l'ATR gia' espresso in percentuale del Close (come lo produce
    `features.normalize_indicators`).
    """
    floor = round_trip_fee * fee_floor_multiple
    atr_fraction = atr_percent.to_numpy(dtype=float) / 100.0

    # Il pavimento si applica allo stop, e il take-profit lo segue mantenendo il rapporto. Se si
    # applicasse a entrambi separatamente, sugli asset a bassa volatilita' (dove il pavimento
    # morde) le due barriere collasserebbero allo stesso valore, riportando il rapporto a 1:1 e
    # con esso il break-even al 66,7% -- esattamente il caso che l'asimmetria evita.
    stop_loss = np.maximum(atr_fraction * sl_multiple, floor)
    take_profit = stop_loss * (tp_multiple / sl_multiple)
    return take_profit, stop_loss


def triple_barrier_events(
    df: pd.DataFrame,
    horizon: int = HORIZON_BARS,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
    round_trip_fee: float = ROUND_TRIP_FEE,
    fee_floor_multiple: float = FEE_FLOOR_MULTIPLE,
) -> pd.DataFrame:
    """Etichetta ogni candela con l'esito del trade che vi si aprirebbe, **e con la sua durata**.

    Ingresso al Close della candela; le barriere vengono verificate sui massimi e minimi delle
    candele **successive** (t+1 .. t+horizon), mai su quella corrente.

    Se entrambe le barriere risultano toccate nella stessa candela l'esito e' ambiguo -- i dati
    OHLC non dicono in che ordine il prezzo le ha raggiunte dentro la barra -- e viene assegnato
    l'esito peggiore (SELL). Assumere il migliore produrrebbe un modello ottimista che in
    esecuzione reale trova sistematicamente meno di quanto si aspetta.

    Colonne restituite:

    - `Label`      esito: 0 hold (timeout), 1 buy (TP per primo), 2 sell (SL per primo)
    - `exit_bar`   posizione della candela di uscita
    - `t_exit`     timestamp della candela di uscita
    - `exit_return` rendimento realizzato, lordo di commissioni
    - `tp_width` / `sl_width`  ampiezza effettiva delle barriere, in frazione del prezzo
    - `ambiguous` le due barriere sono state toccate nella **stessa** candela

    `ambiguous` esiste perche' la convenzione pessimistica non e' universale. Con barriere
    asimmetriche e un consumatore solo lungo, assegnare SELL e' la scelta prudente. Con barriere
    **simmetriche** e un modello che sceglie la direzione, quelle righe direbbero al modello
    "scende" mentre il dato non dice niente: chi etichetta in modo simmetrico le toglie invece di
    crederci, e senza questa colonna non puo' nemmeno sapere quante sono.

    `t_exit` non e' un dettaglio diagnostico: e' il dato senza cui il **purging** della
    cross-validation non e' calcolabile. Due osservazioni le cui vite si sovrappongono
    condividono futuro, e per saperlo serve sapere quando ciascuna finisce.
    """
    if "ATR" not in df.columns:
        raise KeyError("triple_barrier_events richiede la colonna ATR normalizzata in percentuale")

    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    take_profit, stop_loss = barrier_widths(df["ATR"], tp_multiple, sl_multiple, round_trip_fee, fee_floor_multiple)

    total = len(df)
    labels = np.zeros(total, dtype=np.int8)
    exit_bar = np.arange(total, dtype=np.int64)
    exit_return = np.zeros(total, dtype=float)
    ambiguous = np.zeros(total, dtype=bool)

    labelable = total - horizon
    if labelable <= 0:
        return pd.DataFrame(
            {
                "Label": labels,
                "exit_bar": exit_bar,
                "t_exit": df.index,
                "exit_return": exit_return,
                "tp_width": take_profit,
                "sl_width": stop_loss,
                "ambiguous": ambiguous,
            },
            index=df.index,
        )

    # Finestra dei futuri: la riga i copre le candele i+1 .. i+horizon.
    future_high = sliding_window_view(high[1:], horizon)[:labelable]
    future_low = sliding_window_view(low[1:], horizon)[:labelable]
    future_close = sliding_window_view(close[1:], horizon)[:labelable]

    upper = close[:labelable] * (1.0 + take_profit[:labelable])
    lower = close[:labelable] * (1.0 - stop_loss[:labelable])

    for start in range(0, labelable, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, labelable)
        hit_upper = future_high[start:stop] >= upper[start:stop, None]
        hit_lower = future_low[start:stop] <= lower[start:stop, None]

        # argmax su un array booleano da il primo True; senza nessun True da 0, quindi il
        # risultato va reso valido solo dove un contatto c'e' stato davvero.
        never = horizon + 1
        first_upper = np.where(hit_upper.any(axis=1), hit_upper.argmax(axis=1), never)
        first_lower = np.where(hit_lower.any(axis=1), hit_lower.argmax(axis=1), never)

        size = stop - start
        chunk = np.full(size, HOLD, dtype=np.int8)
        # Timeout: uscita a mercato sull'ultima candela dell'orizzonte.
        bars = np.full(size, horizon, dtype=np.int64)
        returns = future_close[start:stop, -1] / close[start:stop] - 1.0

        won = first_upper < first_lower
        # `<=` e non `<`: a parita' di candela vince lo stop, per la ragione nel docstring.
        lost = (first_lower <= first_upper) & (first_lower != never)

        chunk[won] = BUY
        bars[won] = first_upper[won] + 1
        returns[won] = take_profit[start:stop][won]

        chunk[lost] = SELL
        bars[lost] = first_lower[lost] + 1
        returns[lost] = -stop_loss[start:stop][lost]

        # Stessa candela per entrambe le barriere: l'OHLC non dice in che ordine il prezzo le ha
        # toccate. La riga resta SELL per la convenzione pessimistica, ma viene segnalata.
        ambiguous[start:stop] = (first_lower == first_upper) & (first_lower != never)

        labels[start:stop] = chunk
        exit_bar[start:stop] = np.arange(start, stop) + bars
        exit_return[start:stop] = returns

    # La coda senza futuro osservabile resta HOLD, con uscita su se stessa: non e' un trade.
    exit_bar[labelable:] = np.arange(labelable, total)
    exit_bar = np.minimum(exit_bar, total - 1)

    return pd.DataFrame(
        {
            "Label": labels,
            "exit_bar": exit_bar,
            "t_exit": df.index[exit_bar],
            "exit_return": exit_return,
            "tp_width": take_profit,
            "sl_width": stop_loss,
            "ambiguous": ambiguous,
        },
        index=df.index,
    )


def triple_barrier_labels(df: pd.DataFrame, **kwargs) -> pd.Series:
    """Solo le etichette. Comodita' per i chiamanti che non hanno bisogno delle durate."""
    return triple_barrier_events(df, **kwargs)["Label"].rename("Label")


def label_distribution(labels: np.ndarray | pd.Series) -> dict[str, float]:
    """Conteggi e percentuali per classe, per la diagnostica a ogni stadio della pipeline."""
    values = np.asarray(labels)
    if values.size == 0:
        return {}
    distribution = {}
    for code, name in LABEL_NAMES.items():
        count = int((values == code).sum())
        distribution[name] = count
        distribution[f"{name}_pct"] = count / values.size
    return distribution


def format_distribution(labels: np.ndarray | pd.Series, stage: str) -> str:
    """Riga di diagnostica leggibile con la distribuzione delle classi."""
    values = np.asarray(labels)
    if values.size == 0:
        return f"[{stage}] nessun campione"
    parts = [
        f"{name}={int((values == code).sum())} ({(values == code).mean():.1%})" for code, name in LABEL_NAMES.items()
    ]
    return f"[{stage}] {values.size} campioni | " + ", ".join(parts)


# ---------------------------------------------------------------------------------------------
# Metodo precedente, mantenuto per poter confrontare i due su dati identici.
# ---------------------------------------------------------------------------------------------


def apply_label_cooldown(labels: pd.Series, cooldown: int) -> pd.Series:
    """Impone una distanza minima in candele tra due segnali consecutivi."""
    if cooldown <= 0 or labels.empty:
        return labels
    values = labels.to_numpy(copy=True)
    last_kept = None
    for position in np.flatnonzero(values != HOLD):
        if last_kept is not None and (position - last_kept) <= cooldown:
            values[position] = HOLD
        else:
            last_kept = position
    return pd.Series(values, index=labels.index, name="Label")


def filter_labels_by_future_return(df: pd.DataFrame, labels: pd.Series, min_return: float, horizon: int) -> pd.Series:
    """Scarta i segnali non seguiti da un movimento di almeno `min_return` entro `horizon`."""
    if min_return <= 0 or horizon <= 0:
        return labels

    future_max = df["High"][::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    future_min = df["Low"][::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    upside = (future_max / df["Close"]) - 1.0
    downside = (future_min / df["Close"]) - 1.0

    filtered = labels.copy()
    # La forma negata scarta anche le ultime righe, dove il futuro non e' osservabile e il
    # confronto darebbe NaN (un `<` su NaN sarebbe False e terrebbe l'etichetta).
    filtered.loc[(filtered == BUY) & ~(upside >= min_return)] = HOLD
    filtered.loc[(filtered == SELL) & ~(downside <= -min_return)] = HOLD
    return filtered


def extrema_labels(
    df: pd.DataFrame,
    window_pivot: int = 25,
    min_return: float = 0.012,
    return_horizon: int = 48,
    cooldown: int = 8,
) -> pd.Series:
    """Minimi e massimi locali filtrati per rendimento futuro e distanziati da un cooldown."""
    order = max(2, int(window_pivot / 2))
    labels = pd.Series(HOLD, index=df.index, dtype="int64", name="Label")
    labels.iloc[argrelextrema(df["High"].values, np.greater, order=order)[0]] = SELL
    labels.iloc[argrelextrema(df["Low"].values, np.less, order=order)[0]] = BUY
    labels = filter_labels_by_future_return(df, labels, min_return, return_horizon)
    return apply_label_cooldown(labels, cooldown)


# --- Prossimita' agli estremi locali ------------------------------------------------------------


def swing_target(close: pd.Series | np.ndarray, window: int, verso: str = "entrambi") -> pd.Series:
    """Target continuo in [-1, 1]: +1 su un massimo locale, -1 su un minimo, ~0 dove non c'e'.

    E' il **rango centrato** della chiusura dentro la finestra di `window` barre per lato,
    riscalato. Chiede al modello una domanda diversa dalla barriera tripla: quella chiede «di
    quanto si muove il prezzo», che e' una proprieta' della volatilita'; questa chiede «dove sta
    questa barra rispetto alle sue vicine», che e' la forma.

    La proprieta' che la rende adatta e' il comportamento **dentro una tendenza**: in una salita
    regolare la barra centrale ha meta' finestra sopra e meta' sotto per costruzione, quindi il
    target vale circa 0 e non +1. Satura solo dove la salita si esaurisce. Un massimo dei prezzi
    futuri, o una distanza da quel massimo, non hanno questa proprieta' e marcherebbero come
    «vicino al massimo» tutta la salita, che e' l'errore che rende quelle etichette inservibili.

    **Guarda `window` barre nel futuro, per costruzione**, e da qui tre vincoli per chi la usa:
    le ultime `window` barre non sono etichettabili (escono NaN da sole), il taglio fra stima e
    verifica vuole un embargo di `window` barre, e se il target rientra fra le feature va ritardato
    di **almeno `window` + 1** barre -- ritardarlo di una sola inserisce look-ahead quasi puro.

    `verso` serve a separare le due meta', ed e' la cosa piu' importante di questa funzione.
    Il rango centrato usa anche le `window` barre **passate**, che le feature gia' descrivono:
    misurato, uno Stochastic senza modello ne spiega IC 0,70, cioe' il 93%. Valutare un modello
    contro il target pieno misura quindi soprattutto la sua capacita' di rifare uno Stochastic --
    e infatti il modello ne prende 0,67, meno dell'indicatore. Il metro va preso contro
    `verso="avanti"`, dove sia lo Stochastic sia il modello stanno attorno a 0,05.

    Il rango si ricava da due rolling causali invece che da una finestra centrata: quello
    all'indietro da' la posizione fra le `window` barre precedenti, quello sulla serie rovesciata
    fra le successive, e la somma meno uno e' il rango centrato esatto. Costa O(n log window)
    invece di materializzare n x (2 window + 1) valori, che a 5m su quindici simboli non ci sta.
    """
    if window < 1:
        raise ValueError("window deve essere >= 1")
    serie = pd.Series(close).astype(float)
    dietro = serie.rolling(window + 1).rank().to_numpy()
    avanti = serie[::-1].rolling(window + 1).rank().to_numpy()[::-1]
    if verso == "dietro":  # la meta' gia' nota: e' uno Stochastic, serve come riferimento
        return pd.Series((dietro - 1.0) / window * 2.0 - 1.0, index=serie.index, name="Target")
    if verso == "avanti":  # la meta' non conoscibile: e' l'unico metro onesto
        return pd.Series((avanti - 1.0) / window * 2.0 - 1.0, index=serie.index, name="Target")
    if verso != "entrambi":
        raise ValueError(f"verso sconosciuto: {verso!r}")
    rango = dietro + avanti - 1.0  # rango centrato esatto, in [1, 2*window + 1]
    return pd.Series((rango - 1.0) / window - 1.0, index=serie.index, name="Target")


# --- Le gambe fra un estremo e il successivo ----------------------------------------------------
#
# `swing_target` chiede «dove sta questa barra fra le sue vicine», ed e' un rango: misurato su BTC
# a 5m con la finestra dell'addestramento, `|target| > 0,9` cade in 6.501 episodi da tre barre --
# nove «estremi» al giorno da un quarto d'ora l'uno. Sono microstruttura, non le cime e i fondi
# delle gambe, e la meta' passata del rango e' uno Stochastic, cioe' una funzione del prezzo
# recente che il target insegue invece di anticipare.
#
# Qui la domanda cambia: **fra quale coppia di estremi ci troviamo, e a che punto della gamba**.
# Gli estremi si cercano una volta sola con una finestra larga, si alternano per costruzione, e il
# target scorre da -1 a +1 fra un minimo e il massimo successivo, e viceversa. Due cose che il
# rango non ha:
#
# - **il tempo entra**. Meta' del percorso e' il prezzo, meta' sono le barre trascorse dall'estremo
#   precedente. Un prezzo fermo a meta' gamba non resta fermo a meta' scala: si avvicina all'estremo
#   che sta per arrivare. E' cio' che rende l'etichetta un'anticipazione e non un inseguimento;
# - **l'ampiezza conta**. Il valore agli estremi non e' +-1 per tutti: e' +-tanh(prominenza /
#   riferimento), dove la prominenza e' la piu' piccola delle due gambe adiacenti -- un estremo e'
#   tale solo se il prezzo ci arriva e poi se ne va -- e il riferimento e' quanto si muoverebbe in
#   `window` barre una passeggiata casuale con la volatilita' locale. Un'oscillazione dentro il
#   rumore vale 0,3, una gamba vera satura. Senza questa scala i quindici simboli e i loro anni
#   non sono confrontabili: la stessa gamba del 5% e' enorme su una stablecoin e rumore su una
#   altcoin nel 2021.

# Lo smoothing temporale: quanta parte del percorso lungo la gamba la dice il **tempo trascorso**
# invece del prezzo. E' l'unico parametro dell'etichetta che vale per tutti e tre i suoi usi --
# addestramento, servizio del grafico e misura -- e sta qui perche' i tre non possono divergere:
# un modello addestrato a 0,7 e un grafico disegnato a 0,5 mostrano due curve diverse chiamandole
# entrambe «l'etichetta». A 0 e' solo prezzo, ed e' il difetto della versione a rango, dove
# l'etichetta insegue il prezzo invece di anticiparlo; a 1 e' solo tempo, e ignora che il prezzo
# possa tornare sui suoi passi.
TIME_WEIGHT = 0.7


def swing_pivots(close: pd.Series | np.ndarray, window: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Gli estremi locali, **alternati**: posizioni e verso (+1 massimo, -1 minimo).

    `argrelextrema` da' massimi e minimi separatamente, e nulla garantisce che si alternino: due
    massimi di fila senza un minimo in mezzo sono comuni appena la finestra e' larga. Qui si
    fondono in una sequenza sola e, quando due estremi dello stesso verso si susseguono, resta il
    piu' estremo dei due -- l'altro non e' il vertice di nessuna gamba.
    """
    if window < 1:
        raise ValueError("window deve essere >= 1")
    prezzi = pd.Series(close).astype(float).to_numpy()
    candidati = sorted(
        [(int(i), 1) for i in argrelextrema(prezzi, np.greater, order=window)[0]]
        + [(int(i), -1) for i in argrelextrema(prezzi, np.less, order=window)[0]]
    )
    # Ai bordi `argrelextrema` confronta con indici ritagliati (`mode="clip"`), cioe' con la barra
    # stessa ripetuta: dichiara estremi che il futuro non ha ancora confermato. Le ultime `window`
    # barre non sono etichettabili per costruzione, e queste sarebbero l'unico punto in cui
    # l'etichetta sa qualcosa che il tempo non ha ancora detto.
    candidati = [(i, verso) for i, verso in candidati if window <= i < len(prezzi) - window]
    indici: list[int] = []
    versi: list[int] = []
    for posizione, verso in candidati:
        if versi and versi[-1] == verso:
            if (prezzi[posizione] - prezzi[indici[-1]]) * verso > 0:
                indici[-1] = posizione
            continue
        indici.append(posizione)
        versi.append(verso)
    return np.array(indici, dtype=int), np.array(versi, dtype=int)


def swing_leg_target(
    close: pd.Series | np.ndarray,
    window: int = 50,
    peso_tempo: float = TIME_WEIGHT,
    saturazione: float = 1.0,
) -> pd.Series:
    """Target continuo in [-1, 1] che scorre fra un estremo locale e il successivo.

    Vale `-forza` su un minimo, `+forza` sul massimo che segue, e in mezzo interpola: meta' su
    quanto prezzo manca all'estremo che arriva, meta' su quante barre. `forza` sta in (0, 1) e
    dice quanto l'estremo e' pronunciato rispetto al rumore di quel tratto, cosi' una gamba vera
    satura e un'oscillazione dentro la volatilita' locale no.

    `peso_tempo` a 1 ignora il prezzo (rampa lineare fra gli estremi), a 0 ignora il tempo. Il
    valore di partenza e' `TIME_WEIGHT` = 0,7: un prezzo che ritraccia a meta' gamba ma consuma
    tre quarti delle barre sta a 0,68 di scala invece che a 0,5, cioe' l'etichetta lo dichiara
    gia' vicino all'estremo che arriva. E' li' che smette di inseguire il prezzo. Non e' una
    manopola libera: e' il valore con cui `swing_model` e' addestrato, e il grafico lo eredita.

    `saturazione` moltiplica il riferimento su cui si misura la prominenza: alzarla chiede gambe
    piu' grandi per lo stesso valore.

    **Guarda avanti fino all'estremo successivo**, che e' un orizzonte variabile e non limitato da
    `window`: le barre dopo l'ultimo estremo confermato escono NaN, e sono almeno `window`. Per lo
    stesso motivo il taglio fra stima e verifica vuole un embargo di almeno una gamba tipica, non
    di `window` barre.
    """
    serie = pd.Series(close).astype(float)
    fuori = pd.Series(np.nan, index=serie.index, name="Target")
    indici, versi = swing_pivots(serie, window)
    if len(indici) < 2:
        return fuori

    log_prezzi = np.log(serie.to_numpy())
    # Il riferimento e' quanto si muove in `window` barre una passeggiata casuale con la
    # volatilita' locale, sigma * sqrt(window): e' causale (la finestra finisce sull'estremo) e
    # rende la prominenza un numero senza unita', confrontabile fra simboli e fra anni.
    sigma = pd.Series(log_prezzi).diff().rolling(window, min_periods=window // 2).std().to_numpy()
    riferimento = np.maximum(sigma[indici] * np.sqrt(window) * saturazione, 1e-9)

    gambe = np.abs(np.diff(log_prezzi[indici]))
    # La prominenza di un estremo e' la piu' piccola delle due gambe che lo toccano: un vertice
    # raggiunto da una salita enorme e lasciato da una discesa minima non e' un vertice, e' una
    # sosta. Ai due estremi della serie c'e' una gamba sola, e vale quella.
    prominenza = np.minimum(np.append(gambe, gambe[-1]), np.insert(gambe, 0, gambe[0]))
    valori = versi * np.tanh(prominenza / riferimento)

    risultato = np.full(len(serie), np.nan)
    for k in range(len(indici) - 1):
        a, b = indici[k], indici[k + 1]
        passo = b - a
        avanzamento_tempo = np.arange(passo + 1) / passo
        salto = log_prezzi[b] - log_prezzi[a]
        if salto == 0:
            avanzamento = avanzamento_tempo
        else:
            avanzamento_prezzo = np.clip((log_prezzi[a : b + 1] - log_prezzi[a]) / salto, 0.0, 1.0)
            avanzamento = peso_tempo * avanzamento_tempo + (1.0 - peso_tempo) * avanzamento_prezzo
        risultato[a : b + 1] = valori[k] + avanzamento * (valori[k + 1] - valori[k])
    fuori.iloc[:] = risultato
    return fuori
