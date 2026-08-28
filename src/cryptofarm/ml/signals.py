"""Dai punteggi del modello alle operazioni.

Il modello risponde a una sola domanda: **se compro su questa candela, il take-profit arriva
prima dello stop-loss?** E' un segnale di ingresso, e non ne esiste uno simmetrico di uscita --
la classe "sell" delle etichette significa "brutto momento per comprare", non "buon momento per
vendere". Trattarla come un segnale di vendita produce un diluvio di vendite (quella classe copre
circa il 60% delle candele) e, cosa peggiore, rompe la corrispondenza fra le etichette su cui il
modello e' stato valutato e le operazioni effettivamente simulate: i numeri di aspettativa non
descriverebbero piu' nulla.

L'uscita e' definita dalle **stesse barriere che definiscono le etichette**: take-profit,
stop-loss e limite temporale, calcolati dall'ATR al momento dell'ingresso. E' cio' che rende il
P&L simulato la traduzione diretta del win rate misurato in validation.

Come effetto collaterale i segnali risultano perfettamente alternati (un acquisto, la sua
vendita, il successivo acquisto), che e' anche l'unico modo in cui l'accoppiamento per indice di
`simulate_trading_with_commisions` produce operazioni sensate.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes
from cryptofarm.ml.dataset import build_design_matrix, cusum_events
from cryptofarm.ml.features import build_feature_frame
from cryptofarm.ml.labeling import BUY, HORIZON_BARS, SL_ATR_MULTIPLE, TP_ATR_MULTIPLE, barrier_widths
from cryptofarm.ml.models import load_model, predict_proba
from cryptofarm.paths import MODELS_DIR


def interval_from_index(index: pd.DatetimeIndex) -> str:
    """Deduce l'intervallo delle candele dalla loro spaziatura mediana."""
    if len(index) < 2:
        return "15m"
    minutes = int(round(np.median(np.diff(index.to_numpy()).astype("timedelta64[m]").astype(float))))
    return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"


def buy_probabilities(df: pd.DataFrame, model) -> pd.Series:
    """P(take-profit prima dello stop-loss) per ogni candela in cui e' calcolabile.

    Le feature vengono ricostruite dai soli OHLCV con le costanti di questo pacchetto invece di
    riusare le colonne che il chiamante ha gia' in tabella: la dashboard le calcola con i periodi
    scelti dai suoi slider, e un modello alimentato con feature diverse da quelle
    dell'addestramento sbaglia senza dare nessun segno.
    """
    features = build_feature_frame(df, interval_from_index(df.index))
    matrix = build_design_matrix(features)
    matrix = matrix[matrix.notna().all(axis=1)]
    if matrix.empty:
        return pd.Series(dtype=float)
    probabilities = predict_proba(model, matrix.to_numpy())
    return pd.Series(probabilities[:, BUY], index=matrix.index, name="P_buy")


def barrier_signals(
    df: pd.DataFrame,
    model,
    threshold: float,
    horizon: int = HORIZON_BARS,
    tp_multiple: float = TP_ATR_MULTIPLE,
    sl_multiple: float = SL_ATR_MULTIPLE,
) -> tuple[list[tuple], list[tuple]]:
    """Genera le operazioni: ingresso sul punteggio del modello, uscita sulle barriere.

    Restituisce `(buy_signals, sell_signals)` come liste di `(timestamp, prezzo)`, alternate e
    della stessa lunghezza salvo una posizione ancora aperta a fine serie.

    Il prezzo di uscita e' il livello della barriera toccata, non la chiusura della candela: e'
    l'approssimazione che corrisponde a un ordine gia' piazzato sul book. Se in una stessa candela
    risultano toccate entrambe le barriere l'esito e' ambiguo -- l'OHLC non dice in che ordine il
    prezzo le ha raggiunte -- e viene assegnato lo stop, la stessa convenzione pessimistica usata
    nell'etichettatura.
    """
    features = build_feature_frame(df, interval_from_index(df.index))
    if features.empty:
        return [], []

    scores = buy_probabilities(df, model)
    if scores.empty:
        return [], []

    take_profit, stop_loss = barrier_widths(features["ATR"], tp_multiple=tp_multiple, sl_multiple=sl_multiple)
    probability = scores.reindex(features.index).to_numpy()
    high = features["High"].to_numpy(dtype=float)
    low = features["Low"].to_numpy(dtype=float)
    close = features["Close"].to_numpy(dtype=float)
    timestamps = features.index

    buy_signals: list[tuple] = []
    sell_signals: list[tuple] = []

    position = 0
    while position < len(close):
        if not (probability[position] >= threshold):
            position += 1
            continue

        entry_price = close[position]
        target = entry_price * (1.0 + take_profit[position])
        stop = entry_price * (1.0 - stop_loss[position])
        deadline = min(position + horizon, len(close) - 1)
        buy_signals.append((timestamps[position], float(entry_price)))

        exit_position = deadline
        exit_price = close[deadline]
        for step in range(position + 1, deadline + 1):
            if low[step] <= stop:
                exit_position, exit_price = step, stop
                break
            if high[step] >= target:
                exit_position, exit_price = step, target
                break

        sell_signals.append((timestamps[exit_position], float(exit_price)))
        # Nessuna nuova posizione prima che la precedente sia chiusa: il modello stima l'esito di
        # un ingresso isolato, non di posizioni sovrapposte.
        position = exit_position + 1

    return buy_signals, sell_signals


def meta_signals(
    df: pd.DataFrame,
    model,
    threshold: float,
    horizon_hours: float = 24.0,
    tp_multiple: float = 1.5,
    sl_multiple: float = 1.0,
    round_trip_fee: float = 0.0012,
    fee_floor_multiple: float = 5.0,
    cusum_sigma: float = 3.0,
    limit_offset_atr: float = 0.5,
    limit_patience: int = 12,
) -> tuple[list[tuple], list[tuple]]:
    """Catena completa della strategia meta: primario CUSUM, secondario, esecuzione a limite.

    Riproduce esattamente cio' che il modello e' stato addestrato a prevedere, e nell'ordine in
    cui e' stato valutato:

    1. il **primario** (filtro CUSUM) propone un candidato quando il prezzo ha accumulato un
       movimento di dimensione rilevante;
    2. il **secondario** assegna la probabilita' che quell'ingresso chiuda in profitto netto, e
       si opera solo sopra soglia;
    3. l'ingresso e' un **ordine limite** sotto il prezzo, che puo' non riempirsi -- e in quel
       caso non c'e' nessun trade, non un trade a prezzo di mercato;
    4. l'uscita e' la barriera toccata per prima, calcolata dall'ATR al momento dell'ingresso.

    Rispettare questa catena non e' pedanteria: e' cio' che rende il P&L simulato la traduzione
    diretta dell'aspettativa misurata in cross-validation. Cambiare un anello -- entrare a
    mercato invece che a limite, uscire su un segnale invece che su una barriera -- scollega i
    due numeri senza dare nessun segnale che sia successo.
    """
    from cryptofarm.ml.execution import limit_fills
    from cryptofarm.ml.labeling import barrier_widths

    interval = interval_from_index(df.index)
    minutes = interval_to_minutes(interval)
    features = build_feature_frame(df, interval)
    if features.empty:
        return [], []

    matrix = build_design_matrix(features)
    usable = matrix.notna().all(axis=1).to_numpy()
    events = cusum_events(features["Close"], cusum_sigma)
    events = events[usable[events]]
    if len(events) == 0:
        return [], []

    scores = predict_proba(model, matrix.iloc[events].to_numpy())[:, 1]
    candidates = events[scores >= threshold]
    if len(candidates) == 0:
        return [], []

    fills = limit_fills(features, candidates, offset_atr=limit_offset_atr, patience=limit_patience)
    take_profit, stop_loss = barrier_widths(
        features["ATR"], tp_multiple, sl_multiple, round_trip_fee, fee_floor_multiple
    )

    high = features["High"].to_numpy(dtype=float)
    low = features["Low"].to_numpy(dtype=float)
    close = features["Close"].to_numpy(dtype=float)
    timestamps = features.index
    horizon_bars = int(horizon_hours * 60 / minutes)

    buy_signals: list[tuple] = []
    sell_signals: list[tuple] = []
    busy_until = -1

    for row, position in enumerate(candidates):
        if position <= busy_until or not fills["filled"].iloc[row]:
            continue
        entry_bar = int(fills["fill_bar"].iloc[row])
        entry_price = float(fills["fill_price"].iloc[row])
        target = entry_price * (1.0 + take_profit[position])
        stop = entry_price * (1.0 - stop_loss[position])
        deadline = min(entry_bar + horizon_bars, len(close) - 1)

        exit_position, exit_price = deadline, close[deadline]
        for step in range(entry_bar + 1, deadline + 1):
            if low[step] <= stop:
                exit_position, exit_price = step, stop
                break
            if high[step] >= target:
                exit_position, exit_price = step, target
                break

        buy_signals.append((timestamps[entry_bar], entry_price))
        sell_signals.append((timestamps[exit_position], float(exit_price)))
        # Una posizione alla volta: il modello stima l'esito di un ingresso isolato.
        busy_until = exit_position

    return buy_signals, sell_signals


def policy_signals(df: pd.DataFrame, model, threshold: float = 0.5) -> tuple[list, list]:
    """Segnali della politica a tre azioni, per il simulatore.

    A differenza di `barrier_signals` qui **non si predice in blocco**: l'azione di adesso decide
    lo stato di dopo, quindi la serie va percorsa una barra alla volta riempendo a ogni passo le
    tre colonne di posizione. E' la ragione per cui questo modello non entra nel percorso
    esistente senza un adattatore.

    L'uscita non viene dalle barriere ma dal modello stesso: e' lui a emettere SELL. I segnali
    risultano alternati per costruzione, perche' il mascheramento delle azioni non valide rende
    impossibile comprare due volte di fila.

    **Il P&L che il simulatore mostrera' e' peggiore di quello di `strategy.md` §12**, e non e'
    un'incoerenza: li' il costo e' 0,08% andata e ritorno in modalita' maker, mentre il simulatore
    applica per default lo 0,1% per lato, cioe' 0,2%. Il modello perde in entrambi i casi
    (§13), ma di quanto dipende dal costo che si applica.
    """
    from cryptofarm.ml.dagger import episode_bounds, rollout
    from cryptofarm.ml.directional_change import BUY, SELL
    from cryptofarm.ml.policy import FLAT, LONG

    features = build_feature_frame(df, interval_from_index(df.index))
    matrix = build_design_matrix(features)
    matrix = matrix[matrix.notna().all(axis=1)]
    if len(matrix) < 2:
        return [], []

    close = features.loc[matrix.index, "Close"].to_numpy(float)
    # Un unico episodio: il simulatore mostra una finestra continua, e spezzarla azzererebbe la
    # posizione a meta' grafico per una ragione che non ha niente a che vedere col mercato.
    visited = rollout(model, matrix.to_numpy(), close, bounds=episode_bounds([len(close)], len(close)))

    entries = visited[(visited["state"] == FLAT) & (visited["action"] == BUY)]["row"].to_numpy()
    exits = visited[(visited["state"] == LONG) & (visited["action"] == SELL)]["row"].to_numpy()
    pairs = min(len(entries), len(exits))
    if pairs == 0:
        return [], []

    stamps = matrix.index
    buy_signals = [(stamps[row], float(close[row])) for row in entries[:pairs]]
    sell_signals = [(stamps[row], float(close[row])) for row in exits[:pairs]]
    return buy_signals, sell_signals


def leg_signals(
    df: pd.DataFrame,
    model,
    threshold: float,
    symbol: str = "",
    uscita_su_modello: bool = True,
    soglia_uscita: float | None = None,
    cross: dict | None = None,
) -> tuple[list[tuple], list[tuple]]:
    """Segnali del modello delle gambe: entra su `P(su)`, esce su `P(giu)` o sulla barriera.

    E' l'unico percorso in cui questo progetto emette un segnale di vendita **dal modello**, ed e'
    possibile solo perche' le barriere dell'etichetta sono simmetriche: `P(giu)` significa «scende
    di k x ATR prima di salirne altrettanti», non «lo stop di una posizione lunga verrebbe
    toccato». La differenza e' spiegata in `ml/leg_trainer`.

    ## Non c'e' nessun take profit, ed e' una scelta misurata

    L'uscita e' la prima fra tre condizioni: lo **stop duro** a `k x ATR`, `P(giu)` sopra soglia, e
    l'orizzonte. Nessuna barriera superiore.

    Il take profit c'era e ne e' uscito misurando. Sulla stessa popolazione di ingressi (tutte le
    barre, un ingresso al giorno) e con lo stesso stop duro, netto medio per ingresso, mediana su
    cinque asset a 4h:

    | uscita | netto |
    |---|---|
    | nessun take profit, fino all'orizzonte | **+0,074%** |
    | TP 3 ATR | -0,226% |
    | TP 1,5 ATR | -0,303% |
    | TP 1,5 poi trailing 2,5 ATR | -0,454% |
    | trailing 4 ATR | -0,473% |
    | trailing 2,5 ATR | -0,753% |

    L'ordine e' monotono: **piu' si interrompe il rialzo, piu' si perde**, e lo stop a trailing e'
    il peggiore. La ragione e' che la coda destra delle gambe cripto *e'* l'aspettativa, e
    qualunque regola che la tronchi taglia via cio' che paga. Lo stop duro resta perche' e' una
    regola di rischio -- limita la coda *sinistra*, che non paga niente.

    Ne segue che `P(giu)` non e' un ornamento: e' **l'unica uscita al rialzo**. Il modello deve
    dire che la gamba e' finita, o si resta fino all'orizzonte.

    `uscita_su_modello=False` toglie anche quella e lascia solo stop e orizzonte: e' l'ablazione
    che misura quanto valga la testa `P(giu)`, e va riportata accanto a qualunque risultato.

    ## Due soglie, non una

    `threshold` vale su `P(su)`, `soglia_uscita` su `P(giu)`, e **non possono essere lo stesso
    numero**. Le due teste hanno distribuzioni diverse: misurato su BTC a 4h, 0,55 su `P(su)`
    seleziona l'8% delle barre e lo stesso 0,55 su `P(giu)` ne seleziona l'80%. Usare un solo
    valore produceva un'uscita alla barra successiva a ogni ingresso -- operazioni tutte lunghe
    una candela, con gli ingressi giusti e le uscite immediate. Senza `soglia_uscita` esplicita si
    legge quella dei metadata dell'artefatto, che l'addestramento calibra sulla distribuzione di
    `P(giu)` e non su quella di `P(su)`.

    `cross` sono le tre colonne trasversali, che dipendono dagli **altri** asset e quindi non
    sono calcolabili da un simbolo solo. Senza, restano NaN -- ed e' uno stato che il modello
    vede in addestramento (`leg_trainer` ne maschera una quota apposta), quindi lo tratta come
    "non so", non come ribasso.

    `symbol` serve alle due feature di posizionamento; senza, restano NaN e il modello a gradienti
    le tratta come una categoria a se'. E' la condizione in cui gira il simulatore su un simbolo
    caricato dall'exchange invece che dallo store.
    """
    from cryptofarm.ml.bar_features import FEATURE_COLUMNS, build_bar_features, cross_from_store
    from cryptofarm.ml.leg_trainer import BARRIERA_ATR, GIRO, PAVIMENTO, _orizzonte
    from cryptofarm.ml.trainer import stored_exit_threshold

    interval = interval_from_index(df.index)
    soglia_uscita = soglia_uscita if soglia_uscita is not None else stored_exit_threshold()
    # Le trasversali dipendono dagli altri asset: chi passa un simbolo solo non puo' calcolarle, e
    # senza il risultato crolla (mediana da +1,9% a -39,5% fuori campione). Se lo store locale
    # c'e', si ricavano da li'; se non c'e' restano NaN e il modello degrada in modo controllato.
    cross = cross if cross is not None else cross_from_store(interval)
    feature = build_bar_features(symbol, df, interval, cross=cross)
    matrice = feature[FEATURE_COLUMNS]
    utilizzabile = matrice.notna().any(axis=1).to_numpy()
    if not utilizzabile.any():
        return [], []

    probabilita = predict_proba(model, matrice.to_numpy(dtype=float))
    p_su, p_giu = probabilita[:, BUY], probabilita[:, 2]

    # Solo la barriera inferiore: `barrier_widths` restituisce anche quella superiore e qui non
    # viene usata. Lo stop conserva il pavimento sulle commissioni -- uno stop piu' stretto del
    # proprio giro si fa toccare dal rumore e paga per farlo.
    _, stop_loss = barrier_widths(
        feature["atr_rel"] * 100.0,
        tp_multiple=BARRIERA_ATR,
        sl_multiple=BARRIERA_ATR,
        round_trip_fee=GIRO,
        fee_floor_multiple=PAVIMENTO,
    )
    orizzonte = _orizzonte(interval)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    quando = df.index

    acquisti: list[tuple] = []
    vendite: list[tuple] = []
    posizione = 0
    while posizione < len(close):
        # Nessun ingresso dove lo stop non e' calcolabile: sulle barre di riscaldamento l'ATR non
        # c'e' ancora, e una posizione senza stop -- o con lo stop del pavimento -- non e' quella
        # che il modello e' stato addestrato a valutare.
        if not (p_su[posizione] >= threshold) or not np.isfinite(stop_loss[posizione]):
            posizione += 1
            continue

        ingresso = close[posizione]
        stop = ingresso * (1.0 - stop_loss[posizione])
        scadenza = min(posizione + orizzonte, len(close) - 1)
        acquisti.append((quando[posizione], float(ingresso), f"P(su)={p_su[posizione]:.2f}"))

        uscita, prezzo, motivo = scadenza, close[scadenza], "horizon"
        for passo in range(posizione + 1, scadenza + 1):
            # Lo stop per primo: dentro una candela l'OHLC non dice l'ordine degli eventi, e la
            # convenzione pessimistica e' la stessa dell'etichettatura.
            if low[passo] <= stop:
                uscita, prezzo, motivo = passo, stop, "stop"
                break
            if uscita_su_modello and p_giu[passo] >= soglia_uscita:
                uscita, prezzo, motivo = passo, close[passo], f"P(giu)={p_giu[passo]:.2f}"
                break

        vendite.append((quando[uscita], float(prezzo), motivo))
        posizione = uscita + 1

    return acquisti, vendite


# --- Il modello a swing --------------------------------------------------------------------------

# Le soglie della regola a esposizione, scelte **sulla validazione** e non fuori campione.
# `.claude/docs/modello-swing.md` §5.2 misura che nessuna coppia va bene in entrambe le finestre:
# 0,50/0,40 rende fuori campione e perde in validazione, 0,35/0,25 il contrario. Prendere la prima
# perche' e' quella che rende sul 2024-2026 sarebbe tararsi sul campione di verifica -- il difetto
# per cui `leg_model` e' uscito dalla catena. Si prende quindi quella scelta dove e' lecito
# sceglierla, sapendo che fuori campione ha reso -0,191% per operazione.
SWING_ENTRA, SWING_ESCI = 0.35, 0.25

# Quante barre deve avere una scala lunga perche' sia calcolabile. Non e' un margine prudenziale:
# `ExtraCache.adx(14)` passa da `ta`, che con meno di due finestre solleva `IndexError` invece
# di restituire NaN. In addestramento non si vede -- le serie sono di centinaia di migliaia di
# barre -- ma la pagina carica per default 240 ore, cioe' dieci barre giornaliere, e li' cadeva.
SCALA_MINIMA_BARRE = 28


def swing_model_disponibile() -> bool:
    """Se l'artefatto del modello a swing e' su disco.

    Separata da `swing_model` perche' la risposta serve **prima** di caricare: la confluenza deve
    decidere se il votante a modello fa parte dell'insieme di default, e in produzione gli
    artefatti sono gitignorati. Legge `MODELS_DIR` a ogni chiamata, invece di ricordarselo, cosi'
    la variabile d'ambiente che lo sposta continua a valere.
    """
    return (MODELS_DIR / "swing_model.joblib").exists()


@lru_cache(maxsize=1)
def swing_model():
    """Il modello a swing addestrato, o `None` se l'artefatto non c'e'.

    In cache perche' il votante della confluenza lo chiede una volta per configurazione della
    griglia. Il rovescio e' che un riaddestramento si vede solo riavviando il processo: e' la
    stessa condizione della pagina, che il modello lo carica una volta in `st.session_state`.
    """
    return load_model(MODELS_DIR / "swing_model.joblib") if swing_model_disponibile() else None


def swing_features(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Le 41 colonne, nell'ordine di `SWING_COLUMNS`, dalle candele che la pagina ha in mano.

    Una sola definizione per i due modelli che le usano -- quello a swing e la politica RL --
    perche' sono state addestrate sulla stessa matrice e un ordine diverso non solleva niente:
    sposta soltanto i numeri.

    Le scale lunghe sono **quelle piu' lunghe della base**, non sempre `1h` e `1d`: a 4h aggregare
    a un'ora vorrebbe dire ricampionare all'insu', cioe' inventare barre. Le colonne che restano
    fuori diventano NaN, che il modello a gradienti tratta come categoria a se'.
    """
    from cryptofarm.ml.bar_features import SWING_COLUMNS, SWING_SCALES, build_swing_features

    minuti = interval_to_minutes(interval_from_index(df.index))
    scale = tuple(
        s
        for s in SWING_SCALES
        if interval_to_minutes(s) > minuti and len(df) * minuti >= SCALA_MINIMA_BARRE * interval_to_minutes(s)
    )
    return build_swing_features(symbol, df, scales=scale).reindex(columns=SWING_COLUMNS)


def swing_predictions(df: pd.DataFrame, model, symbol: str = "") -> np.ndarray:
    """La previsione del modello a swing per ogni barra: -1 su un minimo locale, +1 su un massimo.

    Le scale lunghe sono **quelle piu' lunghe della base**, non sempre `1h` e `1d`: a 4h
    aggregare a un'ora vorrebbe dire ricampionare all'insu', cioe' inventare barre. Le colonne
    che restano fuori diventano NaN, che il modello a gradienti tratta come categoria a se' --
    ed e' una degradazione **misurata**, non sperata: senza `@1d` l'IC passa da +0,0540 a +0,0524,
    senza `@1h` e `@1d` a +0,0542 (`.claude/docs/modello-swing.md` §4).

    `symbol` serve alle due colonne di posizionamento; senza, restano NaN e l'IC misurato scende
    di un decimillesimo. Non e' un percorso degradato da evitare, e' la condizione normale della
    pagina.
    """
    frame = swing_features(df, symbol)
    previsto = model.predict(frame.to_numpy(dtype=float))
    # `atr_rel` NaN sono le barre di riscaldamento, dove le feature strutturali non esistono
    # ancora: e' lo stesso criterio con cui `swing_trainer.campione_simbolo` le scarta.
    previsto[frame["atr_rel"].isna().to_numpy()] = np.nan
    return previsto


def swing_exposure(previsto: np.ndarray, entra: float, esci: float, cadenza: int) -> np.ndarray:
    """Dentro quando `|previsione|` e' alta, fuori quando e' bassa: vero per barra.

    **Non e' una regola direzionale, ed e' il punto.** Il segno della previsione non dice il
    verso: `.claude/docs/modello-swing.md` §5.1 misura che *entrambi* i poli precedono rendimenti
    sopra la media -- il polo +1 non e' «vendi» ma «tendenza forte in corso», e in cripto la
    continuazione paga. Comprare i minimi previsti e vendere i massimi perde a tutte le soglie e
    tutte le cadenze, in validazione come fuori campione. Cio' che la forma a U sostiene e' solo
    *quanto* stare esposti, non da che parte.

    L'isteresi non e' un abbellimento: con `entra == esci` ogni oscillazione della previsione
    attorno alla soglia costa un giro di commissioni. La decisione si prende una volta ogni
    `cadenza` barre e resta ferma in mezzo.
    """
    dentro = np.zeros(len(previsto), dtype=bool)
    stato, inizio = False, 0
    for i in range(0, len(previsto), max(int(cadenza), 1)):
        forza = abs(previsto[i])
        if not np.isfinite(forza):
            continue
        nuovo = bool(forza >= (esci if stato else entra))
        if nuovo != stato:
            dentro[inizio:i] = stato
            stato, inizio = nuovo, i
    dentro[inizio:] = stato
    return dentro


def rl_model_disponibile() -> bool:
    """Se l'artefatto della politica RL e' su disco. Come per il modello a swing, la risposta serve
    **prima** di caricare: in produzione gli artefatti sono gitignorati."""
    return (MODELS_DIR / "rl_model.joblib").exists()


@lru_cache(maxsize=1)
def rl_model():
    """I due regressori `Q[0]` e `Q[1]` della politica, o `None` se l'artefatto non c'e'."""
    return load_model(MODELS_DIR / "rl_model.joblib") if rl_model_disponibile() else None


def rl_exposure(Q, df: pd.DataFrame, symbol: str = "", cadenza: int | None = None) -> np.ndarray:
    """Dentro o fuori per barra, secondo la politica appresa. Vero per barra.

    La decisione si prende una volta al giorno -- la cadenza a cui la politica e' stata addestrata
    e misurata -- e resta ferma in mezzo. Cambiarla non regola una manopola: cambia il problema,
    perche' il costo dentro la ricompensa e' calibrato su quel passo.

    Le barre di riscaldamento, dove `atr_rel` non esiste ancora, sono fuori: e' lo stesso criterio
    con cui l'addestramento le scarta, e senza, la politica deciderebbe su uno stato di soli NaN.
    """
    from cryptofarm.ml.rl import posizioni

    frame = swing_features(df, symbol)
    stato = frame.to_numpy(dtype=float)
    passo = max(int(cadenza or swing_cadenza(df.index)), 1)
    decisioni = np.arange(0, len(stato), passo)
    pronte = decisioni[frame["atr_rel"].to_numpy()[decisioni] == frame["atr_rel"].to_numpy()[decisioni]]
    dentro = np.zeros(len(stato), dtype=bool)
    if not len(pronte):
        return dentro
    azioni = posizioni(Q, stato[pronte])
    for i, (inizio, azione) in enumerate(zip(pronte, azioni)):
        fine = pronte[i + 1] if i + 1 < len(pronte) else len(stato)
        dentro[inizio:fine] = bool(azione)
    return dentro


def rl_signals(Q, df: pd.DataFrame, symbol: str = "") -> tuple[list[tuple], list[tuple]]:
    """La politica tradotta in acquisti e vendite alternati, come `swing_signals`."""
    return _eventi(rl_exposure(Q, df, symbol=symbol), df)


def _eventi(dentro: np.ndarray, df: pd.DataFrame) -> tuple[list[tuple], list[tuple]]:
    """Da esposizione per barra a due liste alternate di `(quando, prezzo)`.

    Se l'ultima posizione e' ancora aperta a fine serie non le si inventa una vendita: resta un
    acquisto spaiato, che `pnl` ignora accoppiando per indice.
    """
    cambi = np.flatnonzero(np.diff(dentro.astype(np.int8), prepend=np.int8(0)))
    close = df["Close"].to_numpy(dtype=float)
    eventi = [(df.index[i], float(close[i])) for i in cambi]
    return eventi[::2], eventi[1::2]


def swing_cadenza(index: pd.DatetimeIndex) -> int:
    """Una decisione al giorno, in barre. E' la cadenza con cui il modello e' stato misurato."""
    return max(round(1440 / interval_to_minutes(interval_from_index(index))), 1)


def swing_signals(
    df: pd.DataFrame,
    model,
    entra: float = SWING_ENTRA,
    esci: float = SWING_ESCI,
    symbol: str = "",
) -> tuple[list[tuple], list[tuple]]:
    """La regola a esposizione tradotta in acquisti e vendite alternati.

    Nessuna barriera e nessuno stop: qui l'uscita e' la stessa condizione dell'ingresso letta al
    contrario, perche' e' la sola forma su cui il modello sia stato misurato. Se l'ultima
    posizione e' ancora aperta a fine serie non le si inventa una vendita: resta un acquisto
    spaiato, che `pnl` ignora accoppiando per indice.
    """
    previsto = swing_predictions(df, model, symbol=symbol)
    return _eventi(swing_exposure(previsto, entra, esci, swing_cadenza(df.index)), df)
