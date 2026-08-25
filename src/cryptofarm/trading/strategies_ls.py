"""Strategie con posizione a due versi: lunga, fuori, corta.

Le strategie di `strategies.py` restituiscono due liste, compri e vendi, e sanno fare una cosa
sola: comprare per poi rivendere. Su nove anni misurati (`.claude/docs/backtest-strategie.md`)
questo costa due volte -- si sta fuori dal mercato meta' del tempo, e nei mercati in discesa non
si guadagna niente per definizione -- e le tre cause di perdita si ripetono uguali:

1. **si entra contro il trend** (le bande ATR comprano ogni minimo, anche il primo di una
   discesa), e nessun indicatore disponibile diceva se un trend c'era;
2. **si opera troppo** (100-3.000 volte l'anno), e il margine lordo per operazione e' dello stesso
   ordine del costo di transazione;
3. **si esce senza un piano**: o mai, o a un livello fisso che non tiene conto della volatilita'.

Le quattro strategie qui sotto sono costruite ognuna attorno a una di quelle cause. Restituiscono
una lista di **cambi di posizione** `(timestamp, prezzo, obiettivo)` con obiettivo in `{+1, 0, -1}`,
che `pnl.simulate_positions` trasforma in operazioni chiuse. Il formato e' diverso da quello
storico perche' l'inversione diretta -- da lungo a corto senza passare per il flat -- nelle due
liste separate non e' rappresentabile.

Convenzioni comuni, uguali per tutte:

- **si decide alla chiusura della barra e si esegue a quel prezzo.** Su mercati aperti 24 ore con
  book profondo e' realistico entro i secondi; non lo sarebbe su un mercato con apertura e
  chiusura.
- **gli stop si attivano dentro la barra**, sul minimo (lungo) o sul massimo (corto), con
  esecuzione al livello dello stop. E' la convenzione standard; sottostima le perdite quando il
  prezzo salta oltre lo stop, che nelle liquidazioni crypto succede.
- **niente look-ahead**: il canale di Donchian e' spostato di una barra, le span di Ichimoku sono
  quelle gia' visibili sul grafico, ogni incrocio si valuta fra `i-1` e `i`. Vale anche *dentro* la
  barra: lo stop a trailing usa l'estremo raggiunto fino a `i-1` e l'ATR di `i-1`, mai quelli della
  barra su cui viene testato, e all'ingresso parte dal prezzo di riempimento invece che
  dall'estremo della barra d'ingresso.
- `allow_short=False` rende ognuna delle quattro una strategia solo lunga, identica in tutto il
  resto: e' il confronto che isola il contributo del verso corto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.trading.indicators_extra import ExtraCache


def _arrays(candles: pd.DataFrame) -> tuple[np.ndarray, ...]:
    return (
        candles.index,
        candles["Open"].to_numpy(),
        candles["High"].to_numpy(),
        candles["Low"].to_numpy(),
        candles["Close"].to_numpy(),
    )


def donchian_breakout(
    candles: pd.DataFrame,
    cache: ExtraCache,
    channel: int = 20,
    adx_window: int = 14,
    adx_min: float = 20.0,
    atr_window: int = 14,
    atr_multiplier: float = 3.0,
    regime_ema: int = 200,
    allow_short: bool = True,
) -> list:
    """Rottura di canale con filtro di forza del trend e uscita a trailing sull'ATR.

    Risponde alla causa 1 e alla 3. Si entra **nella** direzione del movimento -- alla rottura del
    massimo (o del minimo) delle ultime `channel` barre -- solo quando l'ADX dice che un trend
    c'e' davvero, e solo dalla parte in cui sta la media lunga. Si esce con uno stop che segue il
    massimo raggiunto a distanza di `atr_multiplier` ATR (chandelier): lascia correre i movimenti
    lunghi, che sono l'unica fonte di guadagno di un sistema di rottura, e taglia presto i falsi
    segnali, che sono la maggioranza.
    """
    index, _, highs, lows, closes = _arrays(candles)
    upper, lower = cache.donchian(channel)
    adx = cache.adx(adx_window)
    atr = cache.atr(atr_window)
    regime = cache.ema(regime_ema) if regime_ema else None

    events: list = []
    position = 0
    extreme = 0.0
    start = max(channel, adx_window, atr_window, regime_ema or 0) + 1

    for i in range(start, len(closes)):
        price = closes[i]
        if position != 0 and not np.isnan(atr[i - 1]):
            # Lo stop in vigore *durante* la barra i e' quello calcolabile alla chiusura della
            # barra precedente: estremo raggiunto fino a i-1 e ATR di i-1. Aggiornarlo con il
            # massimo (o il minimo) della barra i prima di confrontarlo con il suo minimo (o
            # massimo) assume che dentro la barra l'estremo favorevole arrivi per primo -- vero
            # meta' delle volte, e sempre a favore: fa uscire a un prezzo che non era ottenibile.
            if position > 0:
                stop = extreme - atr_multiplier * atr[i - 1]
                if lows[i] <= stop:
                    events.append((index[i], float(stop), 0))
                    position = 0
                else:
                    extreme = max(extreme, highs[i])
            else:
                stop = extreme + atr_multiplier * atr[i - 1]
                if highs[i] >= stop:
                    events.append((index[i], float(stop), 0))
                    position = 0
                else:
                    extreme = min(extreme, lows[i])

        if np.isnan(upper[i]) or np.isnan(adx[i]) or adx[i] < adx_min:
            continue
        trend_up = regime is None or price > regime[i]
        trend_down = regime is None or price < regime[i]

        if position <= 0 and price > upper[i] and trend_up:
            events.append((index[i], float(price), 1))
            position = 1
            extreme = price
        elif allow_short and position >= 0 and price < lower[i] and trend_down:
            events.append((index[i], float(price), -1))
            position = -1
            extreme = price

    return events


def squeeze_breakout(
    candles: pd.DataFrame,
    cache: ExtraCache,
    bb_window: int = 20,
    bb_dev: float = 2.0,
    kc_window: int = 20,
    kc_multiplier: float = 1.5,
    atr_window: int = 14,
    atr_multiplier: float = 2.5,
    confirm_volume: bool = True,
    obv_window: int = 20,
    allow_short: bool = True,
) -> list:
    """Rottura dopo compressione: bande di Bollinger dentro il canale di Keltner, poi espansione.

    Risponde alla causa 2. La compressione e' rara per costruzione -- e' il mercato che smette di
    muoversi -- quindi la strategia opera poche volte l'anno senza che serva un filtro arbitrario
    sul numero di segnali. La direzione della rottura la decide la posizione del prezzo rispetto
    alla media delle bande; il volume, se richiesto, deve confermarla (OBV in salita per un lungo).
    L'uscita e' a trailing ATR come sopra.
    """
    index, _, highs, lows, closes = _arrays(candles)
    bb_high, bb_low, bb_mid = cache.bollinger(bb_window, bb_dev)
    kc_high, kc_low = cache.keltner(kc_window, atr_window, kc_multiplier)
    atr = cache.atr(atr_window)
    obv = cache.obv_slope(obv_window) if confirm_volume else None

    squeeze = (bb_high < kc_high) & (bb_low > kc_low)

    events: list = []
    position = 0
    extreme = 0.0
    start = max(bb_window, kc_window, atr_window, obv_window) + 1

    for i in range(start, len(closes)):
        price = closes[i]
        if position != 0 and not np.isnan(atr[i - 1]):
            # Lo stop in vigore *durante* la barra i e' quello calcolabile alla chiusura della
            # barra precedente: estremo raggiunto fino a i-1 e ATR di i-1. Aggiornarlo con il
            # massimo (o il minimo) della barra i prima di confrontarlo con il suo minimo (o
            # massimo) assume che dentro la barra l'estremo favorevole arrivi per primo -- vero
            # meta' delle volte, e sempre a favore: fa uscire a un prezzo che non era ottenibile.
            if position > 0:
                stop = extreme - atr_multiplier * atr[i - 1]
                if lows[i] <= stop:
                    events.append((index[i], float(stop), 0))
                    position = 0
                else:
                    extreme = max(extreme, highs[i])
            else:
                stop = extreme + atr_multiplier * atr[i - 1]
                if highs[i] >= stop:
                    events.append((index[i], float(stop), 0))
                    position = 0
                else:
                    extreme = min(extreme, lows[i])

        released = squeeze[i - 1] and not squeeze[i]
        if not released or position != 0:
            continue
        if obv is not None:
            # Conferma richiesta ma non calcolabile (volume nullo sulla finestra): si sta fuori.
            # Cadere qui nel caso "nessuna conferma" faceva entrare senza il filtro che il
            # chiamante credeva attivo.
            if np.isnan(obv[i]):
                continue
            long_side = price > bb_mid[i] and obv[i] > 0
            short_ok = price < bb_mid[i] and obv[i] < 0
        else:
            long_side = price > bb_mid[i]
            short_ok = price < bb_mid[i]

        if long_side:
            events.append((index[i], float(price), 1))
            position = 1
            extreme = price
        elif allow_short and short_ok:
            events.append((index[i], float(price), -1))
            position = -1
            extreme = price

    return events


def trend_pullback(
    candles: pd.DataFrame,
    cache: ExtraCache,
    regime_ema: int = 200,
    stochrsi_window: int = 14,
    stochrsi_smooth: int = 3,
    oversold: float = 0.2,
    overbought: float = 0.8,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    allow_short: bool = True,
) -> list:
    """Rientro dall'ipervenduto **dentro** un trend gia' stabilito, e simmetrico al ribasso.

    Risponde alla causa 1 nel modo opposto alla rottura: qui si compra un ritracciamento, come
    "Close ATR", ma solo dalla parte della media lunga. La differenza fra le due misure -- questa
    e la stessa senza filtro di regime -- e' la misura diretta di quanto valga il filtro.

    Lo stop e' fisso a `atr_multiplier` ATR dall'ingresso; l'uscita in guadagno e' il ritorno
    dell'oscillatore in zona opposta, cioe' quando il ritracciamento e' finito.
    """
    index, _, highs, lows, closes = _arrays(candles)
    # `regime_ema=0` toglie il filtro di trend e lascia solo l'oscillatore: e' l'ablazione che
    # misura quanto vale il filtro, cioe' la differenza fra questa strategia e "Close ATR".
    regime = cache.ema(regime_ema) if regime_ema else None
    stoch = cache.stochrsi(stochrsi_window, stochrsi_smooth)
    atr = cache.atr(atr_window)

    events: list = []
    position = 0
    stop_price = 0.0
    start = max(regime_ema or 0, stochrsi_window * 2, atr_window) + 2

    for i in range(start, len(closes)):
        price = closes[i]
        if position > 0:
            if lows[i] <= stop_price:
                events.append((index[i], float(stop_price), 0))
                position = 0
            elif stoch[i] >= overbought or (regime is not None and price < regime[i]):
                events.append((index[i], float(price), 0))
                position = 0
        elif position < 0:
            if highs[i] >= stop_price:
                events.append((index[i], float(stop_price), 0))
                position = 0
            elif stoch[i] <= oversold or (regime is not None and price > regime[i]):
                events.append((index[i], float(price), 0))
                position = 0

        if position != 0 or np.isnan(stoch[i]) or np.isnan(atr[i]):
            continue
        if regime is not None and np.isnan(regime[i]):
            continue
        if (regime is None or price > regime[i]) and stoch[i - 1] < oversold <= stoch[i]:
            events.append((index[i], float(price), 1))
            position = 1
            stop_price = price - atr_multiplier * atr[i]
        elif allow_short and (regime is None or price < regime[i]) and stoch[i - 1] > overbought >= stoch[i]:
            events.append((index[i], float(price), -1))
            position = -1
            stop_price = price + atr_multiplier * atr[i]

    return events


def ichimoku_trend(
    candles: pd.DataFrame,
    cache: ExtraCache,
    fast: int = 9,
    slow: int = 26,
    span: int = 52,
    require_cloud: bool = True,
    allow_short: bool = True,
) -> list:
    """Incrocio Tenkan/Kijun con conferma della nuvola: il sistema di trend completo, come metro.

    Non introduce nulla di nuovo rispetto alle altre tre -- e' un sistema di trend con uscita
    incorporata, gia' pronto e molto usato -- e serve proprio per questo: se una strategia
    costruita apposta non batte Ichimoku preso dal manuale, non vale il lavoro che costa.
    """
    index, _, _, _, closes = _arrays(candles)
    tenkan, kijun, span_a, span_b = cache.ichimoku(fast, slow, span)
    cloud_top = np.maximum(span_a, span_b)
    cloud_bottom = np.minimum(span_a, span_b)

    events: list = []
    position = 0
    start = slow + span + 2

    for i in range(start, len(closes)):
        price = closes[i]
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(span_b[i]):
            continue
        cross_up = tenkan[i - 1] <= kijun[i - 1] and tenkan[i] > kijun[i]
        cross_down = tenkan[i - 1] >= kijun[i - 1] and tenkan[i] < kijun[i]
        above_cloud = price > cloud_top[i] if require_cloud else True
        below_cloud = price < cloud_bottom[i] if require_cloud else True

        if position > 0 and (cross_down or price < kijun[i]):
            events.append((index[i], float(price), 0))
            position = 0
        elif position < 0 and (cross_up or price > kijun[i]):
            events.append((index[i], float(price), 0))
            position = 0

        if position == 0 and cross_up and above_cloud:
            events.append((index[i], float(price), 1))
            position = 1
        elif position == 0 and allow_short and cross_down and below_cloud:
            events.append((index[i], float(price), -1))
            position = -1

    return events


def band_reversion_gated(
    candles: pd.DataFrame,
    cache: ExtraCache,
    kama_window: int = 10,
    atr_window: int = 14,
    band_multiplier: float = 2.5,
    adx_window: int = 14,
    adx_max: float = 20.0,
    stop_multiplier: float = 2.0,
    regime_ema: int = 0,
    allow_short: bool = True,
) -> list:
    """ "Close ATR" con il filtro che le mancava: si torna verso la media **solo dentro un range**.

    E' la combinazione diretta fra la strategia storica peggiore per mediana e l'indicatore che
    non c'era. L'ingresso e' lo stesso -- chiusura sotto la banda inferiore costruita su KAMA piu'
    o meno `band_multiplier` ATR -- ma avviene solo quando l'ADX dice che **non** c'e' trend
    (`adx < adx_max`): comprare i minimi funziona in un mercato che oscilla e distrugge il conto
    in uno che scende, e nei nove anni misurati era sempre attiva.

    L'uscita e' il ritorno alla media (KAMA), che e' l'obiettivo naturale di una strategia di
    ritorno alla media, oppure lo stop a `stop_multiplier` ATR. Con `allow_short` la simmetrica
    sopra la banda superiore. `regime_ema=0` lascia entrambi i versi liberi; un valore positivo
    limita i lunghi a sopra la media lunga e i corti a sotto.
    """
    index, _, highs, lows, closes = _arrays(candles)
    kama = cache.kama(kama_window)
    atr = cache.atr(atr_window)
    adx = cache.adx(adx_window)
    regime = cache.ema(regime_ema) if regime_ema else None

    events: list = []
    position = 0
    stop_price = 0.0
    start = max(kama_window, atr_window, adx_window, regime_ema or 0) + 2

    for i in range(start, len(closes)):
        price = closes[i]
        if np.isnan(atr[i]) or np.isnan(kama[i]) or np.isnan(adx[i]):
            continue
        upper = kama[i] + band_multiplier * atr[i]
        lower = kama[i] - band_multiplier * atr[i]

        if position > 0:
            if lows[i] <= stop_price:
                events.append((index[i], float(stop_price), 0))
                position = 0
            elif price >= kama[i]:
                events.append((index[i], float(price), 0))
                position = 0
        elif position < 0:
            if highs[i] >= stop_price:
                events.append((index[i], float(stop_price), 0))
                position = 0
            elif price <= kama[i]:
                events.append((index[i], float(price), 0))
                position = 0

        if position != 0 or adx[i] >= adx_max:
            continue
        if price <= lower and (regime is None or price > regime[i]):
            events.append((index[i], float(price), 1))
            position = 1
            stop_price = price - stop_multiplier * atr[i]
        elif allow_short and price >= upper and (regime is None or price < regime[i]):
            events.append((index[i], float(price), -1))
            position = -1
            stop_price = price + stop_multiplier * atr[i]

    return events


STRATEGIES = {
    "donchian_breakout": donchian_breakout,
    "squeeze_breakout": squeeze_breakout,
    "trend_pullback": trend_pullback,
    "ichimoku_trend": ichimoku_trend,
    "band_reversion_gated": band_reversion_gated,
}
