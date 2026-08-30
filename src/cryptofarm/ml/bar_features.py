"""Le feature per barra: **una sola definizione**, condivisa da addestramento e inferenza.

Questo modulo esiste per una ragione che il progetto ha gia' pagato due volte: ogni artefatto che
addestramento e inferenza devono tenere allineati a mano e' un modo in cui i due divergono in
silenzio. Qui non c'e' niente di appreso dai dati -- nessuno scaler, nessun parametro salvato
accanto al modello -- e la funzione che costruisce le colonne e' letteralmente la stessa nei due
percorsi.

Sostituisce `ml/features.py` **solo per il modello nuovo**. Quello vecchio resta dov'e' perche'
`signal_model` e `meta_model` sono stati addestrati con le sue colonne, e cambiargliele sotto
significherebbe farli sbagliare senza dare nessun segno.

## Le quattro famiglie, e perche' ognuna e' li'

1. **struttura e volatilita' dell'asset** (13 colonne) -- sono quelle di `scripts/meta_gate.py`,
   spostate qui invece che copiate: erano gia' l'insieme buono, scale-free e verificato.
2. **contesto trasversale** (3 colonne) -- rango di forza nell'universo, ampiezza di mercato,
   forza contro BTC. `ricerca-quant-ml.md` §1.5.1: un vantaggio debole diventa eseguibile in
   sezione, non nel tempo.
3. **posizionamento** (2 colonne) -- quanto sono affollati i lunghi, fra i conti al dettaglio e
   fra le posizioni dei top trader. E' l'unica informazione che questo progetto non aveva mai
   avuto, e sono le **sole due** derivate su dodici che hanno superato il pannello a 5 asset x 2
   finestre (`data/positioning.py` riporta la tabella).
4. **il timeframe** (1 colonna) -- il modello e' unico su 1h/4h/1d e deve potersi condizionare
   sulla granularita' invece di mediare comportamenti diversi.

## Cosa e' stato misurato e **non** e' entrato

Il rango di compressione delle bande (l'idea che «le bande strette precedono il movimento»). Sul
pannello, contro l'escursione dei cinque giorni successivi in unita' di prezzo:

| feature | \\|IC\\| medio | celle con segno concorde |
|---|---|---|
| `atr_rel` (il livello) | **0,422** | 10 su 10 |
| rango di compressione dell'ATR su 200 barre | 0,229 | 10 su 10 |
| rango di compressione delle bande su 200 barre | 0,158 | 10 su 10 |

Il rango e' una versione **piu' debole** del livello, che e' gia' in tabella: la volatilita' si
raggruppa, non si alterna. Aggiungerlo sarebbe una colonna correlata con una che c'e' gia'.

Vale la pena registrare il rapporto fra i due numeri: **l'ampiezza del movimento e' prevedibile
(IC 0,42), la sua direzione quasi no (IC 0,06 nel caso migliore).** Le barriere sono scalate
sull'ATR proprio per questo -- normalizzano via la parte prevedibile, e lasciano al modello la
domanda difficile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.data.positioning import load_positioning
from cryptofarm.trading.indicators_extra import ExtraCache

ASSET_COLUMNS = [
    "dist_ema50_atr",
    "dist_ema200_atr",
    "pos_canale",
    "pos_bollinger",
    "atr_rel",
    "adx",
    "larghezza_bollinger",
    "stochrsi",
    "mfi",
    "obv_pendenza",
    "volume_rel",
    "escursione_rel",
    "sopra_ema200",
]
CROSS_COLUMNS = ["rango_forza", "ampiezza_mercato", "forza_su_btc"]
# Quante barre al massimo si puo' riportare avanti un valore trasversale preso da uno store piu'
# vecchio delle candele. Oltre, e' meglio un NaN dichiarato di un numero vecchio.
POSITIONING_COLUMNS = ["affollamento_conti", "affollamento_posizioni"]


def asset_features(candles: pd.DataFrame, cache: ExtraCache) -> pd.DataFrame:
    """Contesto dell'asset alla barra `t`, in unita' confrontabili fra BTC e DOGE.

    Niente prezzi assoluti, niente ATR grezzo: un modello unico su quindici asset con feature
    dimensionali impara l'identita' dell'asset, non il suo comportamento.
    """
    close = candles["Close"].to_numpy(dtype=float)
    high = candles["High"].to_numpy(dtype=float)
    low = candles["Low"].to_numpy(dtype=float)
    volume = candles["Volume"].to_numpy(dtype=float)

    atr = cache.atr(14)
    ema50, ema200 = cache.ema(50), cache.ema(200)
    upper, lower = cache.donchian(20)
    boll_up, boll_mid, boll_low = cache.bollinger(20, 2.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        frame = pd.DataFrame(
            {
                # dove sta il prezzo rispetto alla struttura, misurato in ATR
                "dist_ema50_atr": (close - ema50) / atr,
                "dist_ema200_atr": (close - ema200) / atr,
                "pos_canale": (close - lower) / (upper - lower),
                "pos_bollinger": (close - boll_low) / (boll_up - boll_low),
                # quanto e' volatile e quanto e' direzionale
                "atr_rel": atr / close,
                "adx": cache.adx(14),
                "larghezza_bollinger": (boll_up - boll_low) / boll_mid,
                # oscillatori e volume
                "stochrsi": cache.stochrsi(14, 3),
                "mfi": cache.mfi(14),
                "obv_pendenza": cache.obv_slope(20),
                "volume_rel": volume / pd.Series(volume).rolling(96).median().to_numpy(),
                "escursione_rel": (high - low) / close,
                # regime dell'asset
                "sopra_ema200": (close > ema200).astype(float),
            },
            index=candles.index,
        )

    # `ExtraCache.atr` restituisce **0,0** finche' la finestra non e' piena, non NaN. Uno zero e'
    # un valore plausibile e sbagliato: `barrier_widths` ci cade sopra il pavimento sulle
    # commissioni e mette uno stop allo 0,3% invece che a k x ATR, che la prima candela tocca.
    # Succede all'inizio di ogni serie caricata, quindi in pagina come in addestramento.
    frame.loc[~(frame["atr_rel"] > 0), "atr_rel"] = np.nan
    # Stessa trappola su `sopra_ema200`: `NaN > x` e' `False`, e `False.astype(float)` e' 0,0.
    # Senza questa riga le prime 199 barre di ogni serie dicono «sotto la EMA200» a prescindere
    # dal vero -- misurato: 199 righe su 199 a 4h. In pagina e' peggio che in addestramento,
    # perche' la finestra caricata dall'exchange e' corta e quelle barre sono una quota grossa.
    frame.loc[np.isnan(ema200), "sopra_ema200"] = np.nan
    return frame.replace([np.inf, -np.inf], np.nan)


def cross_features(closes: pd.DataFrame, lookback: int = 30) -> dict[str, pd.DataFrame | pd.Series]:
    """Contesto **trasversale**: come sta questo asset rispetto agli altri, e come sta il mercato.

    E' l'informazione che nessuna misura precedente del progetto ha mai dato a un modello, perche'
    tutte guardavano un simbolo alla volta.
    """
    returns = closes / closes.shift(lookback) - 1.0
    rank = returns.rank(axis=1, pct=True)
    breadth = (closes > closes.rolling(50).mean()).mean(axis=1)
    btc_rel = (closes.div(closes["BTCUSDT"], axis=0)).pipe(lambda f: f / f.shift(lookback) - 1.0)
    return {"rango_forza": rank, "ampiezza_mercato": breadth, "forza_su_btc": btc_rel}


def positioning_features(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Quanto sono affollati i lunghi, fra i conti al dettaglio e fra i top trader.

    In logaritmo perche' sono rapporti: un rapporto di 3 e uno di 1/3 sono lo stesso squilibrio
    di verso opposto, e in scala lineare non lo sarebbero. Il segno misurato e' **negativo** in
    tutte e dieci le celle del pannello: quando i lunghi sono affollati, i giorni successivi
    rendono meno.

    Restituisce NaN dove lo store non arriva -- prima del 2021, o su un simbolo senza perpetuo.
    Il modello a gradienti tratta i NaN come una categoria a se', quindi la mancanza non va
    riempita con uno zero che assomiglierebbe a un equilibrio.
    """
    raw = load_positioning(symbol, index)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame(
            {
                "affollamento_conti": np.log(raw["retail_accounts_ratio"].replace(0, np.nan)),
                "affollamento_posizioni": np.log(raw["top_positions_ratio"].replace(0, np.nan)),
            },
            index=index,
        ).replace([np.inf, -np.inf], np.nan)


# --- Le colonne del modello a swing -------------------------------------------------------------

# Scale lunghe da affiancare alla barra base. **Misurate, non scelte**: sulla meta' futura del
# target, base 5m sola da' IC 0,0502; con 1h sale a 0,0524; con 1h e 1d a 0,0540; aggiungendo
# anche 15m e 4h resta 0,0539 con 67 colonne invece di 41. Le due che restano sono quelle che
# portano informazione che la barra base non ha gia' -- 15m e 4h stanno troppo vicine a cio' che
# EMA200 e ADX sulla base gia' descrivono.
SWING_SCALES = ("1h", "1d")
SWING_BASE_COLUMNS = [*ASSET_COLUMNS, *POSITIONING_COLUMNS]
SWING_COLUMNS = [
    *SWING_BASE_COLUMNS,
    *[f"{c}@{s}" for s in SWING_SCALES for c in ASSET_COLUMNS],
]


def build_swing_features(
    symbol: str,
    candles: pd.DataFrame,
    scales: tuple[str, ...] = SWING_SCALES,
    cache: ExtraCache | None = None,
) -> pd.DataFrame:
    """Le 41 colonne del modello a swing: 15 sulla barra base piu' 13 per ogni scala lunga.

    `candles` sono barre **5m**. Le scale lunghe si ricavano aggregando qui invece che leggendole
    da fuori, cosi' addestramento e pagina passano dalla stessa riga di codice: il disallineamento
    fra le due definizioni e' il difetto che non si vede leggendo nessuno dei due file.

    Le colonne lunghe passano da `mtf.align_to_lower`, che rende disponibile la barra di 1h solo
    **dopo** che ha chiuso. Senza, il modello leggerebbe la barra oraria in formazione, cioe' un
    valore che contiene i cinque minuti che sta cercando di prevedere.

    Niente colonne trasversali: dipendono dagli altri quattordici asset, e in pagina si carica un
    simbolo alla volta. Niente `timeframe`: la base e' 5m e basta.
    """
    from cryptofarm.data.klines import resample_klines
    from cryptofarm.trading.mtf import align_to_lower

    cache = cache if cache is not None else ExtraCache(candles)
    frame = asset_features(candles, cache)
    frame[POSITIONING_COLUMNS] = positioning_features(symbol, candles.index)

    for scala in scales:
        lunghe = resample_klines(candles, scala)
        feature_lunghe = asset_features(lunghe, ExtraCache(lunghe))
        for colonna in ASSET_COLUMNS:
            frame[f"{colonna}@{scala}"] = align_to_lower(feature_lunghe[colonna], lunghe.index, scala, candles.index)
    return frame[[c for c in SWING_COLUMNS if c in frame.columns]]
