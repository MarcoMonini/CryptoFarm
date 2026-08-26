"""Rotazione trasversale: invece di prevedere *quando*, scegliere *quale*.

Tutto il resto di `trading/` decide su **un asset alla volta** -- dentro o fuori dal mercato, in
base alla storia di quell'asset. Fuori campione, su cinque asset, quella famiglia non batte il
possesso passivo (`.claude/docs/ricerca-quant-ml.md` §2.3). Questa e' l'altra famiglia: a ogni
ribilanciamento si ordinano gli asset per forza relativa e si tengono i primi `top` a peso uguale.
Il segnale non e' "il mercato sale" ma "questo sale piu' di quelli".

E' anche la risposta alla domanda sulle **coppie** (BTC/ETH, ETH/SOL): tenere la piu' forte fra due
e' questo codice con un universo di due e `top=1`. Non serve una gamba corta, che il mandato
esclude.

Il modulo sta nel pacchetto e non in `scripts/` perche' lo usa anche la pagina Streamlit, e la
dipendenza deve andare in quella direzione: il pacchetto tiene il meccanismo, `scripts/` ci
costruisce sopra le griglie e i confronti.

**Niente look-ahead**: la classifica alla barra `t` usa solo chiusure fino a `t` incluse, la
posizione si prende alla chiusura di `t` e il rendimento e' quello da `t` al ribilanciamento
successivo -- la stessa convenzione del resto del repository.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.data.klines import load_klines
from cryptofarm.trading.pnl import annualised, drawdown

# I cinque ad alta capitalizzazione, e l'universo largo dello store. Allargare **peggiora**: a
# quindici asset la mediana fuori campione passa da +62% a -0,9% (§2.5 del documento). `WIDE`
# resta perche' e' il controllo che lo dimostra, non perche' sia un'alternativa.
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
WIDE = MAJORS + [
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
]
UNIVERSI = {"majors": MAJORS, "wide": WIDE, "btc_eth": ["BTCUSDT", "ETHUSDT"]}

WALLET = 100.0
# A pronti si paga il listino taker di Binance, non quello dei perpetui: 0,1% per gamba.
FEE_PERCENT = 0.1

# I parametri centrali raccomandati. **Non sono l'ottimo di una griglia, di proposito**: la
# correlazione fra resa in stima e resa in verifica sulle prime dieci configurazioni e' -0,69,
# cioe' scegliere la migliore in campione e' peggio che prenderne una a caso.
LOOKBACK = 20
TOP = 2
EVERY = 7
REGIME_WINDOW = 50


def load_universe(symbols: list[str], interval: str, since: str, until: str | None = None) -> pd.DataFrame:
    """Chiusure allineate su un indice comune. Le celle prima del listing restano NaN, non zero."""
    closes = {}
    for symbol in symbols:
        candles = load_klines(symbol, interval)
        if candles.empty:
            continue
        closes[symbol] = candles["Close"]
    frame = pd.DataFrame(closes).sort_index()
    frame = frame[frame.index >= since]
    if until:
        frame = frame[frame.index < until]
    return frame


def backtest(
    closes: pd.DataFrame,
    lookback: int = LOOKBACK,
    top: int = TOP,
    every: int = EVERY,
    fee: float = FEE_PERCENT,
    regime: str = "none",
    regime_window: int = REGIME_WINDOW,
    skip: int = 0,
) -> dict:
    """Rotazione a peso uguale sui primi `top` per forza relativa.

    `skip` salta le ultime barre nel calcolo del momento (il classico 12-1 delle azioni serve a
    evitare l'inversione di breve).

    `regime="btc"` spegne **tutto il portafoglio** quando BTC sta sotto la sua media a
    `regime_window` barre. E' un interruttore unico e non uno per asset di proposito: in cripto la
    correlazione fra asset in caduta va a uno, quindi selezionare il migliore fra cinque che
    scendono non protegge da niente.
    """
    prices = closes.to_numpy(dtype=float)
    n_bars, n_assets = prices.shape
    if n_bars <= lookback + skip + 2:
        raise ValueError(f"servono piu' di {lookback + skip + 2} barre per questo lookback, ce ne sono {n_bars}")
    if regime == "btc" and "BTCUSDT" not in closes.columns:
        raise ValueError("il filtro di regime guarda BTCUSDT, che non e' nell'universo")

    # Momento causale: rendimento fra `t-lookback-skip` e `t-skip`, entrambi noti alla barra t.
    momentum = np.full_like(prices, np.nan)
    start = prices[: n_bars - lookback - skip]
    end = prices[lookback : n_bars - skip]
    momentum[lookback + skip :] = end / start - 1.0

    gate = np.ones(n_bars, dtype=bool)
    if regime == "btc":
        btc = closes["BTCUSDT"].to_numpy(dtype=float)
        mean = pd.Series(btc).rolling(regime_window).mean().to_numpy()
        gate = btc > mean
        gate[np.isnan(mean)] = False

    rebalances = np.arange(0, n_bars, every)
    equity = np.full(n_bars, WALLET, dtype=float)
    # Si contabilizza in **valore per asset**, non in pesi normalizzati: il portafoglio puo' stare
    # parzialmente in contanti (meno di `top` asset con forza positiva), e una normalizzazione a
    # somma uno cancellerebbe quella quota invece di tenerla ferma.
    holdings = np.zeros(n_assets)
    cash = WALLET
    turnover_total = 0.0
    n_rebalances = 0
    holdings_log: list[tuple[pd.Timestamp, tuple[str, ...]]] = []

    for row, next_row in zip(rebalances, list(rebalances[1:]) + [n_bars - 1]):
        value = cash + holdings.sum()
        scores = momentum[row]
        eligible = np.where(~np.isnan(scores) & ~np.isnan(prices[row]))[0]
        target = np.zeros(n_assets)
        if gate[row] and len(eligible) >= top:
            chosen = eligible[np.argsort(scores[eligible])[::-1][:top]]
            # Forza relativa **negativa non si compra**: essere il meno peggio di un mercato che
            # scende non e' un segnale. La quota resta 1/top, quindi cio' che si scarta va in
            # contanti invece di concentrarsi su chi resta -- degradare verso il contante, non
            # verso una scommessa piu' grossa.
            chosen = chosen[scores[chosen] > 0]
            target[chosen] = value / top

        traded = float(np.abs(target - holdings).sum())
        cost = traded * fee / 100.0
        turnover_total += traded / value if value > 0 else 0.0
        if traded > 0:
            n_rebalances += 1
            holdings_log.append((closes.index[row], tuple(closes.columns[i] for i in np.where(target > 0)[0])))
        holdings = target
        cash = value - holdings.sum() - cost

        # Segnatura a mercato barra per barra fino al prossimo ribilanciamento.
        held = np.where(holdings > 0)[0]
        base = prices[row]
        for step in range(row, next_row + 1):
            grown = float(np.nansum(holdings[held] * prices[step][held] / base[held])) if len(held) else 0.0
            equity[step] = cash + grown
        if len(held):
            holdings[held] = holdings[held] * prices[next_row][held] / base[held]

    cagr, _, sharpe = annualised(equity, closes.index)
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    return {
        "rendimento_%": round((equity[-1] / WALLET - 1) * 100, 1),
        "CAGR_%": round(cagr, 1),
        "Sharpe": round(sharpe, 2),
        "drawdown_%": round(drawdown(equity), 1),
        "ribilanciamenti": n_rebalances,
        "turnover_annuo": round(turnover_total / max(years, 1e-9), 1),
        "_equity": equity,
        "_holdings": holdings_log,
    }


def benchmarks(closes: pd.DataFrame) -> dict[str, dict]:
    """I due metri di paragone: BTC tenuto fermo, e l'universo a peso uguale tenuto fermo.

    Il secondo e' quello che conta. Porta la **stessa distorsione da sopravvivenza** della
    rotazione -- l'universo e' fatto dei grandi capitalizzati di oggi -- quindi confrontarsi con
    lui isola cio' che la rotazione aggiunge, invece di misurare quanto e' stato fortunato
    l'universo.
    """
    out = {}
    for nome, equity in (
        ("BTC comprare e tenere", _hold(closes[["BTCUSDT"]]) if "BTCUSDT" in closes.columns else None),
        ("universo a peso uguale", _hold(closes)),
    ):
        if equity is None:
            continue
        cagr, _, sharpe = annualised(equity, closes.index)
        out[nome] = {
            "rendimento_%": round((equity[-1] / WALLET - 1) * 100, 1),
            "CAGR_%": round(cagr, 1),
            "Sharpe": round(sharpe, 2),
            "drawdown_%": round(drawdown(equity), 1),
            "ribilanciamenti": 0,
            "turnover_annuo": 0.0,
            "_equity": equity,
        }
    return out


def _hold(closes: pd.DataFrame) -> np.ndarray:
    """Peso uguale su cio' che esiste alla prima barra, poi lasciato correre senza ribilanciare."""
    prices = closes.to_numpy(dtype=float)
    first = prices[0]
    live = ~np.isnan(first)
    weights = live / live.sum()
    return WALLET * np.nansum(weights * prices / first, axis=1)
