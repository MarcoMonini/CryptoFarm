"""Backtest sistematico delle strategie del simulatore al variare dei loro parametri.

`trading/simulator.py` misura una configurazione alla volta, sulle ultime `time_hours` ore, e
mostra il risultato in un grafico. Serve a guardare un caso; non risponde alle due domande che
contano prima di mettere soldi su una strategia: **quanto rende su tutto lo storico** e **quanto
il risultato dipende dai parametri** che i widget lasciano muovere.

Questo modulo esegue le stesse funzioni di `trading/strategies.py`, senza copiarne la logica, su
griglie di parametri e sull'intero archivio, e produce due tabelle: una riga per configurazione
con le metriche aggregate, e una riga per configurazione e anno per vedere se il risultato viene
da tutto il periodo o da un singolo trimestre.

Tre scelte che cambiano l'interpretazione dei numeri:

- **Il P&L e' quello del simulatore**, `pnl.simulate_trading_with_commisions`: capitale sempre
  reinvestito per intero, commissione su ogni gamba, segnali accoppiati per indice. Se una
  strategia resta aperta alla fine, quell'ultima posizione non entra nel conto -- come nella
  pagina.
- **Gli indicatori sono quelli di `indicators.add_technical_indicator`**. `indicator_frame` li
  ricalcola con le stesse chiamate a `ta`, ma riceve il PSAR gia' pronto e memorizza le colonne
  per parametro: il PSAR costa 26 secondi sulle 338.000 barre a 15m e non dipende da nessuno dei
  parametri che si spazzolano, quindi ricalcolarlo per ognuna delle migliaia di configurazioni
  sarebbe l'intero costo dell'analisi. L'equivalenza con la funzione di produzione e' verificata
  colonna per colonna da `tests/test_strategy_sweep.py`.
- **L'equity e' segnata a mercato barra per barra**, non solo alla chiusura di ogni operazione:
  drawdown e volatilita' calcolati sulla sola curva dei trade chiusi ignorano quanto si e' stati
  sotto mentre la posizione era aperta, che e' esattamente cio' che fa smettere di seguire una
  strategia.

    python -m scripts.strategy_sweep --list                 # le griglie disponibili
    python -m scripts.strategy_sweep --grid close_atr       # una griglia
    python -m scripts.strategy_sweep --all --interval 15m
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from ta.momentum import KAMAIndicator, RSIIndicator, StochasticOscillator, TSIIndicator
from ta.trend import EMAIndicator, PSARIndicator
from ta.volatility import AverageTrueRange

from cryptofarm.data.klines import load_klines
from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import strategies as strat
from cryptofarm.trading.pnl import simulate_trading_with_commisions

SYMBOL = "BTCUSD"
SINCE = "2017-01-01"  # prima di questa data la fonte e' fatta per meta' di barre piatte
WALLET = 100.0
FEE_PERCENT = 0.1  # commissione per gamba, il default della pagina
# Il PSAR ha due parametri, `step` e `max_step`, ma i widget che li muovevano sono commentati in
# `simulator.py`: la pagina usa sempre i default di `trading_analysis`, e questi sono quelli.
PSAR_STEP = 0.01
PSAR_MAX_STEP = 0.4

OUTPUT_DIR = PROJECT_ROOT / "analysis_cache" / "sweeps"


# ---------------------------------------------------------------------------------------------
# Indicatori
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Indicators:
    """I parametri che la barra laterale espone per gli indicatori, con i default di `config.py`."""

    rsi_window: int = 12
    rsi_window2: int = 24
    rsi_window3: int = 36
    ema_window: int = 10
    ema_window2: int = 50
    ema_window3: int = 200
    atr_window: int = 5
    atr_multiplier: float = 1.6
    kama_pow1: int = 2
    kama_pow2: int = 30


class _ColumnCache:
    """Colonne indicizzate per i soli parametri da cui dipendono.

    Le griglie muovono un parametro alla volta: senza questa memoria l'RSI a finestra 12 verrebbe
    ricalcolato per ognuna delle sette moltiplicazioni dell'ATR che non lo toccano.
    """

    def __init__(self, candles: pd.DataFrame, psar: np.ndarray):
        self.candles = candles
        self.psar = psar
        self._cache: dict[tuple, pd.Series] = {}

    def _get(self, key: tuple, build) -> pd.Series:
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    def rsi(self, window: int) -> pd.Series:
        return self._get(("rsi", window), lambda: RSIIndicator(close=self.candles["Close"], window=window).rsi())

    def atr(self, window: int) -> pd.Series:
        return self._get(
            ("atr", window),
            lambda: AverageTrueRange(
                high=self.candles["High"], low=self.candles["Low"], close=self.candles["Close"], window=window
            ).average_true_range(),
        )

    def ema(self, column: str, window: int) -> pd.Series:
        return self._get(
            ("ema", column, window),
            lambda: EMAIndicator(close=self.candles[column], window=window).ema_indicator(),
        )

    def kama(self, window: int, pow1: int, pow2: int) -> pd.Series:
        return self._get(
            ("kama", window, pow1, pow2),
            lambda: KAMAIndicator(close=self.candles["Close"], window=window, pow1=pow1, pow2=pow2).kama(),
        )

    def stoch(self, window: int) -> tuple[pd.Series, pd.Series]:
        key = ("stoch", window)
        if key not in self._cache:
            indicator = StochasticOscillator(
                high=self.candles["High"],
                low=self.candles["Low"],
                close=self.candles["Close"],
                window=window,
                smooth_window=3,
            )
            self._cache[key] = (indicator.stoch(), indicator.stoch_signal())
        return self._cache[key]

    def tsi(self) -> pd.Series:
        return self._get(
            ("tsi",), lambda: TSIIndicator(close=self.candles["Close"], window_slow=25, window_fast=13).tsi()
        )


def psar_column(candles: pd.DataFrame, step: float = PSAR_STEP, max_step: float = PSAR_MAX_STEP) -> np.ndarray:
    indicator = PSARIndicator(
        high=candles["High"], low=candles["Low"], close=candles["Close"], step=step, max_step=max_step
    )
    return indicator.psar().to_numpy()


def indicator_frame(cache: _ColumnCache, params: Indicators) -> pd.DataFrame:
    """La stessa tabella di `add_technical_indicator`, con il PSAR gia' calcolato.

    L'ordine dei passaggi e le formule seguono quella funzione riga per riga, comprese le prime
    `atr_window` bande azzerate.
    """
    df = cache.candles.copy()
    df["PSAR"] = cache.psar
    df["PSARVP"] = df["PSAR"] / df["Close"]
    df["RSI"] = cache.rsi(params.rsi_window)
    df["RSI2"] = cache.rsi(params.rsi_window2)
    df["RSI3"] = cache.rsi(params.rsi_window3)
    df["ATR"] = cache.atr(params.atr_window)
    df["EMA20"] = cache.ema("Close", params.ema_window)
    df["EMA50"] = cache.ema("Close", params.ema_window2)
    df["EMA100"] = cache.ema("Close", params.ema_window3)
    df["KAMA"] = cache.kama(params.ema_window, params.kama_pow1, params.kama_pow2)
    df["Upper_Band"] = df["KAMA"] + params.atr_multiplier * df["ATR"]
    df["Lower_Band"] = df["KAMA"] - params.atr_multiplier * df["ATR"]
    df.iloc[: params.atr_window, df.columns.get_loc("Upper_Band")] = None
    df.iloc[: params.atr_window, df.columns.get_loc("Lower_Band")] = None
    stoch, stoch_signal = cache.stoch(params.rsi_window)
    df["STOCH"] = stoch
    df["STOCH_S"] = stoch_signal
    df["TSI"] = cache.tsi()
    return df


# ---------------------------------------------------------------------------------------------
# Metriche
# ---------------------------------------------------------------------------------------------


def _bar_equity(candles: pd.DataFrame, operations: list[dict], wallet: float) -> np.ndarray:
    """Valore del conto su ogni barra: contante quando si e' fuori, quantita' per prezzo dentro."""
    closes = candles["Close"].to_numpy()
    equity = np.full(len(candles), wallet, dtype=float)
    cash = wallet
    cursor = 0
    for operation in operations:
        entry = int(candles.index.searchsorted(operation["Buy_Time"]))
        exit_ = int(candles.index.searchsorted(operation["Sell_Time"]))
        equity[cursor:entry] = cash
        equity[entry:exit_] = operation["Quantity"] * closes[entry:exit_]
        cash = operation["Wallet_After"]
        cursor = exit_
    equity[cursor:] = cash
    return equity


def _drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((1 - equity / peak).max() * 100)


def _annualised(equity: np.ndarray, index: pd.DatetimeIndex) -> tuple[float, float, float]:
    """CAGR, volatilita' annualizzata e Sharpe (tasso privo di rischio zero) su rendimenti giornalieri."""
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100 if equity[-1] > 0 and years > 0 else float("nan")
    daily = pd.Series(equity, index=index).resample("1D").last().dropna().pct_change().dropna()
    if len(daily) < 2 or daily.std() == 0:
        return cagr, float("nan"), float("nan")
    volatility = float(daily.std() * np.sqrt(365) * 100)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(365))
    return cagr, volatility, sharpe


def evaluate(
    candles: pd.DataFrame,
    operations: list[dict],
    wallet: float = WALLET,
    fee_percent: float = FEE_PERCENT,
) -> dict:
    """Metriche di una singola configurazione, piu' il confronto con il possesso passivo."""
    closes = candles["Close"].to_numpy()
    buy_and_hold = (closes[-1] / closes[0] - 1) * 100
    equity = _bar_equity(candles, operations, wallet)
    cagr, volatility, sharpe = _annualised(equity, candles.index)
    result = {
        "n_trade": len(operations),
        "rendimento_%": (equity[-1] / wallet - 1) * 100,
        "buy_hold_%": buy_and_hold,
        "cagr_%": cagr,
        "volatilita_%": volatility,
        "sharpe": sharpe,
        "max_drawdown_%": _drawdown(equity),
        "buy_hold_drawdown_%": _drawdown(closes / closes[0] * wallet),
        "win_rate_%": float("nan"),
        "profit_factor": float("nan"),
        "trade_medio_%": float("nan"),
        "trade_mediano_%": float("nan"),
        "esposizione_%": 0.0,
        "durata_media_h": float("nan"),
        "trade_per_anno": 0.0,
        "commissioni_%": 0.0,
    }
    if not operations:
        return result

    frame = pd.DataFrame(operations)
    invested = frame["Quantity"] * frame["Buy_Price"] * (1 + fee_percent / 100)
    returns = frame["Profit"] / invested * 100
    wins = returns > 0
    gains = frame.loc[wins, "Profit"].sum()
    losses = -frame.loc[~wins, "Profit"].sum()
    holding = (frame["Sell_Time"] - frame["Buy_Time"]).dt.total_seconds()
    span = (candles.index[-1] - candles.index[0]).total_seconds()
    # Due gambe per operazione, ognuna con la sua commissione sul controvalore scambiato.
    fees = (frame["Quantity"] * frame["Buy_Price"] + frame["Quantity"] * frame["Sell_Price"]) * fee_percent / 100

    result.update(
        {
            "win_rate_%": float(wins.mean() * 100),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "trade_medio_%": float(returns.mean()),
            "trade_mediano_%": float(returns.median()),
            "esposizione_%": float(holding.sum() / span * 100),
            "durata_media_h": float(holding.mean() / 3600),
            "trade_per_anno": float(len(frame) / (span / (365.25 * 24 * 3600))),
            # Le commissioni si accumulano su un capitale che cresce: rapportarle al capitale
            # iniziale e' l'unico modo per confrontarle con il rendimento.
            "commissioni_%": float(fees.sum() / wallet * 100),
        }
    )
    return result


def by_year(operations: list[dict], wallet: float = WALLET, fee_percent: float = FEE_PERCENT) -> list[dict]:
    """Rendimento composto anno per anno, calcolato sui soli trade aperti in quell'anno."""
    if not operations:
        return []
    frame = pd.DataFrame(operations)
    invested = frame["Quantity"] * frame["Buy_Price"] * (1 + fee_percent / 100)
    frame = frame.assign(anno=frame["Buy_Time"].dt.year, ritorno=frame["Profit"] / invested)
    rows = []
    for anno, group in frame.groupby("anno"):
        rows.append(
            {
                "anno": int(anno),
                "n_trade": len(group),
                "rendimento_%": float(((1 + group["ritorno"]).prod() - 1) * 100),
                "win_rate_%": float((group["ritorno"] > 0).mean() * 100),
            }
        )
    return rows


# ---------------------------------------------------------------------------------------------
# Strategie e griglie
# ---------------------------------------------------------------------------------------------


def _run_strategy(name: str, df: pd.DataFrame, params: dict) -> tuple[list, list]:
    """Chiama la funzione di `strategies.py` che il dispatch di `trading_analysis` associa al nome."""
    if name == "Close Buy/Sell Limits":
        return strat.buy_sell_limits_close_simulation(
            df=df,
            rsi_buy_limit=params["rsi_buy_limit"],
            rsi_sell_limit=params["rsi_sell_limit"],
            num_cond=params["num_cond"],
            stop_loss_percent=params.get("stop_loss", 99.0),
        )
    if name == "Close ATR":
        return strat.close_atr_buy_sell_simulation(df=df, stop_loss_percent=params.get("stop_loss", 99.0))
    if name == "ATR Bands":
        return strat.atr_buy_sell_simulation(df=df, stop_loss_percent=params.get("stop_loss", 99.0))
    if name == "Close Bullish EMA":
        return strat.close_bullish_ema_simulation(
            df=df, rsi_buy_limit=params["rsi_buy_limit"], rsi_sell_limit=params["rsi_sell_limit"]
        )
    if name == "Close EMA Crossover":
        return strat.close_ema_crossover_simulation(df=df)
    if name == "Close RSI Reverse":
        return strat.close_rsi_buy_sell_limits_simulation(df=df)
    if name == "Supertrend":
        return strat.supertrend_simulation(df=df)
    if name == "Trend Zones":
        return strat.trend_zone_simulation(df=df)
    if name == "TP/SL with ATR":
        return strat.tp_sl_simulation(df=df)
    if name == "Green Candles":
        return strat.green_candles_simulation(df=df)
    if name == "ATR Live Trade":
        # La versione con cache di Streamlit rifiuta i DataFrame non hashabili fuori dalla pagina.
        return strat.simulate_candles.__wrapped__(
            raw_df=df,
            atr_window=params["atr_window"],
            atr_multiplier=params["atr_multiplier"],
            step=PSAR_STEP,
            max_step=PSAR_MAX_STEP,
            stop_loss_percent=params.get("stop_loss", 99.0),
        )
    raise KeyError(name)


def _product(**axes) -> list[dict]:
    """Prodotto cartesiano di assi nominati, in dizionari."""
    keys = list(axes)
    return [dict(zip(keys, values)) for values in itertools.product(*(axes[key] for key in keys))]


# Griglie: ogni voce e' (nome della strategia, assi degli indicatori, assi della strategia).
# Gli assi degli indicatori elencano solo cio' che quella strategia legge davvero -- muovere una
# finestra che non entra in nessuna delle sue condizioni moltiplica il tempo di calcolo e produce
# righe identiche.
BAND_AXES = {"atr_window": [5, 7, 10, 14, 20, 30], "atr_multiplier": [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0]}
EMA_TRIPLETS = [
    (10, 50, 200),  # il default della pagina
    (9, 21, 55),
    (12, 26, 50),
    (20, 50, 100),
    (5, 20, 60),
    (50, 100, 200),
    (8, 13, 21),
]

GRIDS: dict[str, dict] = {
    "close_buy_sell_limits": {
        "strategy": "Close Buy/Sell Limits",
        "indicators": {
            "rsi_window": [7, 12, 24],
            "atr_window": [5, 14],
            "atr_multiplier": [1.6, 2.5],
            "ema_window": [10, 50],
        },
        "params": {
            "rsi_buy_limit": [15, 20, 25, 30, 35, 40],
            "rsi_sell_limit": [60, 65, 70, 75, 80, 85],
            "num_cond": [1, 2],
        },
    },
    "close_atr": {
        "strategy": "Close ATR",
        "indicators": {**BAND_AXES, "ema_window": [10, 20, 50]},
        "params": {"stop_loss": [99.0, 10.0, 5.0, 2.0]},
    },
    "atr_bands": {
        "strategy": "ATR Bands",
        "indicators": {**BAND_AXES, "ema_window": [10, 50]},
        "params": {"stop_loss": [99.0, 5.0]},
    },
    "supertrend": {
        "strategy": "Supertrend",
        "indicators": {**BAND_AXES, "ema_window": [10, 20, 50]},
        "params": {},
    },
    "tp_sl_atr": {
        "strategy": "TP/SL with ATR",
        "indicators": {**BAND_AXES, "ema_window": [10, 20, 50]},
        "params": {},
    },
    "close_bullish_ema": {
        "strategy": "Close Bullish EMA",
        "indicators": {"ema_triplet": EMA_TRIPLETS, "rsi_window": [7, 12, 24]},
        "params": {"rsi_buy_limit": [40, 45, 50, 55], "rsi_sell_limit": [60, 65, 70, 75, 80]},
    },
    "close_ema_crossover": {
        "strategy": "Close EMA Crossover",
        "indicators": {"ema_triplet": EMA_TRIPLETS},
        "params": {},
    },
    "trend_zones": {
        "strategy": "Trend Zones",
        "indicators": {"ema_window": [5, 10, 20, 50, 100, 200]},
        "params": {},
    },
    "close_rsi_reverse": {
        "strategy": "Close RSI Reverse",
        "indicators": {"rsi_window": [5, 7, 12, 14, 21], "rsi_window2": [14, 21, 24, 30, 50]},
        "params": {},
    },
    "green_candles": {
        "strategy": "Green Candles",
        "indicators": {},
        "params": {},
    },
    "atr_live_trade": {
        "strategy": "ATR Live Trade",
        "indicators": {"atr_window": [5, 10, 14], "atr_multiplier": [1.2, 1.6, 2.5], "ema_window": [10]},
        "params": {"stop_loss": [99.0, 5.0]},
    },
}


def _cells(grid: dict) -> list[tuple[Indicators, dict]]:
    """Espande una griglia in coppie (parametri degli indicatori, parametri della strategia)."""
    axes = dict(grid["indicators"])
    triplets = axes.pop("ema_triplet", None)
    combos = _product(**axes) if axes else [{}]
    if triplets is not None:
        combos = [
            {**combo, "ema_window": short, "ema_window2": medium, "ema_window3": long}
            for combo in combos
            for short, medium, long in triplets
        ]
    strategy_combos = _product(**grid["params"]) if grid["params"] else [{}]
    return [(Indicators(**combo), params) for combo in combos for params in strategy_combos]


# ---------------------------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------------------------

# Le candele e il PSAR si calcolano una volta nel processo padre: i worker li ereditano con il
# fork, senza rileggere il parquet ne' ripagare i 26 secondi del PSAR per ognuno.
_CANDLES: dict[tuple[str, str], pd.DataFrame] = {}
_PSAR: dict[tuple[str, str], np.ndarray] = {}


def load_interval(interval: str, since: str = SINCE, until: str | None = None, symbol: str = SYMBOL) -> pd.DataFrame:
    candles = load_klines(symbol, interval)
    candles = candles[candles.index >= since]
    if until:
        candles = candles[candles.index < until]
    return candles


def prepare(interval: str, since: str = SINCE, until: str | None = None, symbol: str = SYMBOL) -> None:
    if (symbol, interval) in _CANDLES:
        return
    candles = load_interval(interval, since, until, symbol)
    if candles.empty:
        raise SystemExit(f"nessuna candela per {symbol} {interval}: lo store e' vuoto o il periodo e' fuori copertura")
    _CANDLES[(symbol, interval)] = candles
    _PSAR[(symbol, interval)] = psar_column(candles)


def _run_group(job: tuple[str, str, str, Indicators, list[dict], float]) -> tuple[list[dict], list[dict]]:
    """Una tabella di indicatori e tutte le configurazioni di strategia che ci girano sopra."""
    symbol, interval, strategy, indicators, param_list, fee, since, until = job
    # macOS avvia i worker con `spawn`, non `fork`: i globali riempiti dal padre non arrivano.
    prepare(interval, since, until, symbol)
    candles = _CANDLES[(symbol, interval)]
    cache = _ColumnCache(candles, _PSAR[(symbol, interval)])
    df = indicator_frame(cache, indicators)

    rows, yearly = [], []
    for params in param_list:
        started = time.time()
        buy_signals, sell_signals = _run_strategy(strategy, df, {**asdict(indicators), **params})
        operations = simulate_trading_with_commisions(
            buy_signals=buy_signals, sell_signals=sell_signals, wallet=WALLET, fee_percent=fee
        )
        row = {
            "simbolo": symbol,
            "intervallo": interval,
            "strategia": strategy,
            "fee_%": fee,
            **asdict(indicators),
            **params,
            **evaluate(candles, operations, WALLET, fee),
            "secondi": round(time.time() - started, 2),
        }
        rows.append(row)
        key = {key: row[key] for key in ("simbolo", "intervallo", "strategia", *asdict(indicators), *params)}
        yearly.extend({**key, **year} for year in by_year(operations, WALLET, fee))
    return rows, yearly


def run_cells(
    strategy: str,
    cells: list[tuple[Indicators, dict]],
    interval: str = "15m",
    fee: float = FEE_PERCENT,
    workers: int = 4,
    since: str = SINCE,
    until: str | None = None,
    label: str = "",
    symbol: str = SYMBOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue un elenco qualunque di configurazioni: e' quello che usano sia le griglie sia le
    analisi mirate (stessa configurazione a commissioni diverse, su intervalli diversi, su
    sotto-periodi diversi)."""
    prepare(interval, since, until, symbol)
    # Le configurazioni che condividono gli stessi indicatori vanno nello stesso lavoro: la
    # tabella si costruisce una volta sola per gruppo.
    grouped: dict[Indicators, list[dict]] = {}
    for indicators, params in cells:
        grouped.setdefault(indicators, []).append(params)
    jobs = [
        (symbol, interval, strategy, indicators, params, fee, since, until) for indicators, params in grouped.items()
    ]

    started = time.time()
    rows, yearly = [], []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for group_rows, group_yearly in pool.map(_run_group, jobs):
                rows.extend(group_rows)
                yearly.extend(group_yearly)
    else:
        for job in jobs:
            group_rows, group_yearly = _run_group(job)
            rows.extend(group_rows)
            yearly.extend(group_yearly)
    print(
        f"{label or strategy} [{symbol} {interval}]: {len(rows)} configurazioni, {len(jobs)} tabelle di indicatori, "
        f"{(time.time() - started) / 60:.1f} minuti",
        flush=True,
    )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def run_grid(
    name: str,
    interval: str = "15m",
    fee: float = FEE_PERCENT,
    workers: int = 4,
    since: str = SINCE,
    until: str | None = None,
    symbol: str = SYMBOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = GRIDS[name]
    return run_cells(grid["strategy"], _cells(grid), interval, fee, workers, since, until, name, symbol)


def save(name: str, interval: str, results: pd.DataFrame, yearly: pd.DataFrame, suffix: str = "") -> None:
    """Il simbolo entra nel nome solo quando non e' quello di riferimento, per non spostare i file
    gia' scritti quando si aggiunge un mercato di controllo."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{interval}{suffix}"
    results.to_parquet(OUTPUT_DIR / f"{stem}.parquet")
    if not yearly.empty:
        yearly.to_parquet(OUTPUT_DIR / f"{stem}_annuale.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="append", help="nome della griglia (ripetibile)")
    parser.add_argument("--all", action="store_true", help="tutte le griglie")
    parser.add_argument("--list", action="store_true", help="elenca le griglie e la loro dimensione")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--fee", type=float, default=FEE_PERCENT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--until", default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    if args.list:
        for name, grid in GRIDS.items():
            print(f"{name:24s} {grid['strategy']:24s} {len(_cells(grid)):5d} configurazioni")
        return

    names = list(GRIDS) if args.all else (args.grid or [])
    if not names:
        parser.error("serve --grid, --all o --list")
    for name in names:
        results, yearly = run_grid(name, args.interval, args.fee, args.workers, args.since, args.until, args.symbol)
        suffix = args.suffix if args.symbol == SYMBOL else f"_{args.symbol}{args.suffix}"
        save(name, args.interval, results, yearly, suffix)


if __name__ == "__main__":
    main()
