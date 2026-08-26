"""Banco di prova delle strategie a due versi: griglie, intervalli, costi, asset.

Stessa impalcatura di `scripts/strategy_sweep.py` -- una riga per configurazione, una riga per
configurazione e anno -- applicata alle strategie di `trading/strategies_ls.py`, che possono stare
lunghe, corte o fuori. Tre differenze che contano:

- **il P&L e' `pnl.simulate_positions`**, non `simulate_trading_with_commisions`: conosce il verso,
  gestisce l'inversione diretta e addebita un costo di mantenimento giornaliero (il funding di un
  perpetuo, 0,03% al giorno per default) che sulle posizioni tenute per settimane non e' un
  dettaglio.
- **la commissione di riferimento e' 0,05% per gamba**, non 0,1%: una strategia che va anche corta
  si esegue sui futures perpetui, dove il listino taker di Binance sta a 0,045% e il maker sotto.
  La sensibilita' al costo resta comunque una delle viste.
- **ogni strategia si misura due volte**, con e senza il verso corto (`allow_short`), lasciando
  tutto il resto identico: e' l'unico modo di attribuire al verso corto la differenza, invece di
  confrontare strategie diverse.

    python -m scripts.strategy_lab --list
    python -m scripts.strategy_lab --all --interval 4h --since 2021-01-01
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import strategies_ls as ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions
from scripts.strategy_sweep import _annualised, _drawdown, load_interval

SYMBOL = "BTCUSD"
SINCE = "2021-01-01"
WALLET = 100.0
FEE_PERCENT = 0.05
CARRY_DAILY_PERCENT = 0.03
OUTPUT_DIR = PROJECT_ROOT / "analysis_cache" / "lab"


# ---------------------------------------------------------------------------------------------
# Metriche, con il verso
# ---------------------------------------------------------------------------------------------


def bar_equity(candles: pd.DataFrame, operations: list[dict], wallet: float = WALLET) -> np.ndarray:
    """Valore del conto barra per barra, con le posizioni corte segnate a mercato al contrario.

    `strategy_sweep._bar_equity` moltiplica quantita' per prezzo, che per una posizione corta
    darebbe una curva che sale quando si sta perdendo.
    """
    closes = candles["Close"].to_numpy()
    equity = np.full(len(candles), wallet, dtype=float)
    cash = wallet
    cursor = 0
    for operation in operations:
        entry = int(candles.index.searchsorted(operation["Buy_Time"]))
        exit_ = int(candles.index.searchsorted(operation["Sell_Time"]))
        equity[cursor:entry] = cash
        direction = 1.0 if operation["Side"] == "long" else -1.0
        moves = direction * (closes[entry:exit_] / operation["Buy_Price"] - 1.0)
        equity[entry:exit_] = np.maximum(0.0, cash * (1.0 + moves))
        cash = operation["Wallet_After"]
        cursor = exit_
    equity[cursor:] = cash
    return equity


def evaluate(
    candles: pd.DataFrame,
    operations: list[dict],
    wallet: float = WALLET,
    fee_percent: float = FEE_PERCENT,
) -> dict:
    closes = candles["Close"].to_numpy()
    equity = bar_equity(candles, operations, wallet)
    cagr, volatility, sharpe = _annualised(equity, candles.index)
    result = {
        "n_trade": len(operations),
        "n_long": 0,
        "n_short": 0,
        "rendimento_%": (equity[-1] / wallet - 1) * 100,
        "buy_hold_%": (closes[-1] / closes[0] - 1) * 100,
        "cagr_%": cagr,
        "volatilita_%": volatility,
        "sharpe": sharpe,
        "max_drawdown_%": _drawdown(equity),
        "buy_hold_drawdown_%": _drawdown(closes / closes[0] * wallet),
        "win_rate_%": float("nan"),
        "win_rate_long_%": float("nan"),
        "win_rate_short_%": float("nan"),
        "rendimento_long_%": float("nan"),
        "rendimento_short_%": float("nan"),
        "profit_factor": float("nan"),
        "trade_medio_%": float("nan"),
        "esposizione_%": 0.0,
        "durata_media_h": float("nan"),
        "trade_per_anno": 0.0,
    }
    if not operations:
        return result

    frame = pd.DataFrame(operations)
    invested = frame["Quantity"] * frame["Buy_Price"]
    returns = frame["Profit"] / invested
    wins = returns > 0
    gains = frame.loc[wins, "Profit"].sum()
    losses = -frame.loc[~wins, "Profit"].sum()
    holding = (frame["Sell_Time"] - frame["Buy_Time"]).dt.total_seconds()
    span = (candles.index[-1] - candles.index[0]).total_seconds()
    long_mask = frame["Side"] == "long"

    result.update(
        {
            "n_long": int(long_mask.sum()),
            "n_short": int((~long_mask).sum()),
            "win_rate_%": float(wins.mean() * 100),
            "win_rate_long_%": float(wins[long_mask].mean() * 100) if long_mask.any() else float("nan"),
            "win_rate_short_%": float(wins[~long_mask].mean() * 100) if (~long_mask).any() else float("nan"),
            # Somma dei rendimenti per verso: non e' un composto, e' il contributo di ciascun lato.
            "rendimento_long_%": float(returns[long_mask].sum() * 100) if long_mask.any() else float("nan"),
            "rendimento_short_%": float(returns[~long_mask].sum() * 100) if (~long_mask).any() else float("nan"),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
            "trade_medio_%": float(returns.mean() * 100),
            "esposizione_%": float(holding.sum() / span * 100),
            "durata_media_h": float(holding.mean() / 3600),
            "trade_per_anno": float(len(frame) / (span / (365.25 * 24 * 3600))),
        }
    )
    return result


def by_year(operations: list[dict]) -> list[dict]:
    if not operations:
        return []
    frame = pd.DataFrame(operations)
    invested = frame["Quantity"] * frame["Buy_Price"]
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
# Griglie
# ---------------------------------------------------------------------------------------------

GRIDS: dict[str, dict] = {
    "donchian_breakout": {
        "params": {
            "channel": [20, 55, 100, 150],
            "adx_min": [0.0, 20.0, 25.0, 30.0],
            "atr_multiplier": [2.0, 3.0, 4.0, 6.0],
            "regime_ema": [0, 200],
            "allow_short": [True, False],
        }
    },
    "squeeze_breakout": {
        "params": {
            "bb_dev": [1.5, 2.0, 2.5],
            "kc_multiplier": [1.0, 1.5, 2.0],
            "atr_multiplier": [2.0, 3.0, 4.0],
            "confirm_volume": [True, False],
            "allow_short": [True, False],
        }
    },
    "trend_pullback": {
        "params": {
            "regime_ema": [0, 50, 100, 200],
            "oversold": [0.1, 0.2, 0.3],
            "atr_multiplier": [1.5, 2.0, 3.0],
            "allow_short": [True, False],
        }
    },
    "ichimoku_trend": {
        "params": {
            "fast": [7, 9, 20],
            "require_cloud": [True, False],
            "allow_short": [True, False],
        }
    },
    "band_reversion_gated": {
        "params": {
            "band_multiplier": [1.6, 2.5, 3.5],
            "adx_max": [15.0, 20.0, 25.0, 100.0],
            "stop_multiplier": [1.5, 2.5, 4.0],
            "regime_ema": [0, 200],
            "allow_short": [True, False],
        }
    },
}

# `oversold` e `overbought` sono simmetrici: muoverli separatamente raddoppierebbe la griglia per
# misurare due volte la stessa cosa.
_SYMMETRIC = {"oversold": lambda value: {"overbought": round(1.0 - value, 2)}}
# Le finestre di Ichimoku si muovono in proporzione, come nell'originale (9/26/52).
_ICHIMOKU_SCALE = {7: (22, 44), 9: (26, 52), 20: (60, 120)}


def cells(name: str) -> list[dict]:
    axes = GRIDS[name]["params"]
    keys = list(axes)
    rows = []
    for values in itertools.product(*(axes[key] for key in keys)):
        params = dict(zip(keys, values))
        for key, derive in _SYMMETRIC.items():
            if key in params:
                params.update(derive(params[key]))
        if name == "ichimoku_trend":
            params["slow"], params["span"] = _ICHIMOKU_SCALE[params["fast"]]
        rows.append(params)
    return rows


# ---------------------------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------------------------

_CANDLES: dict[tuple[str, str], pd.DataFrame] = {}
_CACHES: dict[tuple[str, str], ExtraCache] = {}


def prepare(symbol: str, interval: str, since: str, until: str | None) -> pd.DataFrame:
    key = (symbol, interval)
    if key not in _CANDLES:
        candles = load_interval(interval, since, until, symbol)
        if candles.empty:
            raise SystemExit(f"nessuna candela per {symbol} {interval}")
        _CANDLES[key] = candles
    return _CANDLES[key]


def _cache(symbol: str, interval: str) -> ExtraCache:
    """Una memoria per processo: gli indicatori si ricalcolano una volta per finestra, non per cella."""
    key = (symbol, interval)
    if key not in _CACHES:
        _CACHES[key] = ExtraCache(_CANDLES[key])
    return _CACHES[key]


def _run_batch(job: tuple) -> tuple[list[dict], list[dict]]:
    name, symbol, interval, batch, fee, carry, since, until = job
    # macOS avvia i worker con `spawn`, non `fork`: i globali riempiti dal padre non arrivano.
    # `prepare` e' idempotente, quindi sotto fork questa riga non costa nulla.
    prepare(symbol, interval, since, until)
    candles = _CANDLES[(symbol, interval)]
    cache = _cache(symbol, interval)
    strategy = ls.STRATEGIES[name]

    rows, yearly = [], []
    for params in batch:
        started = time.time()
        events = strategy(candles, cache, **params)
        operations = simulate_positions(events, WALLET, fee, carry)
        row = {
            "simbolo": symbol,
            "intervallo": interval,
            "strategia": name,
            "fee_%": fee,
            "carry_%": carry,
            **params,
            **evaluate(candles, operations, WALLET, fee),
            "secondi": round(time.time() - started, 2),
        }
        rows.append(row)
        key = {column: row[column] for column in ("simbolo", "intervallo", "strategia", *params)}
        yearly.extend({**key, **year} for year in by_year(operations))
    return rows, yearly


def run_grid(
    name: str,
    symbol: str = SYMBOL,
    interval: str = "4h",
    since: str = SINCE,
    until: str | None = None,
    fee: float = FEE_PERCENT,
    carry: float = CARRY_DAILY_PERCENT,
    workers: int = 4,
    param_list: list[dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare(symbol, interval, since, until)
    todo = param_list if param_list is not None else cells(name)
    size = max(1, len(todo) // (workers * 2) + 1)
    batches = [todo[start : start + size] for start in range(0, len(todo), size)]
    jobs = [(name, symbol, interval, batch, fee, carry, since, until) for batch in batches]

    started = time.time()
    rows, yearly = [], []
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for batch_rows, batch_yearly in pool.map(_run_batch, jobs):
                rows.extend(batch_rows)
                yearly.extend(batch_yearly)
    else:
        for job in jobs:
            batch_rows, batch_yearly = _run_batch(job)
            rows.extend(batch_rows)
            yearly.extend(batch_yearly)
    print(
        f"{name} [{symbol} {interval}]: {len(rows)} configurazioni in {(time.time() - started) / 60:.1f} minuti",
        flush=True,
    )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def save(name: str, symbol: str, interval: str, results: pd.DataFrame, yearly: pd.DataFrame, suffix: str = "") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{symbol}_{interval}{suffix}"
    results.to_parquet(OUTPUT_DIR / f"{stem}.parquet")
    if not yearly.empty:
        yearly.to_parquet(OUTPUT_DIR / f"{stem}_annuale.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="append")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--until", default=None)
    parser.add_argument("--fee", type=float, default=FEE_PERCENT)
    parser.add_argument("--carry", type=float, default=CARRY_DAILY_PERCENT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    if args.list:
        for name in GRIDS:
            print(f"{name:22s} {len(cells(name)):4d} configurazioni")
        return

    names = list(GRIDS) if args.all else (args.grid or [])
    if not names:
        parser.error("serve --grid, --all o --list")
    for name in names:
        results, yearly = run_grid(
            name, args.symbol, args.interval, args.since, args.until, args.fee, args.carry, args.workers
        )
        save(name, args.symbol, args.interval, results, yearly, args.suffix)


if __name__ == "__main__":
    main()
