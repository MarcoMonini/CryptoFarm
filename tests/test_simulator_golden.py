"""Golden master di `trading/simulator.py`: fissa il comportamento prima di riorganizzarlo.

Il simulatore non aveva alcun test. Queste funzioni vanno spostate in moduli separati e i loro
cicli riscritti per farli girare piu' in fretta, e l'unico modo di dimostrare che il risultato non
cambia e' registrare l'output attuale su dati sintetici deterministici e confrontarlo a ogni passo.

Lo snapshot sta in `tests/data/simulator_golden.json` e si rigenera con:

    SIMULATOR_GOLDEN_REGEN=1 pytest tests/test_simulator_golden.py

Rigenerarlo *accetta* qualunque differenza di comportamento: farlo solo dopo aver verificato a mano
che la differenza sia voluta.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import indicators, market_data, pnl, strategies

# I moduli in cui `simulator.py` e' stato spezzato. `simulator.py` resta la pagina Streamlit e
# non espone piu' nulla da fissare qui.
MODULES = (market_data, indicators, strategies, pnl)

SNAPSHOT = Path(__file__).parent / "data" / "simulator_golden.json"
REGEN = os.environ.get("SIMULATOR_GOLDEN_REGEN") == "1"


def _ohlcv(rows: int, seed: int, drifts: tuple[float, ...], vol: float) -> pd.DataFrame:
    """Random walk riproducibile con la forma OHLCV che il simulatore si aspetta.

    `drifts` sono i regimi che si susseguono in parti uguali: servono a far scattare le strategie
    che pretendono una sequenza precisa di incroci, non del rumore.
    """
    rng = np.random.default_rng(seed)
    drift = np.repeat(np.array(drifts), rows // len(drifts))
    drift = np.r_[drift, np.full(rows - len(drift), drifts[-1])]
    close = 100 * np.exp(np.cumsum(drift + rng.normal(0, vol, rows)))
    spread = np.abs(rng.normal(0, vol * 0.8, rows)) * close
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) + spread,
            "Low": np.minimum(open_, close) - spread,
            "Close": close,
            "Volume": rng.uniform(10, 1000, rows),
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="15min", name="Open time"),
    )


# Nessuno scenario da solo copre tutte le strategie: `close_ema_crossover_simulation` pretende tre
# incroci EMA nell'ordine giusto e scatta solo quando il trend si inverte davvero, mentre
# `close_bullish_ema_simulation` scatta solo in laterale. Servono entrambi.
SCENARIOS = {
    "regimi": dict(rows=2000, seed=101, drifts=(-0.002, 0.002, -0.002, 0.002), vol=0.0025),
    "laterale": dict(rows=600, seed=31, drifts=(0.0,), vol=0.010),
    "calmo": dict(rows=600, seed=47, drifts=(0.0002,), vol=0.002),
    # Regimi che si ribaltano in fretta: producono incroci EMA fuori sequenza, che e' l'unico
    # modo di distinguere la macchina a stati di `close_ema_crossover_simulation` da un singolo
    # incrocio non condizionato.
    "sbandate": dict(
        rows=1600, seed=77, drifts=(0.004, -0.004, 0.004, -0.004, 0.004, -0.004, 0.004, -0.004), vol=0.003
    ),
}


def scenario_frame(name: str) -> pd.DataFrame:
    return _ohlcv(**SCENARIOS[name])


def _signals(pair) -> dict:
    """Coppia (buy, sell) di liste di (timestamp, prezzo) in forma confrontabile."""
    buy, sell = pair
    return {
        "buy": [[str(t), round(float(p), 8)] for t, p in buy],
        "sell": [[str(t), round(float(p), 8)] for t, p in sell],
    }


def _frame(df: pd.DataFrame) -> dict:
    """Impronta di un DataFrame: colonne, e per ognuna somma e conteggio dei NaN."""
    out = {"columns": list(df.columns), "rows": len(df)}
    for column in df.columns:
        values = pd.to_numeric(df[column], errors="coerce")
        out[column] = [round(float(np.nansum(values.to_numpy())), 6), int(values.isna().sum())]
    return out


def _capture(call) -> dict:
    """Esegue `call` registrando anche l'eccezione: alcune strategie oggi sollevano KeyError.

    `buy_sell_limits_simulation` legge `MACD` e solleva sempre; `atr_buy_sell_simulation` e
    `close_atr_buy_sell_simulation` leggono `PSAR` dietro un corto circuito e sollevano solo in
    alcuni scenari, in altri restituiscono segnali. Nessuna delle due colonne viene piu' prodotta
    da `add_technical_indicator`. Sono rotte *prima* di questa riorganizzazione, e lo snapshot
    registra per ogni scenario quale dei due esiti si verifica, cosi' il refactoring resta fedele
    invece di nascondere il difetto.
    """
    try:
        return {"value": call()}
    except Exception as error:  # noqa: BLE001 - registrare il tipo e' esattamente lo scopo
        return {"raises": f"{type(error).__name__}: {error}"}


STRATEGIES = {
    "buy_sell_limits_simulation": lambda f: strategies.buy_sell_limits_simulation(f, 0.0, 0.0, 30, 70, 2),
    "buy_sell_limits_close_simulation": strategies.buy_sell_limits_close_simulation,
    "close_rsi_buy_sell_limits_simulation": strategies.close_rsi_buy_sell_limits_simulation,
    "atr_buy_sell_simulation": lambda f: strategies.atr_buy_sell_simulation(f, 2.0),
    "close_atr_buy_sell_simulation": lambda f: strategies.close_atr_buy_sell_simulation(f, 2.0),
    "close_ema_crossover_simulation": strategies.close_ema_crossover_simulation,
    "close_bullish_ema_simulation": strategies.close_bullish_ema_simulation,
    "tp_sl_simulation": strategies.tp_sl_simulation,
    "green_candles_simulation": strategies.green_candles_simulation,
    "supertrend_simulation": strategies.supertrend_simulation,
    "trend_zone_simulation": strategies.trend_zone_simulation,
}

KEYS = [
    "interval_to_minutes",
    "calculate_latest_indicators",
    "simulate_candles",
    "simulate_trading_with_commisions",
    "simulate_trading_with_commisions_multiple_buy",
    "simulate_positions",
    *(
        f"{scenario}/{name}"
        for scenario in SCENARIOS
        for name in (
            "add_technical_indicator",
            "get_green_red_percentage",
            "identify_trend_zones",
            "bullish_condition",
            "bearish_condition",
            *STRATEGIES,
        )
    ),
]


def build_snapshot() -> dict:
    snapshot: dict = {
        "interval_to_minutes": {i: market_data.interval_to_minutes(i) for i in ("1m", "5m", "15m", "1h", "4h", "1d")},
    }

    for scenario in SCENARIOS:
        raw = scenario_frame(scenario)
        table = indicators.add_technical_indicator(raw)
        snapshot[f"{scenario}/add_technical_indicator"] = _frame(table)
        snapshot[f"{scenario}/get_green_red_percentage"] = round(float(strategies.get_green_red_percentage(table)), 8)
        snapshot[f"{scenario}/identify_trend_zones"] = _capture(lambda f=table: len(strategies.identify_trend_zones(f)))
        probes = [i for i in (210, 250, 300, 350, 500) if i < len(table)]
        snapshot[f"{scenario}/bullish_condition"] = [bool(strategies.bullish_condition(table, i)) for i in probes]
        snapshot[f"{scenario}/bearish_condition"] = [bool(strategies.bearish_condition(table, i)) for i in probes]
        # `latest_bands` e' il nucleo numpy su cui gira `simulate_candles`: va fissato anche lui.
        closes = table["Close"].to_numpy()
        snapshot[f"{scenario}/latest_bands"] = [
            [None if b is None else round(float(b), 8) for b in latest]
            for latest in (
                indicators.latest_bands(
                    table["High"].to_numpy()[:end], table["Low"].to_numpy()[:end], closes[:end], window, 2.0
                )
                for window in (3, 6, 14)
                for end in (2, 5, 20, len(closes))
            )
        ]
        for name, call in STRATEGIES.items():
            snapshot[f"{scenario}/{name}"] = _capture(lambda c=call, f=table: _signals(c(f)))

    # `simulate_candles` ricalcola gli indicatori a ogni candela: e' troppo lento per girare su
    # tutti gli scenari, quindi resta su una finestra corta.
    short = scenario_frame("laterale").iloc[:300]
    snapshot["calculate_latest_indicators"] = _frame(indicators.calculate_latest_indicators(short, 120))
    snapshot["simulate_candles"] = _capture(lambda: _signals(strategies.simulate_candles(short)))

    # P&L sui segnali di una strategia che produce parecchie operazioni in entrambi i versi.
    # Entrambe restituiscono la lista delle operazioni: vanno registrati i valori, non la lunghezza.
    buy, sell = strategies.green_candles_simulation(indicators.add_technical_indicator(scenario_frame("regimi")))
    for name in ("simulate_trading_with_commisions", "simulate_trading_with_commisions_multiple_buy"):
        operations = getattr(pnl, name)(buy, sell, wallet=100, fee_percent=0.1)
        snapshot[name] = [
            {k: (round(float(v), 8) if isinstance(v, (int, float)) else str(v)) for k, v in operation.items()}
            for operation in operations
        ]

    # Il motore a due versi: gli stessi segnali riscritti come cambi di posizione, piu'
    # un'inversione diretta da lungo a corto, che nel formato a due liste non e' esprimibile.
    events = [(time, price, 1) for time, price in buy[:20]]
    events += [(time, price, 0) for time, price in sell[:20]]
    events.sort(key=lambda event: event[0])
    if len(events) > 4:
        events[3] = (events[3][0], events[3][1], -1)
    snapshot["simulate_positions"] = [
        {k: (round(float(v), 8) if isinstance(v, (int, float)) else str(v)) for k, v in operation.items()}
        for operation in pnl.simulate_positions(events, wallet=100, fee_percent=0.05)
    ]

    return snapshot


@pytest.fixture(scope="module")
def golden() -> dict:
    if REGEN or not SNAPSHOT.exists():
        SNAPSHOT.write_text(json.dumps(build_snapshot(), indent=2, sort_keys=True) + "\n")
    return json.loads(SNAPSHOT.read_text())


@pytest.fixture(scope="module")
def current() -> dict:
    return json.loads(json.dumps(build_snapshot(), sort_keys=True))


def test_snapshot_covers_every_public_function(golden):
    """Lo snapshot non deve perdere pezzi ora che il simulatore e' spezzato in piu' moduli."""
    public = {
        name
        for module in MODULES
        for name in dir(module)
        if not name.startswith("_")
        and callable(getattr(module, name))
        and getattr(getattr(module, name), "__module__", None) == module.__name__
    }
    covered = {key.split("/")[-1] for key in golden}
    # Escluse: I/O di rete (`get_market_data*`, `download_market_data`) e `ai_model_simulation`,
    # che richiede un modello addestrato.
    uncovered = (
        public
        - covered
        - {
            "get_market_data",
            "get_market_data_between_dates",
            "download_market_data",
            "ai_model_simulation",
        }
    )
    assert not uncovered, f"funzioni non coperte dal golden master: {sorted(uncovered)}"


@pytest.mark.parametrize("key", KEYS)
def test_behaviour_unchanged(golden, current, key):
    assert current[key] == golden[key]
