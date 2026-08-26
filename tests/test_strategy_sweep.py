"""Verifica che lo sweep misuri la stessa cosa che misura la pagina.

Due proprieta' vanno tenute ferme, perche' senza di esse i numeri prodotti da
`scripts/strategy_sweep.py` non parlerebbero delle strategie del simulatore ma di qualcos'altro:

1. `indicator_frame` produce la stessa tabella di `indicators.add_technical_indicator`. Lo sweep
   ricalcola gli indicatori per proprio conto, saltando il PSAR e memorizzando le colonne, e
   basta una formula fuori posto perche' ogni misura scivoli senza che nulla si rompa.
2. Le metriche derivano dalle operazioni di `pnl.simulate_trading_with_commisions`: il
   rendimento e' quello del capitale finale, e su un caso costruito a mano si sa quanto deve
   valere.
3. A parita' di parametri, lo sweep produce **le stesse operazioni** che produrrebbe
   `trading_analysis`, cioe' la pagina. Le prime due proprieta' valgono sui pezzi; questa lega il
   risultato al percorso intero, dispatch delle strategie compreso.

Le candele sono sintetiche: il test non deve dipendere dallo store, che nella CI non esiste.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import config
from cryptofarm.trading.indicators import add_technical_indicator
from cryptofarm.trading.pnl import simulate_trading_with_commisions
from cryptofarm.trading.simulator import trading_analysis
from scripts.strategy_sweep import (
    GRIDS,
    PSAR_MAX_STEP,
    PSAR_STEP,
    Indicators,
    _ColumnCache,
    _run_strategy,
    evaluate,
    indicator_frame,
    psar_column,
)


@pytest.fixture(scope="module")
def candles() -> pd.DataFrame:
    """Cammino casuale con volatilita' plausibile, abbastanza lungo per le finestre lunghe."""
    generator = np.random.default_rng(20260824)
    steps = generator.normal(0, 0.004, 2_000)
    close = 30_000 * np.exp(np.cumsum(steps))
    spread = np.abs(generator.normal(0, 0.003, 2_000)) * close
    frame = pd.DataFrame(
        {
            "Open": np.concatenate([[close[0]], close[:-1]]),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": generator.uniform(1, 100, 2_000),
        },
        index=pd.date_range("2024-01-01", periods=2_000, freq="15min", name="Open time"),
    )
    frame["High"] = frame[["High", "Open", "Close"]].max(axis=1)
    frame["Low"] = frame[["Low", "Open", "Close"]].min(axis=1)
    return frame


@pytest.mark.parametrize(
    "params",
    [
        Indicators(),
        Indicators(rsi_window=7, ema_window=20, ema_window2=60, ema_window3=120, atr_window=14, atr_multiplier=2.5),
        Indicators(
            rsi_window=24,
            rsi_window2=36,
            rsi_window3=48,
            ema_window=50,
            atr_window=30,
            atr_multiplier=0.8,
            kama_pow1=3,
            kama_pow2=20,
        ),
    ],
)
def test_indicator_frame_matches_production(candles: pd.DataFrame, params: Indicators) -> None:
    cache = _ColumnCache(candles, psar_column(candles))
    fast = indicator_frame(cache, params)
    reference = add_technical_indicator.__wrapped__(
        candles,
        step=0.01,
        max_step=0.4,
        rsi_window=params.rsi_window,
        rsi_window2=params.rsi_window2,
        rsi_window3=params.rsi_window3,
        ema_window=params.ema_window,
        ema_window2=params.ema_window2,
        ema_window3=params.ema_window3,
        atr_window=params.atr_window,
        atr_multiplier=params.atr_multiplier,
        kama_pow1=params.kama_pow1,
        kama_pow2=params.kama_pow2,
    )
    assert list(fast.columns) == list(reference.columns)
    pd.testing.assert_frame_equal(fast, reference)


def test_column_cache_reuses_columns(candles: pd.DataFrame) -> None:
    """La memoria e' l'unica ragione per cui lo sweep e' eseguibile: se smette di funzionare va visto."""
    cache = _ColumnCache(candles, psar_column(candles))
    indicator_frame(cache, Indicators())
    indicator_frame(cache, Indicators(atr_multiplier=3.0))  # cambia solo la banda, non le colonne
    assert cache.rsi(12) is cache.rsi(12)
    assert len([key for key in cache._cache if key[0] == "rsi"]) == 3


def test_evaluate_counts_the_final_wallet(candles: pd.DataFrame) -> None:
    """Un trade in guadagno e uno in perdita, con numeri verificabili a mano."""
    operations = [
        {
            "Buy_Time": candles.index[10],
            "Buy_Price": 100.0,
            "Sell_Time": candles.index[20],
            "Sell_Price": 110.0,
            "Quantity": 1.0,
            "Profit": 9.79,
            "Wallet_After": 110.0,
        },
        {
            "Buy_Time": candles.index[30],
            "Buy_Price": 110.0,
            "Sell_Time": candles.index[40],
            "Sell_Price": 99.0,
            "Quantity": 1.0,
            "Profit": -11.2,
            "Wallet_After": 99.0,
        },
    ]
    metrics = evaluate(candles, operations, wallet=100.0, fee_percent=0.1)
    assert metrics["n_trade"] == 2
    assert metrics["rendimento_%"] == pytest.approx(-1.0)
    assert metrics["win_rate_%"] == pytest.approx(50.0)
    # Il drawdown si misura anche mentre la posizione e' aperta, non solo sui trade chiusi.
    assert metrics["max_drawdown_%"] > 0
    assert 0 < metrics["esposizione_%"] < 100


# I casi non sono elencati a mano: sono l'intersezione fra il menu della pagina e le griglie dello
# sweep, calcolata all'import. Elencandoli, potare il menu lasciava qui dentro nomi che la pagina
# non sa piu' eseguire, e il test falliva con "0 operazioni" invece di dire che la voce non c'e'
# piu'. Cosi' la copertura segue il menu da sola, in entrambe le direzioni.
STRATEGIE_DELLO_SWEEP = {grid["strategy"] for grid in GRIDS.values()}
PARAMETRI_EXTRA = {
    # L'unica voce rimasta con un parametro proprio che valga la pena muovere: lo stop inerte al
    # 99% e lo stop stretto sono due percorsi diversi dentro la stessa funzione.
    "ATR Bands": [{"stop_loss": 99.0}, {"stop_loss": 5.0}],
}
CASI_DI_PARITA = [
    (voce, params)
    for voce in config.STRATEGIES
    if voce in STRATEGIE_DELLO_SWEEP
    for params in PARAMETRI_EXTRA.get(voce, [{}])
]


def test_ci_sono_casi_di_parita_da_verificare() -> None:
    """Se il menu e le griglie smettessero di intersecarsi, il test sopra passerebbe a vuoto."""
    assert len(CASI_DI_PARITA) >= 5, CASI_DI_PARITA


@pytest.mark.parametrize("strategia, params", CASI_DI_PARITA)
def test_sweep_matches_the_page(candles: pd.DataFrame, strategia: str, params: dict) -> None:
    """Le stesse operazioni che `trading_analysis` scriverebbe nella tabella della pagina."""
    indicators = Indicators()
    cache = _ColumnCache(candles, psar_column(candles))
    buy_signals, sell_signals = _run_strategy(
        strategia, indicator_frame(cache, indicators), {**indicators.__dict__, **params}
    )
    nostre = pd.DataFrame(simulate_trading_with_commisions(buy_signals, sell_signals, wallet=100.0, fee_percent=0.1))

    _, trades, _ = trading_analysis(
        asset="TEST",
        interval="15m",
        wallet=100.0,
        # La pagina prende i parametri in un dizionario con i nomi delle costanti di `config`:
        # la barra laterale mostra solo quelli della strategia scelta, e chi non ha widget resta
        # al valore iniziale. Qui si passano espliciti quelli che lo sweep sta usando.
        valori={
            "STOP_LOSS_PERCENT": params.get("stop_loss", 99.0),
            "RSI_BUY_LIMIT": params.get("rsi_buy_limit", 25),
            "RSI_SELL_LIMIT": params.get("rsi_sell_limit", 75),
            "NUM_CONDITIONS": params.get("num_cond", 1),
            "ATR_WINDOW": indicators.atr_window,
            "ATR_MULTIPLIER": indicators.atr_multiplier,
            "RSI_SHORT": indicators.rsi_window,
            "RSI_MEDIUM": indicators.rsi_window2,
            "RSI_LONG": indicators.rsi_window3,
            "EMA_SHORT": indicators.ema_window,
            "EMA_MEDIUM": indicators.ema_window2,
            "EMA_LONG": indicators.ema_window3,
            "KAMA_POW1": indicators.kama_pow1,
            "KAMA_POW2": indicators.kama_pow2,
        },
        strategia=strategia,
        fee_percent=0.1,
        show=False,
        market_data=candles,
    )

    colonne = ["Buy_Time", "Buy_Price", "Sell_Time", "Sell_Price", "Quantity", "Profit", "Wallet_After"]
    assert len(nostre) == len(trades)
    if not nostre.empty:
        pd.testing.assert_frame_equal(nostre[colonne], trades[colonne])


def test_evaluate_without_trades(candles: pd.DataFrame) -> None:
    metrics = evaluate(candles, [], wallet=100.0)
    assert metrics["n_trade"] == 0
    assert metrics["rendimento_%"] == pytest.approx(0.0)
    assert metrics["esposizione_%"] == 0.0


def test_la_pagina_e_lo_sweep_usano_lo_stesso_psar() -> None:
    """Il confronto qui sopra regge solo se i due calcolano il PSAR con gli stessi passi.

    La pagina li prende da `config`, lo sweep li tiene suoi: sono due costanti in due file, e se
    una si sposta il test end-to-end fallirebbe senza dire perche'.
    """
    assert (config.PSAR_STEP, config.PSAR_MAX_STEP) == (PSAR_STEP, PSAR_MAX_STEP)
