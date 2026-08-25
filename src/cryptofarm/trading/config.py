"""Valori di partenza del simulatore, fuori dal codice della pagina.

La pagina Streamlit portava una sessantina di numeri scritti in mezzo alle chiamate ai widget:
per cambiare il default di una finestra bisognava cercarla dentro il layout. Qui ognuno ha un
nome, e `simulator.py` si limita a disporli.

I valori sono quelli di prima, invariati. L'unica eccezione e' `CSV_FILE`, che conteneva un
percorso Windows di un'altra macchina: ora si legge da `MARKET_DATA_CSV` e altrimenti resta vuoto.
"""

from __future__ import annotations

import os
from typing import NamedTuple


class Param(NamedTuple):
    """Un campo numerico della barra laterale: valore iniziale e limiti.

    `widget` si espande direttamente in `st.number_input`. I tipi contano: interi e decimali
    scritti come tali qui producono lo stesso widget di prima.
    """

    value: float
    minimum: float
    maximum: float
    step: float

    @property
    def widget(self) -> dict:
        return {"value": self.value, "min_value": self.minimum, "max_value": self.maximum, "step": self.step}


# Mercato
ASSET = "BTC"
CURRENCY = "USDC"
INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
INTERVAL_INDEX = 3
TIME_HOURS = {"value": 240, "min_value": 0, "step": 24}
WALLET = {"value": 100, "min_value": 0, "step": 1}

STRATEGIES = [
    "-",
    "Close Buy/Sell Limits",
    "Close ATR",
    "ATR Bands",
    "Close Bullish EMA",
    "Close EMA Crossover",
    "Supertrend",
    "Trend Zones",
    "TP/SL with ATR",
    "Green Candles",
    "ATR Live Trade",
    "AI Model",
    "Donchian Breakout",
    "Squeeze Breakout",
    "Trend Pullback",
    "Ichimoku Trend",
    "Band Reversion",
]

# Indicatori
ATR_MULTIPLIER = Param(1.6, 0.1, 50.0, 0.1)
ATR_WINDOW = Param(5, 1, 100, 1)
RSI_SHORT = Param(12, 2, 500, 1)
RSI_MEDIUM = Param(24, 2, 500, 1)
RSI_LONG = Param(36, 2, 500, 1)
EMA_SHORT = Param(10, 1, 500, 1)
EMA_MEDIUM = Param(50, 1, 500, 1)
EMA_LONG = Param(200, 1, 500, 1)
KAMA_POW1 = Param(2, 1, 1000, 1)
KAMA_POW2 = Param(30, 1, 1000, 1)

# Strategia
RSI_BUY_LIMIT = Param(25, 0, 100, 1)
RSI_SELL_LIMIT = Param(75, 0, 100, 1)
STOP_LOSS_PERCENT = Param(99.0, 0.1, 100.0, 1.0)
NUM_CONDITIONS = Param(1, 1, 10, 1)
PIVOT_WINDOW = Param(100, 2, 500, 2)

# Le strategie nuove (`strategies_ls.py`), sempre solo lunghe nella pagina.
# I default sono quelli con cui sono state misurate in `.claude/docs/strategie-nuove.md`: dove due
# strategie usano lo stesso indicatore con un default diverso -- il moltiplicatore dell'ATR e' 3,0
# per la rottura di canale, 2,5 per la compressione, 2,0 per il rientro -- ognuna tiene il suo,
# invece di allinearle a un valore medio che non e' stato misurato per nessuna.
ADX_WINDOW = Param(14, 2, 100, 1)
ADX_MIN = Param(20.0, 0.0, 100.0, 1.0)
ADX_MAX = Param(20.0, 0.0, 100.0, 1.0)
REGIME_EMA = Param(200, 0, 500, 10)
TRAIL_ATR_WINDOW = Param(14, 2, 100, 1)

DONCHIAN_CHANNEL = Param(20, 5, 200, 1)
DONCHIAN_ATR_MULT = Param(3.0, 0.5, 10.0, 0.1)

BB_WINDOW = Param(20, 5, 200, 1)
BB_DEV = Param(2.0, 0.5, 5.0, 0.1)
KC_WINDOW = Param(20, 5, 200, 1)
KC_MULTIPLIER = Param(1.5, 0.5, 5.0, 0.1)
OBV_WINDOW = Param(20, 2, 200, 1)
SQUEEZE_ATR_MULT = Param(2.5, 0.5, 10.0, 0.1)

STOCHRSI_WINDOW = Param(14, 2, 100, 1)
STOCHRSI_SMOOTH = Param(3, 1, 20, 1)
STOCH_OVERSOLD = Param(0.2, 0.0, 1.0, 0.05)
STOCH_OVERBOUGHT = Param(0.8, 0.0, 1.0, 0.05)
PULLBACK_ATR_MULT = Param(2.0, 0.5, 10.0, 0.1)

ICHIMOKU_FAST = Param(9, 2, 100, 1)
ICHIMOKU_SLOW = Param(26, 2, 200, 1)
ICHIMOKU_SPAN = Param(52, 2, 400, 1)

REVERSION_KAMA_WINDOW = Param(10, 2, 200, 1)
REVERSION_BAND_MULT = Param(2.5, 0.5, 10.0, 0.1)
REVERSION_STOP_MULT = Param(2.0, 0.5, 10.0, 0.1)
# Il filtro di regime qui e' spento di default, come nella misura: la strategia lavora dentro un
# intervallo, e limitarla a un lato della media lunga le toglie meta' delle occasioni.
REVERSION_REGIME_EMA = Param(0, 0, 500, 10)

# Il PSAR non ha widget: questi erano i valori con cui la pagina lo calcolava, e restano tali.
# I default di `add_technical_indicator` sono diversi (0,02), quindi vanno passati per esteso.
PSAR_STEP = 0.01
PSAR_MAX_STEP = 0.4

# Interruttori delle strategie nuove.
CONFIRM_VOLUME = True
REQUIRE_CLOUD = True

# Percorso del CSV storico, specifico della macchina: si passa da variabile d'ambiente.
CSV_FILE = os.environ.get("MARKET_DATA_CSV", "")
SHOW_GRAPHS = True
