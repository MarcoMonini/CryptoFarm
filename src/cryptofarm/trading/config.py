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

# Percorso del CSV storico, specifico della macchina: si passa da variabile d'ambiente.
CSV_FILE = os.environ.get("MARKET_DATA_CSV", "")
SHOW_GRAPHS = True
