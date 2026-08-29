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

# Il nome sta in una costante perche' lo usano sia il menu sia il dispatch di
# `trading_analysis`, e perche' la pagina lo toglie dalle opzioni quando manca il modello.
AI_STRATEGY = "AI Model"

# La confluenza non e' una strategia come le altre: legge quattro piani temporali ricavati
# dall'intervallo scelto, e sotto una certa lunghezza di storia i piani lunghi non esistono. Il
# nome sta in una costante perche' lo usano il menu, il registro e la pagina, che la avvisa.
CONFLUENCE_STRATEGY = "Confluence"

# Il menu e' stato potato sulle misure di `.claude/docs/ricerca-quant-ml.md` §2: scelta della
# configurazione sul 2021-2023, resa sul 2024-2026, su BTC/ETH/SOL/XRP/BNB a 1d e 4h. Restano le
# voci che fuori campione hanno mediana positiva o almeno due celle su dieci sopra il possesso
# passivo. Le sette tolte -- Close Buy/Sell Limits, Close ATR, Close Bullish EMA, Green Candles,
# ATR Live Trade, Trend Pullback, Band Reversion -- restano in `strategies.py` e nel golden master:
# sono uscite dal menu, non dal repository, e la misura si rifa' con `scripts/strategy_sweep`.
STRATEGIES = [
    "-",
    "ATR Bands",
    "Trend Zones",
    "Close EMA Crossover",
    "Close RSI Reverse",
    "Supertrend",
    "TP/SL with ATR",
    AI_STRATEGY,
    "Ichimoku Trend",
    "Squeeze Breakout",
    "Donchian Breakout",
    CONFLUENCE_STRATEGY,
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

# La finestra dell'etichetta del modello a swing, in **barre per lato**. 144 e' quella con cui il
# modello e' addestrato, che a 5m sono dodici ore per lato: sulla pagina l'intervallo cambia, e con
# esso il tratto di calendario che la finestra copre. Non e' un parametro di strategia -- l'etichetta
# guarda avanti e non e' operabile -- ma il modo di guardare cosa il modello sta imparando.
SWING_TARGET_WINDOW = Param(144, 4, 2000, 4)

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

# Confluenza. I parametri dei sei votanti **non** sono qui di proposito: sono congelati ai valori
# misurati (`tuned_defaults`) e ritararli dentro l'insieme porterebbe il conto dei parametri liberi
# da nove a oltre venticinque, dove niente e' piu' distinguibile dalla fortuna.
CONF_THETA_BASE = Param(0.35, 0.0, 1.0, 0.05)
CONF_THETA_MACRO = Param(0.15, 0.0, 0.5, 0.05)
CONF_ISTERESI = Param(0.10, 0.0, 0.5, 0.01)
# Il pavimento e il soffitto dell'isteresi. La banda da sola sbagliava in tutte e due le direzioni:
# si apriva e si chiudeva in due barre, e il punteggio che decade piano teneva aperto per ore oltre
# il primo segnale di uscita. Nessuno dei due limite tocca lo stop o il cancello, che sono regole
# di rischio e non di opinione.
CONF_BARRE_MINIME = Param(4, 0, 100, 1)
CONF_PAZIENZA = Param(24, 1, 500, 1)
CONF_EMIVITA = Param(6.0, 0.5, 50.0, 0.5)
CONF_W_MAX = Param(0.30, 0.15, 1.0, 0.05)
CONF_K_FAMIGLIE = Param(2, 1, 6, 1)
CONF_INNESCO = Param(0, 0, 50, 1)
CONF_ATR_WINDOW = Param(14, 2, 100, 1)
CONF_ATR_MULT = Param(3.0, 0.5, 10.0, 0.1)
CONF_REGIME_EMA = Param(50, 5, 300, 5)
CONF_STRUTTURA_EMA = Param(50, 5, 300, 5)

# I parametri dei sei votanti della confluenza. Sono qui, e non riusano i nomi delle strategie
# omonime del menu, per una ragione precisa: un votante gira sul **suo** piano -- struttura o
# conferma -- non sull'intervallo scelto nella pagina, quindi il suo valore misurato e' quello di
# un altro intervallo. Riusare i nomi darebbe alla confluenza i default sbagliati in silenzio.
#
# I valori scritti qui sono i default delle funzioni in `strategies_ls`. Quelli **misurati** li
# sovrascrive `panels.valori_misurati` con i `tuned_defaults` dell'intervallo del piano, dove una
# misura c'e'; dove non c'e', restano questi.
CONF_ICHIMOKU_FAST = Param(9, 2, 100, 1)
CONF_ICHIMOKU_SLOW = Param(26, 2, 200, 1)
CONF_ICHIMOKU_SPAN = Param(52, 2, 400, 1)
CONF_ICHIMOKU_CLOUD = Param(1, 0, 1, 1)

CONF_DONCHIAN_CHANNEL = Param(20, 5, 300, 1)
CONF_DONCHIAN_ADX_WINDOW = Param(14, 2, 100, 1)
CONF_DONCHIAN_ADX_MIN = Param(20.0, 0.0, 100.0, 1.0)
CONF_DONCHIAN_ATR_WINDOW = Param(14, 2, 100, 1)
CONF_DONCHIAN_ATR_MULT = Param(3.0, 0.5, 12.0, 0.1)
CONF_DONCHIAN_REGIME_EMA = Param(200, 0, 500, 10)

CONF_FLOW_WINDOW = Param(20, 2, 200, 1)
CONF_FLOW_MFI_ALTO = Param(80.0, 50.0, 100.0, 1.0)
CONF_FLOW_MFI_BASSO = Param(20.0, 0.0, 50.0, 1.0)

CONF_SQUEEZE_BB_WINDOW = Param(20, 5, 200, 1)
CONF_SQUEEZE_BB_DEV = Param(2.0, 0.5, 5.0, 0.1)
CONF_SQUEEZE_KC_WINDOW = Param(20, 5, 200, 1)
CONF_SQUEEZE_KC_MULT = Param(1.5, 0.5, 5.0, 0.1)
CONF_SQUEEZE_ATR_WINDOW = Param(14, 2, 100, 1)
CONF_SQUEEZE_ATR_MULT = Param(2.5, 0.5, 12.0, 0.1)
CONF_SQUEEZE_VOLUME = Param(1, 0, 1, 1)
CONF_SQUEEZE_OBV_WINDOW = Param(20, 2, 200, 1)

CONF_PULLBACK_REGIME_EMA = Param(200, 0, 500, 10)
CONF_PULLBACK_STOCH_WINDOW = Param(14, 2, 100, 1)
CONF_PULLBACK_STOCH_SMOOTH = Param(3, 1, 30, 1)
CONF_PULLBACK_OVERSOLD = Param(0.2, 0.0, 0.5, 0.05)
CONF_PULLBACK_OVERBOUGHT = Param(0.8, 0.5, 1.0, 0.05)
CONF_PULLBACK_ATR_MULT = Param(2.0, 0.5, 12.0, 0.1)

CONF_REVERSION_KAMA = Param(10, 2, 100, 1)
CONF_REVERSION_BAND_MULT = Param(2.5, 0.5, 8.0, 0.1)
CONF_REVERSION_ADX_MAX = Param(20.0, 0.0, 100.0, 1.0)
CONF_REVERSION_STOP_MULT = Param(2.0, 0.5, 12.0, 0.1)

# Le bande ATR, senza il cancello di range che tiene zitto `reversione`. Registrate due volte su
# due piani con moltiplicatori diversi: la stessa domanda posta a due scale, che e' il motivo per
# cui la famiglia esiste come concetto separato dal votante.
CONF_BANDE_KAMA = Param(10, 2, 100, 1)
CONF_BANDE_BAND_MULT = Param(2.5, 0.5, 8.0, 0.1)
CONF_BANDE_STOP_MULT = Param(3.0, 0.5, 12.0, 0.1)
CONF_BANDE_KAMA_VELOCE = Param(6, 2, 100, 1)
CONF_BANDE_BAND_MULT_VELOCE = Param(1.8, 0.5, 8.0, 0.1)
CONF_BANDE_STOP_MULT_VELOCE = Param(2.5, 0.5, 12.0, 0.1)

# Le zone di trend: la macrostruttura come stato. Anche queste due volte, sul piano di regime e su
# quello di struttura, perche' «sopra o sotto» a un giorno e a quattro ore non sono la stessa cosa.
CONF_ZONE_FAST = Param(20, 2, 200, 1)
CONF_ZONE_SLOW = Param(100, 5, 400, 1)
CONF_ZONE_FAST_STRUTTURA = Param(12, 2, 200, 1)
CONF_ZONE_SLOW_STRUTTURA = Param(50, 5, 400, 1)

# Le due soglie del votante a modello, sulla stessa scala della previsione ([-1, 1] in valore
# assoluto). Sono i valori scelti sulla validazione in `ml/signals.SWING_ENTRA/SWING_ESCI`; qui
# sono manopole perche' §5.2 misura che la coppia buona cambia da una finestra all'altra, e
# tenerla nascosta in una costante farebbe credere che ne esista una giusta.
CONF_MODELLO_ENTRA = Param(0.35, 0.0, 1.0, 0.05)
CONF_MODELLO_ESCI = Param(0.25, 0.0, 1.0, 0.05)

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
CONF_IN_FORMAZIONE = True
CONFIRM_VOLUME = True
REQUIRE_CLOUD = True

# --- Rotazione trasversale ---------------------------------------------------------------------
# La seconda vista della pagina. I valori iniziali sono quelli **centrali** raccomandati da
# `.claude/docs/ricerca-quant-ml.md` §7, non l'ottimo di una griglia: la correlazione fra resa in
# stima e resa in verifica sulle prime dieci configurazioni e' -0,69, quindi cercare il massimo in
# campione e' peggio che prendere una configurazione qualunque.
ROTATION_MODES = ["Single asset", "Cross-asset rotation"]
ROTATION_UNIVERSES = ["majors", "wide"]
ROTATION_INTERVALS = ["4h", "1d"]
ROTATION_SINCE = "2021-01-01"
ROTATION_LOOKBACK = Param(20, 5, 200, 1)
ROTATION_TOP = Param(2, 1, 10, 1)
ROTATION_EVERY = Param(7, 1, 60, 1)
ROTATION_FEE = Param(0.1, 0.0, 1.0, 0.01)

# Percorso del CSV storico, specifico della macchina: si passa da variabile d'ambiente.
CSV_FILE = os.environ.get("MARKET_DATA_CSV", "")
SHOW_GRAPHS = True
