"""Valori di partenza misurati, per intervallo. **Generato da `scripts/tune_defaults.py`.**

Non si modifica a mano: si rigenera con `python -m scripts.tune_defaults --all-intervals --save`
dopo aver rifatto le griglie. Ogni numero qui e' il valore la cui **mediana dei ranghi** su cinque
asset e' la piu' alta, fra quelli provati dalla griglia -- non il valore della configurazione che ha
reso di piu', che su questi dati e' l'errore misurato (ρ = −0,69 fra resa in stima e in verifica).

Compaiono solo i parametri che superano **due** controlli: spostano la mediana dei ranghi di almeno
0.06 (altrimenti il default esistente resta, perche' cambiarlo sarebbe inseguire rumore), e
scelgono lo stesso valore anche guardando il solo 2021-2023. Chi non compare tiene il valore di
`config.py`.

Generato il 2026-08-26, dal 2021-01-01, commissione 0,05% per gamba, solo lunghe, su:
BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT.
"""

from __future__ import annotations

# {intervallo: {voce di menu: {costante di config: valore}}}
PER_INTERVALLO: dict[str, dict[str, dict[str, float | int | bool]]] = {
    "15m": {
        "ATR Bands": {"ATR_MULTIPLIER": 3, "ATR_WINDOW": 5, "EMA_SHORT": 10, "STOP_LOSS_PERCENT": 99},
        "Close RSI Reverse": {"RSI_MEDIUM": 50, "RSI_SHORT": 21},
        "Donchian Breakout": {"ADX_MIN": 30, "DONCHIAN_ATR_MULT": 6, "DONCHIAN_CHANNEL": 150},
        "Ichimoku Trend": {"ICHIMOKU_FAST": 20, "ICHIMOKU_SLOW": 60, "ICHIMOKU_SPAN": 120, "REQUIRE_CLOUD": 1},
        "Squeeze Breakout": {"BB_DEV": 2.5, "CONFIRM_VOLUME": 1, "KC_MULTIPLIER": 1, "SQUEEZE_ATR_MULT": 4},
        "Supertrend": {"ATR_MULTIPLIER": 4, "ATR_WINDOW": 30, "EMA_SHORT": 10},
        "TP/SL with ATR": {"ATR_MULTIPLIER": 4, "ATR_WINDOW": 14, "EMA_SHORT": 50},
    },
    "1d": {
        "ATR Bands": {"ATR_MULTIPLIER": 3, "ATR_WINDOW": 5, "EMA_SHORT": 10},
        "Close EMA Crossover": {"EMA_LONG": 21, "EMA_MEDIUM": 13, "EMA_SHORT": 8},
        "Close RSI Reverse": {"RSI_MEDIUM": 21},
        "Donchian Breakout": {"ADX_MIN": 25, "DONCHIAN_ATR_MULT": 6, "DONCHIAN_CHANNEL": 20, "REGIME_EMA": 0},
        "Ichimoku Trend": {"REQUIRE_CLOUD": 0},
        "Squeeze Breakout": {"BB_DEV": 1.5, "KC_MULTIPLIER": 2, "SQUEEZE_ATR_MULT": 3},
        "Supertrend": {"ATR_MULTIPLIER": 0.8, "ATR_WINDOW": 14, "EMA_SHORT": 10},
        "TP/SL with ATR": {"ATR_WINDOW": 7, "EMA_SHORT": 20},
        "Trend Zones": {"EMA_SHORT": 10},
    },
    "1h": {
        "ATR Bands": {"ATR_MULTIPLIER": 1.6, "ATR_WINDOW": 5, "EMA_SHORT": 10, "STOP_LOSS_PERCENT": 99},
        "Close RSI Reverse": {"RSI_SHORT": 21},
        "Donchian Breakout": {"ADX_MIN": 0, "DONCHIAN_ATR_MULT": 6, "DONCHIAN_CHANNEL": 150, "REGIME_EMA": 200},
        "Ichimoku Trend": {"ICHIMOKU_FAST": 20, "ICHIMOKU_SLOW": 60, "ICHIMOKU_SPAN": 120, "REQUIRE_CLOUD": 1},
        "Squeeze Breakout": {"BB_DEV": 2, "KC_MULTIPLIER": 1, "SQUEEZE_ATR_MULT": 4},
        "Supertrend": {"EMA_SHORT": 50},
        "TP/SL with ATR": {"EMA_SHORT": 50},
        "Trend Zones": {"EMA_SHORT": 100},
    },
    "4h": {
        "ATR Bands": {"ATR_MULTIPLIER": 1.2, "ATR_WINDOW": 5, "EMA_SHORT": 10, "STOP_LOSS_PERCENT": 99},
        "Close EMA Crossover": {"EMA_LONG": 50, "EMA_MEDIUM": 26, "EMA_SHORT": 12},
        "Close RSI Reverse": {"RSI_SHORT": 21},
        "Donchian Breakout": {"DONCHIAN_ATR_MULT": 6},
        "Ichimoku Trend": {"REQUIRE_CLOUD": 1},
        "Squeeze Breakout": {"BB_DEV": 2, "SQUEEZE_ATR_MULT": 4},
        "Supertrend": {"ATR_MULTIPLIER": 2, "ATR_WINDOW": 30, "EMA_SHORT": 50},
        "TP/SL with ATR": {"ATR_MULTIPLIER": 2},
    },
}
