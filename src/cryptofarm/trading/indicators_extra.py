"""Indicatori che le strategie storiche non usano, e che servono a quelle nuove.

`indicators.add_technical_indicator` calcola sempre le stesse quindici colonne, tutte costruite
attorno a tre famiglie: medie mobili, ATR e RSI. Le strategie misurate in
`.claude/docs/backtest-strategie.md` falliscono quasi tutte per la stessa ragione -- entrano
contro il trend e operano troppo -- e nessuna delle colonne disponibili permette di dire *se* il
mercato ha un trend, *quanto* e' compresso, o *se* il volume conferma il movimento.

Qui stanno le colonne che rispondono a quelle tre domande, tutte da `ta`:

- **ADX** (`ta.trend.ADXIndicator`) misura la forza del trend indipendentemente dal verso: e' il
  filtro che manca a "Close ATR", che compra sui minimi anche quando il minimo e' l'inizio di una
  discesa.
- **Canale di Donchian** (massimo e minimo delle ultime N barre) e' la rottura vera, quella che
  fa entrare *nella* direzione del movimento invece che contro.
- **Bande di Bollinger e canale di Keltner** insieme danno lo *squeeze*: quando le bande stanno
  dentro il canale la volatilita' e' compressa, ed e' l'unica condizione che seleziona pochi
  momenti all'anno invece di centinaia.
- **StochRSI** e' l'oscillatore per i rientri dentro un trend gia' stabilito.
- **OBV e MFI** (volume) confermano o smentiscono una rottura.
- **Ichimoku** e' un sistema completo di trend con uscita incorporata, ed e' il termine di
  paragone naturale per le altre tre.

**Il ritardo e' esplicito.** Il canale di Donchian di `ta` include la barra corrente, quindi
`close > hband` puo' accadere solo se la chiusura coincide con il massimo: qui il canale e'
spostato di una barra (`shift(1)`), che e' cio' che vede chi opera. Ichimoku usa `visual=True`,
che sposta le due span in avanti di `window2`: la nuvola sopra la barra di oggi e' quella
calcolata ventisei barre fa, come sul grafico.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import KAMAIndicator, StochRSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, IchimokuIndicator
from ta.volatility import AverageTrueRange, BollingerBands, KeltnerChannel
from ta.volume import MFIIndicator, OnBalanceVolumeIndicator


@dataclass(frozen=True)
class ExtraParams:
    """Le finestre degli indicatori nuovi. I default sono quelli classici della letteratura."""

    donchian_window: int = 20
    adx_window: int = 14
    atr_window: int = 14
    regime_ema: int = 200
    bb_window: int = 20
    bb_dev: float = 2.0
    kc_window: int = 20
    kc_atr_window: int = 10
    kc_multiplier: float = 1.5
    stochrsi_window: int = 14
    stochrsi_smooth: int = 3
    mfi_window: int = 14
    ichimoku_fast: int = 9
    ichimoku_slow: int = 26
    ichimoku_span: int = 52


class ExtraCache:
    """Le colonne calcolate una volta per finestra, come nello sweep delle strategie storiche.

    Le griglie muovono un parametro alla volta: senza memoria, l'ADX a 14 verrebbe ricalcolato per
    ogni ampiezza di canale che non lo tocca.
    """

    def __init__(self, candles: pd.DataFrame):
        self.candles = candles
        self._cache: dict[tuple, object] = {}

    def _get(self, key: tuple, build):
        if key not in self._cache:
            self._cache[key] = build()
        return self._cache[key]

    # --- trend -------------------------------------------------------------------------------
    def adx(self, window: int) -> np.ndarray:
        return self._get(
            ("adx", window),
            lambda: ADXIndicator(
                high=self.candles["High"], low=self.candles["Low"], close=self.candles["Close"], window=window
            )
            .adx()
            .to_numpy(),
        )

    def ema(self, window: int) -> np.ndarray:
        return self._get(
            ("ema", window),
            lambda: EMAIndicator(close=self.candles["Close"], window=window).ema_indicator().to_numpy(),
        )

    def atr(self, window: int) -> np.ndarray:
        return self._get(
            ("atr", window),
            lambda: AverageTrueRange(
                high=self.candles["High"], low=self.candles["Low"], close=self.candles["Close"], window=window
            )
            .average_true_range()
            .to_numpy(),
        )

    def kama(self, window: int, pow1: int = 2, pow2: int = 30) -> np.ndarray:
        """La media adattiva su cui poggiano le bande delle strategie storiche, per poterle
        riusare nella versione con filtro di regime."""
        return self._get(
            ("kama", window, pow1, pow2),
            lambda: KAMAIndicator(close=self.candles["Close"], window=window, pow1=pow1, pow2=pow2).kama().to_numpy(),
        )

    # --- canali ------------------------------------------------------------------------------
    def donchian(self, window: int) -> tuple[np.ndarray, np.ndarray]:
        """Massimo e minimo delle `window` barre **precedenti**: il canale che si vede all'apertura."""

        def build():
            high = self.candles["High"].rolling(window).max().shift(1)
            low = self.candles["Low"].rolling(window).min().shift(1)
            return high.to_numpy(), low.to_numpy()

        return self._get(("donchian", window), build)

    def bollinger(self, window: int, dev: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        def build():
            bands = BollingerBands(close=self.candles["Close"], window=window, window_dev=dev)
            return (
                bands.bollinger_hband().to_numpy(),
                bands.bollinger_lband().to_numpy(),
                bands.bollinger_mavg().to_numpy(),
            )

        return self._get(("bb", window, dev), build)

    def keltner(self, window: int, atr_window: int, multiplier: float) -> tuple[np.ndarray, np.ndarray]:
        def build():
            channel = KeltnerChannel(
                high=self.candles["High"],
                low=self.candles["Low"],
                close=self.candles["Close"],
                window=window,
                window_atr=atr_window,
                multiplier=multiplier,
                original_version=False,
            )
            return channel.keltner_channel_hband().to_numpy(), channel.keltner_channel_lband().to_numpy()

        return self._get(("kc", window, atr_window, multiplier), build)

    # --- oscillatori e volume ----------------------------------------------------------------
    def stochrsi(self, window: int, smooth: int) -> np.ndarray:
        return self._get(
            ("stochrsi", window, smooth),
            lambda: StochRSIIndicator(close=self.candles["Close"], window=window, smooth1=smooth, smooth2=smooth)
            .stochrsi_k()
            .to_numpy(),
        )

    def mfi(self, window: int) -> np.ndarray:
        return self._get(
            ("mfi", window),
            lambda: MFIIndicator(
                high=self.candles["High"],
                low=self.candles["Low"],
                close=self.candles["Close"],
                volume=self.candles["Volume"],
                window=window,
            )
            .money_flow_index()
            .to_numpy(),
        )

    def obv_slope(self, window: int) -> np.ndarray:
        """Variazione dell'On Balance Volume sulle ultime `window` barre, normalizzata.

        Il livello dell'OBV non dice nulla (dipende da quanto volume e' passato da quando esiste
        la serie); la sua *pendenza* dice se il volume sta accompagnando il movimento."""

        def build():
            obv = OnBalanceVolumeIndicator(
                close=self.candles["Close"], volume=self.candles["Volume"]
            ).on_balance_volume()
            change = obv.diff(window)
            scale = self.candles["Volume"].rolling(window).sum().replace(0, np.nan)
            return (change / scale).to_numpy()

        return self._get(("obv", window), build)

    # --- ichimoku ----------------------------------------------------------------------------
    def ichimoku(self, fast: int, slow: int, span: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tenkan, Kijun e le due span **gia' spostate in avanti** (`visual=True`)."""

        def build():
            indicator = IchimokuIndicator(
                high=self.candles["High"],
                low=self.candles["Low"],
                window1=fast,
                window2=slow,
                window3=span,
                visual=True,
            )
            high, low = self.candles["High"], self.candles["Low"]
            tenkan = ((high.rolling(fast).max() + low.rolling(fast).min()) / 2).to_numpy()
            kijun = ((high.rolling(slow).max() + low.rolling(slow).min()) / 2).to_numpy()
            return tenkan, kijun, indicator.ichimoku_a().to_numpy(), indicator.ichimoku_b().to_numpy()

        return self._get(("ichimoku", fast, slow, span), build)
