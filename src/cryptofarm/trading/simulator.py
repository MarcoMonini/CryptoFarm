import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

from cryptofarm.ml.trainer import (
    load_signal_model,
)
from cryptofarm.paths import MODELS_DIR
from cryptofarm.trading import config, panels
from cryptofarm.trading.indicators import add_technical_indicator
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.market_data import (
    get_market_data,
    get_market_data_between_dates,
)
from cryptofarm.trading.pnl import simulate_trading_with_commisions
from cryptofarm.trading.strategies import identify_trend_zones

# Disattiva i FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)

MODEL_PATH = str(MODELS_DIR / "signal_model.joblib")


def trading_analysis(
    asset: str,
    interval: str,
    wallet: float,
    valori: dict,
    strategia: str = panels.VUOTA,
    time_hours: int = 24,
    fee_percent: float = 0.1,
    show: bool = True,
    market_data=None,
):
    """Scarica le candele, calcola gli indicatori, esegue la strategia e costruisce il grafico.

    Restituisce `(figura, operazioni, ore effettive)`.

    `valori` sono i parametri letti dalla barra laterale, con i nomi delle costanti di `config`.
    Prima erano una trentina di argomenti nominali, uno per widget: aggiungere una strategia
    voleva dire allungare la firma, il punto di chiamata e la catena di `if` qui sotto. Ora la
    corrispondenza fra strategia, parametri e indicatori sta in `panels.py`, e questa funzione non
    sa piu' quali strategie esistono.

    Chi non compare in `valori` prende il valore iniziale: la barra laterale mostra solo i
    parametri della strategia scelta, quindi gli altri non hanno un widget da cui arrivare.
    """
    valori = {**panels.valori_predefiniti(), **valori}

    # ======================================
    # Scarica i dati di mercato e calcola il SAR
    if market_data is None:
        # Otteniamo i dati di mercato (funzione esterna da definire)
        # df = get_market_data(asset=asset, interval=interval, limit=limit)
        df, actual_hours = get_market_data(asset=asset, interval=interval, time_hours=time_hours)
    else:
        df = market_data
        actual_hours = time_hours

    # Aggiungiamo una colonna per i massimi e i minimi relativi
    # # Utilizziamo i prezzi massimi ('High') e minimi ('Low')
    price_high = df["High"]
    price_low = df["Low"]
    # Trova gli indici dei massimi e minimi relativi
    order = int(valori["PIVOT_WINDOW"] / 2)
    max_idx = argrelextrema(price_high.values, np.greater, order=order)[0]
    min_idx = argrelextrema(price_low.values, np.less, order=order)[0]
    # Inizializza gli array per massimi e minimi
    rel_max = []
    rel_min = []
    # Popola gli array con tuple (indice, prezzo)
    for i in min_idx:
        rel_min.append((df.index[i], df.loc[df.index[i], "Low"]))
    for i in max_idx:
        rel_max.append((df.index[i], df.loc[df.index[i], "High"]))

    # dinamic_atr = False
    # if strategia == "Dinamic ATR Bands" or strategia == "Dinamic Close ATR":
    #     dinamic_atr = True
    # df = calculate_relative_extrema(df)

    df = add_technical_indicator(
        df,
        step=config.PSAR_STEP,
        max_step=config.PSAR_MAX_STEP,
        rsi_window=int(valori["RSI_SHORT"]),
        rsi_window2=int(valori["RSI_MEDIUM"]),
        rsi_window3=int(valori["RSI_LONG"]),
        ema_window=int(valori["EMA_SHORT"]),
        ema_window2=int(valori["EMA_MEDIUM"]),
        ema_window3=int(valori["EMA_LONG"]),
        atr_window=int(valori["ATR_WINDOW"]),
        atr_multiplier=float(valori["ATR_MULTIPLIER"]),
        kama_pow1=int(valori["KAMA_POW1"]),
        kama_pow2=int(valori["KAMA_POW2"]),
    )

    # ======================================
    # La strategia, presa dal registro invece che da una catena di confronti su stringhe.
    # Era quella catena a far divergere il menu dal codice: `"Supetrend"` scritto male non
    # eseguiva niente e nessuno se ne accorgeva, perche' una stringa che non corrisponde a nulla
    # non e' un errore. Ora una voce senza riga nel registro non arriva nemmeno al menu, e un
    # test lo verifica.
    cache = ExtraCache(df)
    voce = panels.STRATEGIE.get(strategia)
    if voce is None:
        buy_signals, sell_signals = [], []
    else:
        buy_signals, sell_signals = voce.esegui(df, cache, valori)

    operations = simulate_trading_with_commisions(
        wallet=wallet, buy_signals=buy_signals, sell_signals=sell_signals, fee_percent=fee_percent
    )

    # ======================================
    # 4. Creazione del grafico
    rows = 2
    candlestick_height_px = 400
    indicators_height_px = candlestick_height_px / 2
    total_height = candlestick_height_px + ((rows - 1) * indicators_height_px)
    nominal_height = 1 / (rows + 1)
    candle_height = 2 * nominal_height
    row_heights = [candle_height] + [nominal_height] * (rows - 1)
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=(
            "Candlestick",
            "True and Relative Strength Index and Stochastic (TSI / RSI / STOCH)",
        ),
    )
    if show:
        trend_shapes = identify_trend_zones(df=df)
        index = 1
        # Candele (candlestick)
        fig.add_trace(
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name=f"{asset}"
            ),
            row=index,
            col=1,
        )

        # Punti SAR (marker rossi)
        # fig.add_trace(go.Scatter(
        #     x=df.index,
        #     y=df['PSAR'],
        #     mode='markers',
        #     marker=dict(size=2, color='yellow', symbol='circle'),
        #     name='PSAR'
        # ),
        #     row=index, col=1
        # )

        # EMA SHORT
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA20"], mode="lines", line=dict(color="Green", width=1), name="EMA SHORT"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA50"], mode="lines", line=dict(color="purple", width=1), name="EMA MED"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["EMA100"], mode="lines", line=dict(color="coral", width=1), name="EMA LONG"),
            row=index,
            col=1,
        )

        # KAMA
        fig.add_trace(
            go.Scatter(x=df.index, y=df["KAMA"], mode="lines", line=dict(color="yellow", width=1), name="KAMA"),
            row=index,
            col=1,
        )

        # Rolling ATR Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Upper_Band"],
                mode="lines",
                line=dict(color="yellow", width=1, dash="dash"),
                name="Upper ATR",
            ),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Lower_Band"],
                mode="lines",
                line=dict(color="yellow", width=1, dash="dash"),
                name="Lower ATR",
            ),
            row=index,
            col=1,
        )

        # Massimi relativi
        if rel_max:
            max_times, max_prices = zip(*rel_max)
            fig.add_trace(
                go.Scatter(
                    x=max_times,
                    y=max_prices,
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="square-open"),
                    name="Local Max",
                ),
                row=index,
                col=1,
            )
        # Minimi relativi
        if rel_min:
            min_times, min_prices = zip(*rel_min)
            fig.add_trace(
                go.Scatter(
                    x=min_times,
                    y=min_prices,
                    mode="markers",
                    marker=dict(size=10, color="green", symbol="square-open"),
                    name="Local Min",
                ),
                row=index,
                col=1,
            )

        # Segnali di acquisto
        if buy_signals:
            buy_times, buy_prices = zip(*buy_signals)
            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode="markers",
                    marker=dict(size=14, color="green", symbol="triangle-up"),
                    name="Buy Signal",
                ),
                row=index,
                col=1,
            )

        # Segnali di vendita
        if sell_signals:
            sell_times, sell_prices = zip(*sell_signals)
            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="triangle-down"),
                    name="Sell Signal",
                ),
                row=index,
                col=1,
            )

        # STOCASTICO
        index += 1
        fig.add_trace(
            go.Scatter(x=df.index, y=df["STOCH"], mode="lines", line=dict(color="darkblue", width=1), name="STOCH"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["STOCH_S"], mode="lines", line=dict(color="darkcyan", width=1), name="STOCH S"),
            row=index,
            col=1,
        )

        # TSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df["TSI"], mode="lines", line=dict(color="yellow", width=1), name="TSI"),
            row=index,
            col=1,
        )

        # RSI
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['RSI_S'], mode='lines',
        #     line=dict(color='orange', width=1),
        #     name='RSI SMOOTH'
        # ), row=index, col=1)
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], mode="lines", line=dict(color="salmon", width=1), name="RSI SHORT"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI2"], mode="lines", line=dict(color="pink", width=1), name="RSI MED"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI3"], mode="lines", line=dict(color="purple", width=1), name="RSI LONG"),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[valori["RSI_SELL_LIMIT"]] * 2,
                mode="lines",
                line=dict(color="red", width=1, dash="dash"),
                name="Sell Limit",
            ),
            row=index,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[df.index.min(), df.index.max()],
                y=[valori["RSI_BUY_LIMIT"]] * 2,
                mode="lines",
                line=dict(color="green", width=1, dash="dash"),
                name="Buy Limit",
            ),
            row=index,
            col=1,
        )

        # MACD
        # index += 1
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_L'], mode='lines',
        #     line=dict(color='fuchsia', width=1, dash='dot'),
        #     name='MACD Line'
        # ), row=index, col=1)
        # fig.add_trace(go.Scatter(
        #     x=df.index, y=df['MACD_S'], mode='lines',
        #     line=dict(color='blue', width=1, dash='dot'),
        #     name='MACD Signal'
        # ), row=index, col=1)
        # fig.add_trace(go.Bar(
        #     x=df.index, y=df['MACD'], name='MACD',
        #     marker=dict(color='lightyellow')
        # ), row=index, col=1
        # )
        #
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_buy_limit, macd_buy_limit], mode='lines',
        #     line=dict(color='green', width=1, dash='dash'),
        #     name='Buy Limit'
        # ), row=index, col=1
        # )
        # fig.add_trace(go.Scatter(
        #     x=[df.index.min(), df.index.max()], y=[macd_sell_limit, macd_sell_limit], mode='lines',
        #     line=dict(color='red', width=1, dash='dash'),
        #     name='Sell Limit'
        # ), row=index, col=1)

        fig.update_layout(
            template="plotly_dark", xaxis_rangeslider_visible=False, height=total_height, shapes=trend_shapes
        )

    # ======================================
    # Creazione del DataFrame finale con le operazioni
    if operations:
        trades_df = pd.DataFrame(operations)
        # Aggiungiamo qualche metrica sul periodo analizzato
        apertura = df["Open"].iloc[0]  # Prezzo di apertura (prima candela)
        chiusura = df["Close"].iloc[-1]  # Prezzo di chiusura (ultima candela)
        high_max = df["High"].max()
        low_min = df["Low"].min()
        # Variazione percentuale (close finale su open iniziale)
        variazione = (chiusura - apertura) / apertura * 100
        # Volatilità: std dei rendimenti "Close-to-Close", in termini %
        volatilita = df["Close"].pct_change().std() * 100
        # Inseriamo questi valori su ogni riga del DataFrame trades_df.
        trades_df["massimo"] = high_max
        trades_df["minimo"] = low_min
        trades_df["variazione(%)"] = variazione
        trades_df["volatilita(%)"] = volatilita
    else:
        # Nessun trade effettuato
        trades_df = pd.DataFrame(
            columns=["Buy_Time", "Buy_Price", "Sell_Time", "Sell_Price", "Quantity", "Profit", "Wallet_After"]
        )

    print(
        f"{wallet} USDC su {asset}, fee={fee_percent}%, {interval}, strategia: {strategia}, "
        f"profitto totale={round(trades_df['Profit'].sum())} USD"
    )

    return fig, trades_df, actual_hours


if __name__ == "__main__":
    st.set_page_config(
        page_title="CryptoFarm Simulator",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "model" not in st.session_state:
        # `load_signal_model` trova da solo il formato del modello addestrato, cosi' cambiare
        # famiglia di modello non richiede di toccare la pagina.
        st.session_state["model"] = load_signal_model()

    text_placeholder = st.empty()
    fig_placeholder = st.empty()

    # --- Mercato ------------------------------------------------------------------------------
    st.sidebar.header("Mercato")
    col1, col2 = st.sidebar.columns(2)
    asset = col1.text_input(label="Asset", placeholder="es. BTC, ETH, XRP...", max_chars=8, value=config.ASSET)
    currency = col2.text_input(label="Valuta", placeholder="es. USDC, USDT, EUR...", max_chars=8, value=config.CURRENCY)
    interval = col1.selectbox(label="Intervallo", options=config.INTERVALS, index=config.INTERVAL_INDEX)
    time_hours = col2.number_input(label="Ore di storico", **config.TIME_HOURS)
    symbol = asset + currency
    wallet = st.sidebar.number_input(label=f"Capitale ({currency})", **config.WALLET)

    if st.sidebar.button("SCARICA CANDELE", use_container_width=True, type="primary"):
        st.session_state["df"], _ = get_market_data(asset=symbol, interval=interval, time_hours=time_hours)

    # --- Strategia ----------------------------------------------------------------------------
    st.sidebar.header("Strategia")
    strategia = st.sidebar.selectbox(
        label="Strategia", options=config.STRATEGIES, index=0, label_visibility="collapsed"
    )
    voce = panels.STRATEGIE.get(strategia)
    if voce is not None and voce.note:
        st.sidebar.caption(voce.note)
    elif voce is None:
        st.sidebar.caption("Nessuna strategia: la pagina mostra tutti gli indicatori disponibili.")

    # --- Parametri, solo quelli che servono ---------------------------------------------------
    # I widget nascono da `panels.gruppi_di`: cambiando strategia cambiano i riquadri, e un
    # parametro che la strategia scelta non usa non compare. Chi non ha widget resta al suo valore
    # iniziale, che e' cio' che `trading_analysis` usa per gli indicatori non mostrati.
    st.sidebar.header("Parametri")
    valori: dict = {}
    for titolo, nomi in panels.gruppi_di(strategia):
        with st.sidebar.expander(titolo, expanded=True):
            colonne = st.columns(2)
            for posizione, nome in enumerate(nomi):
                valori[nome] = colonne[posizione % 2].number_input(
                    label=panels.ETICHETTE[nome], key=f"par_{nome}", **getattr(config, nome).widget
                )

    if strategia == "Squeeze Breakout":
        valori["CONFIRM_VOLUME"] = st.sidebar.checkbox("Richiedi conferma dal volume", value=config.CONFIRM_VOLUME)
    if strategia == "Ichimoku Trend":
        valori["REQUIRE_CLOUD"] = st.sidebar.checkbox("Richiedi conferma dalla nuvola", value=config.REQUIRE_CLOUD)
    valori["MODELLO"] = st.session_state["model"]

    # --- Dati e visualizzazione ---------------------------------------------------------------
    st.sidebar.header("Dati")
    show_graph = st.sidebar.checkbox(label="Mostra il grafico", value=config.SHOW_GRAPHS)
    with st.sidebar.expander("Altre sorgenti", expanded=False):
        csv_file = st.text_input(label="File CSV", value=config.CSV_FILE)
        if st.button("Leggi dal CSV", use_container_width=True):
            letto = pd.read_csv(csv_file)
            letto.set_index("Open time", inplace=True)
            st.session_state["df"] = letto[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        col1, col2 = st.columns(2)
        start_date = col1.date_input(label="Da")
        end_date = col2.date_input(label="A")
        if st.button("Scarica per date", use_container_width=True):
            st.session_state["df"], _ = get_market_data_between_dates(
                asset=symbol, interval=interval, start_date=start_date, end_date=end_date
            )
        if st.session_state["df"] is not None and st.button("Mostra la tabella", use_container_width=True):
            st.write(st.session_state["df"])

    # --- Risultato ----------------------------------------------------------------------------
    if st.session_state["df"] is None:
        text_placeholder.info("Scarica le candele per cominciare.")
    else:
        fig, trades_df, actual_hours = trading_analysis(
            asset=symbol,
            interval=interval,
            wallet=wallet,
            valori=valori,
            strategia=strategia,
            time_hours=time_hours,
            fee_percent=0.1,
            market_data=st.session_state["df"],
        )
        with text_placeholder.container():
            st.subheader("Operazioni")
            if trades_df.empty:
                st.info("Nessuna operazione con questi parametri.")
            else:
                profitto = trades_df["Profit"].sum()
                in_utile = len(trades_df[trades_df["Profit"] > 0])
                col1, col2, col3 = st.columns(3)
                col1.metric("Profitto totale", f"{profitto:.2f} {currency}")
                col2.metric("Operazioni", f"{len(trades_df)}")
                col3.metric("Quota in utile", f"{in_utile / len(trades_df) * 100:.1f}%")
            if strategia in panels.NUOVE_SENZA_MANTENIMENTO:
                st.caption(
                    "Il motore della pagina non addebita il costo di mantenimento giornaliero e non "
                    "conosce la leva: questi numeri sono piu' ottimisti di quelli di `reports/lab_*.csv`."
                )
        if show_graph:
            fig_placeholder.plotly_chart(fig, use_container_width=True)
