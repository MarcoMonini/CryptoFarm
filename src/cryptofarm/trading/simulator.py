import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema

from cryptofarm.ml.trainer import (
    active_model_name,
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


def modello_di_sessione():
    """Il modello con cui parte la pagina, oppure `None` se non ne e' stato addestrato nessuno.

    E' una funzione e non una riga dentro `__main__` perche' e' proprio quella riga ad aver
    mandato in errore il servizio in produzione: `load_signal_model()` chiamata senza condizione
    solleva `FileNotFoundError`, e la pagina non si apriva affatto -- non solo la strategia che il
    modello lo usa. I pezzi erano coperti da test, l'assemblaggio no, e il guasto e' passato di li'.
    """
    return load_signal_model() if active_model_name() else None


def available_strategies(model) -> list[str]:
    """Le voci offerte dal menu delle strategie.

    Senza un modello addestrato la strategia AI non e' selezionabile: Streamlit non sa
    disabilitare una singola voce di un selectbox, quindi l'unico modo per non renderla
    selezionabile e' non metterla in elenco.
    """
    if model is not None:
        return list(config.STRATEGIES)
    return [name for name in config.STRATEGIES if name != config.AI_STRATEGY]


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
        if strategia == config.AI_STRATEGY and valori.get("MODELLO") is None:
            # Il menu non offre la voce quando manca l'artefatto, ma `trading_analysis` e'
            # chiamabile anche da fuori: meglio fermarsi con un messaggio che con un traceback.
            st.error(f"No trained model in {MODELS_DIR}: the «{config.AI_STRATEGY}» strategy is unavailable.")
            st.stop()
        buy_signals, sell_signals = voce.esegui(df, cache, valori)

    operations = simulate_trading_with_commisions(
        wallet=wallet, buy_signals=buy_signals, sell_signals=sell_signals, fee_percent=fee_percent
    )

    # ======================================
    # Il grafico, costruito da `panels` invece che da un elenco fisso.
    #
    # Prima erano due riquadri sempre uguali e una dozzina di tracce sempre presenti: chi guardava
    # "Trend Zones" vedeva tre RSI, uno stocastico e un TSI che quella strategia non tocca, e
    # doveva sapere a memoria quali linee contassero. Ora la figura ha un riquadro per ogni
    # oscillatore che la strategia usa davvero, e sulle candele solo i suoi overlay.
    indicatori = panels.indicatori_di(strategia)
    pannelli = panels.pannelli_di(strategia)
    riga_di = {titolo: numero for numero, titolo in enumerate(pannelli, start=2)}

    ALTEZZA_CANDELE = 460
    ALTEZZA_PANNELLO = 170
    righe = 1 + len(pannelli)
    fig = make_subplots(
        rows=righe,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04 if righe > 1 else 0.0,
        row_heights=[ALTEZZA_CANDELE] + [ALTEZZA_PANNELLO] * len(pannelli),
        subplot_titles=("", *pannelli),
    )

    if show:
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=asset,
                increasing_line_color=panels.RIALZO,
                decreasing_line_color=panels.RIBASSO,
                increasing_fillcolor=panels.RIALZO,
                decreasing_fillcolor=panels.RIBASSO,
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )

        for chiave in indicatori:
            indicatore = panels.INDICATORI[chiave]
            riga = 1 if indicatore.pannello is None else riga_di[indicatore.pannello]
            calcolate = indicatore.serie(df, cache, valori)
            for traccia in indicatore.tracce:
                # Una serie puo' mancare per scelta: la media di regime a finestra zero e' spenta,
                # e disegnarla come linea vuota metterebbe in legenda un indicatore inattivo.
                if traccia.serie not in calcolate:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=calcolate[traccia.serie],
                        mode=traccia.modo,
                        name=traccia.nome,
                        legendgroup=chiave,
                        line=dict(color=traccia.colore, width=traccia.larghezza, dash=traccia.tratteggio),
                        marker=dict(size=traccia.dimensione, color=traccia.colore, symbol=traccia.simbolo),
                    ),
                    row=riga,
                    col=1,
                )

        if "estremi" in indicatori:
            for punti, etichetta, simbolo in (
                (rel_max, "Swing highs", "triangle-down-open"),
                (rel_min, "Swing lows", "triangle-up-open"),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[quando for quando, _ in punti],
                        y=[prezzo for _, prezzo in punti],
                        mode="markers",
                        marker=dict(size=9, color=panels.ACQUA, symbol=simbolo),
                        name=etichetta,
                        legendgroup="estremi",
                    ),
                    row=1,
                    col=1,
                )

        # Le soglie dell'RSI si disegnano solo dove significano qualcosa: sono i livelli su cui la
        # strategia decide, non una decorazione del pannello.
        propri = panels.STRATEGIE[strategia].parametri if strategia in panels.STRATEGIE else ()
        if "rsi" in indicatori and "RSI_BUY_LIMIT" in propri:
            for nome_parametro, etichetta, colore in (
                ("RSI_SELL_LIMIT", "Sell threshold", panels.RIBASSO),
                ("RSI_BUY_LIMIT", "Buy threshold", panels.RIALZO),
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[df.index.min(), df.index.max()],
                        y=[valori[nome_parametro]] * 2,
                        mode="lines",
                        line=dict(color=colore, width=1, dash="dot"),
                        name=etichetta,
                        legendgroup="soglie",
                    ),
                    row=riga_di["RSI"],
                    col=1,
                )

        for punti, etichetta, simbolo, colore in (
            (buy_signals, "Buy", "triangle-up", panels.RIALZO),
            (sell_signals, "Sell", "triangle-down", panels.RIBASSO),
        ):
            if punti:
                fig.add_trace(
                    go.Scatter(
                        x=[quando for quando, _ in punti],
                        y=[prezzo for _, prezzo in punti],
                        mode="markers",
                        marker=dict(size=13, color=colore, symbol=simbolo, line=dict(width=1, color="#1a1a19")),
                        name=etichetta,
                        legendgroup="segnali",
                    ),
                    row=1,
                    col=1,
                )

        # Le fasce di trend leggono EMA20 ed EMA100: si mostrano dove quelle due ci sono davvero.
        fasce = identify_trend_zones(df=df) if "medie_trend" in indicatori else []

        fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=ALTEZZA_CANDELE + ALTEZZA_PANNELLO * len(pannelli) + 80,
            shapes=fasce,
            margin=dict(l=8, r=8, t=28, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
            hovermode="x unified",
        )
        # Griglia e assi in secondo piano: le linee dei dati devono essere la cosa piu' evidente.
        fig.update_xaxes(showgrid=False, showspikes=True, spikemode="across", spikethickness=1)
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)

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
        # Il modello e' opzionale: un clone del repository non ne ha (gli artefatti sono
        # gitignorati) e nemmeno l'immagine che va in produzione. Senza, la pagina serve
        # comunque tutte le strategie classiche, che dal modello non dipendono.
        #
        # `active_model_name` risponde senza sollevare; `load_signal_model` trova da solo il
        # formato dell'artefatto, cosi' cambiare famiglia di modello non tocca la dashboard.
        st.session_state["model"] = modello_di_sessione()

    text_placeholder = st.empty()
    fig_placeholder = st.empty()

    # --- Mercato ------------------------------------------------------------------------------
    st.sidebar.header("Market")
    col1, col2 = st.sidebar.columns(2)
    asset = col1.text_input(label="Asset", placeholder="e.g. BTC, ETH, XRP...", max_chars=8, value=config.ASSET)
    currency = col2.text_input(
        label="Currency", placeholder="e.g. USDC, USDT, EUR...", max_chars=8, value=config.CURRENCY
    )
    interval = col1.selectbox(label="Candle interval", options=config.INTERVALS, index=config.INTERVAL_INDEX)
    time_hours = col2.number_input(label="Hours of history", **config.TIME_HOURS)
    symbol = asset + currency
    wallet = st.sidebar.number_input(label=f"Wallet ({currency})", **config.WALLET)

    if st.sidebar.button("FETCH CANDLES", use_container_width=True, type="primary"):
        st.session_state["df"], _ = get_market_data(asset=symbol, interval=interval, time_hours=time_hours)

    # --- Strategia ----------------------------------------------------------------------------
    st.sidebar.header("Strategy")
    strategia = st.sidebar.selectbox(
        label="Strategy",
        options=available_strategies(st.session_state["model"]),
        index=0,
        label_visibility="collapsed",
    )
    if st.session_state["model"] is None:
        st.sidebar.caption(
            f"«{config.AI_STRATEGY}» is not listed: no model in `{MODELS_DIR}`. "
            "Train one with `python -m cryptofarm.ml.trainer`."
        )
    voce = panels.STRATEGIE.get(strategia)
    if voce is not None and voce.note:
        st.sidebar.caption(voce.note)
    elif voce is None:
        st.sidebar.caption("No strategy selected: every available indicator is shown.")

    # --- Parametri, solo quelli che servono ---------------------------------------------------
    # I widget nascono da `panels.gruppi_di`: cambiando strategia cambiano i riquadri, e un
    # parametro che la strategia scelta non usa non compare. Chi non ha widget resta al suo valore
    # iniziale, che e' cio' che `trading_analysis` usa per gli indicatori non mostrati.
    st.sidebar.header("Parameters")
    valori: dict = {}
    for titolo, nomi in panels.gruppi_di(strategia):
        with st.sidebar.expander(titolo, expanded=True):
            colonne = st.columns(2)
            for posizione, nome in enumerate(nomi):
                valori[nome] = colonne[posizione % 2].number_input(
                    label=panels.ETICHETTE[nome], key=f"par_{nome}", **getattr(config, nome).widget
                )

    if strategia == "Squeeze Breakout":
        valori["CONFIRM_VOLUME"] = st.sidebar.checkbox("Require volume confirmation", value=config.CONFIRM_VOLUME)
    if strategia == "Ichimoku Trend":
        valori["REQUIRE_CLOUD"] = st.sidebar.checkbox("Require cloud confirmation", value=config.REQUIRE_CLOUD)
    valori["MODELLO"] = st.session_state["model"]

    # --- Dati e visualizzazione ---------------------------------------------------------------
    st.sidebar.header("Data")
    show_graph = st.sidebar.checkbox(label="Show chart", value=config.SHOW_GRAPHS)
    with st.sidebar.expander("Other sources", expanded=False):
        csv_file = st.text_input(label="CSV file", value=config.CSV_FILE)
        if st.button("Load from CSV", use_container_width=True):
            letto = pd.read_csv(csv_file)
            letto.set_index("Open time", inplace=True)
            st.session_state["df"] = letto[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        col1, col2 = st.columns(2)
        start_date = col1.date_input(label="From")
        end_date = col2.date_input(label="To")
        if st.button("Fetch by date range", use_container_width=True):
            st.session_state["df"], _ = get_market_data_between_dates(
                asset=symbol, interval=interval, start_date=start_date, end_date=end_date
            )
        if st.session_state["df"] is not None and st.button("Show raw table", use_container_width=True):
            st.write(st.session_state["df"])

    # --- Risultato ----------------------------------------------------------------------------
    if st.session_state["df"] is None:
        text_placeholder.info("Fetch candles to get started.")
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
            st.subheader("Trades")
            if trades_df.empty:
                st.info("No trades with these parameters.")
            else:
                profitto = trades_df["Profit"].sum()
                in_utile = len(trades_df[trades_df["Profit"] > 0])
                col1, col2, col3 = st.columns(3)
                col1.metric("Total profit", f"{profitto:.2f} {currency}")
                col2.metric("Trades", f"{len(trades_df)}")
                col3.metric("Win rate", f"{in_utile / len(trades_df) * 100:.1f}%")
            if strategia in panels.NUOVE_SENZA_MANTENIMENTO:
                st.caption(
                    "The page engine charges no daily carry and knows nothing about leverage: "
                    "these figures are more optimistic than the ones in `reports/lab_*.csv`."
                )
        if show_graph:
            fig_placeholder.plotly_chart(fig, use_container_width=True)
