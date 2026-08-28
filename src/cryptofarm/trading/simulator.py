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
from cryptofarm.trading import config, confluence, panels, rotation
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
    # La confluenza ricava i suoi quattro piani da qui: e' l'intervallo delle candele, non una
    # preferenza. Sovrascrive sempre, perche' nessun chiamante lo passa a mano.
    valori["INTERVALLO"] = interval
    # Il simbolo serve al modello delle gambe per le due feature di posizionamento, che vivono in
    # uno store separato indicizzato per coppia. Se la coppia scelta non c'e' (o non e' un
    # perpetuo) restano NaN, e il modello a gradienti le tratta come categoria a se'.
    valori["SIMBOLO"] = asset

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

        # Un segnale puo' portare un terzo elemento: la spiegazione di chi l'ha generato. Le
        # strategie a indicatore singolo non ne hanno bisogno -- il segnale *e'* l'indicatore, che
        # e' gia' disegnato -- ma quella a confluenza si', perche' li' la decisione viene da sei
        # votanti e la sola posizione del marcatore non dice quali abbiano parlato.
        for punti, etichetta, simbolo, colore in (
            (buy_signals, "Buy", "triangle-up", panels.RIALZO),
            (sell_signals, "Sell", "triangle-down", panels.RIBASSO),
        ):
            if punti:
                spiegazioni = [punto[2] if len(punto) > 2 else "" for punto in punti]
                fig.add_trace(
                    go.Scatter(
                        x=[punto[0] for punto in punti],
                        y=[punto[1] for punto in punti],
                        mode="markers",
                        marker=dict(size=13, color=colore, symbol=simbolo, line=dict(width=1, color="#1a1a19")),
                        name=etichetta,
                        legendgroup="segnali",
                        text=spiegazioni,
                        hovertemplate="%{text}<extra></extra>" if any(spiegazioni) else None,
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


# -------------------------------------------------------------------------------------------------
# La seconda vista: rotazione fra asset
# -------------------------------------------------------------------------------------------------
# Non e' una voce del menu, e' un'altra pagina. Il menu sceglie *quando* stare in un asset; qui si
# sceglie *quale* fra piu' asset, e la domanda non e' esprimibile in `trading_analysis`, che carica
# un simbolo solo. I due riferimenti -- BTC tenuto fermo e l'universo a peso uguale tenuto fermo --
# sono disegnati sempre: il secondo e' quello che conta, perche' porta la stessa distorsione da
# sopravvivenza della rotazione.


@st.cache_data(ttl=3600, max_entries=8)
def universo_di_sessione(symbols: tuple[str, ...], interval: str, since: str) -> pd.DataFrame:
    """Le chiusure dell'universo, dallo store locale delle candele.

    Il tetto sulle voci c'e' per la stessa ragione degli altri quattro della cartella: i parametri
    arrivano dai widget, quindi la cardinalita' la decide chi muove i controlli.

    **Legge lo store, non la rete.** In produzione `market_data/` e' vuota (il piano non ha dischi
    persistenti), quindi la vista si spegne da sola invece di provare quindici scarichi.
    """
    return rotation.load_universe(list(symbols), interval, since)


def rotation_analysis(closes: pd.DataFrame, parametri: dict) -> tuple[go.Figure, dict, dict]:
    """La curva del capitale della rotazione contro i due riferimenti, piu' le metriche."""
    esito = rotation.backtest(
        closes,
        lookback=int(parametri["lookback"]),
        top=int(parametri["top"]),
        every=int(parametri["every"]),
        fee=float(parametri["fee"]),
        regime="btc" if parametri["regime"] else "none",
    )
    riferimenti = rotation.benchmarks(closes)

    fig = go.Figure()
    # Il capitale della rotazione e' l'unica linea che il lettore deve seguire: prende l'arancio,
    # i riferimenti restano blu, nella rampa chiaro/scuro che il registro usa per le serie
    # ordinate. Verde e rosso restano allo stato, come ovunque nella pagina.
    for nome, curva, colore, tratteggio in (
        ("BTC buy and hold", riferimenti.get("BTC comprare e tenere", {}).get("_equity"), panels.BLU_CHIARO, "dot"),
        ("Equal-weight universe", riferimenti["universo a peso uguale"]["_equity"], panels.BLU_SCURO, "dash"),
        ("Rotation", esito["_equity"], panels.ARANCIO, None),
    ):
        if curva is None:
            continue
        fig.add_trace(
            go.Scatter(
                x=closes.index,
                y=curva,
                name=nome,
                mode="lines",
                line={"color": colore, "width": 2.6 if tratteggio is None else 1.6, "dash": tratteggio},
            )
        )
    fig.update_layout(
        title="Capital, rebased to 100",
        template="plotly_dark",
        height=520,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
    )
    fig.update_yaxes(type="log", title="Capital (log)")
    return fig, esito, riferimenti


def rotation_page(text_placeholder, fig_placeholder) -> None:
    """La barra laterale e il corpo della vista di rotazione."""
    st.sidebar.header("Universe")
    universo = st.sidebar.selectbox(label="Assets", options=config.ROTATION_UNIVERSES, index=0)
    interval = st.sidebar.selectbox(label="Candle interval", options=config.ROTATION_INTERVALS, index=1)
    since = st.sidebar.text_input(label="From", value=config.ROTATION_SINCE)
    if universo == "wide":
        st.sidebar.caption(
            "Measured: widening the universe **hurts**. Out of sample the median goes from +62% "
            "on the five majors to −0.9% on fifteen. It is here as the control that shows it."
        )

    st.sidebar.header("Rotation")
    parametri = {
        "lookback": st.sidebar.number_input(label="Lookback (bars)", **config.ROTATION_LOOKBACK.widget),
        "top": st.sidebar.number_input(label="Assets held", **config.ROTATION_TOP.widget),
        "every": st.sidebar.number_input(label="Rebalance every (bars)", **config.ROTATION_EVERY.widget),
        "fee": st.sidebar.number_input(label="Fee per leg %", **config.ROTATION_FEE.widget),
        "regime": st.sidebar.checkbox("Cash out when BTC is below its 50-bar average", value=True),
    }
    st.sidebar.caption(
        "These defaults are the **central** values, not a grid optimum. Picking the best "
        "in-sample configuration transfers worse than picking one at random (ρ = −0.69)."
    )

    closes = universo_di_sessione(tuple(rotation.UNIVERSI[universo]), interval, since)
    if closes.empty or closes.shape[1] < 2:
        text_placeholder.warning(
            "The candle store is empty or holds fewer than two assets. This view reads "
            "`market_data/`, not the exchange: fill it with "
            "`python -m cryptofarm.data.klines --update`."
        )
        return

    try:
        fig, esito, riferimenti = rotation_analysis(closes, parametri)
    except ValueError as errore:
        text_placeholder.warning(str(errore))
        return

    with text_placeholder.container():
        st.subheader("Rotation vs holding")
        colonne = st.columns(4)
        colonne[0].metric("Rotation", f"{esito['rendimento_%']:.1f}%", f"Sharpe {esito['Sharpe']:.2f}")
        universo_fermo = riferimenti["universo a peso uguale"]
        colonne[1].metric(
            "Equal-weight universe",
            f"{universo_fermo['rendimento_%']:.1f}%",
            f"Sharpe {universo_fermo['Sharpe']:.2f}",
        )
        btc_fermo = riferimenti.get("BTC comprare e tenere")
        if btc_fermo:
            colonne[2].metric(
                "BTC buy and hold",
                f"{btc_fermo['rendimento_%']:.1f}%",
                f"Sharpe {btc_fermo['Sharpe']:.2f}",
            )
        colonne[3].metric(
            "Max drawdown",
            f"{esito['drawdown_%']:.1f}%",
            f"{esito['drawdown_%'] - universo_fermo['drawdown_%']:+.1f} pts vs universe",
            delta_color="inverse",
        )
        st.caption(
            f"{esito['ribilanciamenti']} rebalances, turnover {esito['turnover_annuo']:.1f}× a year. "
            "The number to beat is the equal-weight universe, not BTC: it carries the same "
            "survivorship bias as the rotation, so the comparison isolates what the rotation adds."
        )

        if esito["_holdings"]:
            recenti = pd.DataFrame(
                [{"When": quando, "Held": ", ".join(nomi) or "cash"} for quando, nomi in esito["_holdings"][-8:]]
            )
            st.dataframe(recenti, use_container_width=True, hide_index=True)
        else:
            st.info("No asset ever had positive relative strength here: the portfolio stayed in cash.")

    fig_placeholder.plotly_chart(fig, use_container_width=True)


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

    # --- Quale delle due viste ----------------------------------------------------------------
    # Non e' una strategia in piu' nel menu: e' un'altra domanda. Il menu sceglie *quando* stare
    # dentro un asset, la rotazione sceglie *quale* fra piu' asset, e le due non condividono ne'
    # i dati (una carica un simbolo, l'altra l'universo) ne' i controlli.
    modalita = st.sidebar.radio("View", options=config.ROTATION_MODES, index=0, horizontal=True)

    if modalita == config.ROTATION_MODES[1]:
        rotation_page(text_placeholder, fig_placeholder)
        st.stop()

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
    if strategia == config.CONFLUENCE_STRATEGY:
        # Quali sono davvero i quattro piani a questo intervallo, e quanta storia chiedono. Sono
        # aggregazioni delle candele caricate (`resample_klines`), non scarichi separati: senza
        # dirlo, «timeframe piu' grandi» resta una promessa che non si vede da nessuna parte.
        scala = confluence.piani(interval)
        st.sidebar.caption(
            "Planes: " + " → ".join(f"{nome} {passo}" for nome, passo in scala.items()) + ". "
            "They are resampled from the loaded candles, not fetched separately."
        )
        ore = confluence.ore_richieste(interval, int(config.CONF_REGIME_EMA.value))
        st.sidebar.caption(
            f"The regime gate needs about **{ore} hours** of history at {interval} before it can "
            "open at all. With less, there are no trades and the reason is history, not the rules."
        )
        fuori = confluence.scala_fuori_misura(interval)
        if fuori:
            st.sidebar.warning(f"{interval}: {fuori}. Use 15m, 30m or 1h.")
    elif voce is None:
        st.sidebar.caption("No strategy selected: every available indicator is shown.")

    # --- Parametri, solo quelli che servono ---------------------------------------------------
    # I widget nascono da `panels.gruppi_di`: cambiando strategia cambiano i riquadri, e un
    # parametro che la strategia scelta non usa non compare. Chi non ha widget resta al suo valore
    # iniziale, che e' cio' che `trading_analysis` usa per gli indicatori non mostrati.
    #
    # I valori iniziali dipendono dall'intervallo, perche' le finestre si contano in **barre**: la
    # stessa regola vuole un canale di 20 barre a un giorno e di 150 a un'ora per coprire lo stesso
    # tratto di calendario. La chiave del widget include l'intervallo apposta -- Streamlit conserva
    # lo stato di un widget con la stessa chiave, quindi senza, cambiando intervallo, i campi
    # resterebbero fermi sui valori del precedente e il default misurato non comparirebbe mai.
    st.sidebar.header("Parameters")
    misurati = panels.valori_misurati(strategia, interval)
    ancora = panels.ancora_di(interval)
    iniziali = panels.valori_predefiniti(strategia, interval)
    valori: dict = {}
    for titolo, nomi in panels.gruppi_di(strategia):
        with st.sidebar.expander(titolo, expanded=True):
            colonne = st.columns(2)
            for posizione, nome in enumerate(nomi):
                campo = getattr(config, nome)
                etichetta = panels.ETICHETTE[nome] + (" ·" if nome in misurati else "")
                valori[nome] = colonne[posizione % 2].number_input(
                    label=etichetta,
                    key=f"par_{nome}_{interval}",
                    **{**campo.widget, "value": type(campo.value)(iniziali[nome])},
                )

    if misurati:
        st.sidebar.caption(
            f"· = starting value measured on five assets at {ancora}"
            + (f" (nearest measured interval to {interval})" if ancora != interval else "")
            + ". It is the value whose **median rank** is highest, not the one from the "
            "best-performing configuration — picking that transfers worse than picking at random."
        )
    elif ancora:
        st.sidebar.caption(f"No parameter of this strategy discriminated at {ancora}: the hand-written defaults stand.")

    if strategia == "Squeeze Breakout":
        valori["CONFIRM_VOLUME"] = st.sidebar.checkbox(
            "Require volume confirmation", value=bool(iniziali["CONFIRM_VOLUME"]), key=f"vol_{interval}"
        )
    if strategia == "Ichimoku Trend":
        valori["REQUIRE_CLOUD"] = st.sidebar.checkbox(
            "Require cloud confirmation", value=bool(iniziali["REQUIRE_CLOUD"]), key=f"cloud_{interval}"
        )
    if strategia == config.CONFLUENCE_STRATEGY:
        valori["CONF_IN_FORMAZIONE"] = st.sidebar.checkbox(
            "React inside forming higher-plane bars",
            value=bool(iniziali["CONF_IN_FORMAZIONE"]),
            key=f"forming_{interval}",
        )
        st.sidebar.caption(
            "On, the regime gate and the structure compare **the price now** with the closed "
            "higher-plane average — what the live bot sees mid-period. Off, they wait for the "
            "long bar to close: it is the ablation that measures what reacting early is worth. "
            "The six voters decide at their own close either way."
        )
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
                if strategia == config.CONFLUENCE_STRATEGY and st.session_state["df"] is not None:
                    st.caption("Why: " + panels.diagnosi_confluenza(st.session_state["df"], valori, interval))
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
