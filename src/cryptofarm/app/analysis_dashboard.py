"""Pagina Streamlit: le misure che stanno dietro alle decisioni di `strategy.md`.

Non e' una dashboard di monitoraggio, e' il **quaderno di laboratorio** della strategia: mostra
i numeri su cui le scelte sono state prese, in modo che si possano rileggere e contestare invece
di doverli ricalcolare a memoria.

    streamlit run src/cryptofarm/app/analysis_dashboard.py

Le misure arrivano da `scripts/analysis.py`, che le mette in cache su disco: la prima esecuzione
richiede qualche minuto, le successive sono immediate. Il calcolo vive li' e non qui, cosi' la
riga di comando e la pagina non possono divergere.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scripts.analysis import CACHE_DIR, MEASURES, cached

st.set_page_config(page_title="CryptoFarm — analisi della strategia", layout="wide")

PALETTE = {
    "primary": "#4C78A8",
    "positive": "#54A24B",
    "negative": "#E45756",
    "muted": "#9AA0A6",
    "accent": "#F58518",
}


def styled(figure: go.Figure, height: int = 380) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


@st.cache_data(show_spinner=False)
def load(name: str) -> pd.DataFrame:
    return cached(name, MEASURES[name])


st.title("Analisi della strategia di trading")
st.caption(
    "Ogni numero e' misurato sui dati in `market_data/`. Il calcolo sta in `scripts/analysis.py`; "
    "questa pagina lo visualizza soltanto."
)

missing = [name for name in MEASURES if not (CACHE_DIR / f"{name}.parquet").exists()]
if missing:
    st.warning(
        f"Misure non ancora in cache: {', '.join(missing)}. "
        "Verranno calcolate alla prima apertura della relativa scheda (puo' richiedere minuti). "
        "In alternativa: `python -m scripts.analysis --all`."
    )

tabs = st.tabs(["Dati e regimi", "Capacita' e frequenza", "Economia", "Campionamento", "Portafoglio"])

# ---------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Copertura dello store")
    coverage = load("store_coverage")
    left, right = st.columns([2, 1])
    with left:
        figure = px.bar(
            coverage.sort_values("rows"),
            x="rows",
            y="symbol",
            orientation="h",
            labels={"rows": "candele 5m", "symbol": ""},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        st.plotly_chart(styled(figure, 420), use_container_width=True)
    with right:
        st.metric("Candele totali", f"{coverage['rows'].sum():,}")
        st.metric("Simboli", len(coverage))
        st.metric("Spazio su disco", f"{coverage['MB'].sum():.0f} MB")
        st.caption(
            "Solo il 5m e' archiviato: 15m, 30m e 1h sono derivati per aggregazione esatta, "
            "verificata contro i dump ufficiali."
        )
    st.dataframe(coverage, use_container_width=True, hide_index=True)

    st.subheader("Regimi di mercato")
    st.caption(
        "Rendimento di BTC su finestra mobile di 30 giorni, soglia ±10%. Conta che i tre regimi "
        "siano tutti presenti **e concentrati in periodi distinti**: e' la condizione che rende "
        "informativa la distribuzione della cross-validation combinatoria."
    )
    regimes = load("market_regimes")
    melted = regimes.melt(id_vars="anno", var_name="regime", value_name="quota")
    figure = px.bar(
        melted,
        x="anno",
        y="quota",
        color="regime",
        barmode="stack",
        color_discrete_map={"bear": PALETTE["negative"], "sideways": PALETTE["muted"], "bull": PALETTE["positive"]},
        labels={"quota": "quota di giorni", "anno": ""},
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(styled(figure), use_container_width=True)

# ---------------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Tempo al target: la misura condizionata inganna")
    st.caption(
        "La mediana calcolata sui soli casi che raggiungono il target e' distorta verso il basso "
        "e sovrastima la capacita' di fare trade frequenti. La versione corretta tratta i casi "
        "non raggiunti come censurati (Kaplan-Meier)."
    )
    timing = load("time_to_target")
    figure = go.Figure()
    figure.add_bar(
        x=timing["target"],
        y=timing["mediana_condizionata_h"],
        name="condizionata (distorta)",
        marker_color=PALETTE["negative"],
    )
    figure.add_bar(
        x=timing["target"],
        y=timing["mediana_reale_h"],
        name="reale (censura inclusa)",
        marker_color=PALETTE["positive"],
    )
    figure.update_layout(barmode="group")
    figure.update_xaxes(tickformat=".2%", title="target di prezzo")
    figure.update_yaxes(title="ore")
    st.plotly_chart(styled(figure), use_container_width=True)
    display = timing.copy()
    for column in ("target", "p_raggiunto", "errore"):
        display[column] = display[column].map("{:.1%}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Capacita' reale per configurazione di barriere")
    st.caption(
        "Il tempo al target non e' il tempo di detenzione: con barriere 2:1 la maggior parte dei "
        "trade chiude sullo stop, che e' piu' vicino. Questa e' la tabella che vincola la frequenza."
    )
    capacity = load("barrier_capacity")
    capacity["etichetta"] = capacity.apply(
        lambda row: f"{row['take_profit']:.2%}/{row['stop_loss']:.2%} · {row['orizzonte_h']:.0f}h", axis=1
    )
    left, right = st.columns(2)
    with left:
        figure = px.bar(
            capacity,
            x="etichetta",
            y="tetto_trade_giorno",
            labels={"tetto_trade_giorno": "tetto trade/giorno", "etichetta": ""},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        figure.add_hline(y=4, line_dash="dash", line_color=PALETTE["accent"], annotation_text="target 4/giorno/simbolo")
        st.plotly_chart(styled(figure), use_container_width=True)
    with right:
        figure = px.bar(
            capacity,
            x="etichetta",
            y="in_mercato_per_target",
            labels={"in_mercato_per_target": "tempo in mercato per 4 trade/giorno", "etichetta": ""},
            color_discrete_sequence=[PALETTE["accent"]],
        )
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled(figure), use_container_width=True)

    st.subheader("Il mercato reale contro una random walk")
    st.caption(
        "Il mercato tocca le barriere circa il 30% piu' in fretta e centra il take-profit ~3 punti "
        "meno spesso di una random walk della stessa volatilita'. Entrambe sono la firma delle code "
        "grasse: **la capacita' e' migliore del previsto, l'economia peggiore.**"
    )
    walk = load("random_walk")
    walk["etichetta"] = walk.apply(lambda r: f"{r['take_profit']:.2%}/{r['stop_loss']:.2%}", axis=1)
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        figure.add_bar(
            x=walk["etichetta"], y=walk["holding_random_walk_h"], name="random walk", marker_color=PALETTE["muted"]
        )
        figure.add_bar(x=walk["etichetta"], y=walk["holding_reale_h"], name="reale", marker_color=PALETTE["primary"])
        figure.update_layout(barmode="group", yaxis_title="holding medio (ore)")
        st.plotly_chart(styled(figure), use_container_width=True)
    with right:
        figure = go.Figure()
        figure.add_bar(
            x=walk["etichetta"], y=walk["p_tp_random_walk"], name="random walk", marker_color=PALETTE["muted"]
        )
        figure.add_bar(x=walk["etichetta"], y=walk["p_tp_reale"], name="reale", marker_color=PALETTE["negative"])
        figure.update_layout(barmode="group", yaxis_title="P(take-profit)")
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(styled(figure), use_container_width=True)

# ---------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Il divario di win rate da colmare")
    st.caption(
        "Break-even calcolato sulla distribuzione completa degli esiti, timeout inclusi al loro "
        "rendimento reale. **La differenza fra i regimi di commissione decide la strategia più di "
        "qualunque scelta di modello.**"
    )
    economics = load("break_even")
    economics["etichetta"] = economics.apply(lambda r: f"{r['take_profit']:.2%}/{r['stop_loss']:.2%}", axis=1)
    figure = px.bar(
        economics,
        x="etichetta",
        y="divario_punti",
        color="regime_fee",
        barmode="group",
        labels={"divario_punti": "punti di win rate da recuperare", "etichetta": ""},
        color_discrete_map={
            "taker 0,20%": PALETTE["negative"],
            "BNB 0,15%": PALETTE["accent"],
            "maker 0,04%": PALETTE["positive"],
        },
    )
    st.plotly_chart(styled(figure, 420), use_container_width=True)

    figure = px.bar(
        economics,
        x="etichetta",
        y="expectancy",
        color="regime_fee",
        barmode="group",
        labels={"expectancy": "aspettativa per operazione senza modello", "etichetta": ""},
        color_discrete_map={
            "taker 0,20%": PALETTE["negative"],
            "BNB 0,15%": PALETTE["accent"],
            "maker 0,04%": PALETTE["positive"],
        },
    )
    figure.update_yaxes(tickformat=".3%")
    figure.add_hline(y=0, line_color="white", line_width=1)
    st.plotly_chart(styled(figure), use_container_width=True)

    display = economics.drop(columns=["etichetta"]).copy()
    for column in ("take_profit", "stop_loss", "fee", "win_rate_misurato", "break_even", "expectancy"):
        display[column] = display[column].map("{:.3%}".format)
    display["divario_punti"] = display["divario_punti"].map("{:+.1f}".format)
    st.dataframe(display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Campionamento a eventi: filtro CUSUM")
    st.caption(
        "La soglia e' in multipli della volatilita' locale. Il risultato che conta e' "
        "l'**uniformita'**: a 3σ tutti i simboli producono 30–35 eventi al giorno, nonostante σ "
        "vari di un fattore tre. Nessuna calibrazione per simbolo."
    )
    rates = load("cusum_rates")
    figure = px.line(
        rates,
        x="soglia_sigma",
        y="eventi_giorno",
        color="symbol",
        markers=True,
        labels={"soglia_sigma": "soglia (multipli di σ)", "eventi_giorno": "eventi al giorno"},
    )
    figure.add_hrect(
        y0=25,
        y1=35,
        fillcolor=PALETTE["positive"],
        opacity=0.12,
        line_width=0,
        annotation_text="fascia utile per 4 trade/giorno",
    )
    st.plotly_chart(styled(figure, 440), use_container_width=True)

    at_three = rates[rates["soglia_sigma"] == 3.0].sort_values("eventi_giorno")
    left, right = st.columns([1, 1])
    with left:
        figure = px.bar(
            at_three,
            x="eventi_giorno",
            y="symbol",
            orientation="h",
            labels={"eventi_giorno": "eventi/giorno a 3σ", "symbol": ""},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        st.plotly_chart(styled(figure, 420), use_container_width=True)
    with right:
        figure = px.scatter(
            at_three,
            x="sigma",
            y="eventi_giorno",
            text="symbol",
            labels={"sigma": "volatilità 5m", "eventi_giorno": "eventi/giorno"},
            color_discrete_sequence=[PALETTE["accent"]],
        )
        figure.update_traces(textposition="top center")
        figure.update_xaxes(tickformat=".3%")
        st.plotly_chart(styled(figure, 420), use_container_width=True)
        st.caption("La nuvola piatta e' il punto: la soglia normalizzata rende i simboli confrontabili.")

# ---------------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Posizioni concorrenti a portafoglio")
    st.caption(
        "Con 4 trade/giorno su 15 simboli. **Il picco e' 4–5 volte la media**, e dimensionare il "
        "capitale sulla media significa non poter aprire meta' delle posizioni proprio quando il "
        "modello vede piu' occasioni. Il picco qui e' un limite inferiore: la simulazione assume "
        "arrivi indipendenti, mentre le criptovalute si muovono insieme."
    )
    concurrency = load("concurrency")
    concurrency["etichetta"] = concurrency.apply(lambda r: f"{r['take_profit']:.2%}/{r['stop_loss']:.2%}", axis=1)
    figure = go.Figure()
    figure.add_bar(
        x=concurrency["etichetta"], y=concurrency["posizioni_medie"], name="media", marker_color=PALETTE["primary"]
    )
    figure.add_bar(
        x=concurrency["etichetta"], y=concurrency["picco_mediano"], name="picco mediano", marker_color=PALETTE["accent"]
    )
    figure.add_bar(
        x=concurrency["etichetta"], y=concurrency["picco_p99"], name="picco 99° pct", marker_color=PALETTE["negative"]
    )
    figure.update_layout(barmode="group", yaxis_title="posizioni aperte")
    st.plotly_chart(styled(figure, 420), use_container_width=True)
    st.dataframe(concurrency.drop(columns=["etichetta"]), use_container_width=True, hide_index=True)
