import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go
import streamlit as st

# Carica il file CSV
file_path = 'C:/Users/marco/Documents/2025-02-08T14-13_export_close_buy_sell_rsi_atr.csv'  # Sostituisci con il percorso corretto del file
data = pd.read_csv(file_path)
target_column = 'Profitto Totale'
parameters = ['Moltiplicatore ATR',
              'Finestra ATR',
              'Finestra SMA',
              'Finestra RSI',
              'RSI Buy Limit',
              'RSI Sell Limit']

# Filtra il dataset per l'intervallo di 15 minuti
# filtered_data = data[data['Asset'] != "AMPUSDT"]
# filtered_data = filtered_data[filtered_data['Moltiplicatore ATR'] == 3.2]
# filtered_data = filtered_data[filtered_data['Step'] == 0.04]
# data = filtered_data

# Titolo dell'app Streamlit
st.title("Analisi delle Performance: Ottimizzazione Parametri")
# Raggruppa per combinazione di parametri e calcola la media del Profitto Totale
# grouped = data.groupby(parameters)[target_column].mean().reset_index()
# grouped['Size'] = 20
# Seleziona i parametri per l'asse X e Y
st.subheader("Parametri per Heatmap")
x_param = st.selectbox("Seleziona un parametro per l'asse X:", parameters, index=2)
y_param = st.selectbox("Seleziona un parametro per l'asse Y:", parameters, index=3)
if y_param != x_param:
    # Heatmap
    subset = data[[x_param, y_param, target_column]]
    heatmap_data = subset.groupby([x_param, y_param])[target_column].mean().reset_index()
    heatmap_fig = px.density_heatmap(
        heatmap_data,
        x=x_param,
        y=y_param,
        z=target_column,
        color_continuous_scale='RdBu',
        title=f'{target_column} in funzione di {x_param} e {y_param}'
    )
    heatmap_fig.update_layout(autosize=True, height=600)
    st.plotly_chart(heatmap_fig, use_container_width=True)

st.subheader("Parametri per 3D Map")
x_param = st.selectbox("Seleziona un parametro per l'asse X:", parameters, index=1)
y_param = st.selectbox("Seleziona un parametro per l'asse Y:", parameters, index=2)
z_param = st.selectbox("Seleziona un parametro per l'asse Z:", parameters, index=3)
if y_param != x_param != z_param:
    # Scatter 3D: Step, Finestra ATR, Moltiplicatore ATR
    st.header(f"{target_column} in funzione di {x_param}, {y_param} e {z_param}")
    subset = data[[x_param, y_param, z_param, target_column]]
    scatter_dati = subset.groupby([x_param, y_param, z_param])[target_column].mean().reset_index()
    scatter_dati['Size'] = 20
    scatter_fig = px.scatter_3d(
        scatter_dati,
        x=x_param,
        y=y_param,
        z=z_param,
        color=target_column,
        size='Size',
        title=f"{target_column} in funzione di {x_param}, {y_param} e {z_param}",
        color_continuous_scale='RdYlGn'
    )
    scatter_fig.update_layout(
        scene=dict(
            xaxis_title=x_param,
            yaxis_title=y_param,
            zaxis_title=z_param
        ),
        height=800
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# Line plot: ROI totale rispetto al Moltiplicatore ATR per ciascun Asset
# st.header("Profitto Totale rispetto al Moltiplicatore ATR per ciascun Asset")
# subset = data[['Moltiplicatore ATR', 'Profitto Totale', 'Asset']]
# line_plot_fig = go.Figure()
# for asset in subset['Asset'].unique():
#     asset_data = subset[subset['Asset'] == asset]
#     asset_mean = asset_data.groupby('Moltiplicatore ATR')['Profitto Totale'].mean().reset_index()
#     line_plot_fig.add_trace(
#         go.Scatter(
#             x=asset_mean['Moltiplicatore ATR'],
#             y=asset_mean['Profitto Totale'],
#             mode='lines+markers',
#             name=asset
#         )
#     )
# line_plot_fig.update_layout(
#     title="Profitto Totale rispetto al Moltiplicatore ATR",
#     xaxis_title="Moltiplicatore ATR",
#     yaxis_title="Profitto Totale",
#     legend_title="Asset",
#     height=600
# )
# st.plotly_chart(line_plot_fig, use_container_width=True)

# Bar Plot: Profitto Totale per Asset e Intervallo
st.header("Profitto Totale per Asset e Intervallo")
bar_data = data.groupby(['Asset', 'Intervallo'])['Profitto Totale'].mean().reset_index()
bar_fig = px.bar(
    bar_data,
    x='Asset',
    y='Profitto Totale',
    color='Intervallo',
    barmode='group',
    title='Profitto Totale Medio per Asset e Intervallo',
    labels={'Profitto Totale': 'Profitto Medio'}
)
st.plotly_chart(bar_fig, use_container_width=True)

# Scatter Plot: Step vs. Max Step con Profitto Totale
# st.header("Step vs. Max Step con Profitto Totale")
# scatter_data = data[['Step', 'Max Step', 'Profitto Totale']]
# scatter_data['Marker Size'] = 20
# scatter_fig = px.scatter(
#     scatter_data,
#     x='Step',
#     y='Max Step',
#     size='Marker Size',
#     color='Profitto Totale',
#     color_continuous_scale='Viridis',
#     title='Relazione tra Step, Max Step e Profitto Totale'
# )
# st.plotly_chart(scatter_fig, use_container_width=True)

for parametro in parameters:
    print(parametro)
    # Violin Plot: Distribuzione del Profitto Totale per Finestra ATR
    st.header(f"Distribuzione del Profitto Totale per {parametro}")
    violin_data = data[[parametro, 'Profitto Totale']]
    violin_fig = px.violin(
        violin_data,
        x=parametro,
        y='Profitto Totale',
        box=True,
        points='all',
        color=parametro,
        title='Distribuzione del Profitto Totale per Intervallo'
    )
    st.plotly_chart(violin_fig, use_container_width=True)


