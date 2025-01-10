import pandas as pd
import numpy as np
import streamlit as st
import simulator


def run_simulation(wallet: float,
                   hours: int,
                   assets: list,
                   intervals: list,
                   steps: list,
                   max_steps: list,
                   atr_multipliers: list,
                   atr_windows: list,
                   window_pivots: list,
                   rsi_windows: list,
                   macd_short_windows: list,
                   macd_long_windows: list,
                   macd_signal_windows: list,
                   market_data: dict = None):
    """
    Esegue una simulazione massiva di strategie Buy/Sell basate sul SAR
    (Parabolic SAR). Per ogni combinazione di:
      - Intervallo (es. 1m, 3m, 5m)
      - Asset (lista di crypto)
      - Step e Max Step per il calcolo del SAR
    viene chiamata la funzione sar_trading_analysis, e si registrano
    statistiche relative ai risultati di trading.

    Infine, mostra una tabella 'simulazioni' con i risultati.
    """

    st.title("Simulazione Buy/Sell su indice SAR")
    st.write("Avvio simulazione massiva...")

    simulazioni = []

    for interval in intervals:
        # Calcolo del tempo totale coperto dalle candele (utile per ROI giornaliero)
        try:
            candlestick_minutes = simulator.interval_to_minutes(interval)
            if candlestick_minutes <= 0:
                st.warning(f"Intervallo '{interval}' non valido (non termina in m/h?), salto.")
                continue
        except Exception as e:
            st.error(f"Errore calcolo minuti per '{interval}': {e}")
            continue

        for asset in assets:
            # Usa i dati già scaricati
            df = market_data.get(asset, {}).get(interval, None)
            for step in steps:
                for max_step in max_steps:
                    for atr_multiplier in atr_multipliers:
                        for atr_window in atr_windows:
                            for window_pivot in window_pivots:
                                for rsi_window in rsi_windows:
                                    for macd_short_window in macd_short_windows:
                                        for macd_long_window in macd_long_windows:
                                            for macd_signal_window in macd_signal_windows:
                                                try:
                                                    _, _, _, trades_df, actual_hours = simulator.sar_trading_analysis(
                                                        asset=asset,
                                                        interval=interval,
                                                        wallet=wallet,
                                                        step=step,
                                                        max_step=max_step,
                                                        time_hours=hours,
                                                        show=False,
                                                        atr_multiplier=atr_multiplier,
                                                        atr_window=atr_window,
                                                        window_pivot=window_pivot,
                                                        rsi_window=rsi_window,
                                                        macd_short_window=macd_short_window,
                                                        macd_long_window=macd_long_window,
                                                        macd_signal_window=macd_signal_window,
                                                        market_data=df
                                                    )
                                                except Exception as e:
                                                    st.error(
                                                        f"Errore durante sar_trading_analysis({asset}, {interval}): {e}")
                                                    continue

                                                total_days = actual_hours / 24
                                                time_string = f"{actual_hours:.2f} ore ({total_days:.2f} giorni)"
                                                # Se trades_df è vuoto, nessuna operazione
                                                if trades_df.empty:
                                                    simulazioni.append({
                                                        'Asset': asset,
                                                        'Intervallo': interval,
                                                        'Tempo': time_string,
                                                        'Step': step,
                                                        'Max Step': max_step,
                                                        'Operazioni Chiuse': 0,
                                                        'Profitto Totale': 0,
                                                        'ROI totale (%)': 0,
                                                        'ROI giornaliero (%)': 0
                                                        # ...e altre colonne a zero/nan
                                                    })
                                                    continue

                                                # Calcola statistiche principali
                                                # (ogni riga di trades_df è un trade completo: Buy+Sell)
                                                num_trades = len(trades_df)
                                                total_profit = trades_df['Profit'].sum()

                                                # Trade in profitto/perdita/pareggio
                                                profitable_trades = trades_df[trades_df['Profit'] > 0]
                                                losing_trades = trades_df[trades_df['Profit'] < 0]
                                                break_even_trades = trades_df[trades_df['Profit'] == 0]

                                                num_profitable = len(profitable_trades)
                                                num_losing = len(losing_trades)
                                                num_break_even = len(break_even_trades)

                                                # Win rate
                                                win_rate = (
                                                        num_profitable / num_trades * 100) if num_trades > 0 else 0.0
                                                # Profitto medio
                                                avg_profit = trades_df['Profit'].mean() if num_trades > 0 else 0.0
                                                # Profitto medio (Gain)
                                                avg_win = profitable_trades[
                                                    'Profit'].mean() if num_profitable > 0 else 0.0
                                                # Perdita media (Loss)
                                                avg_loss = losing_trades['Profit'].mean() if num_losing > 0 else 0.0

                                                # Max e Min profit
                                                max_profit_trade = trades_df['Profit'].max() if num_trades > 0 else 0.0
                                                min_profit_trade = trades_df['Profit'].min() if num_trades > 0 else 0.0

                                                # ROI totale
                                                roi_percent = (total_profit / wallet * 100) if wallet > 0 else 0.0

                                                # ROI giornaliero (composto)
                                                final_wallet = wallet + total_profit
                                                if final_wallet > 0:
                                                    daily_roi = (final_wallet / wallet) ** (1 / total_days) - 1
                                                    daily_roi_percent = daily_roi * 100
                                                else:
                                                    daily_roi_percent = -100.0  # Nel caso di portafoglio azzerato o negativo

                                                # Infine, ricaviamo eventuali metriche sul prezzo (se esistono nel trades_df)
                                                if 'massimo' in trades_df.columns:
                                                    prezzo_massimo = trades_df['massimo'].max()
                                                else:
                                                    prezzo_massimo = np.nan

                                                if 'minimo' in trades_df.columns:
                                                    prezzo_minimo = trades_df['minimo'].min()
                                                else:
                                                    prezzo_minimo = np.nan

                                                if 'variazione(%)' in trades_df.columns:
                                                    # L'ultima riga o la prima: dipende da come hai popolato i dati
                                                    variazione_prezzo = trades_df['variazione(%)'].iloc[-1]
                                                else:
                                                    variazione_prezzo = np.nan

                                                if 'volatilita(%)' in trades_df.columns:
                                                    volatilita = trades_df['volatilita(%)'].iloc[-1]
                                                else:
                                                    volatilita = np.nan

                                                # Salviamo i risultati
                                                simulazioni.append({
                                                    'Asset': asset,
                                                    'Intervallo': interval,
                                                    'Tempo': time_string,
                                                    'Step': step,
                                                    'Max Step': max_step,
                                                    'Moltiplicatore ATR': atr_multiplier,
                                                    'Finestra ATR': atr_window,
                                                    'Finestra Min/Max': window_pivot,
                                                    'Finestra RSI': rsi_window,
                                                    'Finestra MACD veloce': macd_short_windows,
                                                    'Finestra MACD lenta': macd_long_windows,
                                                    'Finestra MACD segnale': macd_signal_windows,
                                                    'Prezzo Massimo': prezzo_massimo,
                                                    'Prezzo Minimo': prezzo_minimo,
                                                    'Variazione di prezzo (%)': variazione_prezzo,
                                                    'Volatilità (%)': volatilita,
                                                    'Profitto Totale': round(total_profit, 4),
                                                    'Profitto Medio': round(avg_profit, 4),
                                                    'Operazioni Chiuse': num_trades,
                                                    'Operazioni in Profitto': num_profitable,
                                                    'Operazioni in Perdita': num_losing,
                                                    'Pareggi': num_break_even,
                                                    'Win Rate (%)': round(win_rate, 2),
                                                    'Profitto Medio (Gain)': round(avg_win, 4),
                                                    'Perdita Media (Loss)': round(avg_loss, 4),
                                                    'Max Profit Trade': round(max_profit_trade, 4),
                                                    'Min Profit Trade': round(min_profit_trade, 4),
                                                    'ROI totale (%)': round(roi_percent, 2),
                                                    'ROI giornaliero (%)': round(daily_roi_percent, 2)
                                                })

    # Terminati tutti i loop, mostriamo i risultati
    if simulazioni:
        results_df = pd.DataFrame(simulazioni)
        st.write("## Risultati delle simulazioni:")
        st.dataframe(results_df)
    else:
        st.warning("Nessuna simulazione eseguita o nessun trade effettuato.")


if __name__ == "__main__":
    # ------------------------------
    # Parametri fissati per l'ottimizzazione
    wallet = 1000.0  # Capitale iniziale
    hours = 1000  # Numero di ore
    assets = ["AAVEUSDC", "AMPUSDT", "ADAUSDC", "AVAXUSDC", "BNBUSDC", "BTCUSDC", "DEXEUSDT", "DOGEUSDC", "DOTUSDC",
              "ETHUSDC", "LINKUSDC", "SOLUSDC", "PEPEUSDC", "RUNEUSDC", "SUIUSDC", "ZENUSDT", "XRPUSDT"]
    intervals = ["15m"]
    steps = [0.04]
    max_steps = [0.4]
    atr_multipliers = [2.4]
    atr_windows = [6]
    window_pivots = [10]
    rsi_windows = [10]
    macd_short_windows = [12]
    macd_long_windows = [26]
    macd_signal_windows = [9]
    dati = simulator.download_market_data(assets, intervals, hours)
    run_simulation(wallet=wallet,
                   hours=hours,
                   assets=assets,
                   intervals=intervals,
                   steps=steps,
                   max_steps=max_steps,
                   atr_multipliers=atr_multipliers,
                   atr_windows=atr_windows,
                   window_pivots=window_pivots,
                   rsi_windows=rsi_windows,
                   macd_short_windows=macd_short_windows,
                   macd_long_windows=macd_long_windows,
                   macd_signal_windows=macd_signal_windows,
                   market_data=dati)
    print("Finito.")
