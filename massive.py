import pandas as pd
import numpy as np
import streamlit as st
from simulator import download_market_data, interval_to_minutes
# import simulator
# from background import atr_window
# from background import atr_multiplier
from simulatorOpt import trading_analysis_opt


def run_simulation(wallet: float,
                   hours: int,
                   assets: list,
                   intervals: list,
                   atr_multipliers: list,
                   atr_windows: list,
                   sma_windows: list,
                   rsi_windows: list,
                   rsi_buy_limits: list,
                   rsi_sell_limits: list,
                   # steps: list,
                   # num_conds: list,
                   # macd_buy_limits: list,
                   # macd_sell_limits: list,
                   # vi_buy_limits: list,
                   # vi_sell_limits: list,
                   # psarvp_buy_limits: list,
                   # psarvp_sell_limits: list,
                   # stop_loss_percents: list,
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
            candlestick_minutes = interval_to_minutes(interval)
            if candlestick_minutes <= 0:
                st.warning(f"Intervallo '{interval}' non valido (non termina in m/h?), salto.")
                continue
        except Exception as e:
            st.error(f"Errore calcolo minuti per '{interval}': {e}")
            continue

        for asset in assets:
            # Usa i dati già scaricati
            df = market_data.get(asset, {}).get(interval, None)
            # for rsi_buy_limit in rsi_buy_limits:
            #     for rsi_sell_limit in rsi_sell_limits:
            #         for macd_buy_limit in macd_buy_limits:
            #             for macd_sell_limit in macd_sell_limits:
            #                 for vi_buy_limit in vi_buy_limits:
            #                     for vi_sell_limit in vi_sell_limits:
            #                         for psarvp_buy_limit in psarvp_buy_limits:
            #                             for psarvp_sell_limit in psarvp_sell_limits:
            #                                 for num_cond in num_conds:
            # for atr_multiplier in atr_multipliers:
            for atr_window in atr_windows:
                for atr_multiplier in atr_multipliers:
                    for sma_window in sma_windows:
                        for rsi_window in rsi_windows:
                            for rsi_buy_limit in rsi_buy_limits:
                                for rsi_sell_limit in rsi_sell_limits:
                                    try:
                                        trades_df, actual_hours = trading_analysis_opt(
                                            asset=asset,
                                            interval=interval,
                                            wallet=wallet,
                                            time_hours=hours,
                                            atr_multiplier=atr_multiplier,
                                            atr_window=atr_window,
                                            sma_window=sma_window,
                                            rsi_window=rsi_window,
                                            rsi_sell_limit=rsi_sell_limit,
                                            rsi_buy_limit=rsi_buy_limit,
                                            market_data=df
                                        )
                                    except Exception as e:
                                        st.error(
                                            f"Errore durante sar_trading_analysis({asset}, {interval}): {e}")
                                        continue

                                    # concateno tutti i risultati degli ottimi
                                    # ottimi.extend(lista_min_max)

                                    total_days = actual_hours / 24
                                    time_string = f"{actual_hours:.2f} ore ({total_days:.2f} giorni)"
                                    # Se trades_df è vuoto, nessuna operazione
                                    if trades_df.empty:
                                        simulazioni.append({
                                            'Asset': asset,
                                            'Intervallo': interval,
                                            'Tempo': time_string,
                                            'Profitto Totale': 0,
                                            'Profitto Medio': 0,
                                            'Operazioni Chiuse': 0,
                                            'Operazioni in Profitto': 0,
                                            'Operazioni in Perdita': 0,
                                            'Pareggi': 0,
                                            'Win Rate (%)': 0,
                                            'Profitto Medio (Gain)': 0,
                                            'Perdita Media (Loss)': 0,
                                            'Max Profit Trade': 0,
                                            'Min Profit Trade': 0,
                                            'ROI totale (%)': 0,
                                            'ROI giornaliero (%)': 0
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
                                        'Moltiplicatore ATR': atr_multiplier,
                                        'Finestra ATR': atr_window,
                                        'Finestra SMA': sma_window,
                                        'Finestra RSI': rsi_window,
                                        'RSI Buy Limit': rsi_buy_limit,
                                        'RSI Sell Limit': rsi_sell_limit,
                                        # 'Stop Loss': stop_loss_percent,
                                        # 'Numero condizioni': num_cond,
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
    return simulazioni


if __name__ == "__main__":
    # ------------------------------
    # Parametri fissati per l'ottimizzazione
    wallet = 100.0  # Capitale iniziale
    hours = 1200  # Numero di ore
    intervals = ["15m"]

    # assets = ["AAVEUSDC", "AMPUSDT", "ADAUSDC", "BNBUSDC", "BTCUSDC", "DEXEUSDT", "DOTUSDC",
    #         "ETHUSDC", "HBARUSDT", "LINKUSDC", "SOLUSDT", "SUIUSDC", "ZENUSDT", "XLMUSDT","XRPUSDT"]
    # assets = ["AMPUSDT", "XRPUSDT","HBARUSDT","SOLUSDT","ADAUSDC","SUIUSDC","XLMUSDT","LTCUSDT", "BTCUSDC"]
    # assets = ["AMPUSDT", "XRPUSDT", "HBARUSDT", "BTCUSDC"]
    # assets = ["XRPBTC","ADABTC","ETHBTC","SOLBTC","DOGEBTC","BNBBTC","SUIBTC","LTCBTC","LINKBTC",
    #           "AVAXBTC","TRXBTC", "DOTBTC"]
    # assets = ["AAVEUSDC","DEXEUSDT"]
    # assets = ["BTCUSDC","SOLUSDC","XRPUSDT","AMPUSDT","ZENUSDT"]
    # steps = [0.01] DEFAULT 0.01
    # max_steps = [0.4] # DEFAULT 0.4
    # atr_multipliers = [2.4] DEFAULT 2.4
    # atr_windows = [6] DEFAULT 6
    # window_pivots = [150]
    # rsi_windows = [10] DEFAULT 12
    # macd_short_windows = [12] DEFAULT 12
    # macd_long_windows = [26] DEFAULT 26
    # macd_signal_windows = [9] DEFAULT 9
    # rsi_buy_limits = [25]
    # rsi_sell_limits = [75]
    # macd_buy_limits = [-0.66]
    # macd_sell_limits = [0.66]
    # vi_buy_limits = [-0.82]
    # vi_sell_limits = [0.82]
    # psarvp_buy_limits = [1.08]
    # psarvp_sell_limits = [0.92]
    # num_conds = [2]
    # din_roc_divs = [12, 13, 14, 15]
    # din_macd_divs = [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
    # steps = [0.01]
    # stop_loss_percents = [99]

    # CHAIN = ["ADAUSDC", "BNBUSDC", "BTCUSDC", "ETHUSDC", "SOLUSDC", "XRPUSDT"]
    # CHAIN SECONDARIE = ["DOTUSDC", "TRXUSDC", "LTCUSDC", "XLMUSDC", "ALGOUSDC", "ATOMUSDC"]

    assets = ["AMPUSDT", "HBARUSDT", "LINKUSDC", "XRPUSDT"]
    atr_windows = [20, 30, 40]
    atr_multipliers = [0.9, 1.1, 1.3, 1.5, 1.6]
    sma_windows = [2, 3, 4, 5]
    rsi_windows = [6, 10, 12, 18, 26]
    rsi_buy_limits = [20, 23, 25, 29]
    rsi_sell_limits = [71, 75, 77, 80]

    dati = download_market_data(assets, intervals, hours)
    simulazioni = run_simulation(wallet=wallet,
                                 hours=hours,
                                 assets=assets,
                                 intervals=intervals,
                                 atr_multipliers=atr_multipliers,
                                 atr_windows=atr_windows,
                                 sma_windows=sma_windows,
                                 rsi_windows=rsi_windows,
                                 rsi_buy_limits=rsi_buy_limits,
                                 rsi_sell_limits=rsi_sell_limits,
                                 # steps=steps,
                                 # din_macd_divs=din_macd_divs,
                                 # din_roc_divs=din_roc_divs,
                                 # macd_buy_limits=macd_buy_limits,
                                 # macd_sell_limits=macd_sell_limits,
                                 # vi_buy_limits=vi_buy_limits,
                                 # vi_sell_limits=vi_sell_limits,
                                 # psarvp_buy_limits=psarvp_buy_limits,
                                 # psarvp_sell_limits=psarvp_sell_limits,
                                 # num_conds=num_conds,
                                 # stop_loss_percents=stop_loss_percents,
                                 market_data=dati)
    if simulazioni:
        st.dataframe(pd.DataFrame(simulazioni))
    else:
        st.warning("Nessuna simulazione eseguita o nessun trade effettuato.")

    print("Finito.")
