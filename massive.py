import pandas as pd
import numpy as np
import streamlit as st
import simulator
from simulatorOpt import trading_analysis_opt


def run_simulation(wallet: float,
                   hours: int,
                   assets: list,
                   intervals: list,
                   rsi_buy_limits: list,
                   rsi_sell_limits: list,
                   macd_buy_limits: list,
                   macd_sell_limits: list,
                   vi_buy_limits: list,
                   vi_sell_limits: list,
                   psarvp_buy_limits: list,
                   psarvp_sell_limits: list,
                   num_conds: list,
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
            for rsi_buy_limit in rsi_buy_limits:
                for rsi_sell_limit in rsi_sell_limits:
                    for macd_buy_limit in macd_buy_limits:
                        for macd_sell_limit in macd_sell_limits:
                            for vi_buy_limit in vi_buy_limits:
                                for vi_sell_limit in vi_sell_limits:
                                    for psarvp_buy_limit in psarvp_buy_limits:
                                        for psarvp_sell_limit in psarvp_sell_limits:
                                            for num_cond in num_conds:
                                                try:
                                                    trades_df, actual_hours = trading_analysis_opt(
                                                        asset=asset,
                                                        interval=interval,
                                                        wallet=wallet,
                                                        time_hours=hours,
                                                        rsi_sell_limit=rsi_sell_limit,
                                                        rsi_buy_limit=rsi_buy_limit,
                                                        macd_buy_limit=macd_buy_limit,
                                                        macd_sell_limit=macd_sell_limit,
                                                        vi_buy_limit=vi_buy_limit,
                                                        vi_sell_limit=vi_sell_limit,
                                                        psarvp_buy_limit=psarvp_buy_limit,
                                                        psarvp_sell_limit=psarvp_sell_limit,
                                                        num_cond=num_cond,
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
                                                        'RSI Buy Limit': rsi_buy_limit,
                                                        'RSI Sell Limit': rsi_sell_limit,
                                                        'MACD Buy Limit': macd_buy_limit,
                                                        'MACD Sell Limit': macd_sell_limit,
                                                        'VI Buy Limit': vi_buy_limit,
                                                        'VI Sell Limit': vi_sell_limit,
                                                        'PSARVP Buy Limit': psarvp_buy_limit,
                                                        'PSARVP Sell Limit': psarvp_sell_limit,
                                                        'Numero condizioni': num_cond,
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
                                                    # 'Step': step,
                                                    # 'Max Step': max_step,
                                                    # 'Moltiplicatore ATR': atr_multiplier,
                                                    # 'Finestra ATR': atr_window,
                                                    # 'Finestra Min/Max': window_pivot,
                                                    # 'Finestra RSI': rsi_window,
                                                    # 'Finestra MACD veloce': macd_short_windows,
                                                    # 'Finestra MACD lenta': macd_long_windows,
                                                    # 'Finestra MACD segnale': macd_signal_windows,
                                                    'RSI Buy Limit': rsi_buy_limit,
                                                    'RSI Sell Limit': rsi_sell_limit,
                                                    'MACD Buy Limit': macd_buy_limit,
                                                    'MACD Sell Limit': macd_sell_limit,
                                                    'VI Buy Limit': vi_buy_limit,
                                                    'VI Sell Limit': vi_sell_limit,
                                                    'PSARVP Buy Limit': psarvp_buy_limit,
                                                    'PSARVP Sell Limit': psarvp_sell_limit,
                                                    'Numero condizioni': num_cond,
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
    wallet = 1000.0  # Capitale iniziale
    hours = 3600  # Numero di ore
    # assets = ["AAVEUSDC", "AMPUSDT", "ADAUSDC", "BNBUSDC", "BTCUSDC", "DEXEUSDT", "DOGEUSDC", "DOTUSDC",
    #         "ETHUSDC", "LINKUSDC", "SOLUSDC", "SUIUSDC", "ZENUSDT", "XRPUSDT"]
    assets = ["XRPUSDT","HBARUSDT","SOLUSDT","ADAUSDC","SUIUSDC","XLMUSDT","LTCUSDT"]
    # assets = ["XRPBTC","ADABTC","ETHBTC","SOLBTC","DOGEBTC","BNBBTC","SUIBTC","LTCBTC","LINKBTC","AVAXBTC","TRXBTC", "DOTBTC"]
    intervals = ["15m"]
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

    rsi_buy_limits = [25]
    rsi_sell_limits = [75]
    macd_buy_limits = [-0.66]
    macd_sell_limits = [0.66]
    vi_buy_limits = [-0.82]
    vi_sell_limits = [0.82]
    psarvp_buy_limits = [1.08]
    psarvp_sell_limits = [0.92]
    num_conds = [2]

    dati = simulator.download_market_data(assets, intervals, hours)
    simulazioni = run_simulation(wallet=wallet,
                   hours=hours,
                   assets=assets,
                   intervals=intervals,
                   rsi_buy_limits=rsi_buy_limits,
                   rsi_sell_limits=rsi_sell_limits,
                   macd_buy_limits=macd_buy_limits,
                   macd_sell_limits=macd_sell_limits,
                   vi_buy_limits=vi_buy_limits,
                   vi_sell_limits=vi_sell_limits,
                   psarvp_buy_limits=psarvp_buy_limits,
                   psarvp_sell_limits=psarvp_sell_limits,
                   num_conds=num_conds,
                   market_data=dati)
    if simulazioni:
        st.dataframe(pd.DataFrame(simulazioni))
    else:
        st.warning("Nessuna simulazione eseguita o nessun trade effettuato.")

    print("Finito.")
