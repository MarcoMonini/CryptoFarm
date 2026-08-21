"""Conto del profitto e delle commissioni sui segnali prodotti dalle strategie.

Estratto da `simulator.py` senza modifiche. Entrambe le funzioni accoppiano i segnali per
indice: ha senso solo se la strategia li produce alternati, come nota `ai_model_simulation`.
Restituiscono la lista delle operazioni, non un totale."""


def simulate_trading_with_commisions(
    buy_signals: list, sell_signals: list, wallet: float = 100, fee_percent: float = 0.1
):
    operations = []
    holding = False  # Flag che indica se stiamo detenendo l'asset
    quantity = 0.0  # Quantità dell'asset comprata
    working_wallet = wallet  # Capitale di partenza (USDT/USDC)
    # Converto fee_percent in forma decimale (es. 1% -> 0.01)
    fee_decimal = fee_percent / 100.0
    # Per semplicità, assumiamo che numero di buy_signals e sell_signals
    # siano (in media) abbinati, usando lo stesso indice i in parallelo.
    for i in range(len(buy_signals)):
        # Se NON stiamo detenendo nulla e c'è un segnale di BUY, compriamo
        if not holding and i < len(buy_signals):
            buy_time, buy_price = buy_signals[i]
            if working_wallet > 0:
                # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
                # dopo la fee rimane working_wallet*(1 - fee_decimal) per comprare
                net_invested = working_wallet * (1 - fee_decimal)
                # quantità di crypto ottenuta
                quantity = net_invested / buy_price
                # Ora working_wallet = 0 (tutto investito)
                working_wallet = 0.0
                holding = True
        # Se ABBIAMO una posizione aperta e c'è un segnale di SELL, vendiamo
        if holding and i < len(sell_signals):
            sell_time, sell_price = sell_signals[i]
            # Ricaviamo USDT vendendo la quantity di crypto
            gross_proceed = quantity * sell_price
            # Applichiamo la commissione di vendita
            # commissions = gross_proceed * fee_decimal
            net_proceed = gross_proceed * (1 - fee_decimal)
            # Calcoliamo il profit: differenza fra l'importo netto incassato e l'importo speso in fase di BUY e le commissioni
            cost_in_usd = (quantity * buy_price) * (1 + fee_decimal)  # spesa inziale
            profit = net_proceed - cost_in_usd
            # Aggiorniamo working_wallet
            working_wallet = net_proceed
            # Registriamo il trade in un'unica riga
            operations.append(
                {
                    "Buy_Time": buy_time,
                    "Buy_Price": buy_price,
                    "Sell_Time": sell_time,
                    "Sell_Price": sell_price,
                    "Quantity": quantity,
                    "Profit": profit,
                    "Wallet_After": working_wallet,
                }
            )
            # Resettiamo lo stato
            holding = False
            quantity = 0.0
    return operations


def simulate_trading_with_commisions_multiple_buy(
    buy_signals: list, sell_signals: list, wallet: float = 100, fee_percent: float = 0.1
):
    operations = []
    holding = False  # Flag che indica se stiamo detenendo l'asset
    quantity = 0.0  # Quantità dell'asset comprata
    total_buy = 0
    working_wallet = wallet  # Capitale di partenza (USDT/USDC)
    # Converto fee_percent in forma decimale (es. 1% -> 0.01)
    fee_decimal = fee_percent / 100.0
    buy_percentage = 0.6
    # Per semplicità, assumiamo che numero di buy_signals e sell_signals
    # siano (in media) abbinati, usando lo stesso indice i in parallelo.
    s = 0
    b = 0
    while b < len(buy_signals):
        # for b in range(len(buy_signals)):
        # if i < len(buy_signals):
        buy_time, buy_price = buy_signals[b]
        # if working_wallet > 0:
        # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
        # utilizzo metà del working wallet
        total_buy = working_wallet * buy_percentage
        net_invested = (working_wallet * buy_percentage) * (1 - fee_decimal)
        # quantità di crypto ottenuta
        quantity = net_invested / buy_price
        working_wallet -= working_wallet * buy_percentage
        holding = True
        next = b + 1
        while next < len(buy_signals):
            if s < len(sell_signals):
                buy_time, buy_price = buy_signals[next]
                sell_time, sell_price = sell_signals[s]
                if buy_time < sell_time:
                    # Paghiamo la commissione in USDT/USDC: se abbiamo working_wallet,
                    # utilizzo metà del working wallet
                    total_buy += working_wallet * buy_percentage
                    net_invested = (working_wallet * buy_percentage) * (1 - fee_decimal)
                    # quantità di crypto ottenuta
                    quantity += net_invested / buy_price
                    working_wallet -= working_wallet * buy_percentage
                    b = next
                else:
                    break
            else:
                break
            next += 1
        b += 1

        if holding and s < len(sell_signals):
            sell_time, sell_price = sell_signals[s]
            # mean_cost = total_buy / quantity
            # Ricaviamo USDT vendendo la quantity di crypto
            gross_proceed = quantity * sell_price
            # Applichiamo la commissione di vendita
            # commissions = gross_proceed * fee_decimal
            net_proceed = gross_proceed * (1 - fee_decimal)
            # Calcoliamo il profit: differenza fra l'importo netto incassato e l'importo speso in fase di BUY
            # cost_in_usd = (quantity * buy_price) * (1 + fee_decimal)  # spesa inziale
            # profit = net_proceed - cost_in_usd
            profit = net_proceed - total_buy
            # Aggiorniamo working_wallet
            working_wallet += net_proceed
            # Registriamo il trade in un'unica riga
            operations.append(
                {
                    "Buy_Time": buy_time,
                    "Buy_Price": buy_price,
                    "Sell_Time": sell_time,
                    "Sell_Price": sell_price,
                    "Quantity": quantity,
                    "Profit": profit,
                    "Wallet_After": working_wallet,
                }
            )
            # Resettiamo lo stato
            holding = False
            quantity = 0.0
            s += 1

    return operations
