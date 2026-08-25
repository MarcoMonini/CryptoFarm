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
            # Profit: differenza fra l'incasso netto della vendita e quanto speso in fase di BUY,
            # commissioni comprese.
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


# -------------------------------------------------------------------------------------------------
# Posizioni con verso: long, flat, short
# -------------------------------------------------------------------------------------------------

# Costo di mantenimento giornaliero della posizione, in percentuale del nozionale. Su un conto
# spot non esiste; su un perpetuo e' il funding, che su Binance oscilla attorno allo 0,01% ogni
# otto ore -- 0,03% al giorno -- e che qui viene addebitato **a entrambi i versi**. Nella realta'
# il funding e' un trasferimento: chi sta dalla parte giusta lo incassa. Addebitarlo sempre e' la
# scelta prudente, ed evita di far dipendere il risultato di una strategia da una previsione sul
# segno del funding.
CARRY_DAILY_PERCENT = 0.03


def simulate_positions(
    events: list,
    wallet: float = 100,
    fee_percent: float = 0.1,
    carry_daily_percent: float = CARRY_DAILY_PERCENT,
    leverage: float = 1.0,
) -> list:
    """Da una sequenza di cambi di posizione alle operazioni chiuse, con il verso.

    `events` e' una lista di `(timestamp, prezzo, obiettivo)` con obiettivo in `{+1, 0, -1}`:
    lungo, fuori, corto. Ogni elemento e' un **cambio** di stato, non uno stato ripetuto.

    Perche' non bastava `simulate_trading_with_commisions`: quella accoppia due liste separate per
    indice e conosce un solo verso. Una strategia che inverte la posizione -- da lungo a corto
    senza passare per il flat -- non e' rappresentabile in quel formato, e la vendita allo scoperto
    nemmeno.

    Convenzioni, tutte a leva 1:

    - il nozionale di ogni operazione e' il capitale disponibile moltiplicato per `leverage`
      (a leva 1, tutto il capitale, come nel simulatore storico). La leva serve a confrontare
      strategie con drawdown molto diversi: una che rende meta' del possesso passivo con un quarto
      del drawdown, portata a leva due, rende quanto il possesso passivo con meta' del suo rischio.
      Commissioni e mantenimento si pagano sul nozionale, quindi crescono con la leva;
    - la commissione si paga su entrambe le gambe, calcolata sul nozionale scambiato (all'uscita il
      nozionale e' cambiato con il prezzo);
    - il costo di mantenimento e' proporzionale ai giorni di posizione aperta;
    - se il capitale arriva a zero la simulazione si ferma: a leva 1 uno short viene azzerato
      quando il prezzo raddoppia, a leva 3 basta un movimento contrario di un terzo. Il conto non
      va sotto zero e la simulazione non prosegue: e' la liquidazione.

    La liquidazione qui e' valutata **alla chiusura dell'operazione**, non barra per barra: una
    posizione che va sotto zero a meta' corsa e torna sopra non viene liquidata come lo sarebbe
    davvero. E' un limite noto, e conta solo a leve alte.
    """
    operations = []
    fee = fee_percent / 100.0
    carry = carry_daily_percent / 100.0
    position = 0
    entry_price = 0.0
    entry_time = None
    notional = 0.0

    for timestamp, price, target in events:
        price = float(price)
        if target == position:
            continue
        if position != 0:
            days = (timestamp - entry_time).total_seconds() / 86400.0
            direction = 1 if position > 0 else -1
            gross = notional * direction * (price - entry_price) / entry_price
            fees = fee * notional * (1 + price / entry_price)
            carrying = carry * notional * days
            profit = gross - fees - carrying
            wallet = max(0.0, wallet + profit)
            operations.append(
                {
                    "Side": "long" if direction > 0 else "short",
                    "Buy_Time": entry_time,
                    "Buy_Price": entry_price,
                    "Sell_Time": timestamp,
                    "Sell_Price": price,
                    "Quantity": notional / entry_price,
                    "Profit": profit,
                    "Wallet_After": wallet,
                }
            )
            position = 0
            if wallet <= 0:
                break
        if target != 0:
            position = target
            entry_price = price
            entry_time = timestamp
            notional = wallet * leverage
    return operations
