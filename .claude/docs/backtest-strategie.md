# Backtest delle strategie a indicatori — 3.129 configurazioni su nove anni

Misure prodotte da `scripts/strategy_sweep.py`, `scripts/sweep_report.py` e
`scripts/strategy_focus.py`; le tabelle complete stanno in `reports/`. Ogni numero qui viene da
quelle tabelle: nessuna stima, nessun risultato riportato a memoria.

## Il risultato in una riga

Delle **3.129 configurazioni** provate — tutte le strategie a indicatori del simulatore, ciascuna
su una griglia dei propri parametri, su **BTC a 15 minuti dal 2017-01-01 al 2026-08-24** —
**il 14,9% chiude in utile, il 45,2% perde piu' del 90% del capitale, e cinque (lo 0,2%) battono
il possesso passivo**, che nello stesso periodo ha fatto **+7.947%**. La mediana e' **−87,3%**.

Le cinque che battono il possesso passivo non sono cinque strategie diverse: sono tre
configurazioni di "Close Buy/Sell Limits" con 3-4 operazioni l'anno, una di "Close ATR" con 5,9, e
una di "ATR Bands". E nessuna di loro sopravvive alla verifica fuori campione (§5).

## 1. Cosa e' stato misurato, e come

| | |
|---|---|
| Mercato | BTC/USD, candele a 5 minuti aggregate agli intervalli richiesti |
| Periodo | 2017-01-01 → 2026-08-24, 9,65 anni, **338.114 barre** a 15 minuti |
| Fonte | dataset pubblico Bitstamp a un minuto ([`ff137/bitstamp-btcusd-minute-data`](https://github.com/ff137/bitstamp-btcusd-minute-data)), importato con `scripts/import_bitstamp.py` |
| Capitale | 100, sempre reinvestito per intero, come in `pnl.simulate_trading_with_commisions` |
| Commissione | 0,1% per gamba (il default della pagina); la sensibilita' e' in §6 |
| Strategie | le 10 funzioni di `trading/strategies.py` raggiungibili dal dispatch di `trading_analysis` |
| Metriche | rendimento composto, CAGR, Sharpe e drawdown su equity **segnata a mercato barra per barra**, win rate, profit factor, esposizione, commissioni pagate |

**Perche' Bitstamp e non Binance.** `data/klines.py` prende le candele dai dump di
`data.binance.vision`. In questa sessione quell'host e' bloccato dalla policy di rete (403 sul
CONNECT, come `api.binance.com`), quindi lo store e' stato riempito da una fonte alternativa con
la stessa struttura. BTC/USD su Bitstamp non e' BTCUSDC su Binance: i prezzi differiscono di
frazioni di punto e il listino commissioni e' diverso. Va bene per misurare **il comportamento
delle strategie**; non e' la replica al centesimo di un conto Binance.

**Le strategie non sono state riscritte.** Lo sweep chiama le funzioni di `trading/strategies.py`
e il P&L di `trading/pnl.py` cosi' come stanno. L'unica parte reimplementata e' il calcolo degli
indicatori, per poter riusare le colonne fra configurazioni invece di ricalcolarle (il PSAR da
solo costa 26 secondi su 338.000 barre e non dipende da nessun parametro spazzolato):
`tests/test_strategy_sweep.py` verifica colonna per colonna che produca **la stessa tabella** di
`indicators.add_technical_indicator`.

**Il metro di paragone.** Il periodo non e' neutro: BTC e' passato da ~1.000 a ~77.500 dollari.
Una strategia long-only che sta fuori dal mercato meta' del tempo parte in svantaggio, e va
giudicata anche sul drawdown, non solo sul rendimento.

| periodo | possesso passivo | drawdown |
|---|---|---|
| 2017-2026 (intero) | +7.947% | 84,0% |
| 2017-2021 (stima) | +4.744% | 84,0% |
| 2022-2026 (verifica) | +67,0% | 67,5% |
| 2019-2026 (walk-forward) | +1.970% | 77,3% |

## 2. Panoramica per strategia

Migliore e mediana sono calcolate sulle configurazioni con almeno 30 operazioni in nove anni: sotto
quella soglia non si sta misurando una strategia ma una singola posizione tenuta per anni.

| strategia | config. | migliore | mediana | in utile | batte B&H | Sharpe migliore | DD del migliore |
|---|---:|---:|---:|---:|---:|---:|---:|
| ATR Bands *(non nel menu)* | 168 | **+20.020%** | −70,2% | 33,3% | 0,6% | 1,36 | 63,7% |
| Close Buy/Sell Limits | 1.728 | +13.230% | −90,4% | 9,4% | 0,2% | 1,11 | 83,8% |
| Close ATR | 504 | +8.778% | −96,7% | 4,6% | 0,2% | 1,25 | 63,2% |
| Close Bullish EMA | 420 | +3.392% | −63,7% | 22,6% | 0% | 0,88 | 83,0% |
| Close EMA Crossover | 7 | +2.834% | −64,2% | 42,9% | 0% | 0,97 | 54,0% |
| ATR Live Trade | 18 | +1.056% | −99,6% | 5,9% | 0% | 0,74 | 85,5% |
| Supertrend *(non nel menu)* | 126 | +354% | −12,8% | 45,2% | 0% | 0,75 | 55,0% |
| TP/SL with ATR | 126 | +120% | −97,3% | 12,2% | 0% | 0,49 | 35,2% |
| Trend Zones | 6 | **−100%** | −100% | 0% | 0% | −1,41 | 100% |
| Close RSI Reverse *(non nel menu)* | 25 | **−100%** | −100% | 0% | 0% | −8,06 | 100% |
| Green Candles | 1 | **−100%** | −100% | 0% | 0% | −5,63 | 100% |

Tre strategie perdono **tutto** il capitale in ogni configurazione provata. Non e' un caso
particolare: operano rispettivamente 2.080, 2.916 e 1.459 volte l'anno (§3).

### Con i parametri di partenza della pagina

E' il caso che conta di piu', perche' e' quello che si vede aprendo il simulatore: ATR 5 / 1,6,
EMA 10-50-200, RSI 12, limiti 25/75, una condizione, stop loss disattivato.

| strategia | rendimento | operazioni/anno | win rate | profit factor |
|---|---:|---:|---:|---:|
| ATR Bands *(non selezionabile)* | +1.331% | 494 | 65,8% | 1,01 |
| Close Bullish EMA | −54,6% | 94 | 68,6% | 0,93 |
| Supertrend *(non selezionabile)* | −91,6% | 88 | 36,1% | 0,80 |
| Close ATR | −97,6% | 284 | 60,7% | 0,95 |
| Close Buy/Sell Limits | −98,2% | 291 | 60,8% | 0,95 |
| TP/SL with ATR | −99,9% | 392 | 50,4% | 0,88 |
| Green Candles | −100% | 1.459 | 27,2% | 0,72 |

**Con i valori di partenza, ogni strategia raggiungibile dal menu perde denaro sul lungo periodo.**
Si noti il win rate: 60-69% di operazioni in utile e profit factor sotto 1. Le vincite ci sono, ma
sono piu' piccole delle perdite di quanto le commissioni permettano.

## 3. La frequenza operativa spiega quasi tutto

Tutte le 3.129 configurazioni, di tutte le strategie, raggruppate per numero di operazioni annue:

| operazioni/anno | config. | mediana | in utile | trade medio | commissioni pagate | drawdown mediano |
|---|---:|---:|---:|---:|---:|---:|
| < 10 | 313 | −2,5% | 39,9% | −0,18% | 5% | 61,4% |
| 10-30 | 265 | **+19,5%** | **53,6%** | +0,38% | 48% | 79,8% |
| 30-100 | 595 | −64,9% | 22,5% | −0,07% | 101% | 89,8% |
| 100-300 | 1.303 | −90,5% | 4,2% | −0,09% | 217% | 95,8% |
| 300-1.000 | 576 | −99,6% | 1,6% | −0,11% | 315% | 99,9% |
| > 1.000 | 57 | −100% | 0% | −0,20% | 96% | 100% |

Il numero di operazioni predice il rendimento meglio di qualunque parametro, e lo predice al
contrario. La colonna "trade medio" dice perche': **il margine lordo medio per operazione, sui
timeframe brevi, e' dello stesso ordine del costo di transazione** (0,2% andata e ritorno). Una
strategia che opera 300 volte l'anno paga il 60% del capitale iniziale in commissioni ogni anno, e
deve guadagnarlo prima di guadagnare qualcosa.

La colonna "commissioni pagate" e' cumulata sui nove anni e rapportata al capitale **iniziale**:
oltre il 100% significa che le commissioni versate valgono piu' di tutto il capitale di partenza.

## 4. Sensibilita' ai parametri

L'escursione mediana e' la differenza fra il valore migliore e il peggiore di un parametro, tenuti
fermi tutti gli altri, mediata su tutte le loro combinazioni. E' la risposta a "quanto conta
questo widget".

| griglia | parametro piu' influente | escursione mediana |
|---|---|---:|
| Supertrend | `atr_multiplier` | 187,6 punti |
| ATR Bands | `atr_multiplier` | 169,0 punti |
| TP/SL with ATR | `atr_multiplier` | 140,6 punti |
| Close ATR | `atr_multiplier` | 93,4 punti |
| Close Bullish EMA | `rsi_window` | 83,9 punti |
| Close Buy/Sell Limits | `rsi_sell_limit` | 25,7 punti |

**`atr_multiplier` domina ovunque compaia**, e sempre nella stessa direzione: bande larghe, poche
operazioni, meno perdite.

| `atr_multiplier` | Close ATR (mediana) | ATR Bands (mediana) | TP/SL (mediana) | operazioni/anno (ATR Bands) |
|---|---:|---:|---:|---:|
| 0,8 | −100% | −100% | −100% | 991 |
| 1,2 | −99,8% | −99,8% | −100% | 624 |
| **1,6 (default)** | −98,6% | −95,2% | −99,7% | 414 |
| 2,0 | −95,5% | −67,6% | −96,9% | 280 |
| 2,5 | −85,5% | +16,6% | −72,0% | 205 |
| 3,0 | −56,6% | +45,0% | −16,6% | 133 |
| 4,0 | −7,7% | +37,0% | +28,8% | 43 |

**Il default 1,6 e' nella parte peggiore dell'intervallo per tutte e tre le strategie che lo
usano.** Il moltiplicatore piu' alto provato (4,0) e' il migliore o il secondo migliore ovunque, il
che dice anche che l'ottimo potrebbe stare oltre il limite della griglia.

Gli altri parametri, in breve:

- **`atr_window`** conta poco al confronto (5,2 punti su Close ATR): sposta il rumore della banda,
  non la sua larghezza relativa.
- **`num_cond`** in "Close Buy/Sell Limits" e' la differenza fra "RSI **o** banda" (1) e "RSI **e**
  banda" (2): mediana −96,2% contro −69,1%, in utile 0,6% contro 23,8%. Due condizioni riducono le
  operazioni da 279 a 53 l'anno. Di nuovo la frequenza.
- **`rsi_sell_limit`** e' monotono: 60 → mediana −94,1%, 85 → −68,0%, con la quota in utile che
  passa dallo 0,7% al 35,1%. Uscire tardi e' sistematicamente meglio che uscire presto — su un
  mercato che ha fatto +7.947%, dove ogni uscita e' una scommessa contro il trend.
- **`stop_loss`** non aiuta mai: su ATR Bands, mediana −86,5% con stop al 5% contro −50,5% senza.
  Lo stop chiude in perdita posizioni che sarebbero tornate, e su Close Buy/Sell Limits **non ha
  alcun effetto**, perche' il codice che lo applicherebbe e' commentato (§7).
- **le terne EMA** sono il parametro decisivo di "Close EMA Crossover": 50/100/200 rende +2.834%,
  10/50/200 +1.441%, 8/13/21 **−100%**. Sette valori, quattro dei quali perdono tutto: non e' un
  parametro da lasciare a un default.

## 5. Fuori campione: qui cade tutto

Le sezioni precedenti guardano il periodo intero, cioe' scelgono i parametri sapendo gia' come e'
andata. La verifica onesta e' scegliere sui primi anni e misurare sui successivi.

**Scelta su 2017-2021, resa su 2022-2026** (possesso passivo nello stesso periodo: **+67,0%**):

| griglia | resa in stima | resa in verifica | mediana delle prime 10 in stima | prime 10 in utile | migliore possibile in verifica | ρ Spearman stima/verifica |
|---|---:|---:|---:|---:|---:|---:|
| Close Buy/Sell Limits | +4.462% | **+192,2%** | −6,3% | 40% | +197,4% | 0,47 |
| Close ATR | +4.257% | **+103,8%** | −48,1% | 20% | +107,1% | 0,78 |
| ATR Bands | +10.327% | **−86,2%** | +6,6% | 50% | +169,6% | 0,65 |
| Close Bullish EMA | +2.775% | −8,7% | +20,4% | 70% | +67,1% | 0,49 |
| Supertrend | +662% | −49,8% | −50,8% | 10% | +58,2% | 0,23 |
| TP/SL with ATR | +334% | −68,0% | −73,2% | 0% | +34,4% | 0,90 |
| Close EMA Crossover | +6.032% | −74,9% | −83,0% | 14% | +0,9% | 0,86 |

Due configurazioni su sette trasferiscono, e battono anche il possesso passivo. **Ma la colonna
che conta e' la quinta**: la mediana delle prime dieci in stima e' negativa in cinque casi su
sette. La prima classificata di "Close Buy/Sell Limits" rende +192% in verifica mentre le sue nove
vicine di classifica fanno mediana −6,3%: non e' una regione di parametri che funziona, e' una
riga fortunata. Su "ATR Bands", la migliore in stima — quella da +10.327%, la piu' redditizia di
tutto lo studio sul periodo intero — perde **l'86%** in verifica.

**Walk-forward.** Piu' realistico ancora: a fine di ogni anno si riottimizza sui soli anni gia'
visti e si tiene quella configurazione per l'anno seguente.

| griglia | 2019-2026 | anni in utile | anno peggiore | cambi di configurazione |
|---|---:|---:|---:|---:|
| ATR Bands | +1.111% | 75% | −33,7% | 3 |
| Close ATR | **+914%** | 87,5% | −5,6% | 2 |
| Close Buy/Sell Limits | +366% | 50% | −50,2% | 3 |
| Close Bullish EMA | +354% | 62,5% | −65,4% | 4 |
| Close EMA Crossover | +126% | 75% | −66,0% | 2 |
| Supertrend | −43,9% | 50% | −37,1% | 2 |
| TP/SL with ATR | −76,1% | 37,5% | −55,1% | 3 |
| Trend Zones | −99,9% | 0% | −85,8% | 1 |
| Green Candles | −100% | 0% | −97,4% | 1 |

Nessuna arriva al **+1.970%** del possesso passivo sullo stesso arco. Close ATR ci si avvicina di
piu' con molta meno sofferenza: 87,5% di anni positivi, anno peggiore −5,6%, contro un possesso
passivo che nel 2022 ha perso il 64,3% e ha attraversato un drawdown del 77,3%. Su base
rischio-rendimento e' l'unico risultato di questo studio che meriti un secondo sguardo — con
l'avvertenza che sono comunque 4-9 operazioni l'anno decise da due parametri riottimizzati due
volte in otto anni, cioe' un campione minuscolo.

## 6. Commissioni: dove sta davvero il margine

Le stesse configurazioni, rieseguite variando solo la commissione (`reports/commissioni.csv`):

| griglia (config. migliore) | oper./anno | 0% | 0,04% | 0,075% | 0,1% | 0,2% |
|---|---:|---:|---:|---:|---:|---:|
| ATR Bands | 141 | +307.578% | +103.283% | +39.697% | +20.020% | +1.212% |
| Trend Zones | 681 | **+10.672%** | −43,7% | −99,4% | **−100%** | −100% |
| Close EMA Crossover | 51 | +7.737% | +5.191% | +3.651% | +2.834% | +997% |
| Close Buy/Sell Limits | 3,9 | +14.283% | +13.853% | +13.486% | +13.230% | +12.253% |
| Close ATR | 5,9 | +9.850% | +9.407% | +9.035% | +8.778% | +7.820% |
| Close Bullish EMA | 13,9 | +4.466% | +4.002% | +3.635% | +3.392% | +2.570% |
| Supertrend | 28 | +681% | +529% | +420% | +354% | +164% |
| TP/SL with ATR | 29 | +284% | +207% | +153% | +120% | +26% |
| Green Candles | 1.459 | −10,1% | −100% | −100% | −100% | −100% |
| Close RSI Reverse | 4.075 | −76,2% | −100% | −100% | −100% | −100% |

Tre gruppi distinti:

1. **Chi ha un margine lordo e lo perde tutto in commissioni**: "Trend Zones" guadagna il
   10.672% a costo zero e perde il 100% a 0,04%. "ATR Bands" divide per 250 il proprio risultato
   passando da 0% a 0,2%. Sono strategie il cui segnale contiene qualcosa, ma non abbastanza da
   pagare l'esecuzione.
2. **Chi non ha margine nemmeno lordo**: "Green Candles" (−10% a commissioni zero) e "Close RSI
   Reverse" (−76%) perdono anche in un mondo senza costi. Nessuna taratura le salva.
3. **Chi e' insensibile perche' opera poco**: Close ATR e Close Buy/Sell Limits nelle loro
   configurazioni migliori cambiano meno del 20% fra commissione nulla e 0,2%. E' l'altra faccia
   della §3.

## 7. Cambiare intervallo: la stessa regola, un altro mestiere

Le configurazioni migliori a 15 minuti, rieseguite **senza ritoccare nulla** sugli altri intervalli
del menu (`reports/intervalli.csv`):

| griglia | 5m | 15m | 30m | 1h | 4h | 1d |
|---|---:|---:|---:|---:|---:|---:|
| ATR Bands | +5.629% | **+20.020%** | +9.143% | +977% | +1.938% | +248% |
| Close ATR | +286% | **+8.778%** | +1.124% | +125% | +34% | 0% |
| Close Buy/Sell Limits | +5.251% | **+13.230%** | +843% | +2.278% | 0% | +1.494% |
| Close EMA Crossover | +461% | +2.834% | +3.986% | +2.855% | **+6.740%** | +628% |
| Close Bullish EMA | +4.028% | +3.392% | +2.838% | +3.384% | +3.646% | **+4.142%** |
| Supertrend | −88% | +354% | +344% | +1.624% | **+2.073%** | +175% |
| TP/SL with ATR | −96% | **+120%** | +44% | +79% | −18% | 0% |
| Trend Zones | −100% | −100% | −42% | +810% | **+9.378%** | +4.123% |
| Green Candles | −100% | −100% | −100% | −100% | −55% | **+1.765%** |
| Close RSI Reverse | −100% | −100% | −100% | −100% | −42% | **+3.284%** |

Due letture opposte, stessa causa.

**Chi era stato scelto a 15 minuti perde quasi tutto altrove.** Close ATR passa da +8.778% a
+125% a un'ora e a zero sul giorno; Close Buy/Sell Limits da +13.230% a zero sulle 4 ore. Un
parametro scelto su un timeframe non e' un parametro: e' un parametro **e** un timeframe.

**Chi perdeva tutto a 15 minuti diventa il migliore sul giorno.** "Green Candles" — comprare dopo
una candela verde che supera il massimo precedente — vale −100% a 15 minuti e **+1.765%** sul
giorno. "Close RSI Reverse" va da −100% a **+3.284%**. "Trend Zones" da −100% a **+9.378%** sulle
4 ore, cioe' **piu' del possesso passivo**. La regola non e' cambiata: sono cambiate le operazioni
all'anno, da 1.459 a 15, da 4.159 a 36, da 681 a 32. E' la §3 vista da un'altra angolazione, ed e'
la conferma piu' netta che il problema di queste strategie non e' il segnale ma la sua frequenza.

Il migliore assoluto di ogni intervallo, fra le configurazioni riesaminate:

| intervallo | strategia | operazioni/anno | rendimento | Sharpe | drawdown |
|---|---|---:|---:|---:|---:|
| 5m | ATR Bands | 506 | +5.629% | 1,1 | 90,9% |
| 15m | ATR Bands | 141 | +20.020% | 1,4 | 63,7% |
| 30m | ATR Bands | 65 | +9.143% | 1,2 | 77,2% |
| 1h | Close EMA Crossover | 27 | +7.186% | 1,2 | 60,4% |
| 4h | Close EMA Crossover | 6,8 | +11.524% | 1,2 | 63,7% |
| 1d | Trend Zones | 6,7 | +8.303% | 1,2 | 75,2% |

Sei intervalli, sei vincitori diversi, tutti con Sharpe fra 1,1 e 1,4 e drawdown fra il 60% e il
91%: nessuno dei sei e' distinguibile dagli altri, e nessuno lo e' davvero dal possesso passivo
(+7.947%, drawdown 84%). "ATR Live Trade" non compare in questa tabella: simula trenta sotto-passi
per candela e sulle barre a 5 minuti costerebbe da solo piu' di venti ore.

## 8. Difetti trovati nel codice, misurando

Nessuno di questi e' stato corretto in questo lavoro: sono osservazioni, con la misura accanto.

1. **La voce di menu `"Supetrend"` non esegue niente.** `config.STRATEGIES` scrive `"Supetrend"`,
   il dispatch di `trading_analysis` confronta con `"Supertrend"`. Selezionandola non si producono
   segnali e la pagina mostra un backtest vuoto. La funzione esiste e funziona: nella griglia rende
   fino a +354%, con la mediana meno negativa di tutte (−12,8%).
2. **`"ATR Bands"` non e' nel menu**, ma e' la strategia con il risultato migliore dello studio
   (+20.020% nella configurazione ottima, +1.331% gia' con i parametri di partenza). Come
   `"Close RSI Reverse"`, ha un ramo nel dispatch e nessuna voce che lo raggiunga.
3. **Lo stop loss di "Close Buy/Sell Limits" non esiste.** `buy_sell_limits_close_simulation`
   accetta `stop_loss_percent` e ha le tre righe che lo userebbero commentate: il widget "Stop
   Loss %" per quella strategia e' inerte. Per Close ATR e ATR Bands invece funziona, e peggiora
   sistematicamente il risultato.
4. **"Trend Zones" confronta una media con se stessa.** La condizione e' `EMA20 > EMA200`, ma
   `add_technical_indicator` costruisce `EMA200` come EMA **dell'apertura** con la **stessa
   finestra** di `EMA20` (`ema_window`, default 10), non a 200 periodi. Le due serie differiscono
   solo per apertura contro chiusura, quindi si incrociano di continuo: 2.080 operazioni l'anno
   con `ema_window=10`, e −100% in tutte e sei le configurazioni. A commissioni zero la stessa
   strategia renderebbe +10.672% (§6): il segnale c'e', e' la frequenza che lo divora.
5. **Le strategie che entrano sul prezzo di banda assumono un'esecuzione ideale.**
   `atr_buy_sell_simulation` compra a `Lower_Band` quando il minimo della candela la tocca, e
   `tp_sl_simulation`/`supertrend_simulation` fanno lo stesso con i loro livelli. E' un ordine
   limite riempito esattamente al prezzo, senza slippage ne' code. Il trade medio della migliore
   ATR Bands vale **+0,46%**: uno slippage di pochi punti base per gamba lo cancella. Le strategie
   `close_*`, che usano la chiusura, non hanno questo problema.
6. **L'ultima posizione aperta non entra nel conto.** `simulate_trading_with_commisions` accoppia
   i segnali per indice: se alla fine del periodo si e' dentro, quell'operazione non e' registrata.
   Lo sweep eredita il comportamento della pagina; e' un motivo in piu' per diffidare delle
   configurazioni con pochissime operazioni.

## 9. Limiti di questa misura

- **Un solo mercato.** BTC/USD. Le conclusioni sulla frequenza e sulle commissioni sono
  strutturali e difficilmente cambiano altrove, ma i valori ottimi dei parametri sono di questo
  mercato e di questo periodo.
- **Un solo verso.** Tutte le strategie sono long-only su un asset che nel periodo ha fatto
  +7.947%: stare fuori dal mercato costa, e il confronto e' severo per costruzione.
- **Niente slippage, niente book.** Le commissioni ci sono, il resto no (§7.5).
- **I wick estremi sono compressi** da `clip_wicks` in lettura, come per tutto il resto del
  progetto.
- **Il campione delle configurazioni migliori e' piccolo.** Le due che passano la verifica fuori
  campione fanno 4-6 operazioni l'anno: 38 e 57 operazioni in totale. Con numeri cosi', la
  differenza fra "strategia" e "fortuna" non e' misurabile con i dati disponibili.

## 10. Come riprodurre

```bash
git clone https://github.com/ff137/bitstamp-btcusd-minute-data /percorso/dati
.venv312/bin/python -m scripts.import_bitstamp --source /percorso/dati
.venv312/bin/python -m scripts.strategy_sweep --all --interval 15m --workers 4   # ~45 minuti
.venv312/bin/python -m scripts.sweep_report --interval 15m                       # tabelle in reports/
.venv312/bin/python -m scripts.strategy_focus --top 3                            # commissioni e intervalli
```

Con lo store Binance gia' popolato (`python -m cryptofarm.data.klines --update`) basta cambiare
`SYMBOL` in `scripts/strategy_sweep.py` per rifare tutto su BTCUSDT.
