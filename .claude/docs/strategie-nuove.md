# Strategie nuove, verso corto, e il dataset giusto

Seguito di [`backtest-strategie.md`](backtest-strategie.md), che misurava le strategie del
simulatore su nove anni. Qui: le quattro correzioni applicate al codice, la scelta di un dataset
diverso e il perche', la lettura da operatore dei punti di forza e di debolezza delle strategie
storiche, cinque strategie nuove costruite su quella lettura, e un motore che sa stare anche
**corto**. Tabelle complete in `reports/` (file `lab_*`).

## Il risultato in tre righe

Sul ciclo 2021-2026 di BTC, **a scala giornaliera e con le correzioni applicate, le strategie
storiche non sono affatto perdenti**: mediana +20% e 70% di configurazioni in utile per "Close
ATR", contro il −96% che davano sui 15 minuti. Il disastro misurato prima era in buona parte un
artefatto della frequenza operativa e del timeframe, non delle regole.

**Le cinque strategie nuove — rottura di canale, squeeze, rientro in trend, Ichimoku, e il ritorno
alla media con filtro di regime — non battono il possesso passivo**, ne' in campione ne' fuori.
Battono pero' il suo **rischio**: 22% di drawdown contro 76%, che a leva 2 diventa +196% contro
+166% con la meta' del drawdown.

**Il verso corto, su questo asset e in questo periodo, toglie invece di aggiungere**: la mediana
peggiora in tutte e cinque le strategie e il contributo del lato corto e' negativo. Paga solo nel
2022, l'unico anno davvero ribassista.

---

## 1. Le quattro correzioni

| difetto | correzione | effetto misurato |
|---|---|---|
| la voce di menu `"Supetrend"` non corrispondeva alla stringa del dispatch (`"Supertrend"`) | corretta la stringa in `config.STRATEGIES` | la voce esegue: su BTC 2021-2026 a 4h la migliore configurazione rende **+450%** (Sharpe 1,01, drawdown 34%) |
| `"ATR Bands"` aveva un ramo nel dispatch e nessuna voce di menu | aggiunta la voce | selezionabile: **+678%** a 4h, la migliore delle storiche su questo periodo |
| lo stop loss di `buy_sell_limits_close_simulation` era commentato | ripristinato: stop fissato all'ingresso, uscita alla prima chiusura sotto | con il default 99% resta inerte (nessun golden cambia); ai valori operativi ora agisce |
| `EMA200` era l'EMA **dell'apertura** sulla finestra corta, e "Trend Zones" la confrontava con `EMA20`, cioe' una media con se stessa | eliminata la colonna, le tre funzioni che la leggevano ora usano `EMA100` (la media lunga vera) | vedi sotto |

**Trend Zones, prima e dopo** (BTC 2021-2026, commissione 0,05%):

| intervallo | finestra | prima: operazioni/anno | prima: rendimento | dopo: operazioni/anno | dopo: rendimento |
|---|---:|---:|---:|---:|---:|
| 15m | 10 | 3.603 | −100% | — | — |
| 4h | 10 | 202 | −21,9% | 10,6 | **+309,3%** |
| 4h | 20 | 135 | +30,6% | 8,5 | **+231,5%** |
| 1d | 20 | 21 | +166,1% | 1,2 | +189,4% |
| 1d | 50 | 12 | +78,5% | 0,5 | +204,1% |

Il golden master e' stato rigenerato: le uniche 17 voci cambiate sono `add_technical_indicator`
(una colonna in meno) e le tre funzioni che leggevano `EMA200`, sui quattro scenari. Nessun'altra
strategia si e' mossa.

## 2. Il dataset: perche' non piu' il 2017

La misura precedente usava 2017-2026 perche' era tutto lo storico disponibile. E' la scelta
sbagliata per decidere cosa fare adesso, e si puo' dimostrare invece che sostenerlo:

| periodo | possesso passivo | CAGR | Sharpe | drawdown |
|---|---:|---:|---:|---:|
| BTC 2017-2020 | +2.803% | 132,3% | 1,44 | 83,2% |
| **BTC 2021-2026** | **+166%** | **18,9%** | **0,59** | **76,5%** |
| ETH 2017-2019 | +1.479% | 151,3% | 1,37 | 93,8% |

Un mercato che cresce del 132% l'anno perdona qualunque errore di sistema; uno che cresce del 19%
no. E i parametri non passano da un regime all'altro: scegliendo la configurazione migliore sul
ciclo 2017-2020 e misurandola sul 2021-2026, delle cinque strategie nuove **quattro su cinque
finiscono in perdita** (da −73% a +21%, con il possesso passivo a +166%). Nella direzione opposta,
le stesse strategie scelte sul 2017-2018 e verificate sul 2019-2020 rendevano **+180%** con la
mediana delle prime cinque a +172%: nel ciclo vecchio il trend-following funzionava, in questo no.

**Il dataset usato qui e' quindi BTC/USD dal 2021-01-01 al 2026-08-24**, che contiene un ciclo
completo — massimo di novembre 2021, −64% nel 2022, ripresa 2023-2024, distribuzione 2025-2026 —
a 1h, 4h e 1d. La fonte e' la stessa del lavoro precedente (dump pubblico Bitstamp a un minuto),
con lo 0,03%-0,37% di barre piatte per anno in questo periodo, cioe' dati puliti.

**Cosa manca, e perche'.** L'ambiente di questa sessione ha egress bloccato verso qualunque
exchange (`data.binance.vision`, `api.binance.com`, Kraken, Coinbase, Bybit, Kucoin, Gate, MEXC),
verso Kaggle e verso ogni aggregatore (CoinGecko, CryptoCompare, Messari, DefiLlama): risponde 403
sul CONNECT. Restano raggiungibili solo GitHub e PyPI, e **nessun repository pubblico contiene
candele intraday recenti di SOL e BNB** -- quelli che le pubblicano si fermano al 2019 (Bitfinex)
o pubblicano su Kaggle (WISEPLAT). Il confronto multi-asset richiesto e' quindi rimasto a metà:
BTC sul ciclo recente ed ETH sul 2017-2019, che e' l'unico secondo mercato disponibile.

Non e' un limite del codice: `data/klines.py` scarica gia' BTCUSDT, ETHUSDT, SOLUSDT e BNBUSDT dai
dump di Binance, e ogni script qui accetta `--symbol`. Su una macchina con rete aperta:

```bash
python -m cryptofarm.data.klines --update --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT
for s in BTCUSDT ETHUSDT SOLUSDT BNBUSDT; do
  python -m scripts.strategy_lab --all --symbol $s --interval 1d --since 2021-01-01
  python -m scripts.lab_report --symbol $s --interval 1d
done
```

## 3. Le storiche, lette da operatore

Cosa fa ognuna, dove ha ragione e dove si rompe, con la misura accanto (BTC 2021-2026).

**Ritorno alla media sulle bande ATR** — *Close ATR, ATR Bands, Close Buy/Sell Limits, ATR Live
Trade*. Comprano quando il prezzo si allontana dalla media di `k` ATR e rivendono al ritorno.
*Forza*: nei mercati laterali il rientro verso la media e' il fenomeno statisticamente piu'
affidabile che esista, e infatti su barre giornaliere in questo ciclo sono le migliori in assoluto
(Close ATR +575%, Sharpe 1,25, drawdown 25%). *Debolezza*: comprano **ogni** minimo, anche il primo
di una discesa strutturale, e non hanno modo di distinguere un ritracciamento da un'inversione; su
15 minuti il margine per operazione (+0,08%) e' sotto il costo di andata e ritorno (0,10%-0,20%) e
il risultato mediano crolla a −96%.

**Seguire il trend con le medie** — *Close EMA Crossover, Trend Zones, Close Bullish EMA*.
*Forza*: nessuna previsione, si sta dentro finche' la struttura tiene; sul ciclo 2017-2020 erano la
famiglia vincente. *Debolezza*: in un mercato che oscilla senza direzione ogni incrocio e' un falso
segnale, e il 2021-2026 e' esattamente quel mercato; con i parametri di partenza (10/50/200) su
15 minuti perdevano tutto.

**Rottura con obiettivo e stop** — *TP/SL with ATR, Supertrend*. *Forza*: il rischio per operazione
e' definito prima di entrare, l'unica famiglia in cui lo sia. *Debolezza*: l'obiettivo simmetrico
allo stop (1:1) o a 1,618 taglia proprio i movimenti lunghi che pagano una strategia di rottura, e
il win rate necessario per andare in pari sale sopra il 50% al netto dei costi.

**Pattern di prezzo puri** — *Green Candles, Close RSI Reverse*. *Forza*: nessuna. *Debolezza*: a
15 minuti perdono **anche a commissioni zero** (−10% e −76%), il che dice che non c'e' segnale, non
che il costo lo mangia. A scala giornaliera tornano positive, ma per la stessa ragione per cui lo
diventa tutto: 15-25 operazioni l'anno invece di 1.500.

Il difetto trasversale, gia' misurato nel documento precedente: **nessuna sa in che regime si
trova**. Fra le colonne prodotte non c'e' un solo indicatore che dica se un trend esiste, se la
volatilita' e' compressa, se il volume conferma. Le cinque strategie nuove nascono da li'.

## 4. Le cinque strategie nuove

Tutte in `src/cryptofarm/trading/strategies_ls.py`, tutte con posizione a tre stati (+1 / 0 / −1),
tutte misurabili con e senza il verso corto. Gli indicatori nuovi stanno in
`indicators_extra.py`: ADX, canale di Donchian, Bollinger + Keltner (squeeze), StochRSI, MFI, OBV,
Ichimoku — tutti da `ta`, nessuno era usato dal progetto.

### 4.1 `donchian_breakout` — rottura di canale con filtro di forza
*Ipotesi*: si perde perche' si compra contro il trend; entrare **nella** direzione del movimento e
lasciar correre inverte il problema.
*Regole*: long alla chiusura sopra il massimo delle ultime `channel` barre (canale spostato di una
barra: nessun look-ahead), con `ADX ≥ adx_min` e prezzo dalla parte giusta della EMA lunga; short
speculare. Uscita a **chandelier stop**: massimo raggiunto meno `k·ATR`, che segue il prezzo.
*Indicatori nuovi*: Donchian, ADX.

### 4.2 `squeeze_breakout` — compressione e rilascio
*Ipotesi*: si opera troppo; la compressione di volatilita' seleziona pochi momenti l'anno per
costruzione, senza filtri arbitrari.
*Regole*: quando le bande di Bollinger rientrano dentro il canale di Keltner il mercato e' in
*squeeze*; alla prima barra in cui lo squeeze si apre si entra nella direzione in cui il prezzo
sta rispetto alla media delle bande, con conferma opzionale della pendenza dell'OBV. Uscita a
trailing ATR.
*Indicatori nuovi*: Bollinger, Keltner, OBV.

### 4.3 `trend_pullback` — rientro dall'ipervenduto dentro un trend
*Ipotesi*: il ritorno alla media funziona, ma solo dalla parte del trend.
*Regole*: sopra la EMA lunga si compra quando lo StochRSI risale sopra la soglia di ipervenduto;
sotto la EMA lunga si vende allo scoperto sul rientro dall'ipercomprato. Stop fisso a `k·ATR`,
uscita in guadagno al ritorno dell'oscillatore in zona opposta. Con `regime_ema=0` il filtro si
spegne: e' l'ablazione che misura quanto vale.
*Indicatori nuovi*: StochRSI.

### 4.4 `ichimoku_trend` — il metro di paragone
*Ipotesi*: nessuna. E' un sistema di trend completo, gia' pronto e diffuso; se una strategia
costruita apposta non lo batte, non vale il lavoro che costa.
*Regole*: incrocio Tenkan/Kijun con il prezzo dalla parte giusta della nuvola (span gia' spostate
in avanti, come sul grafico); uscita all'incrocio opposto o alla rottura della Kijun.

### 4.5 `band_reversion_gated` — la combinazione
*Ipotesi*: "Close ATR" fallisce per il regime, non per l'idea. Stessa entrata, ma solo dove ha
senso.
*Regole*: entrata identica alle bande storiche (KAMA ± `k·ATR`) **solo quando `ADX < adx_max`**,
cioe' in assenza di trend; uscita al ritorno sulla KAMA o allo stop a `k·ATR`. Filtro di regime
opzionale sulla EMA lunga.
*Indicatori nuovi*: ADX sopra la struttura storica.

## 5. Il verso corto: come e' simulato, e quanto vale

`pnl.simulate_trading_with_commisions` accoppia due liste di segnali e conosce un solo verso:
l'inversione diretta da lungo a corto non e' rappresentabile. Il motore nuovo,
`pnl.simulate_positions`, prende una lista di **cambi di posizione** `(tempo, prezzo, obiettivo)`
con obiettivo in `{+1, 0, −1}` e produce le operazioni chiuse con il lato. Convenzioni:

- nozionale pari al capitale per `leverage` (default 1), commissione su entrambe le gambe calcolata
  sul nozionale scambiato;
- **costo di mantenimento** giornaliero (`carry`, default 0,03% al giorno) addebitato a entrambi i
  versi: e' il funding di un perpetuo, che su Binance oscilla intorno allo 0,01% ogni otto ore.
  Nella realta' e' un trasferimento e chi sta dalla parte giusta lo incassa; addebitarlo sempre e'
  la scelta prudente;
- capitale che tocca zero: simulazione ferma. E' la liquidazione, e a leva 3 basta un movimento
  contrario di un terzo.

**Quanto vale il verso corto** (BTC 2021-2026, 1h + 4h + 1d, stesse configurazioni con e senza):

| strategia | coppie | mediana solo long | mediana con short | dove lo short migliora | contributo mediano del lato corto | win rate short |
|---|---:|---:|---:|---:|---:|---:|
| donchian_breakout | 384 | −22,1% | −56,9% | 2,6% | −53,3% | 31,6% |
| squeeze_breakout | 162 | −41,6% | −71,9% | 6,8% | −71,7% | 29,6% |
| trend_pullback | 108 | −36,1% | −60,0% | 8,3% | −35,4% | 49,3% |
| ichimoku_trend | 18 | +15,2% | −25,1% | 5,6% | −56,9% | 29,6% |
| band_reversion_gated | 216 | −0,5% | −5,5% | 23,6% | −3,6% | **52,3%** |

La lettura non e' "lo short non funziona": e' che **su un asset con deriva positiva, e in un
periodo in cui l'unico anno ribassista e' il 2022, il lato corto paga il costo di stare dalla parte
sbagliata della deriva** per quattro anni su cinque. La sola eccezione e' il ritorno alla media
(`band_reversion_gated`), dove il corto ha win rate 52% e costa quasi niente: vendere un'estensione
sopra la media in un mercato laterale e' simmetrico al comprarne una sotto.

Chi volesse comunque il lato corto ha due strade misurabili con questi strumenti: attivarlo solo
quando la media lunga **scende** (non basta il prezzo sotto la media), oppure usarlo solo nelle
strategie di ritorno alla media.

## 6. I risultati

**Classifica su BTC 2021-2026, barre giornaliere, commissione 0,05%** (possesso passivo: +166%,
drawdown 76,5%, Sharpe 0,59):

| famiglia | strategia | migliore | Sharpe | drawdown | oper./anno | mediana della griglia | in utile |
|---|---|---:|---:|---:|---:|---:|---:|
| storica | Close ATR | +575% | 1,25 | 25,5% | 3,4 | +20,5% | 70,6% |
| storica | Close Buy/Sell Limits | +335% | 0,86 | 59,3% | 3,9 | +25,0% | 71,6% |
| storica | TP/SL with ATR | +288% | 0,87 | 54,3% | 2,8 | +87,6% | 82,1% |
| storica | ATR Bands *(ora nel menu)* | +212% | 0,67 | 61,4% | 5,5 | +33,1% | 78,2% |
| nuova | squeeze_breakout | +120% | 0,59 | 55,6% | 3,4 | −34,0% | 19,8% |
| nuova | ichimoku_trend | +106% | 0,58 | 32,8% | 7,3 | +11,7% | **75,0%** |
| nuova | trend_pullback | +89% | 0,50 | 37,4% | 26,8 | −24,2% | 29,2% |
| nuova | band_reversion_gated | +84% | **0,78** | **22,1%** | 4,4 | −7,0% | 43,6% |
| nuova | donchian_breakout | +63% | 0,45 | 42,2% | 3,2 | −25,4% | 15,6% |
| storica | Trend Zones *(corretta)* | +60% | 0,42 | 49,9% | 2,7 | +60,2% | 100% |

Le "migliori" sono massimi su griglie di dimensione molto diversa (1.728 configurazioni per Close
Buy/Sell Limits, 12 per Ichimoku): la colonna onesta e' la mediana, e li' Ichimoku long-only con il
75% di configurazioni in utile e' il piu' solido fra i nuovi.

**Fuori campione — scelta sul 2021-2023, resa sul 2024-2026** (possesso passivo: +46% poi +80%):

| famiglia | strategia | scelta in stima | resa in verifica | mediana delle prime 5 | ρ stima/verifica |
|---|---|---:|---:|---:|---:|
| storica | Close RSI Reverse | +99% | **+57,5%** | +46,5% | 0,58 |
| storica | ATR Live Trade | +118% | +47,2% | +14,7% | 0,61 |
| storica | Close Bullish EMA | +38% | +38,2% | +42,1% | 0,57 |
| storica | Supertrend | +80% | +35,1% | +35,1% | 0,02 |
| storica | Close ATR | +487% | +15,0% | +0,7% | 0,08 |
| nuova | band_reversion_gated | +65% | **+11,1%** | +9,4% | 0,52 |
| nuova | squeeze_breakout | +31% | −3,5% | −3,5% | 0,49 |
| nuova | ichimoku_trend | +161% | −21,3% | −2,5% | −0,36 |
| nuova | trend_pullback | +154% | −35,1% | −15,9% | 0,25 |
| nuova | donchian_breakout | +77% | −39,9% | −23,6% | 0,10 |

**Nessuna, di nessuna famiglia, batte il possesso passivo fuori campione.** Le strategie di ritorno
alla media trasferiscono meglio di quelle di trend, il che e' coerente con il regime: in un ciclo
senza una direzione netta, la scommessa sul rientro paga piu' della scommessa sulla continuazione.

**Le ablazioni** (stessi tre intervalli) dicono che gli indicatori nuovi servono, ma per la robustezza,
non per il picco:

| strategia | filtro spento | mediana senza | mediana con | operazioni/anno senza → con |
|---|---|---:|---:|---:|
| ichimoku_trend | conferma della nuvola | −20,7% | **+8,4%** | 60 → 23 |
| donchian_breakout | filtro di trend (EMA lunga) | −46,8% | −34,8% | 20,4 → 19,3 |
| donchian_breakout | filtro ADX | −40,3% | −41,5% | 23,9 → 18,8 |
| squeeze_breakout | conferma di volume (OBV) | −60,5% | −49,8% | 22,2 → 15,9 |
| trend_pullback | filtro di trend (EMA lunga) | −57,1% | −44,7% | 116 → 52 |
| band_reversion_gated | filtro di range (ADX) | −11,0% | −1,9% | 5,8 → 1,1 |
| band_reversion_gated | filtro di trend | −9,0% | 0,0% | 3,9 → 0,5 |

Ogni filtro migliora la mediana e riduce le operazioni; l'unico ininfluente e' l'ADX come soglia
minima nella rottura di canale, dove il canale largo fa gia' quel lavoro.

**Leva e costi.** Una strategia con un quarto del drawdown del possesso passivo non e' peggiore:
e' lo stesso rischio a una leva diversa.

| configurazione | leva 1 | leva 2 | leva 3 |
|---|---|---|---|
| `band_reversion_gated` 1d | +84% / DD 22% | **+196% / DD 41%** | +319% / DD 58% |
| `ichimoku_trend` 1d | +106% / DD 33% | +179% / DD 54% | +169% / DD 71% |
| possesso passivo | +166% / DD 76,5% | — | — |

A leva 2 il ritorno alla media con filtro di regime **batte il possesso passivo su entrambi gli
assi** (+196% contro +166%, drawdown 41% contro 76%). Vale in campione: fuori campione la stessa
configurazione rende +11% contro +80%. Sulla sensibilita' al costo, le strategie a 1d perdono il
15-30% del risultato passando da 0,02% a 0,10% per gamba; quelle a 4h ne perdono la meta' o piu'.

## 7. Cosa ne farei

1. **Scala giornaliera, non 15 minuti.** E' la conclusione piu' robusta di entrambi i documenti: la
   stessa regola cambia segno cambiando timeframe, e la direzione e' sempre la stessa.
2. **Ritorno alla media con filtro di regime, solo lungo, a leva 1,5-2.** E' l'unica combinazione
   che nelle misure batte il possesso passivo a parita' di rischio, ed e' anche l'unica delle nuove
   che trasferisce fuori campione con segno positivo.
3. **Niente short su BTC in un mercato senza tendenza ribassista confermata.** Il costo e'
   misurato, non teorico.
4. **Ichimoku long-only come riferimento**: il 75% delle sue configurazioni chiude in utile a 1d.
   Qualunque strategia nuova che non lo batta su quella metrica non merita di essere messa in
   produzione.
5. **Ripetere tutto su SOL e BNB prima di decidere.** Sono gli asset dove il ciclo 2021-2026 ha
   avuto la volatilita' piu' alta, e nessuna delle conclusioni qui e' stata verificata su di loro.

## 8. Limiti

- **I numeri di `donchian_breakout` e `squeeze_breakout` sono precedenti a una correzione dello
  stop a trailing, e vanno rifatti.** Lo stop in vigore durante una barra veniva costruito con il
  massimo e l'ATR di quella stessa barra, poi confrontato con il suo minimo: assumeva che dentro
  la barra l'estremo favorevole arrivasse per primo. Per singolo riempimento il bias e' a senso
  unico -- si usciva a un prezzo non ottenibile, +0,9% sullo scenario dei test perturbando il solo
  massimo della barra dell'uscita, +2,6% su una perturbazione piu' larga. Sul **netto di
  portafoglio** invece il segno non e' prevedibile, perche' lo stop gonfiato scattava anche prima
  del dovuto: sulla serie sintetica dei test la correzione porta `donchian_breakout` da −6,5% a
  −1,3% e `squeeze_breakout` da −2,6% a −2,2%, cioe' migliora. Su BTC 2021-2026 non e' stato
  possibile rimisurare (serve lo store di candele, assente nell'ambiente in cui la correzione e'
  stata fatta): rilanciare i comandi di §9 e rifare §6 per queste due righe. Le altre tre
  strategie non usano lo stop a trailing e non sono toccate.
- **Un asset e un ciclo.** Le conclusioni sul verso corto e sui regimi valgono per BTC 2021-2026.
- **Selezione.** Le colonne "migliore" sono massimi su griglie: vanno lette con la mediana accanto.
- **Esecuzione ideale.** Ingressi alla chiusura della barra, stop eseguiti al livello esatto, niente
  slippage ne' impatto. Sui gap di liquidazione crypto e' ottimistico.
- **Funding fisso.** 0,03% al giorno addebitato a entrambi i versi; nella realta' varia e cambia
  segno.
- **La liquidazione e' valutata alla chiusura dell'operazione**, non barra per barra: a leve alte
  sottostima il rischio di essere chiusi durante l'escursione.

## 9. Riprodurre

```bash
python -m scripts.strategy_lab --all --interval 1d --since 2021-01-01     # le nuove
python -m scripts.strategy_sweep --all --interval 1d --since 2021-01-01 \
    --fee 0.05 --suffix _2021_fee005                                       # le storiche, stesso costo
python -m scripts.lab_report --symbol BTCUSD --interval 1d                 # tabelle in reports/
```
