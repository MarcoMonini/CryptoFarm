# Piano — Riscrittura della pipeline di addestramento CryptoFarm

Stato: **piano approvato nelle scelte di fondo, in attesa di conferma finale sui dettagli
operativi** (posizione dello store dati, lista asset). Nessun file di progetto modificato
rispetto a questo piano.

## Perché si riscrive (evidenza, non opinione)

Misurato il 2026-08-19 sulle stesse sequenze e le stesse etichette:

| modello | tempo | macro F1 | lift buy | lift sell |
|---|---|---|---|---|
| LSTM 3 layer, 747.267 parametri | ~25 min (10 epoche) | 0.092 | 3,5× | 2,8× |
| HistGradientBoostingClassifier | **3,9 s** | **0.111** | 3,6× | 4,2× |

Due famiglie di modelli molto diverse convergono sullo stesso risultato debole (~5% precision).
Quando questo accade, il limite non è l'architettura ma **la definizione del target**. Il
labeling per estremi locali chiede di indovinare la candela esatta di un minimo/massimo: il 97%
delle righe è "hold" per costruzione, un segnale una candela in anticipo conta come errore
totale, e il bilanciamento necessario a compensare distrugge la calibrazione (priore di training
17% contro 1,5% reale, da cui recall 0,78–0,86 con precision 0,05).

Diagnosi accessoria: su 350.400 candele scaricate il labeling produce ~12.300 segnali (3,5%), e
dopo il downsampling si addestra su 30.114 sequenze — **l'8,6% dei dati scaricati**.

## Decisioni prese

1. **Target**: triple-barrier labeling (take-profit / stop-loss / limite temporale) al posto degli
   estremi locali. Etichetta definita su *ogni* candela, distribuzione naturalmente equilibrata,
   e soprattutto l'etichetta coincide con l'esito di un trade reale: la precision *è* il win rate.
   Barriere **scalate su ATR** con pavimento a un multiplo delle commissioni, non percentuali
   fisse — così si adattano da sole tra timeframe e tra asset con volatilità diverse.
2. **Modello**: `HistGradientBoostingClassifier` come default (4 s per iterazione permette di
   tarare labeling e feature decine di volte al giorno in locale); LSTM/GRU/CNN 1D restano
   selezionabili dietro la stessa interfaccia.
3. **Dati**: 15 asset, storico massimo, timeframe **5m / 15m / 30m / 1h** (~12,5 M candele,
   ~275 MB in parquet, ~60 min di download una tantum, poi aggiornamenti incrementali).
4. **Granularità**: un modello unico su tutti i timeframe, con il timeframe come feature
   esplicita. Reso possibile dalla normalizzazione su ATR di barriere e feature.
5. **Struttura**: `trainer.py` (800 righe, sei responsabilità) diviso in moduli.

## Rischi noti e mitigazioni

| rischio | mitigazione |
|---|---|
| Sovrapposizione delle etichette: su 5m con orizzonte lungo, due etichette adiacenti condividono >99% del futuro → validation ottimista | stride > 1, pesi di unicità del campione, embargo dimensionato **sull'orizzonte** e non su un valore fisso |
| I modelli `.keras` esistenti diventano invalidi (feature nuove) | attesi e sostituiti dal retrain; i file vecchi **non vengono cancellati** |
| La strategia "AI Model" del simulatore si rompe se cambia il formato del modello | `get_model_predictions` resta il punto di ingresso pubblico e smista sul tipo di modello (keras o sklearn); `simulator.py` non cambia import |
| Download lungo, può fallire a metà | store incrementale e ripartibile: si riprende dall'ultimo timestamp salvato |
| Commissioni (0,2% andata e ritorno) mangiano i movimenti su 5m | pavimento sulle barriere a un multiplo delle fee; metriche di valutazione **al netto** delle commissioni |
| Regressioni silenziose train/inferenza (già capitate due volte) | JSON di metadata salvato accanto al modello (feature, finestra, parametri di labeling, copertura dati, commit, metriche) + test di parità |

## Struttura dei file

**Nuovi**

- `src/cryptofarm/data/klines.py` — download Binance + store parquet incrementale, manifest di
  copertura, CLI per aggiornare lo store.
- `src/cryptofarm/ml/features.py` — indicatori, normalizzazione scale-free, variazioni
  percentuali, feature del timeframe. Puro, condiviso da training e inferenza.
- `src/cryptofarm/ml/labeling.py` — triple-barrier (default) + estremi locali filtrati (legacy,
  mantenuto per confronto).
- `src/cryptofarm/ml/dataset.py` — sequenze (float32, stride configurabile), gap masking, split
  con embargo, pesi di unicità.
- `src/cryptofarm/ml/models.py` — `gbdt` / `gru` / `cnn` / `lstm` dietro un'interfaccia unica.
- `src/cryptofarm/ml/evaluate.py` — metriche per classe **e** metriche economiche (win rate,
  expectancy, numero di trade, P&L netto commissioni, confronto con buy & hold), sweep di soglia.

**Riscritti**

- `src/cryptofarm/ml/trainer.py` — sola orchestrazione + CLI.
- `tests/` — diviso per modulo; i 13 test attuali si trasferiscono e si estendono.

**Non toccati**

- `src/cryptofarm/trading/simulator.py`, `app/`, `backup/v2/`.

## Ordine di esecuzione

1. `data/klines.py` + store, e **avvio del download** (è il percorso critico, ~60 min in background).
2. `ml/features.py` e `ml/labeling.py` + test.
3. `ml/dataset.py` + test.
4. `ml/models.py` e `ml/evaluate.py` + test.
5. `ml/trainer.py` (orchestrazione) + metadata JSON.
6. Primo addestramento GBDT, sweep di soglia, metriche economiche.
7. Verifica end-to-end nel simulatore; aggiornamento di `CLAUDE.md`.

Commit incrementale a ogni punto, con riepilogo.

## Cosa si conserva dal lavoro già fatto

Embargo sullo split, gap masking, normalizzazione scale-free dell'ATR, parità train/inferenza in
`get_model_predictions`, e i 13 test — tutti validi, si trasferiscono nei nuovi moduli. I 6
commit sul branch `ai-labeling-rewrite` restano nella history.
