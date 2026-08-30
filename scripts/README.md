# `scripts/` — le misure

Diciotto moduli da riga di comando. Nessuno è importato dal pacchetto: `src/cryptofarm/` non sa
che esistono. Vanno nell'altro verso — leggono lo store, girano strategie e modelli, e **producono
i numeri che stanno nei documenti di `.claude/docs/`**. Si lanciano come moduli
(`.venv312/bin/python -m scripts.entry_lab`), mai come file.

Un test (`tests/test_scripts_importabili.py`) verifica solo che ognuno si importi: è la rete
minima contro un modulo che si rompe e nessuno se ne accorge per mesi.

## I banchi di prova

| file | righe | cosa misura | documento |
|---|---|---|---|
| `analysis.py` | 612 | le misure dietro `strategy.md`, in forma riutilizzabile | `strategy.md` |
| `strategy_sweep.py` | 591 | backtest sistematico delle strategie del menu al variare dei parametri | `backtest-strategie.md` |
| `sweep_report.py` | 456 | legge le tabelle dello sweep e ne ricava le **risposte**, invece delle righe | `backtest-strategie.md` |
| `strategy_focus.py` | 150 | le tre verifiche che si fanno **dopo** aver scelto una configurazione | `backtest-strategie.md` |
| `strategy_lab.py` | 359 | banco delle strategie a due versi: griglie, intervalli, costi, asset | `strategie-nuove.md` |
| `lab_report.py` | 471 | cosa regge, cosa è rumore, cosa aggiunge lo short | `strategie-nuove.md` |
| `confluence_lab.py` | 557 | la confluenza su una griglia larga e su un paniere | `strategia-confluenza.md` |
| `confluence_audit.py` | 450 | le stesse configurazioni su molti asset, dentro e fuori campione | `strategia-confluenza.md` |
| `swing_lab.py` | 178 | decili, P&L e controllo casuale del modello a swing | `modello-swing.md` |
| `entry_lab.py` | 143 | quanto vale il cancello del modello lento, e quanto costa operare di più | `modello-ingresso.md` |
| `rl_lab.py` | 183 | la politica RL batte il possesso passivo? e il caso a pari esposizione? | `politica-rl.md` |
| `cross_section.py` | 232 | rotazione trasversale: scegliere *quale* invece di *quando* | `ricerca-quant-ml.md` |
| `meta_gate.py` | 360 | meta-etichettatura sopra una strategia primaria vera | `ricerca-quant-ml.md` |
| `multiplicity.py` | 203 | correzione per molteplicità delle griglie già misurate: DSR e PBO | `ricerca-quant-ml.md` |
| `ai_voter.py` | 287 | il votante a modello addestrato **sulle operazioni della confluenza stessa** | `strategia-confluenza.md` |

## Gli strumenti

| file | righe | a cosa serve |
|---|---|---|
| `tune_defaults.py` | 406 | sceglie i valori di partenza dei widget e **rigenera `trading/tuned_defaults.py`** |
| `import_candles.py` | 139 | costruisce lo store 5m da un dataset locale, dove `data.binance.vision` non si raggiunge |

## Dove finisce l'output

**`reports/`** (tracciato) tiene le tabelle finali, quelle che i documenti citano.
**`analysis_cache/`** (gitignorato, ~31 MB) tiene gli sweep grezzi e i risultati intermedi: sono
decine di MB e si rigenerano rilanciando lo script.

## Tre cose da sapere

**Il massimo della griglia non è la risposta.** È la cella più fortunata: su questi dati la scelta
del massimo trasferisce peggio della mediana, e sulla rotazione la correlazione fra resa in stima e
resa in verifica è **−0,69**. `tune_defaults.py` sceglie infatti una coordinata alla volta, sul
**rango percentile dentro il proprio simbolo**, e adotta un valore solo se supera due controlli.
`multiplicity.py` esiste per la stessa ragione: dice quanto di un risultato è la griglia.

**La parte cara di `confluence_lab` non dipende dalla griglia.** I votanti congelati hanno uno
stato che dipende solo da (simbolo, intervallo): `stati_dei_votanti` lo calcola una volta e lo
riusa su tutte le celle. Misurato su 11.520 barre: 351 ms per cella contro 104 ms.

**`--selfcheck` gira senza store.** `confluence_lab`, e per la stessa ragione i trainer di `ml/`,
lo accettano: costruiscono dati finti e verificano la meccanica. È il modo di provare una modifica
senza i 4 GB di candele.
