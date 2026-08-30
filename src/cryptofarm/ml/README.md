# `ml/` — la pipeline di addestramento

Da candele a modello a segnale. Il percorso è sempre lo stesso e ogni stadio è un modulo:

```
store 5m  →  feature  →  etichette  →  matrice  →  modello  →  valutazione  →  servizio
(data/)      features     labeling     dataset     models      evaluate       signals
             bar_features directional_change       validation  execution
```

`trainer.py` non contiene logica propria: assembla questi pezzi e tiene la configurazione.
Le decisioni che hanno prodotto questa forma — e le strade escluse misurandole — stanno in
[`.claude/docs/strategy.md`](../../../.claude/docs/strategy.md). **Leggerlo prima di
modificare qualcosa qui**: contiene misure che chiudono diverse idee ragionevoli a prima vista.

## I file

| file | righe | a cosa serve |
|---|---|---|
| `features.py` | 131 | feature dalle candele grezze, **scale-free** per costruzione: modulo puro, nessun parametro appreso, nessuno scaler da ricaricare accanto al modello |
| `bar_features.py` | 211 | le feature per barra dei modelli recenti (swing, ingresso, RL), **una sola definizione** condivisa da addestramento e servizio |
| `labeling.py` | 458 | le etichette. Il riferimento è la barriera tripla; in fondo restano i minimi/massimi locali per confronto |
| `directional_change.py` | 229 | etichettatura a pivot confermati: il pivot è datato alla barra di **conferma**, non all'estremo |
| `dataset.py` | 207 | dalla coppia (feature, etichette) alla matrice: campionamento a eventi CUSUM, split temporale, sequenze |
| `models.py` | 148 | i modelli dietro un'interfaccia sola. Default `gbdt`; `gru`/`cnn`/`lstm` restano dietro `--model` e vogliono l'extra `[dl]` |
| `evaluate.py` | 346 | le metriche, in termini economici prima che statistici: la precision di break-even è il numero che decide |
| `validation.py` | 245 | cross-validation per serie temporali: purging, embargo, pesi di unicità, CPCV |
| `execution.py` | 174 | esecuzione simulata degli ordini limite: il riempimento non è gratuito e non è certo |
| `meta.py` | 111 | il meta-labeling in due stadi: primario CUSUM permissivo, secondario che decide se il trade vale |
| `rl.py` | 222 | la politica a due azioni, fitted Q-iteration, **col costo dentro la ricompensa** |
| `signals.py` | 581 | l'unico punto di **servizio**: da modello su disco a segnali per la pagina |
| `trainer.py` | 444 | assemblaggio, configurazione, e `MODEL_PRECEDENCE` — quale artefatto governa la pagina |
| `meta_trainer.py` | 391 | addestra il secondario del meta-labeling, validato in CPCV |
| `swing_trainer.py` | 272 | addestra la prossimità agli estremi locali (`swing_model`) |
| `entry_trainer.py` | 325 | addestra il rendimento delle prossime H barre (`entry_model`, `entry_model_veloce`) |
| `rl_trainer.py` | 202 | addestra la politica a rinforzo (`rl_model`) |

## Le funzioni

**`features.py`** — `add_technical_indicators`, `normalize_indicators`, `build_feature_frame`;
l'elenco delle colonne come dato in `FEATURES` e `PRICE_FEATURES`. Ogni feature è un rapporto o un
rango: confrontabile fra BTC a 100.000 e DOGE a 0,2, e fra lo stesso asset a cinque anni di
distanza. Senza questa proprietà un modello unico su più simboli impara il livello del prezzo
invece della sua forma.

**`bar_features.py`** — `asset_features`, `cross_features`, `positioning_features`,
`build_swing_features`; le colonne come dati in `ASSET_COLUMNS`, `CROSS_COLUMNS`,
`POSITIONING_COLUMNS`, `SWING_SCALES`, `SWING_BASE_COLUMNS`, `SWING_COLUMNS`.
`cross_features` sopravvive perché la usa `scripts/meta_gate.py`, non solo i trainer.

**`labeling.py`** — `barrier_widths`, `triple_barrier_events`, `triple_barrier_labels`,
`label_distribution`, `format_distribution`, `apply_label_cooldown`,
`filter_labels_by_future_return`, `extrema_labels`, `swing_target`, `swing_pivots`,
`swing_leg_target`. `swing_target` è il rango centrato della chiusura (−1 su un minimo, +1 su un
massimo, 0 dentro una tendenza); `swing_leg_target` pesa la stessa domanda per la forza della
gamba. Il secondo ha IC alto e denaro basso: è la misura che ha spostato il bersaglio verso
`entry_trainer`. `extrema_labels` è il metodo precedente, tenuto per confronto.

**`directional_change.py`** — `directional_change_pivots`, `leg_table`, `capturable_fraction`,
`soft_labels`, `label_distribution`, `tune_threshold`, `confirmed_reversal_rows`.

**`dataset.py`** — `cusum_events`, `build_design_matrix`, `build_samples`, `time_split`,
`create_sequences`. I ritardi (`LAGS`) sono in scala di Fibonacci (1, 2, 3, 5, 8, 13, …):
risoluzione fine sul passato prossimo, grossolana su quello remoto.

**`models.py`** — `build_model`, `fit_model`, `predict_proba`, `save_model`, `load_model`;
le architetture disponibili in `MODEL_KINDS`.

**`evaluate.py`** — `break_even_precision` e `trade_expectancy` sono i due che decidono; attorno
stanno `ranking_auc`, `classification_summary`, `threshold_sweep`, `quantile_sweep`,
`fee_sensitivity`, `best_threshold`, `lift_over_base_rate`, `signal_summary`,
`precisione_estremi` e i loro `format_*`.

**`validation.py`** — `PurgedKFold`, `CombinatorialPurgedCV`, `purge_train_indices`,
`sample_uniqueness`, più la correzione per molteplicità: `probability_of_backtest_overfitting`,
`expected_max_sharpe`, `deflated_sharpe_ratio`.

**`execution.py`** — `limit_fills`, `apply_execution`, `round_trip_cost`,
`adverse_selection_report`; le commissioni come dati in `MAKER_FEE`, `TAKER_FEE`, `FEE_MODES`.

**`meta.py`** — `build_meta_labels`, `expectancy_by_quantile`.

**`rl.py`** — `Transizioni`, `transizioni_simbolo`, `unisci`, `fitted_q`, `posizioni`, `rendimento`.

**`signals.py`** — 25 nomi pubblici, un gruppo per famiglia di modello:
`interval_from_index`, `buy_probabilities`, `barrier_signals`, `meta_signals` (le due famiglie
storiche); `swing_model_disponibile`, `swing_model`, `swing_features`, `swing_predictions`,
`swing_exposure`, `swing_cadenza`, `swing_signals`; `rl_model_disponibile`, `rl_model`,
`rl_exposure`, `rl_signals`; `entry_metadata`, `entry_model_disponibile`, `entry_model`,
`barre_equivalenti`, `entry_tenuta`, `entry_fuori_misura`, `entry_exposure`, `entry_gate`,
`entry_signals`, `entry_predictions`.

**`trainer.py`** — `build_dataset`, `train`, `main` per addestrare; `active_model_name`,
`load_signal_model`, `meta_parameters`, `stored_decision_threshold`, `stored_exit_threshold`,
`get_model_predictions` per **servire**. `meta_parameters()` legge barriere, soglia CUSUM e
parametri di esecuzione dai metadata dell'artefatto e non da costanti: devono essere esattamente
quelli con cui il modello è stato addestrato.

**I trainer recenti** condividono la stessa forma — `addestra`, `selfcheck`, `main` — e si lanciano
come moduli (`python -m cryptofarm.ml.entry_trainer --selfcheck`). Il `--selfcheck` gira su dati
finti e **non** richiede lo store: è il modo di verificare una modifica senza i 4 GB di candele.
`trainer.py` e `meta_trainer.py`, più vecchi, usano invece `build_dataset` + `train`.

## Due cose da sapere prima di toccare `trainer.py`

**`MODEL_PRECEDENCE` è l'unica fonte di verità.** `active_model_name()` decide sia quale artefatto
si carica sia quale strategia di servizio lo interpreta, quindi i due non possono divergere. Per
tornare al modello precedente si sposta altrove l'artefatto di quello più recente — non si tocca
il codice. Oggi la testa è `entry_model_veloce`.

**Due famiglie sono state chiuse in negativo e il loro codice non è più qui.** La politica a tre
azioni (`policy_model`) è chiusa da `strategy.md` §12-13: entrare e uscire alla conferma cattura
zero in media, prima dei costi. Il modello a gambe (`leg_model`) è caduto nell'audit del 2026-08-28
(`modello-swing.md` §1): netto per ingresso negativo a tutte e sei le soglie. Rimettere il nome
nella tupla non li fa girare — il ramo di dispatch non c'è più. La misura che li ha chiusi sta nei
documenti, ed è lì che va riletta prima di rifarli.

## Documenti

[`strategy.md`](../../../.claude/docs/strategy.md) (labeling, feature, validazione) ·
[`modello-swing.md`](../../../.claude/docs/modello-swing.md) ·
[`modello-ingresso.md`](../../../.claude/docs/modello-ingresso.md) ·
[`politica-rl.md`](../../../.claude/docs/politica-rl.md)
