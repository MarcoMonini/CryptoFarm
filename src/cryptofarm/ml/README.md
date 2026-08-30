# `ml/` — the training pipeline

From candles to model to signal. The path is always the same and every stage is a module:

```
5m store  →  features  →  labels    →  matrix    →  model     →  evaluation  →  serving
(data/)      features     labeling     dataset     models      evaluate      signals
             bar_features directional_change       validation  execution
```

`trainer.py` holds no logic of its own: it assembles these pieces and keeps the configuration.
The decisions that produced this shape — and the paths ruled out by measuring them — are in
[`.claude/docs/strategy.md`](../../../.claude/docs/strategy.md). **Read it before changing anything
here**: it contains measurements that close several ideas that look reasonable at first sight. For
the labels specifically, read
[`.claude/docs/labeling-strategy.md`](../../../.claude/docs/labeling-strategy.md).

## The files

| file | lines | what it is for |
|---|---|---|
| `features.py` | 131 | features from the raw candles, **scale-free** by construction: a pure module, no learned parameters, no scaler to reload alongside the model |
| `bar_features.py` | 211 | the per-bar features of the recent models (swing, entry, RL), **a single definition** shared by training and serving |
| `labeling.py` | 469 | the labels. The reference is the triple barrier; at the end sit the local extrema and the leg target |
| `directional_change.py` | 229 | confirmed-pivot labeling: the pivot is dated at the **confirmation** bar, not at the extreme |
| `dataset.py` | 207 | from the (features, labels) pair to the matrix: CUSUM event sampling, temporal split, sequences |
| `models.py` | 148 | the models behind a single interface. Default `gbdt`; `gru`/`cnn`/`lstm` stay behind `--model` and want the `[dl]` extra |
| `evaluate.py` | 346 | the metrics, in economic rather than statistical terms: break-even precision is the number that decides |
| `validation.py` | 245 | cross-validation for time series: purging, embargo, uniqueness weights, CPCV |
| `execution.py` | 174 | simulated execution of limit orders: the fill is neither free nor certain |
| `meta.py` | 111 | two-stage meta-labeling: a permissive CUSUM primary, a secondary that decides whether the trade is worth it |
| `rl.py` | 222 | the two-action policy, fitted Q-iteration, **with the cost inside the reward** |
| `signals.py` | 581 | the single **serving** point: from a model on disk to signals for the page |
| `trainer.py` | 444 | assembly, configuration, and `MODEL_PRECEDENCE` — which artifact governs the page |
| `meta_trainer.py` | 391 | trains the meta-labeling secondary, validated with CPCV |
| `swing_trainer.py` | 272 | trains proximity to the local extremes (`swing_model`) |
| `entry_trainer.py` | 325 | trains the return of the next H bars (`entry_model`, `entry_model_veloce`) |
| `rl_trainer.py` | 202 | trains the RL policy (`rl_model`) |

## The functions

**`features.py`** — `add_technical_indicators`, `normalize_indicators`, `build_feature_frame`; the
column list as data in `FEATURES` and `PRICE_FEATURES`. Every feature is a ratio or a rank:
comparable between BTC at 100,000 and DOGE at 0.2, and between the same asset five years apart.
Without that property a single model over several symbols learns the price level instead of its
shape.

**`bar_features.py`** — `asset_features`, `cross_features`, `positioning_features`,
`build_swing_features`; the columns as data in `ASSET_COLUMNS`, `CROSS_COLUMNS`,
`POSITIONING_COLUMNS`, `SWING_SCALES`, `SWING_BASE_COLUMNS`, `SWING_COLUMNS`.
`cross_features` survives because `scripts/meta_gate.py` uses it, not only the trainers.

**`labeling.py`** — `barrier_widths`, `triple_barrier_events`, `triple_barrier_labels`,
`label_distribution`, `format_distribution`, `apply_label_cooldown`,
`filter_labels_by_future_return`, `extrema_labels`, `swing_target`, `swing_pivots`,
`swing_leg_target`, and the constant `TIME_WEIGHT`.

- `swing_leg_target` is **the label the swing model is trained on**: it slides continuously from −1
  at a local minimum to +1 at the next local maximum, and the position along the leg is
  `TIME_WEIGHT` elapsed time and `1 − TIME_WEIGHT` price covered — **a temporal smoothing at 0.7**.
- `swing_target` is the centered rank of the close (−1 on a low, +1 on a high, 0 inside a trend). It
  has no temporal component by construction, and it is kept as the **yardstick**: with
  `verso="avanti"` it gives the forward-only half, which is what every IC in the documents is scored
  against.
- `extrema_labels` is the earlier method, kept for comparison.

**`TIME_WEIGHT = 0.7` has three consumers and they must not drift**: `swing_trainer.PESO_TEMPO`,
`trading/config.SWING_TARGET_TEMPO` (a copied literal, because that module imports nothing by
design) and `tests/test_swing_target.py`. A drift is silent — the page would draw one label and the
model would train on another.

**`directional_change.py`** — `directional_change_pivots`, `leg_table`, `capturable_fraction`,
`soft_labels`, `label_distribution`, `tune_threshold`, `confirmed_reversal_rows`.

**`dataset.py`** — `cusum_events`, `build_design_matrix`, `build_samples`, `time_split`,
`create_sequences`. The lags (`LAGS`) are on a Fibonacci scale (1, 2, 3, 5, 8, 13, …): fine
resolution on the near past, coarse on the remote one.

**`models.py`** — `build_model`, `fit_model`, `predict_proba`, `save_model`, `load_model`; the
available architectures in `MODEL_KINDS`.

**`evaluate.py`** — `break_even_precision` and `trade_expectancy` are the two that decide; around
them sit `ranking_auc`, `classification_summary`, `threshold_sweep`, `quantile_sweep`,
`fee_sensitivity`, `best_threshold`, `lift_over_base_rate`, `signal_summary`, `precisione_estremi`
and their `format_*`.

**`validation.py`** — `PurgedKFold`, `CombinatorialPurgedCV`, `purge_train_indices`,
`sample_uniqueness`, plus the multiplicity correction: `probability_of_backtest_overfitting`,
`expected_max_sharpe`, `deflated_sharpe_ratio`.

**`execution.py`** — `limit_fills`, `apply_execution`, `round_trip_cost`,
`adverse_selection_report`; the commissions as data in `MAKER_FEE`, `TAKER_FEE`, `FEE_MODES`.

**`meta.py`** — `build_meta_labels`, `expectancy_by_quantile`.

**`rl.py`** — `Transizioni`, `transizioni_simbolo`, `unisci`, `fitted_q`, `posizioni`, `rendimento`.

**`signals.py`** — 25 public names, one group per model family:
`interval_from_index`, `buy_probabilities`, `barrier_signals`, `meta_signals` (the two historical
families); `swing_model_disponibile`, `swing_model`, `swing_features`, `swing_predictions`,
`swing_exposure`, `swing_cadenza`, `swing_signals`; `rl_model_disponibile`, `rl_model`,
`rl_exposure`, `rl_signals`; `entry_metadata`, `entry_model_disponibile`, `entry_model`,
`barre_equivalenti`, `entry_tenuta`, `entry_fuori_misura`, `entry_exposure`, `entry_gate`,
`entry_signals`, `entry_predictions`.

**`trainer.py`** — `build_dataset`, `train`, `main` to train; `active_model_name`,
`load_signal_model`, `meta_parameters`, `stored_decision_threshold`, `stored_exit_threshold`,
`get_model_predictions` to **serve**. `meta_parameters()` reads barriers, CUSUM threshold and
execution parameters from the artifact's metadata and not from constants: they must be exactly the
ones the model was trained with.

**The recent trainers** share the same shape — `addestra`, `selfcheck`, `main` — and are launched as
modules (`python -m cryptofarm.ml.entry_trainer --selfcheck`). The `--selfcheck` runs on fake data
and does **not** require the store: it is the way to verify a change without the 4 GB of candles.
`trainer.py` and `meta_trainer.py`, older, use `build_dataset` + `train` instead.

## Two things to know before touching `trainer.py`

**`MODEL_PRECEDENCE` is the only source of truth.** `active_model_name()` decides both which
artifact is loaded and which serving strategy interprets it, so the two cannot diverge. To go back to
the previous model you move the most recent one's artifact elsewhere — you do not touch the code.
Today the head is `entry_model_veloce`.

**Two families were closed with a negative result and their code is no longer here.** The
three-action policy (`policy_model`) is closed by `strategy.md` §12-13: entering and exiting on the
confirmation captures zero on average, before costs. The leg model (`leg_model`) fell in the
2026-08-28 audit (`modello-swing.md` §1): net per entry negative at all six thresholds. Putting the
name back in the tuple does not make them run — the dispatch branch is gone. The measurement that
closed them is in the documents, and that is where it must be reread before redoing them.

## Documents

[`strategy.md`](../../../.claude/docs/strategy.md) (labeling, features, validation) ·
[`labeling-strategy.md`](../../../.claude/docs/labeling-strategy.md) (the label in detail) ·
[`modello-swing.md`](../../../.claude/docs/modello-swing.md) ·
[`modello-ingresso.md`](../../../.claude/docs/modello-ingresso.md) ·
[`politica-rl.md`](../../../.claude/docs/politica-rl.md)
