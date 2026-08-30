# `models/` — the trained artifacts

Every model lives in **two files with the same name**: `name.joblib` (the model) and `name.json`
(the metadata). **Neither is tracked** — this folder's `.gitignore` covers `*.joblib`, `*.json` and
`*.keras`, and keeps only this README. A clone of the repository finds the folder empty, and that is
also the condition the public service runs in: Render has no persistent disks.

They are not edited by hand. They are regenerated with the trainers.

## Which artifacts, and who produces them

| artifact | command | what it predicts | document |
|---|---|---|---|
| `entry_model_veloce` | `python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce` | the return of the next 20 bars of 5m | `modello-ingresso.md` |
| `entry_model` | `python -m cryptofarm.ml.entry_trainer` | the same over 150 bars: it **gates** the fast one | `modello-ingresso.md` |
| `rl_model` | `python -m cryptofarm.ml.rl_trainer` | the position, with the cost inside the reward | `politica-rl.md` |
| `swing_model` | `python -m cryptofarm.ml.swing_trainer` | proximity to the local extremes | `modello-swing.md` |
| `meta_model` | `python -m cryptofarm.ml.meta_trainer` | whether a primary entry closes in net profit | `strategy.md` |
| `signal_model` | `python -m cryptofarm.ml.trainer` | which barrier is touched first | `strategy.md` |

## Who governs the page

`ml/trainer.MODEL_PRECEDENCE` in that order, and `active_model_name()` is **the only source of
truth**: it decides both which artifact is loaded and which serving strategy interprets it, so the
two cannot diverge. **To go back to the previous model you move the most recent one's artifact
elsewhere** — you do not touch the code.

At the head today is `entry_model_veloce`, and the two entry artifacts work **as a pair**: the fast
one generates the trades, the slow one gates the entry bar only. Without the slow one the fast one
trades alone, and the out-of-sample net per trade drops from +2.071% to +1.360%.

## The metadata are not decoration

`meta_parameters()` reads barriers, CUSUM threshold and execution parameters **from the artifact's
`.json`** and not from constants, because they must be exactly the ones the model was trained with.
For the entry model the threshold, the holding and the gate count too: they are in the metadata and
not in the widgets, because selectivity **is** the model's advantage — changing it does not tune a
knob, it calls for a different strategy.

For `swing_model` the metadata also record **the label**
(`labeling: {method, window, peso_tempo, embargo_finestre, base_interval}`), so a drift between the
label drawn on the page and the one the model was trained on is detectable from disk instead of by
reading the code. See [`.claude/docs/labeling-strategy.md`](../.claude/docs/labeling-strategy.md).

## If you find an artifact in here whose trainer does not exist

They are the remains of two families **closed with a negative result**, and can be deleted: the
three-action policy (`policy_model`, `policy_alta`, `policy_bassa` — `strategy.md` §12-13), the leg
model (`leg_model` — `modello-swing.md` §1) and the three `.keras` files from the era before gradient
boosting (`optimized_model.keras`, `trained_model.keras`, `trained_model1.keras`). Their code is
gone: putting their name back in `MODEL_PRECEDENCE` is not enough to make them run, and a test
verifies it.
