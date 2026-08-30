# `tests/` — 35 files, 1,027 cases (1,024 passed, 3 skipped)

> The numbers in brackets are the cases **collected** by pytest, not the functions written: where
> there is `parametrize` the two diverge a lot (`test_panels.py` has 25 functions and 410 cases).

`.venv312/bin/python -m pytest`. No test touches the network and none requires the candle store:
where data is needed, it is built in memory. That is a condition, not a convenience — it is the one
CI runs in and the one whoever has just cloned the repository must be able to run in.

## How they are organised

One file per module, with the same name (`test_confluence.py` ↔ `trading/confluence.py`), plus three
files that cover a **level** instead of a module.

### The three at level

| file | tests | what it protects |
|---|---|---|
| `test_simulator_golden.py` | 75 | the **behaviour** of 21 functions over four synthetic scenarios, against `data/simulator_golden.json` |
| `test_simulator_page.py` | 8 | the page **as a page**, run with `streamlit.testing.v1.AppTest` |
| `test_scripts_importabili.py` | 18 | that every module in `scripts/` at least imports |

`test_simulator_page.py` is the level the fault that took the simulator out of production passed
through: every function had its own tests and they all passed, while an unconditional
`load_signal_model()` prevented the page from opening. It also covers degradation without the store
and without models, which is the public service's condition.

### The model and its signals

`test_features.py` (7) · `test_labeling.py` (13) · `test_directional_change.py` (12) ·
`test_dataset.py` (14) · `test_evaluate.py` (14) · `test_validation.py` (16) ·
`test_execution.py` (7) · `test_signals.py` (10) · `test_model_discovery.py` (12)

`test_swing_target.py` (15) · `test_swing_features.py` (4) · `test_swing_signals.py` (8) ·
`test_swing_lab.py` (5) — the swing model, from target to serving.

`test_entry_trainer.py` (7) · `test_entry_signals.py` (10) · `test_entry_panel.py` (6) — the entry
model, which is the one at the head today.

`test_rl.py` (6) · `test_rl_signals.py` (8) — the RL policy.

### The strategies and the accounting

`test_confluence.py` (57) · `test_confluence_lab.py` (8) · `test_confluence_audit.py` (4) ·
`test_ai_voter.py` (3) · `test_voters.py` (10) · `test_mtf.py` (5) · `test_long_short.py` (25) ·
`test_portfolio.py` (13) · `test_rotation.py` (9) · `test_panels.py` (410) ·
`test_tuned_defaults.py` (177) · `test_strategy_sweep.py` (15)

### The data

`test_klines_store.py` (12) · `test_positioning.py` (4) — no network: the dumps are built in memory.

## Six tests to read before changing their module

They are the ones defending against a defect that is **invisible when reading the code**, and
rewriting them without understanding them is the quickest way to reintroduce it.

- **`test_mtf.py`** cuts *inside* a long bar that has already begun. A cut aligned to the boundaries
  passes even with the look-ahead reintroduced, and that is how it was written the first time.
- **`test_tuned_defaults.py`** asserts on the widget's **key**, not on the value: Streamlit keeps
  state by key, and `AppTest` rebuilds state on every run, so it would not see the real defect (the
  fields staying put when the interval changes).
- **`test_panels.py`** counts the *Voters* panel's traces against `len(VOTANTI)`: it is the only list
  in the confluence that has to be kept aligned by hand.
- **`test_model_discovery.py`** verifies that an old artifact in `models/` does **not** bring a design
  already closed with a negative result back into service. It is the name, not the branch, that
  decides what gets loaded.
- **`test_swing_signals.py`** pins the exposure rule *and what it is not*: `sign(prediction)` is the
  natural reading of a target in [−1, 1] and is measured at a loss at every threshold.
- **`test_swing_target.py`** pins `labeling.TIME_WEIGHT` — the temporal smoothing at 0.7 — against
  its two copies, `swing_trainer.PESO_TEMPO` and `trading/config.SWING_TARGET_TEMPO`. Without it the
  three drift silently, and the page draws one label while the model trains on another. It also pins
  the label's monotonicity along the leg and the fact that `swing_target(verso="avanti")` looks only
  forward, which is the yardstick every IC in the documents is scored against.

## The golden master

`test_simulator_golden.py` **must pass before a change and pass again afterwards, without being
regenerated**. Regenerating (`SIMULATOR_GOLDEN_REGEN=1 pytest tests/test_simulator_golden.py`)
accepts any behaviour difference: do it only after verifying by hand that the difference is intended,
and check that the JSON diff contains only the expected lines.

The scenarios are not interchangeable: `close_ema_crossover_simulation` demands three EMA crossovers
in sequence and only fires on a real reversal, `close_bullish_ema_simulation` only in the sideways
one. Removing a scenario uncovers strategies without any test failing.

## Lint

`ruff check src scripts tests` and `black src scripts tests`. The configuration is in
`pyproject.toml` (120-character line).
