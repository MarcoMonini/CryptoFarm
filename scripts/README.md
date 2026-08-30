# `scripts/` — the measurements

Eighteen command-line modules. None is imported by the package: `src/cryptofarm/` does not know they
exist. They go the other way — they read the store, run strategies and models, and **produce the
numbers that appear in the documents in `.claude/docs/`**. They are launched as modules
(`.venv312/bin/python -m scripts.entry_lab`), never as files.

One test (`tests/test_scripts_importabili.py`) only verifies that each of them imports: it is the
minimum net against a module that breaks with nobody noticing for months.

## The benches

| file | lines | what it measures | document |
|---|---|---|---|
| `analysis.py` | 612 | the measurements behind `strategy.md`, in reusable form | `strategy.md` |
| `strategy_sweep.py` | 591 | systematic backtest of the menu strategies as the parameters vary | `backtest-strategie.md` |
| `sweep_report.py` | 456 | reads the sweep tables and derives the **answers** from them, instead of the rows | `backtest-strategie.md` |
| `strategy_focus.py` | 150 | the three checks made **after** choosing a configuration | `backtest-strategie.md` |
| `strategy_lab.py` | 359 | bench for the two-sided strategies: grids, intervals, costs, assets | `strategie-nuove.md` |
| `lab_report.py` | 471 | what holds, what is noise, what the short adds | `strategie-nuove.md` |
| `confluence_lab.py` | 557 | the confluence over a wide grid and over a basket | `strategia-confluenza.md` |
| `confluence_audit.py` | 450 | the same configurations over many assets, in and out of sample | `strategia-confluenza.md` |
| `swing_lab.py` | 178 | deciles, P&L and random control of the swing model | `modello-swing.md` |
| `entry_lab.py` | 143 | what the slow model's gate is worth, and what trading more costs | `modello-ingresso.md` |
| `rl_lab.py` | 183 | does the RL policy beat passive holding? and chance at equal exposure? | `politica-rl.md` |
| `cross_section.py` | 232 | cross-sectional rotation: choosing *which* instead of *when* | `ricerca-quant-ml.md` |
| `meta_gate.py` | 360 | meta-labeling on top of a real primary strategy | `ricerca-quant-ml.md` |
| `multiplicity.py` | 203 | multiplicity correction for grids already measured: DSR and PBO | `ricerca-quant-ml.md` |
| `ai_voter.py` | 287 | the model voter trained **on the confluence's own trades** | `strategia-confluenza.md` |

## The tools

| file | lines | what it is for |
|---|---|---|
| `tune_defaults.py` | 406 | chooses the widgets' starting values and **regenerates `trading/tuned_defaults.py`** |
| `import_candles.py` | 139 | builds the 5m store from a local dataset, where `data.binance.vision` is unreachable |

## Where the output goes

**`reports/`** (tracked) keeps the final tables, the ones the documents quote.
**`analysis_cache/`** (gitignored, ~31 MB) keeps the raw sweeps and the intermediate results: they
are tens of MB and regenerate by rerunning the script.

## Three things to know

**The grid maximum is not the answer.** It is the luckiest cell: on this data choosing the maximum
transfers worse than the median, and on the rotation the correlation between in-sample and
out-of-sample return is **−0.69**. `tune_defaults.py` therefore chooses one coordinate at a time, on
the **percentile rank within its own symbol**, and adopts a value only if it passes two checks.
`multiplicity.py` exists for the same reason: it says how much of a result is the grid.

**The expensive part of `confluence_lab` does not depend on the grid.** The frozen voters have a
state that depends only on (symbol, interval): `stati_dei_votanti` computes it once and reuses it
across all the cells. Measured over 11,520 bars: 351 ms per cell against 104 ms.

**`--selfcheck` runs without the store.** `confluence_lab`, and for the same reason the trainers in
`ml/`, accept it: they build fake data and verify the mechanics. It is the way to try a change
without the 4 GB of candles.
