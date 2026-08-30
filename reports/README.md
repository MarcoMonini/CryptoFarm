# Strategy backtest tables

Produced by `scripts/sweep_report.py` and `scripts/strategy_focus.py` from the sweeps in
`analysis_cache/sweeps/` (which are not tracked: they are tens of MB and regenerate). The reasoned
reading of these numbers is in
[`.claude/docs/backtest-strategie.md`](../.claude/docs/backtest-strategie.md).

All the measurements are on BTC/USD at 15 minutes from 2017-01-01 to 2026-08-24, capital 100 always
reinvested, 0.1% commission per leg except where stated.

| file | what it holds |
|---|---|
| `riferimento_15m.csv` | passive holding per period and per year: the yardstick for everything else |
| `panoramica_15m.csv` | one row per strategy: best, median, worst, profitable share, share beating passive holding |
| `frequenza_15m.csv` | all the configurations grouped by trades per year — the relationship that explains more than any other |
| `sensibilita_15m.csv` | for every parameter and every one of its values: median, best, profitable share |
| `escursione_15m.csv` | how much each parameter moves the result with all the others held fixed |
| `stabilita_15m.csv` | the year-by-year return of each grid's best configuration |
| `fuori_campione_15m.csv` | chosen on 2017-2021, returned on 2022-2026, with the median of the top ten |
| `walk_forward_15m.csv`, `walk_forward_dettaglio_15m.csv` | annual re-optimisation on the years already seen |
| `commissioni.csv` | the best configurations rerun at 0%, 0.04%, 0.075%, 0.1%, 0.2% commission |
| `intervalli.csv` | the same configurations rerun from 5m to 1d, without retouching the parameters |
| `*_ETHUSD.csv` | the same views on the control market: Bitfinex's ETH/USD, 2017-2019 (§9 of the document) |
| `lab_panoramica_*.csv` | two-sided strategies: best, median and profitable share per interval |
| `lab_effetto_short_*.csv` | the same configuration with and without the short side |
| `lab_ablazioni_*.csv` | each filter switched off in turn: what ADX, regime, volume and cloud are worth |
| `lab_classifica_*.csv` | historical and new ones over the same period and at the same cost |
| `lab_fuori_campione_*.csv` | chosen on 2021-2023, returned on 2024-2026, for both families |
| `lab_leva_costi_*.csv` | the best ones at 1×, 2× and 3× leverage and at three levels of commission and funding |

The `lab_*` views are on BTC/USD 2021-2026 (and ETH/USD 2017-2019 as a control), 0.05% commission per
leg plus 0.03% a day of carry: the reading is in
[`.claude/docs/strategie-nuove.md`](../.claude/docs/strategie-nuove.md).

## Cross-sectional rotation and the meta filter

Two families produced by different scripts, answering different questions from the previous ones. The
reading is in [`.claude/docs/ricerca-quant-ml.md`](../.claude/docs/ricerca-quant-ml.md).

| file | what it holds |
|---|---|
| `cs_majors_1d.csv` | rotation over the *majors* at daily scale: it chooses **which** asset, not when |
| `cs_majors_1d_oos.csv` | the same grid, chosen in sample and returned out of it |
| `cs_pairs.csv` | the pairs universe |
| `cs_pairs_2024.csv` | the same pairs on 2024 alone |
| `cs_wide_1d_oos.csv` | the wide 15-asset universe, out of sample |
| `meta_donchian_breakout_4h*.csv` | the meta-labeling secondary on top of Donchian Breakout at 4h |
| `meta_trend_pullback_4h*.csv` | the same on top of Trend Pullback at 4h |

Produced by `scripts/cross_section.py` and `scripts/meta_gate.py`:

```bash
.venv312/bin/python -m scripts.cross_section --universe majors --interval 1d --grid
.venv312/bin/python -m scripts.meta_gate --strategy donchian_breakout --interval 4h --oos 2024-01-01
```

Two warnings that apply to every table in this folder:

**The rotation's reference is the equal-weight universe, not BTC.** It carries the same survivorship
bias, so the comparison isolates what the rotation adds. Against BTC the rotation wins in 95.6% of the
configurations; against the universe, in 44.4%.

**The best row of a grid is not a result.** It is the luckiest cell: the correlation between
in-sample and out-of-sample return over the rotation's top ten configurations is **−0.69**.
`scripts/multiplicity.py` (DSR and PBO) exists to say how much of a maximum is the grid.
