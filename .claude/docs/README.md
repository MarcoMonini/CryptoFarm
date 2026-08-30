# Working documentation — CryptoFarm

Everything needed to pick the work back up is here. `CLAUDE.md` stays in the root because Claude
Code loads it automatically from there, and it points at this folder.

> **Language rule.** Every document in this repository is written in English. See `CLAUDE.md`
> § *Documentation language*. It is a standing instruction, not a style preference.

## Reading order

Whoever starts from scratch reads **`HANDOFF.md`** and nothing else: it is the current state, and it
references the other documents. Whoever has to touch one specific piece jumps to that piece's
document.

Whoever wants to understand *how far the project has got*, in chronological order of result:
`backtest-strategie.md` → `strategie-nuove.md` → `ricerca-quant-ml.md` → `strategia-confluenza.md`
→ `politica-rl.md` → `modello-swing.md` → **`modello-ingresso.md`**, which is the only one with
numbers that pass the matched-exposure control.

Whoever is about to touch labels or training reads **`labeling-strategy.md`** first: it is the
document a whole family of models depends on.

## The documents

| document | when it is needed |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | **read this first.** State of the two strands with the most recent results, what is still open, environment and measurement traps, rules of engagement. It does not duplicate the others, it references them. **To be updated at the end of a session.** |
| [`labeling-strategy.md`](labeling-strategy.md) | **how the data is labeled.** Labels in [-1, +1] oscillating between local lows and highs, the pivot windows, and the **temporal smoothing** (`TIME_WEIGHT = 0.7`) that makes the label lead the price instead of following it. Also the embargo the variable look-ahead demands, and why the training target is not the measuring stick. |
| [`strategy.md`](strategy.md) | **source of truth for the decisions** on labeling, features, model and validation, with the measurements that justify them. It has a revision table at the top. To be updated in place when something is decided. |
| [`backtest-strategie.md`](backtest-strategie.md) | **the indicator strategies, measured.** 3,129 configurations over nine years of BTC: what returns, how much it depends on the parameters, what survives out of sample, and the code defects found by measuring. The tables are in `reports/`, the scripts that produce them in `scripts/{strategy_sweep,sweep_report,strategy_focus}.py`. |
| [`strategie-nuove.md`](strategie-nuove.md) | **the operational sequel to the backtest.** The four code corrections and what they changed, the 2021-2026 cycle as a dataset, five new strategies and the engine that can also go short (`trading/strategies_ls.py`, `pnl.simulate_positions`). |
| [`ricerca-quant-ml.md`](ricerca-quant-ml.md) | the measurements on five assets: cross-sectional rotation (`scripts/cross_section.py`) and the meta filter (`scripts/meta_gate.py`), plus §2, which is the reason seven entries left the menu. |
| [`piano-strategie.md`](piano-strategie.md) | the plan agreed with the user on 2026-08-27. **Step 1 done, 2bis became the confluence, 2-5 not executed**: the work moved to the model strand. The multiplicity control (DSR/PBO) is here, and so is step 5, which is the last clean verification window left. |
| [`strategia-confluenza.md`](strategia-confluenza.md) | **the multi-timeframe multi-signal strategy, measured.** Four planes with disjoint questions, six voters chosen by family, threshold decided by the higher planes. On 15 assets and seven years it **does not beat passive holding**: no look-ahead, uncorrelated voters, but the gradient of every parameter points at not trading. The conclusions are at the bottom. |
| [`politica-rl.md`](politica-rl.md) | **the reinforcement policy, wired in** (2026-08-28). It starts from a premise of the user's — "buy shortly before the crashes" — and the measurement proves it false: entries have the same drawdown as any other bar, and every stop level makes the net worse. The cause is the commission. From there, the shape of the agent, with the cost inside the reward. It beats passive holding 11/15 out of sample and **halves the maximum drawdown**; the *when* is only weakly above chance. |
| [`modello-swing.md`](modello-swing.md) | **the AI model redone and measured** (2026-08-28). The audit that closed the leg model (§1), the label and why 93% of that target is free, and the measurements for which the signal exists but **does not beat chance at matched exposure** (1 symbol out of 15). §5.4: what was wired in and what deliberately was not. |
| [`modello-ingresso.md`](modello-ingresso.md) | **the model leading today, wired in** (2026-08-29). It changes the question: not "how close are we to an extreme" but "what does buying here return". The lever is **selectivity**, not accuracy. These are the project's first numbers that pass the matched-exposure control: +2.071% net per trade out of sample, 14/15 symbols profitable, 100th percentile. The fast one trades, the slow one gates it. |
| [`MAPPA-modello-ai.md`](MAPPA-modello-ai.md) | **a pre-registration**: the success criteria of the AI-model work, declared *before* the measurements. It is here on purpose, to check the target was not moved after seeing it — and it moved twice. It is not a specification to execute. |

## Rules

- documentation is written **in English**, always, including new documents and edited lines;
- whatever is decided goes **in that piece's document**, not just in the commit message;
- `git log` remains the densest source on the *why* of every choice: the messages are long on purpose;
- measurements come with their numbers and with the command that reproduces them, otherwise they are
  not verifiable;
- a negative result is written up the way a positive one is. Half of these documents close a road,
  and that is what stops it being reopened for the third time.
