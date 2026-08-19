# models/

This folder holds the trained `.keras` model files (`optimized_model.keras`,
`trained_model.keras`, `trained_model1.keras`). They are gitignored (see `.gitignore` in
this folder) — a fresh clone will find this directory empty.

Regenerate them with:
```bash
python src/cryptofarm/ml/trainer.py
```
(edit the hardcoded training CSV path in its `if __name__ == "__main__":` block first).

`src/cryptofarm/trading/simulator.py`'s "AI Model" backtest strategy expects
`models/optimized_model.keras` to exist.
