# `tests/data/`

A single file: **`simulator_golden.json`**, the expectation of `../test_simulator_golden.py`.

It holds the output of 21 functions of `trading/simulator.py` and its modules over four synthetic
market scenarios (trend, sideways, regimes, spikes). It is not a configuration file and is not edited
by hand: it is regenerated with

```bash
SIMULATOR_GOLDEN_REGEN=1 .venv312/bin/python -m pytest tests/test_simulator_golden.py
```

**Regenerating accepts any behaviour difference**, including a regression. The correct flow is:
understand why the test fails, verify by hand that the difference is intended, regenerate, and then
**read the JSON diff** checking it contains only the expected lines. A diff wider than expected means
the change also touched strategies nobody was looking at — which is exactly the defect this file
exists for.
