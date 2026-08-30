# `src/`

`src/` layout, a single package: [`cryptofarm/`](cryptofarm/). There is nothing else here —
`cryptofarm.egg-info/` is produced by `pip install -e` and is not tracked.

The `src/` layout is not decorative: it prevents `import cryptofarm` from resolving to the repository
folder instead of the installed package, and therefore prevents the tests from passing for a
different reason than the one they will run under in production.

Installation: `.venv312/bin/pip install -e ".[app,data,dev]"`. The pre-existing `.venv` is Python 3.9
without `scikit-learn`; the project requires Python >= 3.12 and the environment to use is
**`.venv312`**.
