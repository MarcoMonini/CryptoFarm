# `.devcontainer/`

A single file: **`devcontainer.json`**, for GitHub Codespaces and for VS Code's "Reopen in
Container".

It starts from `mcr.microsoft.com/devcontainers/python:1-3.12-bullseye`, installs the package in
editable mode with all the extras (`pip install --user -e ".[app,data,dev]"`) and **on attach starts
the simulator by itself** on 8501, which is forwarded and opened in preview.

It is not the project's `Dockerfile` and does not replace it: that one ships the image to production
(four targets, `web` last), this one only provides a ready development environment. The practical
difference is that here the package is editable and the data directories stay relative to the
repository root, whereas in the image it lives in `site-packages` and `CRYPTOFARM_MODELS_DIR` and
`CRYPTOFARM_MARKET_DATA_DIR` are needed.

Two consequences to know: the container starts **with no candle store and no models**, so the page
opens in degraded mode (classic strategies yes, "AI Model" and rotation no); and Python is 3.12, as
in CI and as in `.venv312`.
