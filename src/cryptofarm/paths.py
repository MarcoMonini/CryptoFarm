"""Filesystem paths shared across the cryptofarm package.

Resolved relative to this file rather than the process's working directory, so model
loading/saving works regardless of where `streamlit run` / `python` is invoked from.

Le due directory di dati si possono spostare con `CRYPTOFARM_MODELS_DIR` e
`CRYPTOFARM_MARKET_DATA_DIR`. Serve in container: il pacchetto e' installato in
`site-packages`, quindi la radice dedotta dal file punterebbe dentro il virtualenv invece
che al volume montato.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _dir_from_env(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else default


MODELS_DIR = _dir_from_env("CRYPTOFARM_MODELS_DIR", PROJECT_ROOT / "models")
# Store locale delle candele scaricate (parquet, un file per coppia simbolo/intervallo).
# Gitignorato: sono centinaia di MB rigenerabili con `python -m cryptofarm.data.klines --update`.
MARKET_DATA_DIR = _dir_from_env("CRYPTOFARM_MARKET_DATA_DIR", PROJECT_ROOT / "market_data")
