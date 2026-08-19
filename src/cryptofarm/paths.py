"""Filesystem paths shared across the cryptofarm package.

Resolved relative to this file rather than the process's working directory, so model
loading/saving works regardless of where `streamlit run` / `python` is invoked from.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
