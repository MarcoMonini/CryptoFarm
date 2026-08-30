# syntax=docker/dockerfile:1.7
#
# Quattro target. L'ultimo e' `web` di proposito: chi costruisce senza `--target` — Render, che
# non ha un campo per sceglierlo — prende l'ultimo stage del file, e deve prendere quello giusto.
#
#   runtime  simulatore, trainer, store delle candele, scripts.analysis
#   dev      runtime + pytest/ruff/black, l'immagine con cui gira la CI
#   dl       runtime + TensorFlow, serve solo a `--model gru|cnn|lstm`
#   web      quello che va in produzione: identico a runtime, ma e' l'ultimo del file
#
#   docker build -t cryptofarm:web .                       # il default
#   docker build -t cryptofarm:runtime --target runtime .
#
# Il pacchetto viene installato nel virtualenv /opt/venv, non in modalita' editable: i dati
# stanno fuori dall'immagine, in /app/models e /app/market_data, indirizzati dalle due
# variabili CRYPTOFARM_* che `cryptofarm/paths.py` legge.

ARG PYTHON_VERSION=3.12


# --- base: interprete e virtualenv, condiviso da tutti gli stage ------------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"


# --- builder: dipendenze complete (app + data) -----------------------------------------
# Il copy in due tempi e' voluto: finche' pyproject.toml non cambia, il layer con le
# dipendenze resta in cache e una modifica al codice non fa riscaricare nulla.
FROM base AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/cryptofarm && touch src/cryptofarm/__init__.py && \
    pip install ".[app,data]"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .



# --- runtime: l'immagine completa per l'uso locale --------------------------------------
FROM base AS runtime

# tini: PID 1 che inoltra i segnali, altrimenti Ctrl-C e `docker stop` non arrivano al
# processo Python e ogni arresto costa i 10 s di timeout prima del SIGKILL.
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 cryptofarm
COPY --from=builder --chown=cryptofarm:cryptofarm /opt/venv /opt/venv

WORKDIR /app
COPY --chown=cryptofarm:cryptofarm src ./src
COPY --chown=cryptofarm:cryptofarm scripts ./scripts
COPY --chown=cryptofarm:cryptofarm .streamlit ./.streamlit
COPY --chown=cryptofarm:cryptofarm docker/healthcheck.py ./docker/healthcheck.py

# I due volumi. Vanno creati qui: montati da un host che non li ha, docker li crea di root
# e l'utente non privilegiato non ci scriverebbe.
RUN mkdir -p /app/models /app/market_data && chown -R cryptofarm:cryptofarm /app

ENV CRYPTOFARM_MODELS_DIR=/app/models \
    CRYPTOFARM_MARKET_DATA_DIR=/app/market_data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    MALLOC_ARENA_MAX=2

USER cryptofarm
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "docker/healthcheck.py"]

# `$PORT` perche' i PaaS assegnano la porta a runtime (Render usa 10000 di default) e
# pretendono il bind su 0.0.0.0; il fallback tiene funzionante `docker compose` in locale.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "streamlit run src/cryptofarm/trading/simulator.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]


# --- dev: l'immagine dei test ----------------------------------------------------------
FROM runtime AS dev

USER root
COPY --chown=cryptofarm:cryptofarm pyproject.toml ./
COPY --chown=cryptofarm:cryptofarm tests ./tests
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pytest==9.1.1 ruff==0.16.3 black==26.5.1
USER cryptofarm

CMD ["pytest"]


# --- dl: modelli sequenziali -----------------------------------------------------------
FROM runtime AS dl

USER root
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install tensorflow==2.19.0
USER cryptofarm

CMD ["python", "-m", "cryptofarm.ml.trainer", "--model", "gru"]


# --- web: quello che va in produzione, e il default del file ----------------------------
# Identico a `runtime`, e sta qui solo per essere **l'ultimo stage**: una build senza
# `--target` prende l'ultimo, ed e' cosi' che Render costruisce. Se un giorno si aggiunge
# uno stage, va aggiunto sopra questo.
#
# Un'immagine piu' magra per la sola pagina non e' possibile: `streamlit` dipende da
# `pyarrow>=7.0`, quindi i 141 MB del motore parquet entrano comunque. Toglierli
# richiederebbe di rinunciare a Streamlit, non a `data/klines.py`.
FROM runtime AS web
