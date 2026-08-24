# syntax=docker/dockerfile:1.7
#
# Tre target, una sola catena di build:
#   runtime  (default) simulatore Streamlit, trainer, store delle candele, scripts.analysis
#   dev                runtime + pytest/ruff/black, e' l'immagine con cui gira la CI
#   dl                 runtime + TensorFlow, serve solo a `--model gru|cnn|lstm`
#
#   docker build -t cryptofarm:runtime .
#   docker build -t cryptofarm:dev --target dev .
#
# Il pacchetto viene installato nel virtualenv /opt/venv, non in modalita' editable: i dati
# stanno fuori dall'immagine, in /app/models e /app/market_data, indirizzati dalle due
# variabili CRYPTOFARM_* che `cryptofarm/paths.py` legge.

ARG PYTHON_VERSION=3.12


# --- base: interprete e virtualenv, condiviso da builder e runtime ---------------------
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m venv "${VIRTUAL_ENV}"


# --- builder: risolve e installa le dipendenze ----------------------------------------
# Il copy in due tempi e' voluto: finche' pyproject.toml non cambia, il layer con le
# dipendenze resta in cache e una modifica al codice non fa riscaricare nulla.
FROM base AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/cryptofarm && touch src/cryptofarm/__init__.py && \
    pip install ".[app]"

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps .


# --- runtime: quello che si spedisce ---------------------------------------------------
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

# I due volumi. Vanno creati qui: montati da un host che non li ha, docker li crea di root
# e l'utente non privilegiato non ci scriverebbe.
RUN mkdir -p /app/models /app/market_data && chown -R cryptofarm:cryptofarm /app

ENV CRYPTOFARM_MODELS_DIR=/app/models \
    CRYPTOFARM_MARKET_DATA_DIR=/app/market_data \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

USER cryptofarm
EXPOSE 8501

# Niente curl nell'immagine: la sonda la fa l'interprete che c'e' gia'.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "src/cryptofarm/trading/simulator.py"]


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
    pip install tensorflow==2.19.0 keras-tuner==1.4.7
USER cryptofarm

CMD ["python", "-m", "cryptofarm.ml.trainer", "--model", "gru"]
