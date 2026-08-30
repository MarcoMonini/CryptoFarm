# `docker/`

Un file solo: **`healthcheck.py`**, la sonda di liveness dell'immagine.

Interroga `http://127.0.0.1:$PORT/_stcore/health`, l'endpoint di salute di Streamlit, ed esce 0 se
risponde 200. Il `Dockerfile` la copia (riga 64) e la usa come `HEALTHCHECK CMD` (riga 81).

Sta in un file, e non dentro il `HEALTHCHECK` a una riga, per due ragioni: la porta si sa solo a
runtime (`$PORT` sui PaaS, 8501 in locale) e la versione inline sarebbe illeggibile. Usa solo la
libreria standard, così l'immagine non deve installare `curl`.

Il resto della configurazione dei container non sta qui ma nella radice:
[`../Dockerfile`](../Dockerfile) (quattro target: `runtime`, `dev`, `dl`, `web`),
[`../compose.yaml`](../compose.yaml) e [`../render.yaml`](../render.yaml).

**`web` è l'ultimo stage del `Dockerfile` e deve restarci**: una build senza `--target` prende
l'ultimo stage, e Render non ha un campo per sceglierlo. Spostarlo significa spedire in produzione
l'immagine con TensorFlow dentro. Uno stage nuovo va aggiunto sopra `web`, mai sotto — e la CI
costruisce anche senza `--target` proprio per accorgersene.
