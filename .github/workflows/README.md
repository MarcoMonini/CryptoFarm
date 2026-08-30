# `.github/workflows/`

Un workflow solo: **`ci.yml`**, su ogni pull request e su ogni push a `main`. Due job.

## `quality` — lint, format e test

Installa `.[app,data,dev]` su Python 3.12 e passa `ruff check`, `black --check` e `pytest` su
`src`, `tests` e `scripts`.

## `docker` — build delle immagini

Costruisce `runtime`, `dev` e `web`, e verifica **quattro cose che dal sorgente non si vedono**:

1. che il pacchetto si importi dentro l'immagine e risolva le directory dei dati a `/app/...`
   (cioè che l'override di `paths.py` funzioni, altrimenti i modelli addestrati in container
   finiscono in un layer usa e getta);
2. che i test passino dentro l'immagine, non solo sulla macchina della CI;
3. che la build **senza `--target`** non porti TensorFlow — cioè che `web` sia ancora l'ultimo
   stage del `Dockerfile`, che è come Render lo costruisce;
4. che il container si leghi davvero a `$PORT`: lo avvia con `PORT=10000` e interroga
   `/_stcore/health`.

## Cosa non fa

**Non pubblica nessuna immagine su un registry.** Render costruisce il `Dockerfile` da sé a ogni
push su `main`; queste build servono a scoprire una rottura prima che ci arrivi, non a produrre
l'artefatto che va in produzione.
