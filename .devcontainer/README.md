# `.devcontainer/`

Un file solo: **`devcontainer.json`**, per GitHub Codespaces e per «Reopen in Container» di VS Code.

Parte da `mcr.microsoft.com/devcontainers/python:1-3.12-bullseye`, installa il pacchetto in
editable con tutti gli extra (`pip install --user -e ".[app,data,dev]"`) e **all'attacco avvia da
sé il simulatore** sulla 8501, che viene inoltrata e aperta in anteprima.

Non è il `Dockerfile` del progetto e non lo sostituisce: quello serve a spedire l'immagine in
produzione (quattro target, `web` in fondo), questo serve solo ad avere un ambiente di sviluppo
pronto. La differenza pratica è che qui il pacchetto è in editable e le directory dei dati restano
relative alla radice del repository, mentre nell'immagine sta in `site-packages` e servono
`CRYPTOFARM_MODELS_DIR` e `CRYPTOFARM_MARKET_DATA_DIR`.

Due conseguenze da conoscere: il container parte **senza store delle candele e senza modelli**,
quindi la pagina si apre in modalità degradata (strategie classiche sì, «AI Model» e rotazione no);
e Python è 3.12, come in CI e come `.venv312`.
