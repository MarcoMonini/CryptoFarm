# `.streamlit/`

Un file solo: **`config.toml`**, che imposta il tema scuro (`[theme] base = "dark"`) e nient'altro.

È l'unica configurazione di Streamlit del progetto, e vale sia in locale sia in container: il
`Dockerfile` copia la cartella nell'immagine. Porta, indirizzo e CORS **non** stanno qui — sono
argomenti della riga di comando, perché in produzione la porta si sa solo a runtime
(`streamlit run ... --server.port ${PORT:-8501} --server.address 0.0.0.0`).

Il tema conta più di quanto sembri: i colori delle tracce del grafico sono scelti per il contrasto
**su superficie scura**, e sono tre — blu, arancio, acquamarina — perché sono le uniche che passano
tutte le coppie del validatore, deuteranopia inclusa. Tre test in `tests/test_panels.py` tengono
ferma questa regola. Cambiare `base` in `"light"` la invaliderebbe senza far fallire niente.
