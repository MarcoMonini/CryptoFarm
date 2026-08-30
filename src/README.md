# `src/`

Layout `src/`, un solo pacchetto: [`cryptofarm/`](cryptofarm/). Non c'è altro qui —
`cryptofarm.egg-info/` è prodotto da `pip install -e` e non si traccia.

Il layout `src/` non è decorativo: impedisce che `import cryptofarm` risolva alla cartella del
repository invece che al pacchetto installato, e quindi che i test passino per una ragione diversa
da quella per cui gireranno in produzione.

Installazione: `.venv312/bin/pip install -e ".[app,data,dev]"`. Il `.venv` preesistente è Python
3.9 senza `scikit-learn`; il progetto richiede Python >= 3.12 e l'ambiente da usare è **`.venv312`**.
