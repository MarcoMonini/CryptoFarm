"""Sonda di liveness per l'immagine: interroga l'endpoint di salute di Streamlit.

Sta in un file perche' la porta si sa solo a runtime (`$PORT` sui PaaS, 8501 in locale) e
perche' infilare questo dentro un `HEALTHCHECK CMD` a una riga lo renderebbe illeggibile.
Nessuna dipendenza esterna: usa la libreria standard, cosi' l'immagine non installa `curl`.
"""

import os
import sys
import urllib.request

PORT = os.environ.get("PORT", "8501")
URL = f"http://127.0.0.1:{PORT}/_stcore/health"

try:
    with urllib.request.urlopen(URL, timeout=4) as response:
        sys.exit(0 if response.status == 200 else 1)
except Exception as error:  # rete, connessione rifiutata, timeout: il servizio non e' pronto
    print(f"healthcheck fallito su {URL}: {error}", file=sys.stderr)
    sys.exit(1)
