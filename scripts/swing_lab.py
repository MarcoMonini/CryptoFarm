"""Banco del modello a swing: il segnale statistico c'e', il vantaggio operativo no.

Tre misure, in quest'ordine, perche' ognuna decide se la successiva ha senso:

1. **Rendimento futuro per decile di previsione.** Dice *che forma* ha il segnale. Sul modello
   addestrato il 2026-08-28 la forma e' a **U**, non monotona: sia il decile piu' basso (che il
   modello legge come «vicino a un minimo locale») sia i tre piu' alti («vicino a un massimo»)
   precedono rendimenti sopra la media, e il centro sta sotto. Non e' una previsione di direzione:
   e' una previsione di *struttura*. Vendere sui massimi previsti -- la lettura naturale di un
   target in [-1, 1] -- vende esattamente le barre migliori.
2. **P&L della regola letta dalla forma**: dentro il mercato quando |previsione| e' alta, fuori
   quando e' bassa, con isteresi. E' l'unica regola che la forma a U sostiene.
3. **Il controllo casuale a esposizione appaiata.** Una regola che sta fuori dal mercato il 76%
   del tempo batte il possesso passivo in un ribasso *per costruzione*. La domanda vera e' se lo
   batte meglio di collocare la stessa esposizione, con le stesse durate, a caso.

Il terzo numero e' quello che chiude la faccenda: **1 simbolo su 15 in validazione e 1 su 15
fuori campione** superano il p95 di duecento estrazioni, contro lo 0,75 atteso dal caso. Il
merito della regola e' stare fuori dal mercato, e per quello non serve un modello.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines
from cryptofarm.ml.bar_features import SWING_COLUMNS, build_swing_features
from cryptofarm.ml.models import load_model
from cryptofarm.paths import MARKET_DATA_DIR, MODELS_DIR
from cryptofarm.trading.pnl import simulate_trading_with_commisions

FEE = 0.1  # percento per lato
VAL0, OOS0, FINE = pd.Timestamp("2023-06-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2027-01-01")
ENTRA, ESCI, CADENZA = 0.50, 0.40, 288  # 288 barre da 5m = una decisione al giorno
ESTRAZIONI = 200
CACHE = MARKET_DATA_DIR / "swing_previsioni.pkl"


def _firma_del_modello() -> str:
    """Data di creazione dell'artefatto: distingue un riaddestramento da un riavvio."""
    return json.loads((MODELS_DIR / "swing_model.json").read_text())["created"]


def previsioni(simboli: list[str], da: pd.Timestamp, riusa: bool = True) -> dict:
    """Previsione del modello per ogni barra 5m. In cache: ricalcolarla costa minuti, non secondi.

    La cache porta la **firma del modello** e si invalida da sola quando l'artefatto cambia. Senza,
    misurare un modello appena riaddestrato restituisce in silenzio i numeri di quello precedente,
    che e' il modo piu' rapido di credere a un miglioramento che non c'e'.
    """
    firma = _firma_del_modello()
    if riusa and CACHE.exists():
        salvato = pickle.loads(CACHE.read_bytes())
        if salvato.get("firma") == firma:
            return salvato["dati"]
        print("  cache ignorata: il modello e' cambiato dall'ultima volta")
    modello = load_model(MODELS_DIR / "swing_model.joblib")
    dati = {}
    for symbol in simboli:
        candele = load_klines(symbol, "5m")
        candele = candele[candele.index >= da]
        if len(candele) < 60_000:
            continue
        X = build_swing_features(symbol, candele)[SWING_COLUMNS].to_numpy()
        previsto = modello.predict(X)
        previsto[np.isnan(X).all(axis=1)] = np.nan
        dati[symbol] = (candele.index, candele["Close"].to_numpy(), previsto)
        print(f"  {symbol}: {len(candele):,} barre", flush=True)
    CACHE.write_bytes(pickle.dumps({"firma": firma, "dati": dati}))
    return dati


def decili(dati: dict, da, a, orizzonte: int = 576) -> pd.Series:
    """Eccesso di rendimento futuro per decile di previsione. `orizzonte` in barre da 5m."""
    pezzi = []
    for _, (idx, close, previsto) in dati.items():
        futuro = np.full(len(close), np.nan)
        futuro[:-orizzonte] = close[orizzonte:] / close[:-orizzonte] - 1.0
        tieni = (idx >= da) & (idx < a) & np.isfinite(previsto) & np.isfinite(futuro)
        pezzi.append(pd.DataFrame({"pred": previsto[tieni], "ret": futuro[tieni] * 100}))
    tutto = pd.concat(pezzi)
    tutto["decile"] = pd.qcut(tutto["pred"], 10, labels=False, duplicates="drop")
    return tutto.groupby("decile")["ret"].mean() - tutto["ret"].mean()


def indici_dei_segnali(previsto: np.ndarray, entra: float, esci: float, cadenza: int):
    """Indici di ingresso e uscita della regola a esposizione, con isteresi.

    L'isteresi non e' un abbellimento: senza, `entra == esci` fa entrare e uscire sulla stessa
    soglia e ogni oscillazione del modello attorno a quel valore costa un giro di commissioni.
    """
    ingressi, uscite, dentro = [], [], False
    for i in range(0, len(previsto), cadenza):
        if not np.isfinite(previsto[i]):
            continue
        if not dentro and abs(previsto[i]) >= entra:
            ingressi.append(i)
            dentro = True
        elif dentro and abs(previsto[i]) < esci:
            uscite.append(i)
            dentro = False
    if dentro:
        uscite.append(len(previsto) - 1)
    return list(zip(ingressi, uscite[: len(ingressi)]))


def composto(idx, close, coppie) -> float:
    """Rendimento composto della lista di operazioni, passando dal motore vero di `pnl.py`."""
    operazioni = simulate_trading_with_commisions(
        [(idx[a], close[a]) for a, _ in coppie],
        [(idx[b], close[b]) for _, b in coppie],
        wallet=100.0,
        fee_percent=FEE,
    )
    return operazioni[-1]["Wallet_After"] - 100 if operazioni else 0.0


def collocazioni_casuali(n_barre: int, durate: list[int], seme: int):
    """Le stesse durate, messe a caso e senza sovrapporsi: il motore tiene una posizione sola."""
    rng = np.random.default_rng(seme)
    inizi = np.sort(rng.integers(0, max(n_barre - max(durate) - 1, 1), size=len(durate)))
    coppie, ultimo = [], -1
    for inizio, durata in zip(inizi, durate):
        inizio = max(int(inizio), ultimo + 1)
        if inizio + durata >= n_barre:
            break
        coppie.append((inizio, inizio + durata))
        ultimo = inizio + durata
    return coppie


def controllo(dati: dict, da, a, estrazioni: int = ESTRAZIONI) -> pd.DataFrame:
    """La regola contro il p95 di `estrazioni` collocazioni casuali di pari esposizione."""
    righe = []
    for symbol, (idx, close, previsto) in dati.items():
        tieni = (idx >= da) & (idx < a)
        i2, c2, p2 = idx[tieni], close[tieni], previsto[tieni]
        coppie = indici_dei_segnali(p2, ENTRA, ESCI, CADENZA)
        if not coppie:
            continue
        vero = composto(i2, c2, coppie)
        durate = [b - a2 for a2, b in coppie]
        casuali = [
            composto(i2, c2, finte)
            for seme in range(estrazioni)
            if (finte := collocazioni_casuali(len(p2), durate, seme))
        ]
        p95 = float(np.percentile(casuali, 95))
        righe.append((symbol, len(coppie), vero, float(np.median(casuali)), p95, vero > p95))
    return pd.DataFrame(righe, columns=["simbolo", "ops", "regola_%", "caso_p50_%", "caso_p95_%", "batte"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--estrazioni", type=int, default=ESTRAZIONI)
    parser.add_argument("--ricalcola", action="store_true", help="ignora la cache delle previsioni")
    args = parser.parse_args()

    dati = previsioni(args.symbols or list(DEFAULT_SYMBOLS), VAL0 - pd.Timedelta(days=150), not args.ricalcola)
    for fase, da, a in [("validazione", VAL0, OOS0), ("fuori campione", OOS0, FINE)]:
        print(f"\n=== {fase} ===")
        eccessi = decili(dati, da, a)
        print("  eccesso per decile (48h): " + " ".join(f"{v:+.3f}" for v in eccessi))
        tabella = controllo(dati, da, a, args.estrazioni)
        battute = int(tabella["batte"].sum())
        print(tabella.to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
        attesi = 0.05 * len(tabella)
        print(f"  -> batte il p95 del caso su {battute}/{len(tabella)} simboli" f" (attesi dal caso: {attesi:.2f})")


if __name__ == "__main__":
    main()
