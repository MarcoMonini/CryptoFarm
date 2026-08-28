"""Banco della politica RL: batte il possesso passivo, e batte il caso a pari esposizione?

Le due domande sono diverse e la seconda e' quella che conta. Una politica fuori dal mercato meta'
del tempo batte il possesso passivo in un ribasso **per costruzione**: e' il metro con cui il
modello a swing era stato dichiarato non cablabile (`.claude/docs/modello-swing.md` §5.3).

Il controllo qui e' piu' stretto di quello di `swing_lab`: invece di collocare a caso durate simili,
**rimescola i blocchi della politica stessa**. Esposizione totale, numero di blocchi e loro durate
restano identici; cambia solo *dove* cadono. Cio' che resta e' esattamente il valore del *quando*.

E si misura il **rango percentile** fra le estrazioni, non quante volte si supera il p95: con 15
simboli il conteggio butta via quasi tutta l'informazione, e 0,75 successi attesi contro 2 osservati
non distinguono niente. Il rango medio atteso, se la politica non sa scegliere il momento, e' 0,50.

Le altre due misure rispondono alla domanda da cui e' partito tutto -- «evita i crolli?»:
la **discesa massima** del capitale contro quella del possesso passivo, e l'**esposizione
condizionata** ai dieci giorni peggiori di ogni simbolo.
"""

from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from cryptofarm.data.klines import DEFAULT_SYMBOLS
from cryptofarm.ml.models import load_model
from cryptofarm.ml.rl import CADENZA, FEE, posizioni, rendimento, transizioni_simbolo
from cryptofarm.ml.rl_trainer import MODEL_NAME, SINCE, dati_simbolo
from cryptofarm.paths import MARKET_DATA_DIR, MODELS_DIR

CACHE = MARKET_DATA_DIR / "rl_stati.pkl"
ESTRAZIONI = 400


def _firma() -> str:
    return json.loads((MODELS_DIR / f"{MODEL_NAME}.json").read_text())["created"]


def stati(simboli: list[str], riusa: bool = True) -> dict:
    """Feature e chiusure per simbolo. In cache: ricostruirle costa minuti.

    La cache porta la firma dell'artefatto e si invalida da sola quando il modello cambia --
    misurare un modello riaddestrato con i numeri del precedente e' il modo piu' rapido di credere
    a un miglioramento che non c'e'.
    """
    firma = _firma()
    if riusa and CACHE.exists():
        salvato = pickle.loads(CACHE.read_bytes())
        if salvato.get("firma") == firma:
            return salvato["dati"]
        print("  cache ignorata: il modello e' cambiato dall'ultima volta")
    dati = {}
    for symbol in simboli:
        pezzo = dati_simbolo(symbol, SINCE)
        if pezzo is not None:
            dati[symbol] = pezzo
            print(f"  {symbol}: {len(pezzo[0]):,} barre", flush=True)
    CACHE.write_bytes(pickle.dumps({"firma": firma, "dati": dati}))
    return dati


def blocchi(azioni: np.ndarray) -> list[tuple[int, int]]:
    """La sequenza compressa in `(valore, lunghezza)`."""
    fuori, valore, n = [], int(azioni[0]), 1
    for a in azioni[1:]:
        if int(a) == valore:
            n += 1
        else:
            fuori.append((valore, n))
            valore, n = int(a), 1
    fuori.append((valore, n))
    return fuori


def rimescolate(azioni: np.ndarray, logret: np.ndarray, estrazioni: int) -> np.ndarray:
    """Gli stessi blocchi in ordine diverso: stessa esposizione, stesse durate, momenti diversi."""
    bl = blocchi(azioni)
    fuori = np.empty(estrazioni)
    for seme in range(estrazioni):
        ordine = np.random.default_rng(seme).permutation(len(bl))
        seq = np.concatenate([np.full(bl[i][1], bl[i][0]) for i in ordine]).astype(np.int8)
        fuori[seme] = rendimento(seq, logret)
    return fuori


def discesa_massima(logret: np.ndarray, azioni: np.ndarray | None = None) -> float:
    """Discesa massima in percento del capitale, costi dei cambi inclusi."""
    if azioni is None:
        passi = logret
    else:
        cambi = np.abs(np.diff(azioni.astype(float), prepend=0.0))
        passi = azioni * logret - FEE * cambi
    equity = np.exp(np.cumsum(passi))
    return float((equity / np.maximum.accumulate(equity) - 1).min()) * 100


def misura(Q, dati: dict, da, a, cadenza: int, estrazioni: int) -> pd.DataFrame:
    righe = []
    for symbol, (idx, features, close) in dati.items():
        m = (idx >= da) & (idx < a)
        t = transizioni_simbolo(features[m], close[m], cadenza, fasi=1)
        if len(t) < 10:
            continue
        azioni = posizioni(Q, t.stato)
        vero = rendimento(azioni, t.logret)
        casuali = rimescolate(azioni, t.logret, estrazioni)
        # I dieci passi peggiori del simbolo: se la politica «evita i crolli» ci sta fuori piu'
        # che nel resto del tempo. Non e' causale -- e' il conto a posteriori, e serve a descrivere
        # il comportamento, non a dichiarare una capacita'.
        crolli = np.argsort(t.logret)[:10]
        righe.append(
            (
                symbol,
                vero,
                float(np.exp(t.logret.sum()) - 1) * 100,
                float(np.median(casuali)),
                float((casuali < vero).mean()),
                float(azioni.mean()) * 100,
                float(azioni[crolli].mean()) * 100,
                discesa_massima(t.logret, azioni),
                discesa_massima(t.logret),
                int(np.abs(np.diff(azioni, prepend=np.int8(0))).sum()),
            )
        )
    return pd.DataFrame(
        righe,
        columns=[
            "simbolo",
            "politica_%",
            "hold_%",
            "caso_p50_%",
            "rango",
            "espos_%",
            "espos_crolli_%",
            "dd_%",
            "dd_hold_%",
            "cambi",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--estrazioni", type=int, default=ESTRAZIONI)
    parser.add_argument("--cadenza", type=int, default=CADENZA)
    parser.add_argument("--ricalcola", action="store_true")
    args = parser.parse_args()

    meta = json.loads((MODELS_DIR / f"{MODEL_NAME}.json").read_text())
    Q = load_model(MODELS_DIR / f"{MODEL_NAME}.joblib")
    dati = stati(args.symbols or list(DEFAULT_SYMBOLS), not args.ricalcola)

    periodi = [
        ("validazione", pd.Timestamp(meta["validation_start"]), pd.Timestamp(meta["oos_start"])),
        ("fuori campione", pd.Timestamp(meta["oos_start"]), pd.Timestamp("2100-01-01")),
    ]
    for nome, da, a in periodi:
        tabella = misura(Q, dati, da, a, args.cadenza, args.estrazioni)
        print(f"\n=== {nome}")
        print(tabella.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))
        ranghi = tabella["rango"].to_numpy()
        p = stats.wilcoxon(ranghi - 0.5).pvalue if len(ranghi) > 5 else float("nan")
        vinte = int((tabella["politica_%"] > tabella["hold_%"]).sum())
        print(
            f"  batte il possesso passivo su {vinte}/{len(tabella)}; "
            f"rango medio fra le {args.estrazioni} rimescolate {ranghi.mean():.3f} "
            f"(atteso 0,500 se il *quando* non conta), Wilcoxon p={p:.3f}"
        )
        print(
            f"  esposizione media {tabella['espos_%'].mean():.0f}%, nei dieci passi peggiori "
            f"{tabella['espos_crolli_%'].mean():.0f}%; discesa massima mediana "
            f"{tabella['dd_%'].median():+.1f}% contro {tabella['dd_hold_%'].median():+.1f}% del possesso"
        )


if __name__ == "__main__":
    main()
