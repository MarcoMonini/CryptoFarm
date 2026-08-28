"""Modello a swing: prevede la **prossimita' agli estremi locali**, non l'ampiezza del movimento.

Cosa cambia rispetto a `leg_trainer`, e perche'. La barriera tripla chiede «il prezzo si muove di
1,5 ATR in su o in giu' entro l'orizzonte»: e' una domanda sulla **volatilita'**, e siccome le
barriere sono gia' scalate sull'ATR l'etichetta normalizza via proprio la parte prevedibile.
Qui l'etichetta e' `labeling.swing_target`, il rango centrato della chiusura: -1 su un minimo
locale, +1 su un massimo, 0 dentro una tendenza regolare. E' una domanda sulla **forma**, ed e'
quella che serve a comprare prima di una gamba invece che a meta'.

**Il metro.** Il rango centrato guarda anche le `W` barre passate, che le feature gia' descrivono:
uno Stochastic senza modello prende IC 0,70 contro il target pieno. Valutare li' misurerebbe la
capacita' di rifare uno Stochastic. Tutte le cifre di questo modulo sono quindi contro
`swing_target(..., verso="avanti")`, la sola meta' non conoscibile, dove il riferimento causale
vale circa 0,043 e c'e' spazio per battere qualcosa.

**Cosa e' stato misurato prima di scrivere questo file** (sette simboli, verifica 2024-2026,
IC di Spearman contro la meta' futura):

| variante                                   | colonne | IC      |
|--------------------------------------------|---------|---------|
| `pos_canale` da solo, nessun modello        |       1 | +0,0433 |
| base 5m                                     |      15 | +0,0502 |
| base + storico a -1 e -2 barre              |      45 | +0,0509 |
| base + storico fino a -32 barre             |     105 | +0,0498 |
| base + Target ritardato di W+1              |      16 | +0,0497 |
| **base + aggregazione 1h e 1d**             |  **41** | **+0,0540** |
| base + Target ritardato di 1 barra          |      16 | +0,6729 |

Da cui le tre decisioni: **niente storico esplicito** (le feature gia' portano EMA200, ADX e OBV
a 20 barre, e ricopiarle indietro costa il triplo delle colonne per due millesimi di IC);
**niente Target fra le feature** -- l'ultima riga non e' un risultato ma la misura del danno, e
serve a ricordare che al ritardo di una barra il target condivide 143 delle sue 144 barre con
quello di oggi, quindi quel +0,67 e' la fuga di informazione, non un modello; **aggregazione a 1h
e 1d**, che sono le due scale che aggiungono qualcosa che la barra base non contiene gia'.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines
from cryptofarm.ml.bar_features import SWING_COLUMNS, build_swing_features
from cryptofarm.ml.labeling import swing_target
from cryptofarm.ml.models import save_model
from cryptofarm.paths import MODELS_DIR

MODEL_NAME = "swing_model"
BASE_INTERVAL = "5m"
# Mezza giornata per lato. Sotto, il target insegue il rumore a cinque minuti; sopra, l'embargo
# mangia il campione e l'estremo smette di essere «locale».
W = 144
PASSO = 6  # una riga ogni trenta minuti: barre vicine condividono quasi tutta la finestra
SINCE = "2018-01-01"
OOS = "2024-01-01"
# Coda dello stima tenuta per scegliere il numero di giri. Il fuori campione non la sceglie:
# tararci sopra anche un solo iperparametro e' il difetto per cui il modello precedente
# dichiarava di passare (`.claude/docs/`, ciclo di dubbio del 2026-08-28).
QUOTA_VALIDAZIONE = 0.15
GIRI = 3  # quanti riaddestramenti su etichette riviste provare
PIEGHE = 3
ALPHA = 0.3  # quanto pesa la predizione fuori piega nella nuova etichetta


def nuovo_modello(seme: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=seme)


def campione_simbolo(symbol: str, since: str, w: int, passo: int) -> pd.DataFrame | None:
    """Feature, target pieno e meta' futura per un simbolo, gia' diradati."""
    candele = load_klines(symbol, BASE_INTERVAL)
    if candele.empty:
        return None
    candele = candele[candele.index >= pd.Timestamp(since)]
    if len(candele) < 20_000:
        return None
    frame = build_swing_features(symbol, candele)
    frame["Target"] = swing_target(candele["Close"], w)
    frame["avanti"] = swing_target(candele["Close"], w, verso="avanti")
    frame = frame.iloc[::passo]
    # `atr_rel` NaN sono le barre di riscaldamento: li' le feature strutturali non esistono ancora.
    frame = frame[frame["Target"].notna() & frame["atr_rel"].notna()]
    frame["simbolo"] = symbol
    for colonna in SWING_COLUMNS:
        frame[colonna] = frame[colonna].astype(np.float32)
    return frame


def costruisci(symbols, since=SINCE, w=W, passo=PASSO, verboso=True) -> pd.DataFrame:
    pezzi = []
    for i, symbol in enumerate(symbols, 1):
        t0 = time.time()
        frame = campione_simbolo(symbol, since, w, passo)
        if frame is None:
            if verboso:
                print(f"  [{i}/{len(symbols)}] {symbol}: saltato, storico insufficiente")
            continue
        pezzi.append(frame)
        if verboso:
            print(
                f"  [{i}/{len(symbols)}] {symbol}: {len(frame):>7,} righe "
                f"({frame.index[0].date()} .. {frame.index[-1].date()}) {time.time() - t0:.0f}s",
                flush=True,
            )
    return pd.concat(pezzi).sort_index()


def taglia(dati: pd.DataFrame, oos: str, w: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stima e verifica, con **embargo di `w` barre** prima del taglio.

    Senza, le ultime righe dello stima hanno un target che guarda dentro la verifica: le due parti
    condividono futuro e il numero fuori campione e' gonfio.
    """
    confine = pd.Timestamp(oos)
    embargo = confine - w * pd.Timedelta(minutes=5)
    return dati[dati.index < embargo], dati[dati.index >= confine]


def rietichetta(X, y, k: int = PIEGHE, alpha: float = ALPHA, seme: int = 0) -> np.ndarray:
    """Etichette riviste con le predizioni **fuori piega** del modello stesso.

    E' la parte di riaddestramento per rinforzo: il modello rilegge i propri dati e le etichette
    si spostano verso cio' che il modello sa gia' spiegare, il che toglie rumore all'etichetta
    dove il rumore non e' apprendibile. Fuori piega e non in campione, perche' un GBDT in campione
    riproduce anche il rumore che ha memorizzato e il giro successivo lo rinforzerebbe.
    """
    predetto = np.empty(len(y), dtype=np.float32)
    for piega in np.array_split(np.arange(len(y)), k):
        resto = np.ones(len(y), dtype=bool)
        resto[piega] = False
        modello = nuovo_modello(seme)
        modello.fit(X[resto], y[resto])
        predetto[piega] = modello.predict(X[piega])
    return (1.0 - alpha) * y + alpha * predetto


def ic(previsto: np.ndarray, vero: np.ndarray) -> float:
    return float(spearmanr(previsto, vero).statistic)


def ic_per_simbolo(previsto: np.ndarray, dati: pd.DataFrame) -> tuple[float, int, int]:
    valori = [
        ic(previsto[(dati["simbolo"] == s).to_numpy()], dati.loc[dati["simbolo"] == s, "avanti"])
        for s in dati["simbolo"].unique()
    ]
    return float(np.median(valori)), int(np.sum(np.array(valori) > 0)), len(valori)


def giri_di_rinforzo(X, y, giri: int, seme: int = 0):
    """Addestra, rietichetta, riaddestra. Restituisce un modello per giro, dal giro 0 in poi."""
    modelli = []
    etichetta = np.asarray(y, dtype=np.float32)
    for giro in range(giri + 1):
        modello = nuovo_modello(seme)
        modello.fit(X, etichetta)
        modelli.append(modello)
        if giro < giri:
            etichetta = rietichetta(X, etichetta, seme=seme)
    return modelli


def addestra(args) -> None:
    simboli = args.symbols or list(DEFAULT_SYMBOLS)
    print(f"Campione: {len(simboli)} simboli a {BASE_INTERVAL}, da {args.since}, W={args.w}")
    dati = costruisci(simboli, args.since, args.w, args.passo)
    dentro, fuori = taglia(dati, args.oos, args.w)
    if len(dentro) < 10_000 or len(fuori) < 5_000:
        raise SystemExit(f"campione insufficiente: stima {len(dentro):,}, verifica {len(fuori):,}")

    # La coda dello stima serve a scegliere i giri. Anche qui l'embargo, per la stessa ragione.
    confine = dentro.index[int(len(dentro) * (1 - QUOTA_VALIDAZIONE))]
    fit = dentro[dentro.index < confine - args.w * pd.Timedelta(minutes=5)]
    val = dentro[dentro.index >= confine]
    print(
        f"\nRighe: stima {len(fit):,} | validazione {len(val):,} | verifica {len(fuori):,}"
        f"  (taglio {args.oos}, embargo {args.w} barre)"
    )

    riferimento = ic(val["pos_canale"].to_numpy(), val["avanti"].to_numpy())
    print(f"\nScelta dei giri sulla validazione (riferimento causale pos_canale: {riferimento:+.4f})")
    modelli = giri_di_rinforzo(fit[SWING_COLUMNS].to_numpy(), fit["Target"].to_numpy(), args.giri)
    ic_val = []
    for giro, modello in enumerate(modelli):
        valore = ic(modello.predict(val[SWING_COLUMNS].to_numpy()), val["avanti"].to_numpy())
        ic_val.append(valore)
        print(f"  giro {giro}: IC validazione {valore:+.4f}")
    scelto = int(np.argmax(ic_val))
    print(f"  -> scelto il giro {scelto}")

    # Solo ora si riaddestra su tutto lo stima e si guarda il fuori campione, una volta sola.
    finali = giri_di_rinforzo(dentro[SWING_COLUMNS].to_numpy(), dentro["Target"].to_numpy(), scelto)
    modello = finali[-1]
    previsto = modello.predict(fuori[SWING_COLUMNS].to_numpy())
    ic_fuori = ic(previsto, fuori["avanti"].to_numpy())
    mediana, concordi, totali = ic_per_simbolo(previsto, fuori)
    rif_fuori = ic(fuori["pos_canale"].to_numpy(), fuori["avanti"].to_numpy())
    ic_pieno = ic(previsto, fuori["Target"].to_numpy())

    print(f"\n=== Fuori campione ({fuori.index[0].date()} .. {fuori.index[-1].date()}) ===")
    print(f"  IC contro la meta' futura   {ic_fuori:+.4f}   <- il numero che conta")
    print(f"  riferimento causale         {rif_fuori:+.4f}   (pos_canale, nessun modello)")
    print(f"  eccesso sul riferimento     {ic_fuori - rif_fuori:+.4f}")
    print(f"  mediana per simbolo         {mediana:+.4f}   ({concordi}/{totali} concordi di segno)")
    print(f"  IC contro il target pieno   {ic_pieno:+.4f}   (non e' un risultato: 93% e' passato)")

    percorso = MODELS_DIR / f"{MODEL_NAME}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(modello, percorso)
    metadata = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "model_kind": "gbdt_regressor",
        "model_path": percorso.name,
        "features": SWING_COLUMNS,
        "labeling": {"method": "swing_target", "window": args.w, "base_interval": BASE_INTERVAL},
        "data": {
            "symbols": simboli,
            "since": args.since,
            "oos": args.oos,
            "passo": args.passo,
            "rows": len(dati),
            "train_rows": len(dentro),
            "test_rows": len(fuori),
        },
        "rinforzo": {"giri_provati": args.giri, "giro_scelto": scelto, "ic_validazione": ic_val},
        "ic_futuro": round(ic_fuori, 4),
        "ic_riferimento": round(rif_fuori, 4),
        "ic_mediana_simbolo": round(mediana, 4),
        "simboli_concordi": f"{concordi}/{totali}",
    }
    (MODELS_DIR / f"{MODEL_NAME}.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nSalvato {percorso.name} + metadata")


def selfcheck() -> None:
    """Gira senza store: verifica che su una passeggiata senza deriva il modello non trovi nulla."""
    rng = np.random.default_rng(0)
    n = 20_000
    close = pd.Series(rng.normal(size=n).cumsum() + 1000.0)
    target = swing_target(close, 48)
    avanti = swing_target(close, 48, verso="avanti")
    dietro = swing_target(close, 48, verso="dietro")
    valide = target.notna()
    assert abs(float(target[valide].mean())) < 0.05, "il target deve essere simmetrico"
    assert -1.0 <= float(target[valide].min()) and float(target[valide].max()) <= 1.0
    # La meta' passata spiega il target pieno; la meta' futura no. E' l'intero argomento del metro.
    assert ic(dietro[valide].to_numpy(), target[valide].to_numpy()) > 0.6
    assert abs(ic(dietro[valide].to_numpy(), avanti[valide].to_numpy())) < 0.1
    print("selfcheck ok: target simmetrico, meta' passata separata dalla meta' futura")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--oos", default=OOS)
    parser.add_argument("--w", type=int, default=W)
    parser.add_argument("--passo", type=int, default=PASSO)
    parser.add_argument("--giri", type=int, default=GIRI)
    args = parser.parse_args()
    selfcheck() if args.selfcheck else addestra(args)


if __name__ == "__main__":
    main()
