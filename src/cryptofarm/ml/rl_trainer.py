"""Addestra la politica a due azioni: `rl_model.joblib` piu' i suoi metadata.

Lo stato sono le **stesse 41 colonne** del modello a swing (`bar_features.SWING_COLUMNS`) piu' la
posizione corrente. Riusarle non e' pigrizia: sono state scelte misurando (`swing_trainer`), e
l'unica cosa che cambia qui e' la domanda che si fa sopra.

## I tre periodi, e chi sceglie cosa

| periodo | quando | a cosa serve |
|---|---|---|
| stima | dal 2019 a `--val` | addestra i due regressori |
| validazione | da `--val` a `--oos` | **sceglie i giri di iterazione** |
| fuori campione | da `--oos` in poi | si guarda una volta e non decide niente |

La validazione parte dal 2022-06 apposta: contiene il ribasso del 2022 **e** il rialzo del 2023.
Sceglierla dentro un solo regime avrebbe scelto «stare fuori dal mercato», che in un ribasso vince
sempre e non e' una capacita'.

Fra stima e validazione c'e' un embargo di `144 + cadenza` barre: 144 e' la finestra del target a
swing, e senza quello le ultime transizioni di stima guarderebbero dentro la validazione.

## Cosa **non** si sceglie qui

`COSTO`, `GAMMA` e la cadenza sono fissati in `ml/rl.py`. Sono tre manopole, e la griglia che le ha
scelte (12 celle) e' stata percorsa una volta sola in validazione: rifarla a ogni addestramento
sarebbe tarare su cio' che poi si dichiara. Chi le vuole cambiare le cambi li', deliberatamente.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines
from cryptofarm.ml.bar_features import SWING_COLUMNS, build_swing_features
from cryptofarm.ml.models import save_model
from cryptofarm.ml.rl import (
    CADENZA,
    COSTO,
    GAMMA,
    fitted_q,
    posizioni,
    rendimento,
    transizioni_simbolo,
    unisci,
)
from cryptofarm.paths import MODELS_DIR

MODEL_NAME = "rl_model"
BASE_INTERVAL = "5m"
SINCE, VAL, OOS = "2019-01-01", "2022-06-01", "2024-01-01"
FINESTRA_TARGET = 144  # la stessa di `swing_trainer.W`: e' cio' che l'embargo deve coprire
FASI = 8  # sfasature di partenza dentro la cadenza, per moltiplicare il campione
GIRI_PROVATI = (1, 2, 3, 5)


def dati_simbolo(symbol: str, since: str) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray] | None:
    candele = load_klines(symbol, BASE_INTERVAL)
    if candele.empty:
        return None
    candele = candele[candele.index >= pd.Timestamp(since)]
    if len(candele) < 60_000:
        return None
    frame = build_swing_features(symbol, candele).reindex(columns=SWING_COLUMNS)
    return candele.index, frame.to_numpy(dtype=float), candele["Close"].to_numpy(dtype=float)


def _finestra(idx, features, close, da, a):
    m = (idx >= da) & (idx < a)
    return features[m], close[m]


def valuta(Q, dati: dict, da, a, cadenza: int) -> pd.DataFrame:
    """Rendimento composto netto per simbolo, contro il possesso passivo dello stesso periodo."""
    righe = []
    for symbol, (idx, features, close) in dati.items():
        t = transizioni_simbolo(*_finestra(idx, features, close, da, a), cadenza, fasi=1)
        if len(t) < 10:
            continue
        azioni = posizioni(Q, t.stato)
        righe.append(
            (
                symbol,
                rendimento(azioni, t.logret),
                float(np.exp(t.logret.sum()) - 1) * 100,
                float(azioni.mean()) * 100,
                int(np.abs(np.diff(azioni, prepend=np.int8(0))).sum()),
            )
        )
    return pd.DataFrame(righe, columns=["simbolo", "politica_%", "hold_%", "esposizione_%", "cambi"])


def addestra(args) -> None:
    val, oos = pd.Timestamp(args.val), pd.Timestamp(args.oos)
    embargo = pd.Timedelta(minutes=5 * (FINESTRA_TARGET + args.cadenza))

    dati = {}
    for i, symbol in enumerate(args.symbols or list(DEFAULT_SYMBOLS), 1):
        t0 = time.time()
        pezzo = dati_simbolo(symbol, args.since)
        if pezzo is None:
            print(f"[{i}] {symbol}: saltato (storia insufficiente)")
            continue
        dati[symbol] = pezzo
        print(f"[{i}] {symbol}: {len(pezzo[0]):,} barre in {time.time() - t0:.0f}s", flush=True)
    if not dati:
        raise SystemExit("nessun simbolo utilizzabile: eseguire prima `python -m cryptofarm.data.klines --update`")

    batch = unisci(
        [
            transizioni_simbolo(*_finestra(*d, pd.Timestamp(args.since), val - embargo), args.cadenza, FASI)
            for d in dati.values()
        ]
    )
    print(f"\nstima: {len(batch):,} transizioni, {batch.stato.shape[1]} colonne di stato")

    migliore, scelta = None, None
    for giri in GIRI_PROVATI:
        Q = fitted_q(batch, giri=giri, gamma=args.gamma, costo=args.costo)
        tabella = valuta(Q, dati, val, oos, args.cadenza)
        punteggio = float(tabella["politica_%"].median())
        vinte = int((tabella["politica_%"] > tabella["hold_%"]).sum())
        hold = tabella["hold_%"].median()
        print(f"  giri={giri}: validazione mediana {punteggio:+.1f}% (hold {hold:+.1f}%), batte {vinte}/{len(tabella)}")
        if migliore is None or punteggio > migliore:
            migliore, scelta, Q_scelto = punteggio, giri, Q

    print(f"\ngiri scelti in validazione: {scelta}")
    for nome, da, a in [("validazione", val, oos), ("fuori campione", oos, pd.Timestamp("2100-01-01"))]:
        tabella = valuta(Q_scelto, dati, da, a, args.cadenza)
        vinte = int((tabella["politica_%"] > tabella["hold_%"]).sum())
        print(f"\n=== {nome}")
        print(tabella.to_string(index=False, float_format=lambda v: f"{v:+.1f}"))
        pol, hold = tabella["politica_%"].median(), tabella["hold_%"].median()
        print(f"  mediana {pol:+.1f}% contro hold {hold:+.1f}%, batte {vinte}/{len(tabella)}")

    save_model(Q_scelto, MODELS_DIR / f"{MODEL_NAME}.joblib")
    (MODELS_DIR / f"{MODEL_NAME}.json").write_text(
        json.dumps(
            {
                "created": pd.Timestamp.utcnow().isoformat(),
                "symbols": list(dati),
                "since": args.since,
                "validation_start": args.val,
                "oos_start": args.oos,
                "cadenza_barre": args.cadenza,
                "base_interval": BASE_INTERVAL,
                "costo": args.costo,
                "gamma": args.gamma,
                "giri": scelta,
                "fasi": FASI,
                "columns": SWING_COLUMNS,
            },
            indent=2,
        )
    )
    print(f"\nsalvato {MODELS_DIR / f'{MODEL_NAME}.joblib'}")


def selfcheck() -> None:
    """Gira senza store: serie sintetiche, e la sola cosa verificata e' che la catena stia in piedi."""
    rng = np.random.default_rng(0)
    n = 4000
    dati = {}
    for k, symbol in enumerate(("AAAUSDT", "BBBUSDT")):
        segnale = rng.normal(size=n)
        close = np.exp(np.cumsum(0.02 * segnale + 0.01 * rng.normal(size=n)))
        idx = pd.date_range("2020-01-01", periods=n, freq="1D")
        dati[symbol] = (idx, np.column_stack([segnale, rng.normal(size=n)]), close)
    batch = unisci([transizioni_simbolo(d[1], d[2], 1, 1) for d in dati.values()])
    Q = fitted_q(batch, giri=1, costo=0.0, max_iter=60)
    tabella = valuta(Q, dati, idx[0], idx[-1], cadenza=1)
    print(tabella.to_string(index=False))
    # Su una serie in cui la prima colonna *e'* il rendimento di domani, la politica deve entrare
    # e uscire davvero e battere il possesso passivo. Controllare solo che l'esposizione stia fra
    # 0 e 100 passerebbe anche con una politica che non fa mai niente -- ed e' quello che faceva.
    assert len(tabella) == 2
    assert tabella["esposizione_%"].between(5, 95).all(), tabella["esposizione_%"].tolist()
    assert (tabella["politica_%"] > tabella["hold_%"]).all(), tabella.to_string(index=False)
    print("selfcheck ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--val", default=VAL)
    parser.add_argument("--oos", default=OOS)
    parser.add_argument("--cadenza", type=int, default=CADENZA)
    parser.add_argument("--costo", type=float, default=COSTO)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    args = parser.parse_args()
    selfcheck() if args.selfcheck else addestra(args)


if __name__ == "__main__":
    main()
