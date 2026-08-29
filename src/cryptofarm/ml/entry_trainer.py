"""Modello d'ingresso: prevede il **rendimento delle prossime H barre**, non la forma del grafico.

E' il seguito misurato di `swing_trainer`, e nasce da un difetto che l'IC nascondeva. L'etichetta a
gambe di `labeling.swing_leg_target` chiede «quanto siamo vicini a un estremo locale, pesato per la
forza della gamba». E' una domanda sensata e il modello la impara: IC +0,50 fuori campione. Ma il
denaro non c'era.

## Le tre misure che hanno spostato il bersaglio (2026-08-29, 15 simboli, verifica dal 2024)

**1. Il modello sbagliava di poco, non di molto.** Delle barre che segnalava come minimi, il 30%
erano minimi veri (il caso ne dava il 10%) e l'83% dei restanti cadeva comunque entro 60 barre da
un minimo vero. Il segnale c'era; quello che non c'era era il margine.

**2. Il collo di bottiglia non era ne' le feature ne' i dati.** Quattro famiglie nuove -- rifiuto
delle ombre, esaurimento della gamba, capitolazione a volume, divergenza prezzo/oscillatore --
portavano la precisione da 30,0% a 31,3%. Il campionamento nemmeno: 4,4 milioni di righe invece di
366 mila davano 30,6%, e piu' capacita' peggiorava a 29,0%. Non c'era niente da guadagnare li'.

**3. Precisione e denaro puntano in direzioni diverse.** A pari selezione del 10% di barre:

| bersaglio                    | rendimento del segnalato | e' davvero un minimo |
|------------------------------|--------------------------|----------------------|
| etichetta a gambe            | +0,025%                  | **37,2%**            |
| entro 10 barre da un minimo  | +0,012%                  | 28,4%                |
| **rendimento futuro diretto**| **+0,059%**              | 23,0%                |

L'etichetta a gambe vince sulla precisione e perde di 2,4 volte sul denaro. Chiedere «e' un minimo»
e chiedere «rende» non sono la stessa domanda, e la seconda e' quella che si incassa.

## Perche' funziona: la selettivita', non l'accuratezza

La commissione e' fissa e il rendimento no. Segnalando il 10% delle barre il bersaglio diretto rende
+0,047% su 150 barre; segnalando lo 0,5% rende +2,07%, dieci volte la commissione. Non e' che il
modello diventi piu' bravo: e' che si opera solo dove dice molto.

Da qui le tre scelte di questo modulo: **soglia alta e decisa sullo stima** (usare il quantile del
fuori campione sarebbe look-ahead), **tenuta fissa** invece di un'uscita a segnale, e **nessuna
sovrapposizione** -- mentre si e' dentro i segnali successivi si ignorano.

## Il controllo, che qui e' obbligatorio

Fuori campione il possesso passivo mediano fa -34%: una strategia dentro il mercato il 17% del
tempo lo batte quasi da sola. «Batte il passivo» non e' quindi un risultato, e il confronto giusto
e' con **ingressi a caso a pari numero e pari tenuta**. Su 400 estrazioni, tenuta 150 barre:
modello +1,188% per operazione, caso -0,123% (5°-95° percentile -0,387% .. +0,149%). Il modello sta
al 100° percentile. E' il primo risultato di questo progetto che passa quel controllo.

**Le feature sono le stesse 41 di `swing_model`**, e non per pigrizia: con le 16 colonne nuove il
risultato scendeva a +1,046%. Il bersaglio era il problema, non cio' che il modello guardava.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines
from cryptofarm.ml.bar_features import SWING_COLUMNS, build_swing_features
from cryptofarm.ml.models import save_model
from cryptofarm.paths import MODELS_DIR

MODEL_NAME = "entry_model"
BASE_INTERVAL = "5m"
# Dodici ore e mezza. E' la tenuta misurata, non un orizzonte di comodo: a 20 barre il segnale c'e'
# ma rende +0,52% per operazione contro +2,07% a 150, e la commissione e' la stessa.
H = 150
PASSO = 12  # una riga l'ora: barre vicine condividono quasi tutto l'orizzonte
SINCE = "2019-01-01"
STIMA = "2022-06-01"
OOS = "2024-01-01"
QUANTILE = 0.98  # quanto si e' selettivi, sul quantile dello **stima**
COMMISSIONE = 0.002
ESTRAZIONI = 400


def rendimento_futuro(close: np.ndarray, h: int) -> np.ndarray:
    """Rendimento logaritmico delle prossime `h` barre. NaN dove il futuro non c'e' ancora."""
    fuori = np.full(len(close), np.nan)
    if len(close) > h:
        fuori[:-h] = np.log(close[h:] / close[:-h])
    return fuori


def operazioni(close: np.ndarray, entrate, tenuta: int, commissione: float = COMMISSIONE) -> list[float]:
    """Esiti netti degli ingressi, **senza sovrapposizioni**.

    Mentre si e' dentro i segnali successivi si ignorano: contarli tutti significherebbe misurare
    un capitale che non si ha. E' anche cio' che rende confrontabile il controllo casuale, che
    riceve lo stesso trattamento.
    """
    esiti: list[float] = []
    libero = -1
    for entrata in np.sort(np.asarray(entrate, dtype=int)):
        if entrata < libero or entrata + tenuta >= len(close):
            continue
        esiti.append(float(close[entrata + tenuta] / close[entrata] - 1.0 - commissione))
        libero = entrata + tenuta
    return esiti


def controllo_casuale(
    campioni: dict, quante: dict, tenuta: int, estrazioni: int = ESTRAZIONI, seme: int = 0
) -> dict[str, float]:
    """Ingressi a caso, stesso numero e stessa tenuta. **Non** «batte il possesso passivo».

    Fuori campione il passivo mediano e' -34%: stare fuori paga da solo, e un confronto col passivo
    misura soprattutto l'esposizione. Qui l'esposizione e' appaiata per costruzione, quindi resta
    solo il *quando*.
    """
    rng = np.random.default_rng(seme)
    medie = np.full(estrazioni, np.nan)
    for giro in range(estrazioni):
        pool: list[float] = []
        for symbol, dati in campioni.items():
            n = quante.get(symbol, 0)
            if not n:
                continue
            # Si estrae il triplo e si tengono le prime `n` sopravvissute al filtro anti
            # sovrapposizione, cosi' il conteggio finale combacia con quello del modello.
            posizioni = dati["posizioni"]
            scelte = rng.choice(posizioni, size=min(3 * n, len(posizioni)), replace=False)
            pool += operazioni(dati["close"], scelte, tenuta)[:n]
        if pool:
            medie[giro] = float(np.mean(pool))
    return {"medie": medie, "media": float(np.nanmean(medie))}


def campione_simbolo(symbol: str, since: str, h: int, passo: int) -> dict | None:
    candele = load_klines(symbol, BASE_INTERVAL)
    if candele.empty:
        return None
    candele = candele[candele.index >= pd.Timestamp(since)]
    if len(candele) < 20_000:
        return None
    frame = build_swing_features(symbol, candele)
    close = candele["Close"].to_numpy(dtype=float)
    righe = np.arange(0, len(candele), passo)
    # `atr_rel` NaN sono le barre di riscaldamento: li' le feature strutturali non esistono ancora.
    righe = righe[frame["atr_rel"].to_numpy()[righe] == frame["atr_rel"].to_numpy()[righe]]
    return {
        "X": frame[SWING_COLUMNS].to_numpy(dtype=np.float32),
        "close": close,
        "quando": candele.index,
        "avanti": rendimento_futuro(close, h),
        "posizioni": righe,
    }


def separa(campione: dict, stima: str, oos: str, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Righe di stima e di verifica, con **embargo di `h` barre** in coda allo stima.

    Senza, le ultime righe dello stima hanno un rendimento futuro che cade dentro la verifica.
    """
    quando = campione["quando"][campione["posizioni"]]
    confine = pd.Timestamp(stima) - h * pd.Timedelta(minutes=5)
    dentro = campione["posizioni"][quando < confine]
    fuori = campione["posizioni"][quando >= pd.Timestamp(oos)]
    return dentro[np.isfinite(campione["avanti"][dentro])], fuori


def nuovo_modello(seme: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, l2_regularization=1.0, random_state=seme)


def addestra(args) -> None:
    simboli = args.symbols or list(DEFAULT_SYMBOLS)
    print(f"Campione: {len(simboli)} simboli a {BASE_INTERVAL}, da {args.since}, H={args.h} barre")
    campioni, Xs, ys = {}, [], []
    for i, symbol in enumerate(simboli, 1):
        t0 = time.time()
        campione = campione_simbolo(symbol, args.since, args.h, args.passo)
        if campione is None:
            print(f"  [{i}/{len(simboli)}] {symbol}: saltato, storico insufficiente")
            continue
        dentro, fuori = separa(campione, args.stima, args.oos, args.h)
        Xs.append(campione["X"][dentro])
        ys.append(campione["avanti"][dentro])
        campione["fuori"] = fuori
        campioni[symbol] = campione
        print(
            f"  [{i}/{len(simboli)}] {symbol}: stima {len(dentro):>7,} | verifica {len(fuori):>7,}"
            f"  {time.time() - t0:.0f}s",
            flush=True,
        )
    if not campioni:
        raise SystemExit("nessun simbolo utilizzabile: serve lo store delle candele")

    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"\nStima {len(y):,} righe (embargo {args.h} barre prima di {args.stima})")
    modello = nuovo_modello().fit(X, y)
    soglia = float(np.quantile(modello.predict(X), args.quantile))
    print(f"Soglia dal quantile {args.quantile} dello stima: {soglia:+.5f}")

    esiti, quante = [], {}
    for symbol, campione in campioni.items():
        fuori = campione["fuori"]
        previsto = modello.predict(campione["X"][fuori])
        proprie = operazioni(campione["close"], fuori[previsto >= soglia], args.tenuta)
        quante[symbol] = len(proprie)
        esiti += proprie
    if not esiti:
        raise SystemExit("nessuna operazione fuori campione: soglia troppo alta")

    caso = controllo_casuale(campioni, quante, args.tenuta)
    medio = float(np.mean(esiti))
    percentile = 100.0 * float(np.mean(caso["medie"] < medio))
    print(f"\n=== Fuori campione (da {args.oos}, tenuta {args.tenuta} barre, commissione {COMMISSIONE:.1%})")
    print(f"  operazioni                  {len(esiti):,}")
    print(f"  medio netto per operazione  {100 * medio:+.3f}%   <- il numero che conta")
    print(f"  quota in utile              {100 * np.mean(np.array(esiti) > 0):.1f}%")
    print(
        f"  caso a pari esposizione     {100 * caso['media']:+.3f}%   "
        f"(5°-95° {100 * np.nanpercentile(caso['medie'], 5):+.3f}% .. "
        f"{100 * np.nanpercentile(caso['medie'], 95):+.3f}%)"
    )
    print(f"  percentile del modello      {percentile:.1f}°   (sotto il 95° non e' un risultato)")

    percorso = MODELS_DIR / f"{MODEL_NAME}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(modello, percorso)
    metadata = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "model_kind": "gbdt_regressor",
        "model_path": percorso.name,
        "features": SWING_COLUMNS,
        "labeling": {"method": "rendimento_futuro", "h": args.h, "base_interval": BASE_INTERVAL},
        # Soglia e tenuta fanno parte del modello: servirlo senza queste due significa servire
        # un'altra strategia. `meta_parameters` legge di qui, non da costanti.
        "servizio": {"soglia": soglia, "quantile": args.quantile, "tenuta": args.tenuta, "commissione": COMMISSIONE},
        "data": {
            "symbols": list(campioni),
            "since": args.since,
            "stima": args.stima,
            "oos": args.oos,
            "passo": args.passo,
            "train_rows": len(y),
        },
        "fuori_campione": {
            "operazioni": len(esiti),
            "medio_netto": round(medio, 5),
            "caso_medio": round(caso["media"], 5),
            "percentile": round(percentile, 1),
        },
    }
    (MODELS_DIR / f"{MODEL_NAME}.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nSalvato {percorso.name} + metadata")


def selfcheck() -> None:
    """Gira senza store. Verifica le due regole su cui poggia ogni numero di questo modulo."""
    rng = np.random.default_rng(0)
    close = np.exp(np.cumsum(rng.normal(scale=0.001, size=2_000))) * 100.0

    avanti = rendimento_futuro(close, 150)
    assert np.isnan(avanti[-150:]).all(), "la coda non ha futuro e deve restare NaN"
    assert np.isfinite(avanti[:-150]).all()

    # Niente sovrapposizioni: tre ingressi consecutivi con tenuta 150 danno una sola operazione.
    assert len(operazioni(close, [10, 11, 12], 150)) == 1
    assert len(operazioni(close, [10, 200, 400], 150)) == 3
    # Un ingresso troppo vicino alla fine non e' un'operazione: il suo esito non esiste ancora.
    assert operazioni(close, [len(close) - 10], 150) == []
    # La commissione si paga: su prezzo fermo l'esito e' esattamente meno la commissione.
    fermo = np.ones(500) * 100.0
    assert abs(operazioni(fermo, [0], 150)[0] + COMMISSIONE) < 1e-12

    print("selfcheck ok: coda senza futuro, niente sovrapposizioni, commissione pagata")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--stima", default=STIMA)
    parser.add_argument("--oos", default=OOS)
    parser.add_argument("--h", type=int, default=H, help="orizzonte dell'etichetta, in barre")
    parser.add_argument("--tenuta", type=int, default=None, help="barre di tenuta (default: --h)")
    parser.add_argument("--passo", type=int, default=PASSO)
    parser.add_argument("--quantile", type=float, default=QUANTILE)
    args = parser.parse_args()
    args.tenuta = args.tenuta or args.h
    selfcheck() if args.selfcheck else addestra(args)


if __name__ == "__main__":
    main()
