"""Il modello delle gambe: un solo addestramento, tre classi, due segnali.

    python -m cryptofarm.ml.leg_trainer                      # addestra e valuta
    python -m cryptofarm.ml.leg_trainer --oos 2024-06-01     # con un altro taglio temporale
    python -m cryptofarm.ml.leg_trainer --selfcheck          # senza store, dati finti

## La domanda, e perche' e' diversa da quelle gia' fallite

Da ogni barra: il prezzo tocca `+k x ATR` **prima** di `-k x ATR` (SU), il contrario (GIU'), o
nessuno dei due entro l'orizzonte (FERMO). Una softmax su tre classi mutuamente esclusive:
`P(su)` e' il segnale d'ingresso, `P(giu)` quello d'uscita e il voto a -1 nella confluenza.

Le barriere sono **simmetriche**, e questa e' l'unica differenza che conta rispetto a
`ml/labeling.py`, dove sono 1,5 : 1. Li' la classe SELL significa «lo stop di una posizione lunga
e' stato toccato per primo»: copre circa il 60% delle barre e confonde «scende» con «scende un po'
e poi sale» -- ed e' la ragione scritta in `ml/signals.py` per cui quella classe non si puo' usare
come segnale di vendita. Qui GIU' significa «e' sceso di `k x ATR` prima di salirne altrettanti»,
cioe' la gamba ribassista, e le due classi direzionali diventano confrontabili fra loro. E' la
proprieta' che serve a mettere `P(su)` contro `P(giu)`.

Il vincolo economico resta **dentro** l'etichetta: `barrier_widths` tiene ogni barriera sopra tre
volte le commissioni di andata e ritorno, quindi una barra SU e' per costruzione un movimento che
paga il proprio giro.

## Cosa questo modello **non** sa

Non sa se una posizione e' aperta. E' una scelta, non una dimenticanza: la politica a tre azioni
condizionata sullo stato e' il disegno chiuso in negativo di `strategy.md` §11-12, e uno stato
nelle feature lega il modello alla strategia che lo ha generato. Qui l'opinione e' sulla barra, e
per questo l'artefatto e' **identico** per i due consumatori -- la voce di menu e il votante.

## Il numero da battere, dichiarato prima di guardarlo

Non l'AUC. Il netto medio per operazione, fuori campione, contro il **p95 di 500 selezioni casuali
della stessa numerosita'** -- lo stesso controllo di `scripts/meta_gate.py` e `scripts/ai_voter.py`,
che nessun disegno di questo progetto ha ancora superato in modo stabile. E deve passare su **due
soglie adiacenti**, perche' un picco solo fra soglie vicine e' rumore ed e' gia' successo.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes, load_klines
from cryptofarm.ml.bar_features import CROSS_COLUMNS, FEATURE_COLUMNS, build_bar_features, cross_features
from cryptofarm.ml.labeling import BUY as SU
from cryptofarm.ml.labeling import HOLD as FERMO
from cryptofarm.ml.labeling import SELL as GIU
from cryptofarm.ml.labeling import triple_barrier_events
from cryptofarm.ml.models import build_model, fit_model, predict_proba, save_model
from cryptofarm.paths import MODELS_DIR
from cryptofarm.trading.indicators_extra import ExtraCache

MODEL_NAME = "leg_model"
INTERVALS = ("1h", "4h", "1d")
# Sotto l'ora nessuna misura di questo progetto ha mai trovato qualcosa che batta il possesso
# passivo, e la correlazione fra operazioni all'anno e resa e' -0,60: quella regione resta fuori.

SINCE = "2021-12-01"  # dove arriva lo store di posizionamento su tutti i simboli tranne BTC
ORIZZONTE_ORE = 168  # sette giorni, uguale su tutti gli intervalli invece che in barre
BARRIERA_ATR = 1.5  # `k`: simmetrica, sopra e sotto
COMMISSIONE = 0.0005  # per lato, listino taker sui perpetui
GIRO = COMMISSIONE * 2
PAVIMENTO = 3.0  # nessuna barriera sotto 3x il giro
# Quota di righe in cui le tre feature trasversali vengono nascoste durante l'addestramento, cosi'
# che il modello impari a comportarsi anche quando al momento di servire non ci sono.
MASCHERA_TRASVERSALI = 0.30
# Quota di barre su cui l'uscita a modello puo' scattare. E' una frequenza dichiarata prima, non
# un livello: l'ablazione mostra che un'uscita frequente distrugge il risultato perche' tronca la
# coda destra, quindi il parametro che conta e' quanto **raramente** scatta.
QUOTA_USCITA = 0.05
NOMI = {FERMO: "fermo", SU: "su", GIU: "giu"}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _orizzonte(interval: str) -> int:
    """L'orizzonte in barre per questo intervallo. Almeno tre barre, o non e' un movimento."""
    return max(3, int(ORIZZONTE_ORE * 60 / interval_to_minutes(interval)))


def campione_simbolo(
    symbol: str,
    interval: str,
    candles: pd.DataFrame,
    cross: dict,
    barriera: float = BARRIERA_ATR,
) -> pd.DataFrame:
    """Feature ed etichetta per ogni barra utilizzabile di un simbolo a un intervallo."""
    cache = ExtraCache(candles)
    feature = build_bar_features(symbol, candles, interval, cross=cross, cache=cache)

    # `triple_barrier_events` vuole l'ATR gia' in percentuale del Close, come lo produce la
    # pipeline vecchia. Qui l'ATR sta in `atr_rel` come frazione, quindi va riportato in scala.
    da_etichettare = candles[["High", "Low", "Close"]].copy()
    da_etichettare["ATR"] = feature["atr_rel"].to_numpy() * 100.0

    orizzonte = _orizzonte(interval)
    eventi = triple_barrier_events(
        da_etichettare,
        horizon=orizzonte,
        tp_multiple=barriera,
        sl_multiple=barriera,
        round_trip_fee=GIRO,
        fee_floor_multiple=PAVIMENTO,
    )

    frame = feature.copy()
    frame["y"] = eventi["Label"].to_numpy()
    frame["rendimento_%"] = eventi["exit_return"].to_numpy() * 100.0
    frame["t_exit"] = eventi["t_exit"].to_numpy()
    frame["simbolo"] = symbol
    frame["intervallo"] = interval
    frame["ambiguo"] = eventi["ambiguous"].to_numpy()

    # Le ultime `orizzonte` barre non hanno futuro osservabile: la loro etichetta e' FERMO per
    # mancanza di dati, non perche' non sia successo niente.
    frame = frame.iloc[:-orizzonte] if orizzonte < len(frame) else frame.iloc[:0]

    # Le righe ambigue -- entrambe le barriere nella stessa candela -- direbbero "scende" mentre
    # il dato non dice niente. Con barriere simmetriche e un modello direzionale si tolgono.
    frame = frame[~frame["ambiguo"]].drop(columns=["ambiguo"])

    # Le barre di riscaldamento hanno un ATR non definito, quindi barriere prese dal pavimento
    # invece che dalla volatilita': l'etichetta che ne esce non descrive lo stesso esperimento
    # delle altre. Si tolgono qui invece di lasciarle pesare come le buone.
    return frame[frame["atr_rel"].notna()].dropna(subset=["y"])


def costruisci(
    symbols: list[str],
    intervals: tuple[str, ...] = INTERVALS,
    since: str = SINCE,
    barriera: float = BARRIERA_ATR,
    verbose: bool = True,
) -> pd.DataFrame:
    """Il campione completo: tutti i simboli, tutti gli intervalli, una riga per barra tenuta.

    Le etichette di barre vicine condividono quasi tutto il loro futuro, quindi si campiona con
    passo: tenerle tutte moltiplica le righe senza aggiungere informazione, e gonfia la fiducia
    nelle metriche. Il passo e' un ottavo dell'orizzonte, cioe' quasi un giorno a ogni scala.
    """
    parti = []
    for interval in intervals:
        per_simbolo = {}
        for symbol in symbols:
            candles = load_klines(symbol, interval)
            candles = candles[candles.index >= since]
            if len(candles) < 400:
                continue
            per_simbolo[symbol] = candles

        if not per_simbolo:
            continue
        chiusure = pd.DataFrame({s: c["Close"] for s, c in per_simbolo.items()}).sort_index()
        cross = cross_features(chiusure)

        passo = max(1, _orizzonte(interval) // 8)
        for symbol, candles in per_simbolo.items():
            frame = campione_simbolo(symbol, interval, candles, cross, barriera)
            if frame.empty:
                continue
            parti.append(frame.iloc[::passo])

        if verbose:
            righe = sum(len(p) for p in parti)
            print(f"  {interval}: {len(per_simbolo)} simboli, passo {passo} -> {righe:,} righe totali", flush=True)

    if not parti:
        raise RuntimeError("Nessun dato: popolare lo store con `cryptofarm.data.klines --update`")
    return pd.concat(parti).sort_index()


def distribuzione(y: np.ndarray, stadio: str) -> str:
    pezzi = [f"{NOMI[c]}={int((y == c).sum()):,} ({(y == c).mean():.1%})" for c in (SU, GIU, FERMO)]
    return f"[{stadio}] {len(y):,} righe | " + ", ".join(pezzi)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC per ranghi di una classe contro tutte le altre."""
    ordine = np.argsort(p)
    ranghi = np.empty(len(p), dtype=float)
    ranghi[ordine] = np.arange(1, len(p) + 1)
    positivi, negativi = y.sum(), (1 - y).sum()
    if positivi == 0 or negativi == 0:
        return float("nan")
    return float((ranghi[y == 1].sum() - positivi * (positivi + 1) / 2) / (positivi * negativi))


def controllo_casuale(netto: np.ndarray, quanti: int, prove: int = 500, seme: int = 12345) -> tuple[float, np.ndarray]:
    """Il netto medio di `prove` selezioni casuali della **stessa numerosita'**.

    Con una distribuzione a coda destra bastano poche righe fortunate ad alzare la media: il
    numero da battere non e' zero, e' il percentile alto del caso.
    """
    rng = np.random.default_rng(seme)
    estrazioni = np.array([netto[rng.choice(len(netto), size=quanti, replace=False)].mean() for _ in range(prove)])
    return float(np.percentile(estrazioni, 95)), estrazioni


def valuta(fuori: pd.DataFrame, probabilita: np.ndarray, soglie: tuple[float, ...]) -> pd.DataFrame:
    """La tabella che decide: per ogni soglia su `P(su)`, il netto contro il controllo casuale.

    Il netto e' il rendimento realizzato dell'etichetta meno il giro di commissioni. Non e' un
    portafoglio -- le righe di quindici simboli si sovrappongono nel tempo -- ma e' esattamente
    l'oggetto su cui il controllo casuale e' definito, e le due cose vanno lette insieme.
    """
    netto = fuori["rendimento_%"].to_numpy() - GIRO * 100
    p_su = probabilita[:, SU]
    righe = []
    for soglia in soglie:
        tieni = p_su >= soglia
        if tieni.sum() < 30:
            continue
        p95, estrazioni = controllo_casuale(netto, int(tieni.sum()))
        righe.append(
            {
                "soglia": soglia,
                "righe": int(tieni.sum()),
                "quota_su_%": round(100 * float((fuori["y"].to_numpy()[tieni] == SU).mean()), 1),
                "netto_medio_%": round(float(netto[tieni].mean()), 3),
                "caso_p95_%": round(p95, 3),
                "percentile": round(float((estrazioni < netto[tieni].mean()).mean() * 100), 1),
                "batte": "si" if netto[tieni].mean() > p95 else "no",
            }
        )
    return pd.DataFrame(righe)


def addestra(
    symbols: list[str],
    intervals: tuple[str, ...] = INTERVALS,
    since: str = SINCE,
    oos: str = "2024-01-01",
    barriera: float = BARRIERA_ATR,
    seme: int = 42,
    maschera_trasversali: float = MASCHERA_TRASVERSALI,
) -> dict:
    started = time.time()
    print(f"Campione: {len(symbols)} simboli x {len(intervals)} intervalli, da {since}\n")
    campione = costruisci(symbols, intervals, since, barriera)

    print(f"\n{distribuzione(campione['y'].to_numpy(), 'completo')}")

    # Taglio temporale, non cross-validation: il purging toglie la sovrapposizione fra righe
    # vicine, non il fatto che in una CV il regime successivo sia gia' nel campione di
    # addestramento. `ai_voter` lo ha misurato -- AUC 0,495 in CV contro 0,536 al taglio vero.
    dentro = campione[campione.index < oos]
    fuori = campione[campione.index >= oos].copy()
    # L'embargo copre l'orizzonte dell'etichetta sull'intervallo piu' lungo: e' li' che due righe
    # a cavallo del taglio condividono futuro.
    embargo = pd.Timedelta(hours=ORIZZONTE_ORE)
    dentro = dentro[dentro["t_exit"] < pd.Timestamp(oos) - embargo]
    print(f"\nTaglio {oos} con embargo {embargo}: stima {len(dentro):,}, verifica {len(fuori):,}")
    print(distribuzione(dentro["y"].to_numpy(), "stima"))
    print(distribuzione(fuori["y"].to_numpy(), "verifica"))

    X = dentro[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dentro["y"].to_numpy(dtype=int)

    # Le tre colonne trasversali dipendono dagli **altri** asset, quindi al momento di servire
    # possono mancare: il simulatore carica un simbolo alla volta dall'exchange, e in produzione
    # non c'e' nemmeno lo store. Un modello che non ha mai visto quello stato lo interpreta come
    # ribassista -- misurato: `P(giu)` media passa da 0,55 a 0,67, e con una soglia d'uscita
    # tarata sulla prima ogni posizione si chiude alla barra successiva.
    #
    # Mascherarne una quota in addestramento rende il NaN uno stato **conosciuto**, che il modello
    # tratta come "non so". Costa un po' del vantaggio trasversale sulle righe mascherate; il
    # confronto fra le due calibrazioni sta nel rapporto finale.
    if maschera_trasversali > 0:
        rng = np.random.default_rng(seme)
        colonne = [FEATURE_COLUMNS.index(c) for c in CROSS_COLUMNS]
        da_mascherare = rng.random(len(X)) < maschera_trasversali
        X[np.ix_(da_mascherare, colonne)] = np.nan
        print(f"Trasversali mascherate su {da_mascherare.sum():,} righe ({maschera_trasversali:.0%})")
    print(f"\nAddestramento su {X.shape[0]:,} x {X.shape[1]}...", flush=True)
    fit_started = time.time()
    modello = build_model("gbdt", random_state=seme)
    fit_model(modello, X, y)
    secondi = time.time() - fit_started
    print(f"Addestrato in {secondi:.1f}s")

    probabilita = predict_proba(modello, fuori[FEATURE_COLUMNS].to_numpy(dtype=float))
    y_fuori = fuori["y"].to_numpy(dtype=int)
    auc_su = _auc((y_fuori == SU).astype(int), probabilita[:, SU])
    auc_giu = _auc((y_fuori == GIU).astype(int), probabilita[:, GIU])
    print(f"\nAUC fuori campione: P(su) {auc_su:.4f}   P(giu) {auc_giu:.4f}")
    print("(0,50 = nessun segnale; il soffitto dello stato dell'arte su questi mercati e' ~0,54)")

    soglie = (0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55)
    tabella = valuta(fuori, probabilita, soglie)
    print("\nNetto per riga contro il controllo casuale di pari numerosita' (500 estrazioni):")
    print(tabella.to_string(index=False))

    passate = tabella[(tabella["batte"] == "si") & (tabella["soglia"] > 0)]["soglia"].to_numpy()
    adiacenti = [(a, b) for a, b in zip(soglie[1:], soglie[2:]) if a in passate and b in passate]  # noqa: B905
    verdetto = (
        f"PASSA: soglie adiacenti sopra il p95 del caso {adiacenti}"
        if adiacenti
        else "NON PASSA: nessuna coppia di soglie adiacenti batte il p95 del caso"
    )
    print(f"\n{verdetto}")

    # La soglia d'ingresso si sceglie fra quelle che passano; se nessuna passa si registra
    # comunque la migliore, con il verdetto accanto, invece di non salvare niente.
    utili = tabella[tabella["soglia"] > 0]
    soglia = float(utili.loc[utili["netto_medio_%"].idxmax(), "soglia"]) if not utili.empty else 0.40

    # La soglia d'**uscita** vive su un'altra distribuzione e va calibrata su quella. Prendere il
    # valore d'ingresso e riusarlo qui e' il difetto che chiudeva ogni posizione alla barra
    # successiva: 0,55 seleziona l'8% delle barre su `P(su)` e l'80% su `P(giu)`.
    #
    # Si fissa al quantile dichiarato di `P(giu)` fuori campione, cioe' "esci sul 5% di barre in
    # cui il modello e' piu' convinto del ribasso". E' una quota, non un numero magico, e sposta
    # la domanda dal livello alla frequenza -- che e' cio' che l'ablazione ha mostrato contare.
    soglia_uscita = float(np.quantile(probabilita[:, GIU], 1.0 - QUOTA_USCITA))
    quota_effettiva = float((probabilita[:, GIU] >= soglia_uscita).mean())
    print(
        f"\nSoglia d'uscita su P(giu): {soglia_uscita:.3f} "
        f"(quota di barre {quota_effettiva:.1%}; P(giu) mediana {np.median(probabilita[:, GIU]):.3f})"
    )
    print("La soglia d'ingresso vale su P(su) e non e' riusabile qui: le due teste hanno distribuzioni diverse.")

    percorso = MODELS_DIR / f"{MODEL_NAME}.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(modello, percorso)

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model_kind": "gbdt",
        "model_path": percorso.name,
        "features": FEATURE_COLUMNS,
        "labeling": {
            "method": "symmetric_triple_barrier",
            "classi": {"su": SU, "giu": GIU, "fermo": FERMO},
            "barriera_atr": barriera,
            "orizzonte_ore": ORIZZONTE_ORE,
            "round_trip_fee": GIRO,
            "fee_floor_multiple": PAVIMENTO,
        },
        "data": {
            "symbols": symbols,
            "intervals": list(intervals),
            "since": since,
            "oos": oos,
            "rows": int(len(campione)),
            "train_rows": int(len(dentro)),
            "test_rows": int(len(fuori)),
        },
        "decision_threshold": soglia,
        "exit_threshold": soglia_uscita,
        "exit_share": round(quota_effettiva, 4),
        "cross_mask": maschera_trasversali,
        "auc_su": round(auc_su, 4),
        "auc_giu": round(auc_giu, 4),
        "verdetto": verdetto,
        "sweep": tabella.to_dict(orient="records"),
        "fit_seconds": round(secondi, 1),
    }
    (MODELS_DIR / f"{MODEL_NAME}.json").write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nModello in {percorso}\nTotale: {(time.time() - started) / 60:.1f} minuti")
    return metadata


def _selfcheck() -> None:
    """Un segnale piantato si deve trovare; rumore no. Gira senza store e senza rete."""
    rng = np.random.default_rng(0)
    n = 4000
    index = pd.date_range("2022-01-01", periods=n, freq="4h")
    # Serie con deriva a tratti: le etichette simmetriche devono trovarci dentro entrambe le classi
    passi = rng.normal(0, 0.004, n) + np.repeat(rng.normal(0, 0.004, n // 100), 100)[:n]
    close = 100 * np.exp(np.cumsum(passi))
    candele = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.002,
            "Low": close * 0.998,
            "Close": close,
            "Volume": rng.uniform(1, 2, n),
        },
        index=index,
    )
    candele.index.name = "Open time"
    cross = cross_features(pd.DataFrame({"BTCUSDT": candele["Close"], "X": candele["Close"] * 1.1}))
    frame = campione_simbolo("BTCUSDT", "4h", candele, cross)
    assert not frame.empty, "il campione non deve essere vuoto"
    quote = {c: (frame["y"] == c).mean() for c in (SU, GIU, FERMO)}
    assert quote[SU] > 0.05 and quote[GIU] > 0.05, f"barriere simmetriche, classi sbilanciate: {quote}"
    # La simmetria e' il punto: su una serie senza deriva le due classi devono quasi pareggiare.
    assert abs(quote[SU] - quote[GIU]) < 0.20, f"su e giu' troppo diverse su una passeggiata: {quote}"
    assert list(frame[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
    assert frame["t_exit"].max() <= candele.index[-1], "un'uscita non puo' cadere fuori dalla serie"
    print(f"leg_trainer selfcheck: ok  (su {quote[SU]:.1%}, giu {quote[GIU]:.1%}, fermo {quote[FERMO]:.1%})")


def main() -> None:
    from cryptofarm.data.klines import DEFAULT_SYMBOLS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--oos", default="2024-01-01", help="inizio della verifica temporale")
    parser.add_argument("--barriera", type=float, default=BARRIERA_ATR)
    parser.add_argument("--maschera", type=float, default=MASCHERA_TRASVERSALI, help="quota di trasversali nascoste")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    addestra(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        intervals=tuple(args.intervals) if args.intervals else INTERVALS,
        since=args.since,
        oos=args.oos,
        barriera=args.barriera,
        maschera_trasversali=args.maschera,
    )


if __name__ == "__main__":
    main()
