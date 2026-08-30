"""Banco del modello d'ingresso: quanto vale il cancello del modello lento.

La domanda e' quella posta all'inizio -- due modelli, uno stretto e uno largo, complementari -- e
la risposta e' che si compongono in un verso solo. **Il veloce fa le operazioni, il lento dice
dentro quali movimenti puo' farle.** L'inverso non e' stato provato perche' non ha senso operativo:
un modello a tenuta 150 barre non entra dentro un'operazione di 20.

Misurato su 15 simboli dal 2024 (fuori campione), tenuta 20 barre, commissione 0,2%:

| cancello del lento | operazioni | netto medio | simboli in utile |
|---|---|---|---|
| nessuno            | 223 | +1,360% | 12/15 |
| mediana dello stima| 161 | +1,806% | 12/15 |
| 90° dello stima    | 148 | **+2,071%** | **14/15** |
| 98° dello stima    | 100 | +2,464% | 13/15 |

La curva e' monotona, quindi il punto servito non e' il rendimento piu' alto -- prenderlo
significherebbe scegliere il massimo del campione di verifica, l'errore che questo progetto ha
gia' misurato altrove. Il 90° e' scelto sulla **concordanza fra simboli**, che e' la differenza fra
un modello e un episodio.

Il controllo e' a esposizione appaiata, non il possesso passivo: fuori campione il passivo mediano
fa -34%, quindi stare fuori dal mercato paga da solo. Il modello sta al 100° percentile di 200
estrazioni in tutte le righe della tabella.
"""

from __future__ import annotations

import argparse

import numpy as np

from cryptofarm.data.klines import DEFAULT_SYMBOLS
from cryptofarm.ml import signals
from cryptofarm.ml.entry_trainer import (
    OOS,
    PASSO,
    SINCE,
    STIMA,
    campione_simbolo,
    controllo_casuale,
    operazioni,
    separa,
)
from cryptofarm.ml.models import load_model
from cryptofarm.paths import MODELS_DIR

CANCELLI = (None, 0.5, 0.8, 0.9, 0.95, 0.98)
# Quanto si opera, cioe' il quantile della soglia del veloce. Serve alla domanda «voglio piu'
# operazioni su intervalli brevi»: la commissione e' fissa allo 0,2% e il rendimento no, quindi
# abbassare la soglia non aggiunge operazioni allo stesso rendimento -- ne aggiunge di peggiori.
# La tabella dice di quanto, invece di lasciarlo credere in un verso o nell'altro.
SOGLIE = (0.95, 0.98, 0.99, 0.995, 0.999)
ESTRAZIONI = 200


def raccogli(simboli: list[str], h: int) -> tuple[dict, dict[str, np.ndarray]]:
    """Previsioni dei due modelli sulle righe di verifica, e le stesse sulle righe di stima.

    Le seconde servono a fissare cancello e soglia **dove e' lecito fissarli**: un quantile preso
    sul fuori campione sarebbe look-ahead, e sposterebbe il risultato senza che si veda. Non e' un
    dettaglio da poco: il quantile 0,995 dello stima seleziona 223 operazioni, lo stesso quantile
    preso sul fuori campione ne seleziona 615, perche' le previsioni fuori campione sono piu'
    basse. Chi tara sul secondo si sta guardando le carte.
    """
    lento = load_model(MODELS_DIR / f"{signals.ENTRY_LENTO}.joblib")
    veloce = load_model(MODELS_DIR / f"{signals.ENTRY_VELOCE}.joblib")
    dati, stima = {}, {"lento": [], "veloce": []}
    for symbol in simboli:
        campione = campione_simbolo(symbol, SINCE, h, PASSO)
        if campione is None:
            continue
        dentro, fuori = separa(campione, STIMA, OOS, h)
        stima["lento"].append(lento.predict(campione["X"][dentro]))
        stima["veloce"].append(veloce.predict(campione["X"][dentro]))
        dati[symbol] = {
            "close": campione["close"],
            "posizioni": campione["posizioni"],
            "fuori": fuori,
            "veloce": veloce.predict(campione["X"][fuori]),
            "lento": lento.predict(campione["X"][fuori]),
        }
        print(f"  {symbol}", flush=True)
    if not dati:
        raise SystemExit("nessun simbolo utilizzabile: serve lo store delle candele")
    return dati, {k: np.concatenate(v) for k, v in stima.items()}


def riga(dati: dict, porta: float, soglia: float, tenuta: int, estrazioni: int) -> str:
    esiti, quante, per_simbolo = [], {}, []
    for symbol, d in dati.items():
        scelte = d["fuori"][(d["veloce"] >= soglia) & (d["lento"] >= porta)]
        proprie = operazioni(d["close"], scelte, tenuta)
        quante[symbol] = len(proprie)
        esiti += proprie
        if proprie:
            per_simbolo.append(float(np.mean(proprie)))
    if not esiti:
        return "nessuna operazione"
    medio = float(np.mean(esiti))
    caso = controllo_casuale(dati, quante, tenuta, estrazioni=estrazioni)
    percentile = 100.0 * float(np.mean(caso["medie"] < medio))
    in_utile = sum(1 for m in per_simbolo if m > 0)
    return (
        f"{len(esiti):5d} {100 * medio:+8.3f}% {100 * np.mean(np.array(esiti) > 0):7.1f}% "
        f"{in_utile:3d}/{len(per_simbolo):<3d} {100 * caso['media']:+7.3f}% {percentile:5.1f}°"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*")
    parser.add_argument("--estrazioni", type=int, default=ESTRAZIONI)
    parser.add_argument("--frequenza", action="store_true", help="quanto costa operare di piu'")
    args = parser.parse_args()

    servizio = signals.entry_metadata(signals.ENTRY_VELOCE)
    if not servizio:
        raise SystemExit(f"manca {signals.ENTRY_VELOCE}: addestrare con `--h 20 --quantile 0.995`")
    tenuta = int(servizio["tenuta"])
    dati, stima = raccogli(args.symbols or list(DEFAULT_SYMBOLS), tenuta)

    intestazione = f"{'op':>5s} {'medio':>9s} {'utile':>8s} {'simboli':>7s} {'caso':>8s} {'perc':>6s}"

    if args.frequenza:
        # Il cancello resta quello servito: qui si muove una cosa sola, quanto si e' selettivi.
        porta = float(np.quantile(stima["lento"], 0.90))
        print(f"\n{'soglia del veloce':22s} {intestazione}")
        for quantile in SOGLIE:
            soglia = float(np.quantile(stima["veloce"], quantile))
            esito = riga(dati, porta, soglia, tenuta, args.estrazioni)
            print(f"{f'{1 - quantile:.1%} delle barre':22s} {esito}", flush=True)
        return

    print(f"\n{'cancello del lento':22s} {intestazione}")
    for quantile in CANCELLI:
        porta = -np.inf if quantile is None else float(np.quantile(stima["lento"], quantile))
        nome = "nessuno" if quantile is None else f"{quantile:.0%} dello stima"
        print(f"{nome:22s} {riga(dati, porta, float(servizio['soglia']), tenuta, args.estrazioni)}", flush=True)


if __name__ == "__main__":
    main()
