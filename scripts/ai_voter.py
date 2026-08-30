"""Il votante a modello: un GBDT addestrato **sulle operazioni della confluenza stessa**.

## Perche' non e' `meta_gate` con un'altra primaria

`scripts/meta_gate.py` mette un filtro sopra una strategia di `strategies_ls`, ed etichetta ogni
operazione col proprio esito. Il disegno e' giusto e qui si riusa quasi tutto -- feature, campione
per operazione, validazione purgata, controllo casuale. Cambia **da dove viene l'etichetta**, ed e'
il punto che rende il modello utilizzabile dentro la confluenza invece che accanto:

- la confluenza chiude **quattro uscite su cinque con lo stop a trailing**, non con un segnale
  della primaria. Un modello addestrato su «questa operazione della primaria e' finita in utile»
  risponderebbe su un'operazione **che non verra' presa**: l'esito e' dominato da una regola di
  uscita che l'etichetta non ha mai visto;
- qui il campione sono le operazioni che la confluenza produce davvero, eseguite dal suo motore,
  coi suoi stop e i suoi cancelli. L'etichetta e' il netto di **quella** operazione. Etichetta ed
  esecuzione coincidono per costruzione.

## A cosa serve il punteggio, e a cosa no

**Non genera ingressi.** Genera l'ordine con cui gli ingressi gia' proposti si contendono il
capitale. E' la decisione che oggi prende `Confluenza.eventi_con_priorita` con
`abs(punteggio) - soglia`, cioe' un'euristica: quanto il voto era netto. Sostituirla con una
probabilita' addestrata e' l'unico posto della strategia in cui un modello a IC basso **puo'**
pagare, perche' li' la domanda e' comparativa -- *quale di questi tre asset merita la quota* -- ed
e' esattamente la forma in cui un vantaggio di ranking debole diventa eseguibile
(`ricerca-quant-ml.md` §1.5.1). Come filtro assoluto, invece, sarebbe il disegno che §3.4 ha gia'
misurato non superare il proprio controllo.

**Il campione viene da una configurazione permissiva**, non da quella stretta: un modello
addestrato solo sugli ingressi che la configurazione stretta lascia passare non vedrebbe mai quelli
che essa scarta, e non potrebbe imparare a ordinarli.

## Il numero da battere non e' zero

Per ogni soglia si estraggono 500 selezioni casuali della **stessa numerosita'**: con una coda
lunga bastano poche operazioni fortunate ad alzare il netto medio. E vale l'avvertenza di §3.4:
avendo provato piu' combinazioni, un 98o percentile e' quello che ci si aspetta dal caso. La
soglia si dichiara **prima**, o non si dichiara.

## Cosa questo modulo non risolve

Il rango di forza e' calcolato fra gli asset che **oggi** sono maggiori: nel gennaio 2021 SOL non
lo era. E' sopravvivenza dentro l'insieme dei pari, e nessun purging la toglie -- va tenuta a mente
leggendo qualunque numero trasversale qui dentro.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import confluence, rotation
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions
from scripts import confluence_lab as lab
from scripts.meta_gate import (
    FEATURE_COLUMNS,
    _auc,
    cross_features,
    features_frame,
    gate_report,
    sequential_equity,
)

USCITA = PROJECT_ROOT / "analysis_cache" / "ai_voter"

# La configurazione permissiva da cui vengono i candidati. Soglia bassa e ampiezza minima: serve
# che proponga **piu'** di quanto si vorrebbe eseguire, altrimenti non c'e' niente da ordinare.
PERMISSIVA = dict(lab.CENTRO)
PERMISSIVA.update(theta_base=0.20, k_famiglie=1, theta_macro=0.0)


def campioni(
    simboli: list[str],
    intervallo: str,
    da: str,
    fino: str | None,
    parametri: dict | None = None,
    fee: float = lab.COMMISSIONE,
    carry: float = lab.MANTENIMENTO,
) -> pd.DataFrame:
    """Una riga per **ingresso della confluenza**, con l'esito che il suo motore gli ha dato."""
    from scripts.strategy_sweep import load_interval

    parametri = parametri or PERMISSIVA
    per_simbolo, chiusure = {}, {}
    for simbolo in simboli:
        candele = load_interval(intervallo, da, fino, simbolo)
        if len(candele) < 500:
            continue
        per_simbolo[simbolo] = candele
        chiusure[simbolo] = candele["Close"]
    if not per_simbolo:
        raise SystemExit("nessun simbolo con abbastanza storia")

    quadro = pd.DataFrame(chiusure).sort_index()
    trasversali = cross_features(quadro)

    righe = []
    for simbolo, candele in per_simbolo.items():
        stati = confluence.stati_dei_votanti(candele, intervallo)
        risultato = confluence.evaluate(candele, intervallo, stati=stati, **parametri)
        operazioni = simulate_positions(risultato.eventi, lab.CAPITALE, fee, carry)
        if not operazioni:
            continue

        frame = features_frame(candele, ExtraCache(candele))
        frame["rango_forza"] = trasversali["rango_forza"][simbolo].reindex(frame.index)
        frame["ampiezza_mercato"] = trasversali["ampiezza_mercato"].reindex(frame.index)
        frame["forza_su_btc"] = trasversali["forza_su_btc"][simbolo].reindex(frame.index)
        # Il margine del punteggio sopra la soglia: e' l'euristica che il modello deve battere,
        # quindi entra come feature -- se il modello non aggiunge niente, si vedra' che il suo
        # vantaggio e' tutto qui dentro.
        margine = pd.Series(risultato.punteggio - risultato.soglia, index=candele.index)

        for operazione in operazioni:
            ingresso = operazione["Buy_Time"]
            if ingresso not in frame.index:
                continue
            lordo = (operazione["Sell_Price"] / operazione["Buy_Price"] - 1.0) * 100
            netto = lordo - fee * (1 + operazione["Sell_Price"] / operazione["Buy_Price"])
            righe.append(
                {
                    "simbolo": simbolo,
                    "t_start": ingresso,
                    "t_exit": operazione["Sell_Time"],
                    "netto_%": netto,
                    "y": int(netto > 0),
                    "margine_punteggio": float(margine.loc[ingresso]),
                    **frame.loc[ingresso].to_dict(),
                }
            )

    campione = pd.DataFrame(righe).sort_values("t_start").reset_index(drop=True)
    return campione.dropna(subset=["netto_%"])


def controllo_casuale(netto: np.ndarray, quanti: int, prove: int = 500, seme: int = 12345) -> dict:
    """Il netto medio di `prove` selezioni casuali della **stessa numerosita'**.

    Le estrazioni si calcolano una volta sola e da quelle si ricavano sia il p95 sia il percentile
    in cui cade il risultato osservato: sono due letture della stessa distribuzione, e calcolarle
    separatamente vorrebbe dire confrontare il risultato con due campioni diversi.
    """
    rng = np.random.default_rng(seme)
    estrazioni = np.array([netto[rng.choice(len(netto), size=quanti, replace=False)].mean() for _ in range(prove)])
    return {
        "caso_medio_%": float(estrazioni.mean()),
        "caso_p95_%": float(np.percentile(estrazioni, 95)),
        "_estrazioni": estrazioni,
    }


def rapporto(campione: pd.DataFrame, fold: int = 6, embargo: int = 24) -> pd.DataFrame:
    """La tabella di `meta_gate.gate_report`, piu' il controllo casuale per ogni soglia."""
    tabella = gate_report(campione, folds=fold, embargo_bars=embargo)
    netto = campione["netto_%"].to_numpy()
    extra = []
    for _, riga in tabella.iterrows():
        caso = controllo_casuale(netto, int(riga["operazioni"]))
        estrazioni = caso.pop("_estrazioni")
        extra.append(
            {
                **{k: round(v, 3) for k, v in caso.items()},
                "percentile_nel_caso": round(float((estrazioni < riga["netto_medio_%"]).mean() * 100), 1),
            }
        )
    unita = pd.concat([tabella.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    unita.attrs = tabella.attrs
    return unita


def verifica_temporale(campione: pd.DataFrame, taglio: str, seme: int = 0) -> pd.DataFrame:
    """Addestra **solo sul passato** e misura **solo sul futuro**. E' il test che decide.

    La validazione purgata dentro una finestra risponde a «il modello sa ordinare le operazioni di
    questo periodo?». La domanda operativa e' un'altra: «un modello che conosce solo il passato sa
    ordinare quello che viene dopo?». Le due possono divergere -- e qui divergono -- perche' la
    prima vede, in ogni fold, righe che vengono da *dopo* le righe che predice: il purging toglie
    la sovrapposizione fra operazioni vicine, non il fatto che il regime successivo sia gia' nel
    campione di addestramento.

    Nessun riaddestramento scorrevole: un solo taglio, dichiarato prima. Rifarlo a piu' tagli e
    scegliere il migliore sarebbe la stessa ricerca su griglia che questo progetto ha gia'
    misurato non trasferire.
    """
    dentro = campione[campione["t_start"] < taglio]
    fuori = campione[campione["t_start"] >= taglio].copy()
    if len(dentro) < 500 or len(fuori) < 200:
        raise SystemExit(f"campione insufficiente per il taglio {taglio}: {len(dentro)}/{len(fuori)}")

    modello = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=seme,
    )
    modello.fit(dentro[FEATURE_COLUMNS].to_numpy(dtype=float), dentro["y"].to_numpy(dtype=int))
    fuori["p"] = modello.predict_proba(fuori[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]

    netto = fuori["netto_%"].to_numpy()
    righe = []
    for soglia in (0.0, 0.40, 0.45, 0.50, 0.55, 0.60):
        tieni = fuori["p"].to_numpy() >= soglia if soglia else np.ones(len(fuori), bool)
        if tieni.sum() < 20:
            continue
        caso = controllo_casuale(netto, int(tieni.sum()))
        estrazioni = caso.pop("_estrazioni")
        righe.append(
            {
                "soglia": soglia or "nessuna",
                "operazioni": int(tieni.sum()),
                "precisione_%": round(100 * float(fuori["y"].to_numpy()[tieni].mean()), 1),
                "netto_medio_%": round(float(netto[tieni].mean()), 3),
                "composto_seq": round(sequential_equity(netto[tieni]), 2),
                **{k: round(v, 3) for k, v in caso.items()},
                "percentile_nel_caso": round(float((estrazioni < netto[tieni].mean()).mean() * 100), 1),
            }
        )
    tabella = pd.DataFrame(righe)
    tabella.attrs["auc"] = _auc(fuori["y"].to_numpy(), fuori["p"].to_numpy())
    tabella.attrs["n_dentro"] = len(dentro)
    tabella.attrs["n_fuori"] = len(fuori)
    return tabella


def _selfcheck() -> None:
    """Un segnale piantato si deve trovare; rumore no. Riusa il controllo gia' scritto."""
    from scripts.meta_gate import selfcheck as meta_selfcheck

    meta_selfcheck()
    netto = np.array([1.0] * 50 + [-1.0] * 50)
    caso = controllo_casuale(netto, 50, prove=200)
    assert abs(caso["caso_medio_%"]) < 0.2, caso
    print("ai_voter selfcheck: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="wide", choices=["majors", "wide"])
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--since", default="2019-01-01")
    parser.add_argument("--oos", default="2024-01-01", help="inizio della verifica temporale")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    simboli = rotation.WIDE if args.universe == "wide" else rotation.UNIVERSI["majors"]
    print(f"campione: confluenza permissiva su {len(simboli)} simboli, {args.interval}, da {args.since}")
    campione = campioni(simboli, args.interval, args.since, None)
    USCITA.mkdir(parents=True, exist_ok=True)
    campione.to_csv(USCITA / f"campione_{args.universe}_{args.interval}.csv", index=False)
    print(f"  {len(campione)} operazioni, quota in utile {campione['y'].mean() * 100:.1f}%")

    dentro = campione[campione["t_start"] < args.oos]
    fuori = campione[campione["t_start"] >= args.oos]
    print(f"  stima {len(dentro)} / verifica {len(fuori)}")

    for nome, parte in (("tutto (CV purgata)", campione), ("solo verifica", fuori)):
        if len(parte) < 300:
            print(f"\n--- {nome}: campione insufficiente ({len(parte)}) ---")
            continue
        tabella = rapporto(parte, fold=args.folds)
        print(f"\n--- {nome} --- AUC {tabella.attrs.get('auc', float('nan')):.3f}  n={tabella.attrs.get('n')}")
        print(tabella.to_string(index=False))

    temporale = verifica_temporale(campione, args.oos)
    print(f"\n--- VERIFICA TEMPORALE (addestrato < {args.oos}, misurato dopo) ---")
    print(
        f"    AUC {temporale.attrs['auc']:.3f}   "
        f"stima {temporale.attrs['n_dentro']}  verifica {temporale.attrs['n_fuori']}"
    )
    print(temporale.to_string(index=False))


if __name__ == "__main__":
    main()
