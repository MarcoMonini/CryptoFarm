"""Banco di prova della sola strategia a confluenza, su una griglia larga e su un paniere.

Le altre strategie qui non ci sono di proposito: `scripts/strategy_lab.py` le misura gia', e
questo banco serve a rispondere a due domande che riguardano solo la confluenza.

**Prima domanda: su questi nove parametri, dove sta e quanto e' larga la regione che funziona?**
Le griglie sono tre e rispondono in due modi diversi. `veloce` e `ampia` sono cartesiane, cioe'
muovono piu' parametri insieme e vedono le interazioni; `coordinate` scandisce l'**intero**
intervallo di ognuno degli undici parametri, uno per volta, tenendo gli altri al centro. La
cartesiana su tutti sarebbe mezzo milione di celle e non si esegue.

Non il massimo -- il massimo e' la cella piu' fortunata, e su questi dati la correlazione fra resa
in stima e resa in verifica e' -0,69, cioe' scegliere il migliore in campione trasferisce peggio
che prendere una configurazione a caso. Quello che conta e' la **frazione di celle in utile** e la
mediana, che dicono se la regione esiste o se il risultato e' un punto isolato.

**Seconda domanda: sorvegliare cinque asset con un capitale solo cambia qualcosa?**
Il rischio dichiarato prima di misurare e' che la confluenza operi troppo poco perche' si possa
dire se ha funzionato. Un paniere non cambia la regola, cambia quante volte la trova. Ma il rimedio
va verificato: `--paniere` riporta accanto al risultato le occasioni perse mentre il capitale era
impegnato e la concentrazione, cioe' la quota dell'asset piu' operato.

## I tre riferimenti, sempre riportati

1. **il possesso passivo**, che e' quello da battere davvero;
2. **`ichimoku_trend` da solo** sul piano di struttura: una strategia complessa che non batte il
   proprio votante migliore non ha guadagnato niente dalla complessita';
3. **il riferimento a frequenza appaiata**: lo stesso `ichimoku_trend` ritarato per fare lo
   *stesso numero di operazioni all'anno* della confluenza. E' il controllo che separa «seleziona
   meglio» da «opera solo di meno», e in questo progetto quella distinzione ha gia' spiegato
   quasi tutto.

## Il conto delle prove

Ogni file di risultati porta in testa il numero di celle girate, e va usato come conteggio delle
prove per la correzione di molteplicita' (`scripts/multiplicity.py`). Guardare la cella migliore
di dodicimila e riportarne lo Sharpe senza scontarlo non e' una misura, e' un aneddoto.

    python -m scripts.confluence_lab --selfcheck                       # senza store, dati finti
    python -m scripts.confluence_lab --grid veloce --symbol BTCUSDT
    python -m scripts.confluence_lab --grid ampia --interval 15m --since 2021-01-01
    python -m scripts.confluence_lab --grid veloce --paniere majors
"""

from __future__ import annotations

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import confluence, rotation, strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import annualised, drawdown, simulate_positions
from cryptofarm.trading.portfolio import curva_capitale, simulate_shared_capital

SIMBOLO = "BTCUSDT"
INTERVALLO = "15m"
DA = "2021-01-01"
CAPITALE = 100.0
# Una strategia che puo' andare corta si esegue sui perpetui: listino taker 0,045%, maker sotto.
COMMISSIONE = 0.05
MANTENIMENTO = 0.03
USCITA = PROJECT_ROOT / "analysis_cache" / "confluence"

# Le griglie. La confluenza ha nove parametri liberi e il prodotto cartesiano su tutti e nove non
# si esegue: queste tre coprono lo stesso spazio a tre risoluzioni, e la piu' fitta muove i sei
# che decidono se e quando si entra, tenendo gli altri tre al valore centrale.
GRIGLIE: dict[str, dict[str, list]] = {
    "veloce": {
        "theta_base": [0.25, 0.35, 0.45],
        "theta_macro": [0.0, 0.15],
        "isteresi": [0.05, 0.15],
        "emivita": [3.0, 6.0, 12.0],
        "k_famiglie": [1, 2, 3],
    },
    "ampia": {
        "theta_base": [0.15, 0.25, 0.35, 0.45, 0.55],
        "theta_macro": [0.0, 0.10, 0.20, 0.30],
        "isteresi": [0.0, 0.05, 0.10, 0.20],
        "emivita": [1.5, 3.0, 6.0, 12.0, 24.0],
        "k_famiglie": [1, 2, 3, 4],
        "atr_multiplier": [2.0, 3.0, 5.0],
    },
}

# Il prodotto cartesiano su tutti e nove i parametri non si esegue: sarebbe mezzo milione di celle,
# cioe' giorni di calcolo per una risposta che nessuno leggerebbe. La scansione **per coordinata**
# copre invece l'intero intervallo di *ognuno* dei parametri, uno per volta, tenendo gli altri al
# centro. E' anche il metodo con cui questo progetto ha gia' scelto i valori di partenza
# (`scripts/tune_defaults.py`): il massimo di una griglia e' la cella piu' fortunata, e su questi
# dati trasferisce fuori campione peggio della mediana.
CENTRO: dict = {
    "theta_base": 0.35,
    "theta_macro": 0.15,
    "isteresi": 0.10,
    "emivita": 6.0,
    "w_max": 0.30,
    "k_famiglie": 2,
    "innesco": 0,
    "atr_window": 14,
    "atr_multiplier": 3.0,
    "regime_ema": 50,
    "struttura_ema": 50,
    "barre_in_formazione": True,
}

SCANSIONE: dict[str, list] = {
    "theta_base": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70],
    "theta_macro": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
    "isteresi": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30],
    "emivita": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0],
    "w_max": [0.17, 0.20, 0.25, 0.30, 0.40, 0.50, 1.0],
    "k_famiglie": [1, 2, 3, 4, 5, 6],
    "innesco": [0, 2, 4, 8, 12, 24, 48],
    "atr_window": [5, 7, 10, 14, 20, 30, 50],
    "atr_multiplier": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0],
    "regime_ema": [10, 20, 30, 50, 80, 120, 200],
    "struttura_ema": [10, 20, 30, 50, 80, 120, 200],
    "barre_in_formazione": [True, False],
}


NOMI_GRIGLIA = [*GRIGLIE, "coordinate"]


def celle(nome: str) -> list[dict]:
    """Tutte le configurazioni della griglia, come dizionari pronti per `confluence.evaluate`.

    Le celle cartesiane portano solo i parametri che la griglia muove, e gli altri restano ai
    default di `confluence.evaluate`. Le celle per coordinata li portano tutti, perche' li' il
    centro e' parte della definizione: senza, non si saprebbe rispetto a cosa si e' scostati.
    """
    if nome == "coordinate":
        configurazioni = [dict(CENTRO)]
        for parametro, valori in SCANSIONE.items():
            configurazioni += [{**CENTRO, parametro: valore} for valore in valori if valore != CENTRO[parametro]]
        return configurazioni
    griglia = GRIGLIE[nome]
    return [dict(zip(griglia, valori)) for valori in itertools.product(*griglia.values())]


# ---------------------------------------------------------------------------------------------
# Le candele, e la parte cara che dalla griglia non dipende
# ---------------------------------------------------------------------------------------------

_CANDELE: dict[tuple, pd.DataFrame] = {}
_STATI: dict[tuple, dict] = {}


def prepara(simbolo: str, intervallo: str, da: str, fino: str | None) -> pd.DataFrame:
    chiave = (simbolo, intervallo)
    if chiave not in _CANDELE:
        from scripts.strategy_sweep import load_interval

        candele = load_interval(intervallo, da, fino, simbolo)
        if candele.empty:
            raise SystemExit(f"nessuna candela per {simbolo} {intervallo}")
        _CANDELE[chiave] = candele
    return _CANDELE[chiave]


def stati(simbolo: str, intervallo: str) -> dict:
    """Gli stati dei votanti, calcolati una volta per (simbolo, intervallo).

    E' la ragione per cui una griglia da dodicimila celle e' eseguibile: i votanti sono congelati,
    quindi il loro stato non dipende da nessun parametro della griglia. Misurato su 11.520 barre,
    351 ms per cella ricalcolando tutto contro 104 ms riusando gli stati.
    """
    chiave = (simbolo, intervallo)
    if chiave not in _STATI:
        _STATI[chiave] = confluence.stati_dei_votanti(_CANDELE[chiave], intervallo)
    return _STATI[chiave]


# ---------------------------------------------------------------------------------------------
# Misura di una configurazione
# ---------------------------------------------------------------------------------------------


def _metriche(operazioni: list, curva: np.ndarray, indice: pd.DatetimeIndex) -> dict:
    if not operazioni:
        return {"n_trade": 0, "rendimento_%": 0.0, "trade_anno": 0.0}
    anni = (indice[-1] - indice[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr, volatilita, sharpe = annualised(curva, indice)
    ritorni = np.array([o["Profit"] / (o["Quantity"] * o["Buy_Price"]) for o in operazioni])
    return {
        "n_trade": len(operazioni),
        "trade_anno": len(operazioni) / anni if anni > 0 else float("nan"),
        "rendimento_%": float(curva[-1] / curva[0] - 1) * 100,
        "cagr_%": cagr,
        "sharpe": sharpe,
        "drawdown_%": drawdown(curva),
        "win_rate_%": float((ritorni > 0).mean() * 100),
    }


def valuta(
    candele: pd.DataFrame,
    intervallo: str,
    parametri: dict,
    stati_votanti: dict | None = None,
    fee: float = COMMISSIONE,
    carry: float = MANTENIMENTO,
) -> dict:
    """Una configurazione su un asset. Riporta le metriche **e** la necessarieta' per votante."""
    risultato = confluence.evaluate(candele, intervallo, stati=stati_votanti, **parametri)
    operazioni = simulate_positions(risultato.eventi, wallet=CAPITALE, fee_percent=fee, carry_daily_percent=carry)
    curva = curva_capitale(operazioni, candele.index, CAPITALE)
    riga = {**parametri, **_metriche(operazioni, curva, candele.index)}
    riga.update({f"nec_{nome}": quota for nome, quota in risultato.necessarieta.items()})
    # Il votante piu' necessario e' il numero da guardare per primo: sopra 0,60 l'insieme e' quel
    # votante travestito, e le metriche sopra parlano di lui, non della confluenza.
    if risultato.necessarieta:
        piu_necessario = max(risultato.necessarieta, key=risultato.necessarieta.get)
        riga["votante_dominante"] = piu_necessario
        riga["necessarieta_max"] = risultato.necessarieta[piu_necessario]
    return riga


def valuta_paniere(
    per_simbolo: dict[str, pd.DataFrame],
    intervallo: str,
    parametri: dict,
    stati_per_simbolo: dict[str, dict] | None = None,
    fee: float = COMMISSIONE,
    carry: float = MANTENIMENTO,
) -> dict:
    """La stessa configurazione su tutto il paniere, con **un** capitale.

    L'indice comune e' l'unione degli indici: un asset che comincia dopo non deve accorciare la
    misura degli altri, e la curva del capitale sta ferma dove non succede niente.
    """
    eventi = {}
    for simbolo, candele in per_simbolo.items():
        risultato = confluence.evaluate(
            candele,
            intervallo,
            stati=(stati_per_simbolo or {}).get(simbolo),
            **parametri,
        )
        eventi[simbolo] = risultato.eventi_con_priorita()

    portafoglio = simulate_shared_capital(eventi, wallet=CAPITALE, fee_percent=fee, carry_daily_percent=carry)
    indice = pd.DatetimeIndex(sorted(set().union(*(c.index for c in per_simbolo.values()))))
    curva = curva_capitale(portafoglio.operazioni, indice, CAPITALE)
    return {
        **parametri,
        **_metriche(portafoglio.operazioni, curva, indice),
        "occasioni_perse": portafoglio.occasioni_perse,
        "concentrazione": portafoglio.concentrazione,
        **{f"trade_{simbolo}": quante for simbolo, quante in portafoglio.per_asset.items()},
    }


# ---------------------------------------------------------------------------------------------
# I tre riferimenti
# ---------------------------------------------------------------------------------------------


def possesso_passivo(candele: pd.DataFrame) -> dict:
    chiusure = candele["Close"].to_numpy()
    cagr, _, sharpe = annualised(chiusure, candele.index)
    return {
        "riferimento": "possesso passivo",
        "rendimento_%": float(chiusure[-1] / chiusure[0] - 1) * 100,
        "cagr_%": cagr,
        "sharpe": sharpe,
        "drawdown_%": drawdown(chiusure),
        "n_trade": 1,
    }


def _ichimoku(candele: pd.DataFrame, intervallo: str, fast: int, slow: int, span: int, fee, carry) -> dict:
    """`ichimoku_trend` sul piano di struttura, cioe' dove la confluenza lo fa votare."""
    from cryptofarm.data.klines import interval_to_minutes, resample_klines

    minuti = interval_to_minutes(intervallo) * confluence.FATTORI["struttura"]
    lungo = resample_klines(candele, confluence._intervallo(minuti))
    eventi = strategies_ls.ichimoku_trend(lungo, ExtraCache(lungo), fast=fast, slow=slow, span=span)
    operazioni = simulate_positions(eventi, wallet=CAPITALE, fee_percent=fee, carry_daily_percent=carry)
    curva = curva_capitale(operazioni, lungo.index, CAPITALE)
    return {"fast": fast, "slow": slow, "span": span, **_metriche(operazioni, curva, lungo.index)}


def riferimenti(
    candele: pd.DataFrame, intervallo: str, trade_anno_obiettivo: float, fee=COMMISSIONE, carry=MANTENIMENTO
) -> list[dict]:
    """I tre riferimenti, con quello a frequenza appaiata scelto **sulla frequenza, non sulla resa**.

    Sceglierlo sulla resa lo renderebbe un secondo massimo di griglia e il confronto non direbbe
    piu' niente: qui si prende la taratura che fa il numero di operazioni piu' vicino a quello
    della confluenza, qualunque cosa renda.
    """
    righe = [possesso_passivo(candele)]

    centrale = _ichimoku(candele, intervallo, 9, 26, 52, fee, carry)
    righe.append({"riferimento": "ichimoku (centrale)", **centrale})

    candidati = [
        _ichimoku(candele, intervallo, f, s, sp, fee, carry)
        for f, s, sp in ((5, 15, 30), (9, 26, 52), (12, 35, 70), (20, 60, 120), (30, 90, 180))
    ]
    appaiato = min(candidati, key=lambda r: abs(r["trade_anno"] - trade_anno_obiettivo))
    righe.append({"riferimento": "ichimoku (frequenza appaiata)", **appaiato})
    return righe


# ---------------------------------------------------------------------------------------------
# Esecuzione
# ---------------------------------------------------------------------------------------------


def _lotto(lavoro: tuple) -> list[dict]:
    simbolo, intervallo, da, fino, batch, fee, carry = lavoro
    # macOS avvia i worker con `spawn`: i globali riempiti dal padre non arrivano. `prepara` e'
    # idempotente, quindi sotto fork questa riga non costa niente.
    candele = prepara(simbolo, intervallo, da, fino)
    votanti = stati(simbolo, intervallo)
    return [valuta(candele, intervallo, parametri, votanti, fee, carry) for parametri in batch]


def esegui_griglia(
    simbolo: str, intervallo: str, da: str, fino: str | None, configurazioni: list[dict], fee, carry, workers: int
) -> pd.DataFrame:
    prepara(simbolo, intervallo, da, fino)
    if workers <= 1:
        return pd.DataFrame(_lotto((simbolo, intervallo, da, fino, configurazioni, fee, carry)))
    passo = max(1, len(configurazioni) // (workers * 4))
    lotti = [configurazioni[i : i + passo] for i in range(0, len(configurazioni), passo)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        righe = pool.map(_lotto, [(simbolo, intervallo, da, fino, b, fee, carry) for b in lotti])
        return pd.DataFrame([riga for lotto in righe for riga in lotto])


def salva(nome: str, risultati: pd.DataFrame, prove: int) -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    percorso = USCITA / f"{nome}.csv"
    with percorso.open("w") as file:
        file.write(f"# prove totali girate: {prove}\n")
        risultati.to_csv(file, index=False)
    print(f"scritto {percorso} ({len(risultati)} righe, {prove} prove)")


def riassunto(risultati: pd.DataFrame) -> str:
    """Cosa guardare: l'ampiezza della regione, non la cella migliore."""
    utili = (risultati["rendimento_%"] > 0).mean() * 100
    dominanti = risultati.get("necessarieta_max")
    righe = [
        f"celle: {len(risultati)}",
        f"in utile: {utili:.1f}%",
        f"mediana rendimento: {risultati['rendimento_%'].median():+.1f}%",
        f"mediana trade/anno: {risultati['trade_anno'].median():.1f}",
    ]
    if dominanti is not None and dominanti.notna().any():
        sopra = (dominanti > 0.60).mean() * 100
        righe.append(f"celle in cui un votante e' necessario oltre il 60%: {sopra:.1f}%")
    return " | ".join(righe)


def _finte(giorni: int = 90, seme: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2024-01-01", periods=96 * giorni, freq="15min", name="Open time")
    passo = 100 + np.cumsum(rng.normal(0.01, 0.4, len(idx)))
    return pd.DataFrame(
        {
            "Open": passo,
            "High": passo + abs(rng.normal(0, 0.5, len(idx))),
            "Low": passo - abs(rng.normal(0, 0.5, len(idx))),
            "Close": passo + rng.normal(0, 0.1, len(idx)),
            "Volume": rng.random(len(idx)) * 10,
        },
        index=idx,
    )


def _selfcheck() -> None:
    """Gira tutto il banco su dati finti: non misura niente, verifica che il banco funzioni.

    Serve perche' lo store delle candele non c'e' ovunque -- una sessione remota non ha ne' i dati
    ne' l'uscita verso l'exchange -- e un banco che si scopre rotto solo sulla macchina giusta e'
    un banco che fa perdere il giro.
    """
    candele = _finte()
    votanti = confluence.stati_dei_votanti(candele, "15m")

    configurazioni = celle("veloce")
    assert len(configurazioni) == 3 * 2 * 2 * 3 * 3, len(configurazioni)
    righe = pd.DataFrame([valuta(candele, "15m", p, votanti) for p in configurazioni[:6]])
    assert {"n_trade", "trade_anno", "necessarieta_max"} <= set(righe.columns)
    assert righe["n_trade"].sum() > 0, "nessuna operazione: il banco non proverebbe niente"

    # Il paniere: due asset scorrelati devono dare piu' operazioni di uno solo, o almeno non meno.
    paniere = {"A": candele, "B": _finte(seme=7)}
    parametri = configurazioni[0]
    uno = valuta(candele, "15m", parametri, votanti)
    due = valuta_paniere(paniere, "15m", parametri)
    assert due["n_trade"] >= uno["n_trade"], (due["n_trade"], uno["n_trade"])
    assert due["occasioni_perse"] >= 0 and 0 <= due["concentrazione"] <= 1

    # I tre riferimenti, e quello appaiato scelto sulla frequenza.
    tre = riferimenti(candele, "15m", uno["trade_anno"])
    assert len(tre) == 3 and all("rendimento_%" in r for r in tre)
    appaiato = tre[-1]
    assert abs(appaiato["trade_anno"] - uno["trade_anno"]) <= max(
        abs(tre[1]["trade_anno"] - uno["trade_anno"]), 1e-9
    ), "il riferimento appaiato non e' quello piu' vicino per frequenza"

    # La scansione per coordinata copre ogni parametro e parte dal centro.
    coordinate = celle("coordinate")
    assert coordinate[0] == CENTRO
    assert set(SCANSIONE) <= set(coordinate[0])
    for parametro, valori in SCANSIONE.items():
        visti = {c[parametro] for c in coordinate}
        assert set(valori) <= visti, f"{parametro}: la scansione non copre {set(valori) - visti}"

    assert "in utile" in riassunto(righe)
    print(f"confluence_lab selfcheck: passato · {riassunto(righe)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default="veloce", choices=NOMI_GRIGLIA)
    parser.add_argument("--symbol", default=SIMBOLO)
    parser.add_argument("--paniere", default=None, choices=list(rotation.UNIVERSI))
    parser.add_argument("--interval", default=INTERVALLO)
    parser.add_argument("--since", default=DA)
    parser.add_argument("--until", default=None)
    parser.add_argument("--fee", type=float, default=COMMISSIONE)
    parser.add_argument("--carry", type=float, default=MANTENIMENTO)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--selfcheck", action="store_true")
    argomenti = parser.parse_args()

    if argomenti.selfcheck:
        _selfcheck()
        return

    configurazioni = celle(argomenti.grid)
    print(f"griglia «{argomenti.grid}»: {len(configurazioni)} configurazioni")
    inizio = time.time()

    if argomenti.paniere:
        simboli = rotation.UNIVERSI[argomenti.paniere]
        per_simbolo = {s: prepara(s, argomenti.interval, argomenti.since, argomenti.until) for s in simboli}
        stati_per_simbolo = {s: stati(s, argomenti.interval) for s in simboli}
        risultati = pd.DataFrame(
            [
                valuta_paniere(per_simbolo, argomenti.interval, p, stati_per_simbolo, argomenti.fee, argomenti.carry)
                for p in configurazioni
            ]
        )
        nome = f"paniere_{argomenti.paniere}_{argomenti.interval}_{argomenti.grid}"
    else:
        risultati = esegui_griglia(
            argomenti.symbol,
            argomenti.interval,
            argomenti.since,
            argomenti.until,
            configurazioni,
            argomenti.fee,
            argomenti.carry,
            argomenti.workers,
        )
        candele = _CANDELE[(argomenti.symbol, argomenti.interval)]
        for riga in riferimenti(
            candele, argomenti.interval, risultati["trade_anno"].median(), argomenti.fee, argomenti.carry
        ):
            print(
                "  "
                + " · ".join(
                    f"{k}={v}" for k, v in riga.items() if k in ("riferimento", "rendimento_%", "sharpe", "trade_anno")
                )
            )
        nome = f"{argomenti.symbol}_{argomenti.interval}_{argomenti.grid}"

    salva(nome + argomenti.suffix, risultati, len(configurazioni))
    print(riassunto(risultati))
    print(f"in {time.time() - inizio:.1f}s")


if __name__ == "__main__":
    main()
