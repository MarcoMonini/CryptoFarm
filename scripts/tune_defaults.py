"""Valori di partenza dei widget, scelti per **robustezza** e per intervallo.

Il modo ovvio di scegliere un default e' prendere la configurazione che ha reso di piu'. Su questi
dati e' l'errore misurato: `.claude/docs/ricerca-quant-ml.md` §2.5 riporta ρ = **−0,69** fra resa
in stima e resa in verifica sulle prime dieci configurazioni della rotazione, e §2.3 mostra che
sulle strategie a un asset la scelta del massimo trasferisce peggio della mediana. Il massimo di
una griglia e' la cella piu' fortunata, non la regola migliore.

Qui si sceglie in un altro modo, **una coordinata alla volta**:

1. dentro ogni coppia (simbolo, strategia) ogni configurazione riceve il suo **rango percentile**
   nella griglia. E' l'unico modo di mettere insieme cinque asset i cui possessi passivi vanno da
   +134% a +4.346%: sommare i rendimenti grezzi farebbe decidere tutto a SOL;
2. per ogni parametro, e per ogni suo valore, si prende la **mediana di quei ranghi** su tutte le
   configurazioni e tutti i simboli. Ogni scelta poggia quindi su decine o centinaia di misure, non
   su una cella;
3. si tiene il valore con la mediana piu' alta -- ma **solo se sposta qualcosa**: se fra il valore
   migliore e il peggiore la mediana dei ranghi cambia meno di `SOGLIA_UTILE`, il parametro non
   discrimina e si lascia il default che c'e' gia', invece di inseguire rumore.

Il quarto passo e' quello che rende la scelta credibile: la stessa procedura gira **due volte**,
sul periodo intero e sul solo 2021-2023, e le due risposte si confrontano. Un parametro il cui
valore scelto cambia fra le due finestre non e' stabile, e viene segnalato.

    python -m scripts.tune_defaults --interval 1d
    python -m scripts.tune_defaults --all-intervals --save
"""

from __future__ import annotations

import argparse
import json
from datetime import date

import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import config
from scripts.strategy_lab import GRIDS as LAB_GRIDS
from scripts.strategy_lab import OUTPUT_DIR as LAB_DIR
from scripts.sweep_report import _fold_triplet, load_sweeps, parameter_columns

SIMBOLI = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
INTERVALLI = ["15m", "1h", "4h", "1d"]
SUFFISSO = "_2021_fee005"
# Sotto questo numero di operazioni non si sta misurando una strategia ma una posizione tenuta per
# anni: il rendimento e' quello del possesso passivo, con il fianco esposto a quale settimana e'
# capitata l'entrata.
MIN_TRADE = 10
# Di quanto deve spostarsi la mediana dei ranghi (0..1) fra il valore migliore e il peggiore perche'
# valga la pena cambiare un default. Sotto, il parametro non discrimina: cambiarlo sarebbe adattarsi
# al rumore, e il valore che c'e' gia' e' stato scelto da qualcuno che guardava un grafico.
SOGLIA_UTILE = 0.06

# Si scrive un modulo Python e non un JSON: cosi' viaggia con il pacchetto senza toccare la
# configurazione di packaging, si importa senza I/O, e il diff di un cambio di default si legge
# nella revisione come si legge il diff di qualunque altra costante.
USCITA = PROJECT_ROOT / "src" / "cryptofarm" / "trading" / "tuned_defaults.py"


# ---------------------------------------------------------------------------------------------
# Dai nomi delle griglie ai nomi dei widget
# ---------------------------------------------------------------------------------------------
# Le griglie usano i nomi degli argomenti delle strategie, `config` quelli dei widget. La mappa e'
# scritta a mano e per strategia, perche' lo stesso nome di griglia finisce in costanti diverse:
# `atr_multiplier` e' `ATR_MULTIPLIER` per le bande, `DONCHIAN_ATR_MULT` per il canale e
# `SQUEEZE_ATR_MULT` per la compressione, e ognuna e' stata misurata col suo.

VERSO_CONFIG: dict[str, dict[str, str]] = {
    "ATR Bands": {
        "ema_window": "EMA_SHORT",
        "atr_window": "ATR_WINDOW",
        "atr_multiplier": "ATR_MULTIPLIER",
        "stop_loss": "STOP_LOSS_PERCENT",
    },
    "Supertrend": {"ema_window": "EMA_SHORT", "atr_window": "ATR_WINDOW", "atr_multiplier": "ATR_MULTIPLIER"},
    "TP/SL with ATR": {"ema_window": "EMA_SHORT", "atr_window": "ATR_WINDOW", "atr_multiplier": "ATR_MULTIPLIER"},
    "Close RSI Reverse": {"rsi_window": "RSI_SHORT", "rsi_window2": "RSI_MEDIUM"},
    "Trend Zones": {"ema_window": "EMA_SHORT"},
    "Close EMA Crossover": {"ema_triplet": "EMA_TRIPLET"},
    "Donchian Breakout": {
        "channel": "DONCHIAN_CHANNEL",
        "adx_min": "ADX_MIN",
        "atr_multiplier": "DONCHIAN_ATR_MULT",
        "regime_ema": "REGIME_EMA",
    },
    "Squeeze Breakout": {
        "bb_dev": "BB_DEV",
        "kc_multiplier": "KC_MULTIPLIER",
        "atr_multiplier": "SQUEEZE_ATR_MULT",
        "confirm_volume": "CONFIRM_VOLUME",
    },
    "Ichimoku Trend": {"fast": "ICHIMOKU_FAST", "require_cloud": "REQUIRE_CLOUD"},
}
# Le finestre di Ichimoku si muovono in proporzione, come nell'originale 9/26/52: scegliere `fast`
# fissa le altre due, e la griglia non le ha mai mosse separatamente.
ICHIMOKU_SCALA = {7: (22, 44), 9: (26, 52), 20: (60, 120)}
# `Close EMA Crossover` muove le tre medie insieme, quindi la griglia le ha piegate in una colonna
# sola: la scelta e' la terna, non tre numeri indipendenti.
TERNA = ("EMA_SHORT", "EMA_MEDIUM", "EMA_LONG")

STORICHE = {"ATR Bands", "Supertrend", "TP/SL with ATR", "Close RSI Reverse", "Trend Zones", "Close EMA Crossover"}
GRIGLIA_DI = {
    "ATR Bands": "atr_bands",
    "Supertrend": "supertrend",
    "TP/SL with ATR": "tp_sl_atr",
    "Close RSI Reverse": "close_rsi_reverse",
    "Trend Zones": "trend_zones",
    "Close EMA Crossover": "close_ema_crossover",
    "Donchian Breakout": "donchian_breakout",
    "Squeeze Breakout": "squeeze_breakout",
    "Ichimoku Trend": "ichimoku_trend",
}


# ---------------------------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------------------------


def _tabella(voce: str, intervallo: str) -> pd.DataFrame:
    """Le griglie di una strategia su tutti i simboli, impilate, solo lunghe e con abbastanza trade."""
    griglia = GRIGLIA_DI[voce]
    pezzi = []
    for simbolo in SIMBOLI:
        if voce in STORICHE:
            trovate = load_sweeps(intervallo, grids=[griglia], symbol=simbolo, suffix=SUFFISSO)
            frame = trovate.get(griglia)
        else:
            percorso = LAB_DIR / f"{griglia}_{simbolo}_{intervallo}.parquet"
            frame = pd.read_parquet(percorso) if percorso.exists() else None
            if frame is not None:
                frame = frame[~frame["allow_short"]]
        if frame is None or frame.empty:
            continue
        frame = frame[frame["n_trade"] >= MIN_TRADE]
        if frame.empty:
            continue
        # Il rango percentile **dentro il simbolo**: e' cio' che rende sommabili cinque asset i cui
        # possessi passivi differiscono di un fattore trenta.
        frame = frame.assign(rango=frame["rendimento_%"].rank(pct=True), simbolo_=simbolo)
        pezzi.append(frame)
    return pd.concat(pezzi, ignore_index=True) if pezzi else pd.DataFrame()


def _assi(voce: str, tabella: pd.DataFrame) -> list[str]:
    """Le colonne che in questa tabella sono davvero parametri, e che sappiamo tradurre."""
    if voce in STORICHE:
        candidate = parameter_columns(tabella)
    else:
        candidate = [nome for nome in LAB_GRIDS[GRIGLIA_DI[voce]]["params"] if nome != "allow_short"]
    verso = VERSO_CONFIG[voce]
    return [nome for nome in candidate if nome in verso and tabella[nome].nunique() > 1]


# ---------------------------------------------------------------------------------------------
# Scelta
# ---------------------------------------------------------------------------------------------


def scegli(voce: str, tabella: pd.DataFrame) -> list[dict]:
    """Una riga per parametro: il valore scelto, quanto sposta, e se vale la pena cambiarlo."""
    righe = []
    for asse in _assi(voce, tabella):
        per_valore = tabella.groupby(asse)["rango"].median().sort_values(ascending=False)
        escursione = float(per_valore.max() - per_valore.min())
        righe.append(
            {
                "strategia": voce,
                "parametro": asse,
                "costante": VERSO_CONFIG[voce][asse],
                "valore": per_valore.index[0],
                "rango_mediano": round(float(per_valore.iloc[0]), 3),
                "peggiore": per_valore.index[-1],
                "escursione": round(escursione, 3),
                "discrimina": escursione >= SOGLIA_UTILE,
                "valori_provati": len(per_valore),
                "configurazioni": len(tabella),
            }
        )
    return righe


def _valore_attuale(costante: str):
    if costante == "EMA_TRIPLET":
        return "/".join(str(int(getattr(config, nome).value)) for nome in TERNA)
    corrente = getattr(config, costante, None)
    return corrente.value if isinstance(corrente, config.Param) else corrente


def defaults_da(scelte: list[dict]) -> dict:
    """Da righe di scelta al dizionario dei default, con le dipendenze espanse."""
    valori: dict = {}
    for riga in scelte:
        # Due condizioni, non una. "Discrimina" dice che il parametro sposta la mediana dei ranghi;
        # "stabile" dice che guardando meta' dei dati si sceglieva lo stesso valore. Senza la
        # seconda si starebbe rifacendo, una coordinata alla volta, l'errore che questo modulo
        # esiste per evitare -- e sui dati di oggi solo la meta' dei parametri la supera.
        if not (riga["discrimina"] and riga.get("stabile_2021_2023")):
            continue
        costante, valore = riga["costante"], riga["valore"]
        if costante == "EMA_TRIPLET":
            for nome, pezzo in zip(TERNA, str(valore).split("/")):
                valori[nome] = int(pezzo)
        elif costante == "ICHIMOKU_FAST":
            valori["ICHIMOKU_FAST"] = int(valore)
            valori["ICHIMOKU_SLOW"], valori["ICHIMOKU_SPAN"] = ICHIMOKU_SCALA[int(valore)]
        elif isinstance(valore, (bool,)):
            valori[costante] = bool(valore)
        else:
            numero = float(valore)
            valori[costante] = int(numero) if numero.is_integer() else round(numero, 3)
    return valori


def stabilita(voce: str, intervallo: str) -> dict[str, bool]:
    """La stessa scelta rifatta sul solo 2021-2023: un default che cambia non e' un default.

    Non e' un fuori campione (la seconda finestra non viene misurata), e' un controllo di
    **stabilita' della scelta**: se guardando meta' dei dati si sceglie un altro valore, quel
    parametro non ha un valore giusto, ne ha uno per periodo.
    """
    tabella = _tabella(voce, intervallo)
    if tabella.empty:
        return {}
    annuale = _annuale(voce, intervallo)
    if annuale.empty:
        return {}
    prima = scegli(voce, annuale)
    intero = {riga["parametro"]: riga["valore"] for riga in scegli(voce, tabella)}
    return {riga["parametro"]: riga["valore"] == intero.get(riga["parametro"]) for riga in prima}


def _annuale(voce: str, intervallo: str) -> pd.DataFrame:
    """Le stesse griglie ricostruite sul solo 2021-2023, dalle tabelle annuali."""
    griglia = GRIGLIA_DI[voce]
    pezzi = []
    for simbolo in SIMBOLI:
        if voce in STORICHE:
            percorso = LAB_DIR.parent / "sweeps" / f"{griglia}_{intervallo}_{simbolo}{SUFFISSO}_annuale.parquet"
        else:
            percorso = LAB_DIR / f"{griglia}_{simbolo}_{intervallo}_annuale.parquet"
        if not percorso.exists():
            continue
        annuale = _fold_triplet(griglia, pd.read_parquet(percorso))
        if "allow_short" in annuale:
            annuale = annuale[~annuale["allow_short"]]
        finestra = annuale[annuale["anno"].between(2021, 2023)]
        if finestra.empty:
            continue
        chiavi = [c for c in finestra.columns if c not in {"anno", "rendimento_%", "n_trade", "win_rate_%"}]
        composto = finestra.groupby(chiavi, dropna=False).agg(
            **{"rendimento_%": ("rendimento_%", lambda s: ((1 + s / 100).prod() - 1) * 100)},
            n_trade=("n_trade", "sum"),
        )
        composto = composto.reset_index()
        # La soglia va scalata sulla finestra: 2021-2023 e' il 53% del periodo, e pretendere lo
        # stesso numero assoluto di operazioni escluderebbe le strategie lente proprio qui, dove
        # servono per dire se la scelta e' stabile.
        composto = composto[composto["n_trade"] >= MIN_TRADE * 3 / 5.6]
        if composto.empty:
            continue
        pezzi.append(composto.assign(rango=composto["rendimento_%"].rank(pct=True), simbolo_=simbolo))
    return pd.concat(pezzi, ignore_index=True) if pezzi else pd.DataFrame()


# ---------------------------------------------------------------------------------------------
# Riga di comando
# ---------------------------------------------------------------------------------------------


def rapporto(intervallo: str) -> tuple[pd.DataFrame, dict]:
    righe, defaults = [], {}
    for voce in config.STRATEGIES:
        if voce not in GRIGLIA_DI:
            continue
        tabella = _tabella(voce, intervallo)
        if tabella.empty:
            continue
        scelte = scegli(voce, tabella)
        stabili = stabilita(voce, intervallo)
        for riga in scelte:
            riga["stabile_2021_2023"] = stabili.get(riga["parametro"])
            riga["attuale"] = _valore_attuale(riga["costante"])
            riga["cambia"] = bool(
                riga["discrimina"] and riga["stabile_2021_2023"] and str(riga["valore"]) != str(riga["attuale"])
            )
        righe.extend(scelte)
        scelti = defaults_da(scelte)
        if scelti:
            defaults[voce] = scelti
    return pd.DataFrame(righe), defaults


INTESTAZIONE = '''"""Valori di partenza misurati, per intervallo. **Generato da `scripts/tune_defaults.py`.**

Non si modifica a mano: si rigenera con `python -m scripts.tune_defaults --all-intervals --save`
dopo aver rifatto le griglie. Ogni numero qui e' il valore la cui **mediana dei ranghi** su cinque
asset e' la piu' alta, fra quelli provati dalla griglia -- non il valore della configurazione che ha
reso di piu', che su questi dati e' l'errore misurato (ρ = −0,69 fra resa in stima e in verifica).

Compaiono solo i parametri che superano **due** controlli: spostano la mediana dei ranghi di almeno
{soglia} (altrimenti il default esistente resta, perche' cambiarlo sarebbe inseguire rumore), e
scelgono lo stesso valore anche guardando il solo 2021-2023. Chi non compare tiene il valore di
`config.py`.

Generato il {data}, dal 2021-01-01, commissione 0,05% per gamba, solo lunghe, su:
{simboli}.
"""

from __future__ import annotations

# {{intervallo: {{voce di menu: {{costante di config: valore}}}}}}
PER_INTERVALLO: dict[str, dict[str, dict[str, float | int | bool]]] = '''


def _modulo(tutti: dict) -> str:
    corpo = json.dumps(tutti, indent=4, sort_keys=True, ensure_ascii=False)
    corpo = corpo.replace(": true", ": True").replace(": false", ": False")
    testata = INTESTAZIONE.format(soglia=SOGLIA_UTILE, data=date.today().isoformat(), simboli=", ".join(SIMBOLI))
    return f"{testata}{corpo}\n"


def accordo(tutti: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Lo stesso parametro visto sui quattro intervalli, uno accanto all'altro.

    Serve a leggere le scelte con la diffidenza giusta. Una finestra che **cresce** man mano che le
    barre si accorciano non e' rumore: e' la stessa regola che copre lo stesso tratto di calendario,
    e va creduta. Un interruttore che **si accende e si spegne** cambiando intervallo invece non ha
    una lettura meccanica, e quasi sempre e' un parametro seduto sulla soglia.
    """
    righe = []
    chiavi = {(r["strategia"], r["parametro"]) for tab in tutti.values() for r in tab.to_dict("records")}
    for strategia, parametro in sorted(chiavi):
        riga = {"strategia": strategia, "parametro": parametro}
        scelti = []
        for intervallo in INTERVALLI:
            tabella = tutti.get(intervallo)
            trovata = (
                tabella[(tabella["strategia"] == strategia) & (tabella["parametro"] == parametro)]
                if tabella is not None and not tabella.empty
                else None
            )
            if trovata is None or trovata.empty:
                riga[intervallo] = "-"
                continue
            voce = trovata.iloc[0]
            preso = bool(voce["discrimina"] and voce["stabile_2021_2023"])
            riga[intervallo] = str(voce["valore"]) if preso else f"({voce['valore']})"
            if preso:
                scelti.append(voce["valore"])
        riga["scelti_diversi"] = len({str(v) for v in scelti})
        righe.append(riga)
    return pd.DataFrame(righe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--all-intervals", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    intervalli = INTERVALLI if args.all_intervals else [args.interval]
    tutti: dict[str, dict] = {}
    tabelle: dict[str, pd.DataFrame] = {}
    pd.set_option("display.width", 220)
    for intervallo in intervalli:
        tabella, defaults = rapporto(intervallo)
        if tabella.empty:
            print(f"\n=== {intervallo}: nessuna misura ===")
            continue
        print(f"\n=== {intervallo} ===")
        colonne = [
            "strategia",
            "parametro",
            "attuale",
            "valore",
            "escursione",
            "discrimina",
            "stabile_2021_2023",
            "cambia",
            "configurazioni",
        ]
        print(tabella[colonne].to_string(index=False))
        discriminanti = int(tabella["discrimina"].sum())
        stabili = int((tabella["discrimina"] & (tabella["stabile_2021_2023"] == True)).sum())  # noqa: E712
        print(
            f"  parametri: {len(tabella)}  |  che discriminano: {discriminanti}  |  "
            f"di questi stabili sul 2021-2023: {stabili}  |  default cambiati: {int(tabella['cambia'].sum())}"
        )
        tutti[intervallo] = defaults
        tabelle[intervallo] = tabella

    if len(tabelle) > 1:
        print("\n=== la stessa scelta sui quattro intervalli ===")
        print("(fra parentesi: misurato ma non adottato -- non discrimina, o non regge sul 2021-2023)")
        print(accordo(tabelle).to_string(index=False))

    if args.save and tutti:
        USCITA.write_text(_modulo(tutti))
        print(f"\nscritto {USCITA.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
