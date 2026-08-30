"""L'audit della confluenza: le stesse configurazioni su molti asset, dentro e fuori campione.

`confluence_lab.py` misura **un** simbolo per invocazione ed e' il banco con cui si guarda una
configurazione. Questo modulo risponde a una domanda diversa e piu' scomoda: *la strategia
funziona*, o funziona la cella fortunata su BTC? Da li' le quattro differenze:

- **molti asset in una passata**, perche' un risultato che non si ripete su quindici mercati non
  e' un risultato. I ranghi si sommano **dentro il simbolo** (rango percentile), che e' l'unico
  modo di mettere insieme asset i cui possessi passivi vanno da +134% a +4.346% -- lo stesso
  metodo con cui `tune_defaults` sceglie i valori di partenza, e per la stessa ragione;
- **la finestra e' divisa**, stima e verifica, e le due si confrontano *per configurazione*. La
  domanda non e' quanto rende la migliore in campione: e' se la classifica in campione dice
  qualcosa di quella fuori. Su questo progetto quella correlazione e' gia' stata misurata a
  **-0,69** sulla rotazione, cioe' peggio che scegliere a caso, e finche' non la si misura anche
  qui non c'e' motivo di credere che qui sia diversa;
- **il riferimento e' il possesso passivo dello stesso asset sulla stessa finestra**, non lo zero.
  Una strategia che rende +40% dove tenere fermo rendeva +300% ha perso il 260%, e leggere quel
  +40% come un successo e' il modo piu' comune di sbagliarsi;
- **Monte Carlo per permutazione delle barre**, che e' la difesa contro il fatto che tutto questo
  e' una ricerca su molte celle. Vedi `permuta`.

Non c'e' niente qui che decida parametri: l'uscita sono CSV in `analysis_cache/confluence_audit/`
e le conclusioni si scrivono a mano guardandoli.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import confluence, rotation
from cryptofarm.trading.pnl import annualised, drawdown
from scripts import confluence_lab as lab

USCITA = PROJECT_ROOT / "analysis_cache" / "confluence_audit"

# Tutto lo store. `rotation.WIDE` e' il controllo che allargare l'universo *peggiora* la
# rotazione, ma qui l'ampiezza non e' una scelta di portafoglio: e' il campione su cui si guarda
# se un risultato si ripete, e piu' e' largo meno una coincidenza sopravvive.
TUTTI = rotation.WIDE

# Il taglio fra stima e verifica. Non e' a meta' della storia: e' l'inizio dell'ultimo ciclo
# intero, cosi' che la verifica contenga un massimo e un minimo invece di mezza salita.
TAGLIO = "2024-01-01"


# ---------------------------------------------------------------------------------------------
# Le candele e gli stati, con la cache che tiene **un** simbolo per volta
# ---------------------------------------------------------------------------------------------

_CACHE: dict = {}


def _dati(simbolo: str, intervallo: str, da: str, fino: str | None):
    """Candele e stati dei votanti per una finestra. Tiene in memoria solo l'ultima chiamata.

    La cache di `confluence_lab` e' chiavata su (simbolo, intervallo) e qui non basterebbe: la
    stessa coppia ricorre con finestre diverse e si riprenderebbe le candele sbagliate. Tenerne
    una sola evita anche che quindici simboli per dieci processi diventino un gigabyte e mezzo.
    """
    chiave = (simbolo, intervallo, da, fino)
    if chiave not in _CACHE:
        _CACHE.clear()
        from scripts.strategy_sweep import load_interval

        candele = load_interval(intervallo, da, fino, simbolo)
        if len(candele) < 500:
            return None, None
        _CACHE[chiave] = (candele, confluence.stati_dei_votanti(candele, intervallo))
    return _CACHE[chiave]


def passivo(candele: pd.DataFrame) -> dict:
    """Il possesso passivo sulla stessa finestra: il numero da battere, non lo zero."""
    chiusure = candele["Close"].to_numpy()
    cagr, _, sharpe = annualised(chiusure, candele.index)
    return {
        "passivo_%": float(chiusure[-1] / chiusure[0] - 1) * 100,
        "passivo_cagr_%": cagr,
        "passivo_sharpe": sharpe,
        "passivo_drawdown_%": drawdown(chiusure),
    }


# ---------------------------------------------------------------------------------------------
# La griglia su molti asset
# ---------------------------------------------------------------------------------------------


def _lotto(lavoro: tuple) -> list[dict]:
    simbolo, intervallo, finestra, da, fino, batch, fee, carry = lavoro
    candele, stati = _dati(simbolo, intervallo, da, fino)
    if candele is None:
        return []
    riferimento = passivo(candele)
    righe = []
    for parametri in batch:
        # Gli stati precalcolati valgono finche' la cella non tocca un votante; `valuta` se ne
        # accorge da sola e li ricalcola, quindi qui si passano sempre.
        riga = lab.valuta(candele, intervallo, parametri, stati, fee, carry)
        righe.append(
            {
                "simbolo": simbolo,
                "intervallo": intervallo,
                "finestra": finestra,
                "barre": len(candele),
                **riga,
                **riferimento,
                "extra_%": riga["rendimento_%"] - riferimento["passivo_%"],
            }
        )
    return righe


def esegui(
    simboli: list[str],
    intervallo: str,
    finestre: dict[str, tuple[str, str | None]],
    configurazioni: list[dict],
    fee: float,
    carry: float,
    workers: int,
) -> pd.DataFrame:
    """Ogni configurazione su ogni simbolo in ogni finestra. Un lotto per (simbolo, finestra)."""
    passo = max(1, len(configurazioni) // 4)
    lavori = [
        (simbolo, intervallo, nome, da, fino, configurazioni[i : i + passo], fee, carry)
        for simbolo in simboli
        for nome, (da, fino) in finestre.items()
        for i in range(0, len(configurazioni), passo)
    ]
    t0 = time.time()
    if workers <= 1:
        righe = [riga for lavoro in lavori for riga in _lotto(lavoro)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            righe = [riga for lotto in pool.map(_lotto, lavori) for riga in lotto]
    print(f"  {len(righe)} misure in {time.time() - t0:.0f}s")
    return pd.DataFrame(righe)


# ---------------------------------------------------------------------------------------------
# Aggregazione: il rango dentro il simbolo, non la media fra simboli
# ---------------------------------------------------------------------------------------------

CHIAVI_NON_PARAMETRO = {
    "simbolo",
    "intervallo",
    "finestra",
    "barre",
    "n_trade",
    "trade_anno",
    "rendimento_%",
    "cagr_%",
    "sharpe",
    "drawdown_%",
    "win_rate_%",
    "votante_dominante",
    "necessarieta_max",
    "passivo_%",
    "passivo_cagr_%",
    "passivo_sharpe",
    "passivo_drawdown_%",
    "extra_%",
    "rango",
    "config",
}


def colonne_parametro(risultati: pd.DataFrame) -> list[str]:
    return [c for c in risultati.columns if c not in CHIAVI_NON_PARAMETRO and not c.startswith("nec_")]


def con_rango(risultati: pd.DataFrame, metrica: str = "extra_%") -> pd.DataFrame:
    """Aggiunge il rango percentile della configurazione **dentro il proprio simbolo e finestra**.

    Sommare le rese fra asset darebbe a SOL un peso venti volte quello di BTC solo perche' e'
    cresciuto di piu'. Il rango toglie la scala e lascia l'ordinamento, che e' l'unica cosa che
    interessa quando si chiede «questa configurazione e' meglio di quell'altra».
    """
    risultati = risultati.copy()
    parametri = colonne_parametro(risultati)
    risultati["config"] = risultati[parametri].astype(str).agg("|".join, axis=1)
    risultati["rango"] = risultati.groupby(["simbolo", "finestra"])[metrica].rank(pct=True)
    return risultati


def classifica(risultati: pd.DataFrame, finestra: str) -> pd.DataFrame:
    """Le configurazioni ordinate per mediana dei ranghi, con quanto spesso battono il passivo."""
    dentro = risultati[risultati["finestra"] == finestra]
    return (
        dentro.groupby("config")
        .agg(
            **{
                "rango_mediano": ("rango", "median"),
                "extra_mediano": ("extra_%", "median"),
                "batte_passivo_%": ("extra_%", lambda s: (s > 0).mean() * 100),
                "rendimento_mediano": ("rendimento_%", "median"),
                "trade_anno": ("trade_anno", "median"),
                "drawdown_mediano": ("drawdown_%", "median"),
                "sharpe_mediano": ("sharpe", "median"),
                "nec_max_mediana": ("necessarieta_max", "median"),
                "simboli": ("simbolo", "nunique"),
            }
        )
        .sort_values("rango_mediano", ascending=False)
    )


def trasferimento(risultati: pd.DataFrame, dentro: str = "stima", fuori: str = "verifica") -> dict:
    """La correlazione fra rango in stima e rango in verifica, che e' la domanda vera.

    Se e' negativa, cercare il massimo in campione e' peggio che scegliere a caso -- ed e' cio'
    che questo progetto ha gia' misurato sulla rotazione (-0,69). Il numero si riporta con
    Spearman perche' interessa l'ordinamento, non la distanza.
    """
    a = classifica(risultati, dentro)["rango_mediano"]
    b = classifica(risultati, fuori)["rango_mediano"]
    comuni = a.index.intersection(b.index)
    if len(comuni) < 5:
        return {"n": len(comuni)}
    return {
        "n": len(comuni),
        "spearman": float(a[comuni].corr(b[comuni], method="spearman")),
        "pearson": float(a[comuni].corr(b[comuni])),
        # Le prime dieci in stima: dove finiscono davvero.
        "rango_fuori_delle_prime_10": float(b[a[comuni].nlargest(10).index].mean()),
    }


# ---------------------------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------------------------


def permuta(candele: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Le stesse barre in ordine diverso: il null contro cui misurare un vantaggio di *tempismo*.

    Ogni barra conserva la propria geometria interna -- dove stanno massimo, minimo e chiusura
    rispetto all'apertura -- e viene riattaccata alla chiusura della barra che la precede nel
    nuovo ordine. Cosi' la distribuzione dei rendimenti di barra e la forma delle candele restano
    **identiche**, e sparisce solo la correlazione seriale: i trend, i canali, gli intrecci di
    medie, cioe' tutto cio' che la confluenza dice di leggere.

    E' il null giusto e non «prezzi casuali»: la deriva dell'asset e' conservata per costruzione,
    quindi una strategia lunga che guadagna solo per esposizione guadagna anche qui. Cio' che
    resta della differenza fra vero e permutato e' tempismo, e nient'altro.
    """
    rendimenti = np.log(candele["Close"].to_numpy() / candele["Open"].to_numpy())
    alto = np.log(candele["High"].to_numpy() / candele["Open"].to_numpy())
    basso = np.log(candele["Low"].to_numpy() / candele["Open"].to_numpy())
    salto = np.log(candele["Open"].to_numpy()[1:] / candele["Close"].to_numpy()[:-1])

    ordine = rng.permutation(len(candele))
    salti = np.concatenate([[0.0], rng.permutation(salto)])

    # Il prezzo ricostruito: apertura della barra i = chiusura della i-1 piu' il salto notturno.
    apertura = np.empty(len(candele))
    log_apertura = np.log(candele["Open"].to_numpy()[0])
    for i, k in enumerate(ordine):
        log_apertura += salti[i]
        apertura[i] = log_apertura
        log_apertura += rendimenti[k]

    fuori = pd.DataFrame(
        {
            "Open": np.exp(apertura),
            "High": np.exp(apertura + alto[ordine]),
            "Low": np.exp(apertura + basso[ordine]),
            "Close": np.exp(apertura + rendimenti[ordine]),
            "Volume": candele["Volume"].to_numpy()[ordine],
        },
        index=candele.index,
    )
    return fuori


def _una_permutazione(lavoro: tuple) -> dict | None:
    simbolo, intervallo, da, fino, parametri, seme, fee, carry = lavoro
    from scripts.strategy_sweep import load_interval

    candele = load_interval(intervallo, da, fino, simbolo)
    if len(candele) < 500:
        return None
    finte = candele if seme < 0 else permuta(candele, np.random.default_rng(seme))
    riga = lab.valuta(finte, intervallo, parametri, None, fee, carry)
    return {
        "simbolo": simbolo,
        "seme": seme,
        "vero": seme < 0,
        **{k: riga[k] for k in ("n_trade", "rendimento_%", "cagr_%", "sharpe", "drawdown_%", "win_rate_%")},
        **{"passivo_%": passivo(finte)["passivo_%"]},
        "extra_%": riga["rendimento_%"] - passivo(finte)["passivo_%"],
    }


def monte_carlo(
    simboli: list[str],
    intervallo: str,
    da: str,
    fino: str | None,
    parametri: dict,
    prove: int,
    fee: float,
    carry: float,
    workers: int,
) -> pd.DataFrame:
    """Il vero contro `prove` permutazioni, per ogni simbolo. Restituisce tutte le righe."""
    lavori = [
        (simbolo, intervallo, da, fino, parametri, seme, fee, carry)
        for simbolo in simboli
        for seme in [-1, *range(prove)]
    ]
    t0 = time.time()
    if workers <= 1:
        righe = [_una_permutazione(lavoro) for lavoro in lavori]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            righe = list(pool.map(_una_permutazione, lavori))
    print(f"  {len(lavori)} prove in {time.time() - t0:.0f}s")
    return pd.DataFrame([r for r in righe if r])


def valore_p(mc: pd.DataFrame, metrica: str = "extra_%") -> pd.DataFrame:
    """Per ogni simbolo, in che frazione delle permutazioni il caso ha fatto **meglio** del vero.

    E' il valore-p a una coda della domanda «questo risultato si spiega col tempismo?». Con la
    convenzione `(k+1)/(n+1)`, che non restituisce mai zero: con duecento permutazioni il minimo
    onesto e' 0,005 e scriverlo come 0 sarebbe una precisione che non si e' comprata.
    """
    righe = []
    for simbolo, gruppo in mc.groupby("simbolo"):
        vero = gruppo[gruppo["vero"]]
        finte = gruppo[~gruppo["vero"]]
        if vero.empty or finte.empty:
            continue
        osservato = float(vero[metrica].iloc[0])
        meglio = int((finte[metrica] >= osservato).sum())
        righe.append(
            {
                "simbolo": simbolo,
                "vero": osservato,
                "mediana_permutata": float(finte[metrica].median()),
                "permutazioni": len(finte),
                "meglio_del_vero": meglio,
                "valore_p": (meglio + 1) / (len(finte) + 1),
                "vero_n_trade": int(vero["n_trade"].iloc[0]),
                "mediana_n_trade_permutato": float(finte["n_trade"].median()),
            }
        )
    return pd.DataFrame(righe)


# ---------------------------------------------------------------------------------------------


def salva(nome: str, dati: pd.DataFrame) -> None:
    USCITA.mkdir(parents=True, exist_ok=True)
    percorso = USCITA / f"{nome}.csv"
    dati.to_csv(percorso, index=False)
    print(f"  scritto {percorso} ({len(dati)} righe)")


def _selfcheck() -> None:
    """Gira senza store: la permutazione e l'aggregazione su candele finte."""
    candele = lab._finte(giorni=120)
    # `_finte` produce massimi **sotto** il corpo nel 6% delle barre: sono candele impossibili, e
    # le candele vere non lo sono mai (verificato: zero violazioni su 50.000 barre di BTC). Qui
    # servono coerenti, perche' la proprieta' da verificare e' «la permutazione **conserva** la
    # coerenza», che su un ingresso gia' incoerente non si potrebbe distinguere dal contrario.
    corpo = candele[["Open", "Close"]]
    candele = candele.assign(
        High=np.maximum(candele["High"], corpo.max(axis=1)), Low=np.minimum(candele["Low"], corpo.min(axis=1))
    )
    rng = np.random.default_rng(0)
    finte = permuta(candele, rng)
    assert len(finte) == len(candele), "la permutazione cambia il numero di barre"
    assert (finte["High"] >= finte[["Open", "Close"]].max(axis=1) - 1e-9).all(), "massimo sotto il corpo"
    assert (finte["Low"] <= finte[["Open", "Close"]].min(axis=1) + 1e-9).all(), "minimo sopra il corpo"
    veri = np.diff(np.log(candele["Close"].to_numpy()))
    permutati = np.diff(np.log(finte["Close"].to_numpy()))
    assert abs(np.std(veri) - np.std(permutati)) < 0.2 * np.std(veri), "la volatilita' di barra non e' conservata"

    print("selfcheck: permutazione ok,", f"deriva vera {veri.sum():+.3f} vs permutata {permutati.sum():+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", default="coordinate", choices=lab.NOMI_GRIGLIA)
    parser.add_argument("--symbols", default="wide", choices=[*rotation.UNIVERSI, "tutti"])
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--since", default="2019-01-01")
    parser.add_argument("--split", default=TAGLIO)
    parser.add_argument("--fee", type=float, default=lab.COMMISSIONE)
    parser.add_argument("--carry", type=float, default=lab.MANTENIMENTO)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--monte-carlo", type=int, default=0, help="quante permutazioni per simbolo")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    simboli = TUTTI if args.symbols == "tutti" else rotation.UNIVERSI[args.symbols]
    nome = f"{args.grid}_{args.symbols}_{args.interval}{args.suffix}"

    if args.monte_carlo:
        print(f"Monte Carlo: {args.monte_carlo} permutazioni x {len(simboli)} simboli")
        mc = monte_carlo(
            simboli,
            args.interval,
            args.since,
            None,
            dict(lab.CENTRO),
            args.monte_carlo,
            args.fee,
            args.carry,
            args.workers,
        )
        salva(f"mc_{nome}", mc)
        p = valore_p(mc)
        salva(f"mc_p_{nome}", p)
        print(p.to_string(index=False))
        return

    configurazioni = lab.celle(args.grid)
    finestre = {"stima": (args.since, args.split), "verifica": (args.split, None), "tutto": (args.since, None)}
    print(f"{len(configurazioni)} configurazioni x {len(simboli)} simboli x {len(finestre)} finestre")
    risultati = esegui(simboli, args.interval, finestre, configurazioni, args.fee, args.carry, args.workers)
    if risultati.empty:
        raise SystemExit("nessuna misura: lo store non copre il periodo chiesto")
    risultati = con_rango(risultati)
    salva(nome, risultati)

    for finestra in finestre:
        top = classifica(risultati, finestra)
        print(f"\n--- {finestra} ---")
        print(top.head(8).to_string())
    print("\ntrasferimento stima -> verifica:", trasferimento(risultati))


if __name__ == "__main__":
    main()
