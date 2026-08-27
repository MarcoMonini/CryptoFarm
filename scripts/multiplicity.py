"""Correzione per molteplicita' delle griglie gia' misurate: DSR e PBO.

Oltre dodicimila configurazioni sono state valutate fra le tre sessioni di misura e nessun
risultato della parte quantitativa e' mai stato corretto per il numero di prove. Questo script
paga quel debito **senza rieseguire nulla**: legge gli Sharpe gia' scritti nelle griglie di
`reports/cs_*.csv` e le matrici (anno x configurazione) gia' in `analysis_cache/*/*_annuale.parquet`.

    python -m scripts.multiplicity --grid reports/cs_majors_1d.csv
    python -m scripts.multiplicity --cache
    python -m scripts.multiplicity --selfcheck

Le due misure rispondono a domande diverse e vanno lette insieme:

- **DSR** — lo Sharpe migliore della griglia supera quello che il caso produrrebbe provando
  altrettante configurazioni? Sotto 0,95 il massimo non e' distinguibile dalla fortuna.
- **PBO** — la configurazione migliore in una meta' del campione finisce sotto la mediana
  nell'altra? Sopra 0,5 la *procedura di selezione* fa peggio di una scelta a caso, ed e'
  esattamente cio' che la correlazione stima/verifica di -0,69 gia' diceva sulla rotazione.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from cryptofarm.ml.validation import expected_max_sharpe, probability_of_backtest_overfitting

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "analysis_cache"

# `pnl.annualised` annualizza rendimenti giornalieri con sqrt(365): per tornare alle unita' per
# osservazione che la formula vuole si divide per la stessa costante.
ANNUALISATION = np.sqrt(365)
# 2021-01-01 -> 2026-08-19, il periodo di ogni misura su cinque asset.
DEFAULT_OBS = 2057


def haircut(sharpe_annuo: np.ndarray, n_obs: int, trials: int | None = None) -> dict:
    """Il taglio da molteplicita' su una griglia di cui si conoscono solo gli Sharpe."""
    ann = np.asarray(sharpe_annuo, dtype=float)
    ann = ann[np.isfinite(ann)]
    per_obs = ann / ANNUALISATION
    n = trials if trials is not None else len(per_obs)
    variance = float(np.var(per_obs, ddof=1)) if len(per_obs) > 1 else 0.0
    soglia = expected_max_sharpe(n, variance)
    return {
        "configurazioni": len(ann),
        "prove_contate": n,
        "sharpe_max": float(ann.max()),
        "sharpe_mediano": float(np.median(ann)),
        "sharpe_min": float(ann.min()),
        "soglia_del_caso": soglia * ANNUALISATION,
        "sopra_la_soglia_%": float((per_obs > soglia).mean() * 100),
        # Senza la serie dei rendimenti non si misurano asimmetria e curtosi: restano gaussiane.
        # ponytail: DSR gaussiano; passare `returns` reali quando gli sweeper li scriveranno.
        "DSR_max": _dsr_da_sharpe(float(per_obs.max()), soglia, n_obs),
        "DSR_mediano": _dsr_da_sharpe(float(np.median(per_obs)), soglia, n_obs),
    }


def _dsr_da_sharpe(osservato: float, soglia: float, n_obs: int) -> float:
    from scipy.stats import norm

    return float(norm.cdf((osservato - soglia) * np.sqrt(n_obs - 1)))


def pbo_da_annuale(annuale: pd.DataFrame, chiavi: list[str]) -> tuple[float, int, int]:
    """PBO combinatorio sulla matrice (anno x configurazione) gia' in cache.

    Ogni combinazione di meta' degli anni fa da stima, il resto da verifica: e' la CSCV di
    Bailey-Lopez de Prado, con l'anno come blocco perche' e' la sola partizione che gli sweep
    hanno gia' scritto.
    """
    matrice = annuale.pivot_table(index="anno", columns=chiavi, values="rendimento_%", aggfunc="mean")
    matrice = matrice.dropna(axis=1)
    anni, configurazioni = matrice.shape
    if anni < 4 or configurazioni < 2:
        return float("nan"), anni, configurazioni

    valori = matrice.to_numpy()
    meta = anni // 2
    dentro, fuori = [], []
    for scelta in combinations(range(anni), meta):
        resto = [i for i in range(anni) if i not in scelta]
        dentro.append(valori[list(scelta)].mean(axis=0))
        fuori.append(valori[resto].mean(axis=0))
    return probability_of_backtest_overfitting(np.array(dentro), np.array(fuori)), anni, configurazioni


def _stampa_griglia(percorso: Path, colonna: str, n_obs: int, trials: int | None) -> None:
    tabella = pd.read_csv(percorso)
    if colonna not in tabella.columns:
        raise SystemExit(f"{percorso.name}: nessuna colonna '{colonna}' (ci sono: {list(tabella.columns)})")
    esito = haircut(tabella[colonna].to_numpy(), n_obs, trials)
    print(f"\n{percorso.name}  [{colonna}]")
    if esito["configurazioni"] < 30:
        # Un file di prime-N ha la dispersione troncata dalla selezione stessa: la soglia del caso
        # ne esce vicina a zero e il DSR vicino a uno, sempre. Non e' un risultato.
        print("  ATTENZIONE: poche righe. Se sono le prime-N di una griglia piu' grande la")
        print("  dispersione e' troncata dalla selezione e questo taglio non si puo' leggere.")
    print(f"  {esito['configurazioni']} configurazioni, {esito['prove_contate']} prove contate, {n_obs} osservazioni")
    print(
        f"  Sharpe annuo   max {esito['sharpe_max']:.2f}   mediano {esito['sharpe_mediano']:.2f}"
        f"   min {esito['sharpe_min']:.2f}"
    )
    print(f"  soglia del caso {esito['soglia_del_caso']:.2f} annuo   ({esito['sopra_la_soglia_%']:.0f}% la supera)")
    for nome in ("DSR_max", "DSR_mediano"):
        valore = esito[nome]
        print(f"  {nome:12s} {valore:.3f}   {'sopravvive' if valore > 0.95 else 'NON sopravvive'} a 0,95")


def _cache(n_obs: int) -> None:
    file_annuali = sorted(CACHE_DIR.glob("*/*_annuale.parquet"))
    if not file_annuali:
        print(f"nessuna matrice annuale in {CACHE_DIR}: gli sweep non sono stati eseguiti qui.")
        print("Rigenerare con scripts.strategy_sweep / scripts.strategy_lab, poi rilanciare.")
        return
    print(f"{len(file_annuali)} matrici annuali in {CACHE_DIR}\n")
    print(f"{'griglia':44s} {'anni':>5} {'conf.':>6} {'PBO':>6}  {'DSR max':>8}")
    for percorso in file_annuali:
        annuale = pd.read_parquet(percorso)
        chiavi = [c for c in annuale.columns if c not in ("anno", "n_trade", "rendimento_%", "win_rate_%")]
        pbo, anni, configurazioni = pbo_da_annuale(annuale, chiavi)
        sommario = percorso.with_name(percorso.name.replace("_annuale", ""))
        dsr = float("nan")
        if sommario.exists():
            colonna = pd.read_parquet(sommario)["sharpe"].to_numpy()
            dsr = haircut(colonna, n_obs)["DSR_max"]
        print(f"{percorso.stem[:44]:44s} {anni:5d} {configurazioni:6d} {pbo:6.2f}  {dsr:8.3f}")
    print("\nPBO > 0,50 = selezionare fa peggio che scegliere a caso.  DSR < 0,95 = il massimo e' fortuna.")


def _selfcheck() -> None:
    rng = np.random.default_rng(0)

    # 1. Nessun edge: cento configurazioni di puro rumore. Il massimo non deve sopravvivere.
    rumore = rng.normal(0, 1.0, (2000, 100))
    sharpe_nullo = rumore.mean(axis=0) / rumore.std(axis=0, ddof=1) * ANNUALISATION
    nullo = haircut(sharpe_nullo, 2000)
    assert nullo["DSR_max"] < 0.95, nullo

    # 2. Edge vero e uguale per tutti: il massimo deve sopravvivere.
    vero = rng.normal(0.05, 1.0, (2000, 100))
    sharpe_vero = vero.mean(axis=0) / vero.std(axis=0, ddof=1) * ANNUALISATION
    assert haircut(sharpe_vero, 2000)["DSR_max"] > 0.95

    # 3. Piu' prove non possono alzare il DSR, a parita' di tutto il resto.
    assert haircut(sharpe_vero, 2000, trials=100)["DSR_max"] >= haircut(sharpe_vero, 2000, trials=10_000)["DSR_max"]

    # 4. PBO su rendimenti annuali senza struttura: la selezione non trasferisce, PBO alto.
    anni = 6
    casuale = pd.DataFrame(
        [{"anno": 2020 + a, "cfg": c, "rendimento_%": float(rng.normal(0, 20))} for a in range(anni) for c in range(40)]
    )
    pbo, _, _ = pbo_da_annuale(casuale, ["cfg"])
    assert pbo > 0.3, pbo

    # 5. PBO con una configurazione buona in ogni anno: la selezione trasferisce, PBO basso.
    strutturato = pd.DataFrame(
        [
            {"anno": 2020 + a, "cfg": c, "rendimento_%": float(rng.normal(0, 5) + (30 if c == 7 else 0))}
            for a in range(anni)
            for c in range(40)
        ]
    )
    pbo_buono, _, _ = pbo_da_annuale(strutturato, ["cfg"])
    assert pbo_buono < 0.1, pbo_buono

    print("selfcheck: 5 controlli passati")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="append", help="CSV di griglia da correggere (ripetibile)")
    parser.add_argument("--column", default="Sharpe", help="colonna degli Sharpe annui")
    parser.add_argument("--obs", type=int, default=DEFAULT_OBS, help="osservazioni giornaliere del periodo")
    parser.add_argument(
        "--trials",
        type=int,
        help="prove da contare, se piu' delle righe del file: e' il conto onesto quando la stessa "
        "famiglia e' stata provata anche altrove",
    )
    parser.add_argument("--cache", action="store_true", help="PBO su analysis_cache/*/*_annuale.parquet")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        return
    for percorso in args.grid or []:
        _stampa_griglia(Path(percorso), args.column, args.obs, args.trials)
    if args.cache:
        _cache(args.obs)
    if not args.grid and not args.cache:
        parser.print_help()


if __name__ == "__main__":
    main()
