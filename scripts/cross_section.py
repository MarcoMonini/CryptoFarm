"""Rotazione trasversale: invece di prevedere *quando*, scegliere *quale*.

Tutto cio' che il progetto ha misurato finora -- `strategy_sweep`, `strategy_lab`, l'intera
pipeline ML -- decide su **un asset alla volta**: dentro o fuori dal mercato, in base alla storia
di quell'asset. E' l'unica famiglia provata, ed e' quella che fuori campione non ha mai battuto il
possesso passivo (`.claude/docs/strategie-nuove.md` §6).

Qui si misura l'altra famiglia. A ogni ribilanciamento si ordinano gli asset per forza relativa e
si tengono i primi `top`, a peso uguale, sempre lunghi e a pronti. Il segnale non e' "il mercato
sale" ma "questo sale piu' di quelli": due informazioni diverse, e la seconda non e' mai stata
provata su questi dati.

E' anche la risposta alla domanda sulle **coppie** (BTC/ETH, ETH/SOL): tenere la piu' forte fra due
e' esattamente questo codice con un universo di due e `top=1`. Non serve una gamba corta, che il
mandato esclude.

Due varianti del filtro di regime, perche' la differenza e' il punto:

- `none` -- sempre investiti nei primi `top`;
- `btc` -- fuori dal mercato quando BTC sta sotto la sua media a `regime_window` barre. E' un
  interruttore **unico per tutto il portafoglio**: in cripto la correlazione fra asset in caduta
  va a uno, quindi selezionare il migliore fra quindici che scendono non protegge da niente.

**Il meccanismo sta in `cryptofarm.trading.rotation`**, non qui: lo usa anche la pagina Streamlit,
e il pacchetto non puo' dipendere da `scripts/`. Qui restano le griglie, il fuori campione, le
coppie e il controllo -- cioe' l'esperimento, non la strategia.

    python -m scripts.cross_section --universe majors --interval 1d
    python -m scripts.cross_section --grid --interval 1d
    python -m scripts.cross_section --pairs

Le proprieta' del motore sono verificate in `tests/test_rotation.py`, che la CI esegue.
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import rotation
from cryptofarm.trading.rotation import FEE_PERCENT, MAJORS, backtest, benchmarks, load_universe

UNIVERSES = rotation.UNIVERSI
SINCE = "2021-01-01"
OUTPUT_DIR = PROJECT_ROOT / "reports"

GRID = {
    "lookback": [10, 20, 30, 60, 90],
    "top": [1, 2, 3, 5],
    "every": [1, 3, 7, 14],
    "regime": ["none", "btc"],
}


def run_grid(closes: pd.DataFrame, fee: float = FEE_PERCENT) -> pd.DataFrame:
    rows = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[key] for key in keys)):
        params = dict(zip(keys, values))
        if params["top"] > closes.shape[1]:
            continue
        result = backtest(closes, fee=fee, **params)
        rows.append({**params, **{k: v for k, v in result.items() if not k.startswith("_")}})
    return pd.DataFrame(rows)


def _report(title: str, table: pd.DataFrame, bench: dict[str, dict]) -> None:
    print(f"\n===== {title} =====")
    for name, row in bench.items():
        print(
            f"  [riferimento] {name:26s} {row['rendimento_%']:>8.1f}%  "
            f"Sharpe {row['Sharpe']:>5.2f}  DD {row['drawdown_%']:>5.1f}%"
        )
    returns = table["rendimento_%"]
    sopra_btc = 100 * (returns > bench["BTC comprare e tenere"]["rendimento_%"]).mean()
    sopra_universo = 100 * (returns > bench["universo a peso uguale"]["rendimento_%"]).mean()
    print(
        f"  configurazioni: {len(table)}  |  mediana rendimento {returns.median():.1f}%  |  "
        f"in utile {100 * (returns > 0).mean():.1f}%  |  "
        f"sopra BTC {sopra_btc:.1f}%  |  sopra l'universo a peso uguale {sopra_universo:.1f}%"
    )
    best = table.sort_values("Sharpe", ascending=False).head(8)
    print(best.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--universe", default="majors", choices=list(UNIVERSES))
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--until", default=None)
    parser.add_argument("--fee", type=float, default=FEE_PERCENT)
    parser.add_argument("--grid", action="store_true", help="tutta la griglia invece di una configurazione")
    parser.add_argument("--split", action="store_true", help="scelta 2021-2023, resa 2024-2026")
    parser.add_argument("--pairs", action="store_true", help="la piu' forte fra due, su tutte le coppie")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--top", type=int, default=2)
    parser.add_argument("--every", type=int, default=7)
    parser.add_argument("--regime", default="none", choices=["none", "btc"])
    parser.add_argument("--save", default="")
    args = parser.parse_args()

    if args.pairs:
        pairs_report(args)
        return

    symbols = UNIVERSES[args.universe]
    closes = load_universe(symbols, args.interval, args.since, args.until)
    print(
        f"{args.universe}: {closes.shape[1]} asset, {len(closes)} barre {args.interval}, "
        f"{closes.index[0].date()} -> {closes.index[-1].date()}"
    )

    if args.split:
        split_report(symbols, args)
        return

    bench = benchmarks(closes)
    if args.grid:
        table = run_grid(closes, args.fee)
        _report(f"{args.universe} {args.interval} fee {args.fee}%", table, bench)
        if args.save:
            OUTPUT_DIR.mkdir(exist_ok=True)
            table.to_csv(OUTPUT_DIR / f"{args.save}.csv", index=False)
            print(f"salvato in reports/{args.save}.csv")
        return

    result = backtest(closes, args.lookback, args.top, args.every, args.fee, args.regime)
    for name, row in bench.items():
        print(
            f"  [riferimento] {name:26s} {row['rendimento_%']:>8.1f}%  Sharpe {row['Sharpe']:>5.2f}  DD"
            f"{row['drawdown_%']:>5.1f}%"
        )
    print({k: v for k, v in result.items() if not k.startswith("_")})
    print("ultime rotazioni:", result["_holdings"][-5:])


def split_report(symbols: list[str], args) -> None:
    """Scelta della configurazione sulla prima meta', resa sulla seconda. Nient'altro conta."""
    estimation = load_universe(symbols, args.interval, args.since, "2024-01-01")
    verification = load_universe(symbols, args.interval, "2024-01-01", args.until)
    table_in = run_grid(estimation, args.fee).sort_values("Sharpe", ascending=False)
    bench_in, bench_out = benchmarks(estimation), benchmarks(verification)

    rows = []
    for _, row in table_in.head(10).iterrows():
        params = {k: row[k] for k in GRID}
        params["lookback"], params["top"], params["every"] = (
            int(params["lookback"]),
            int(params["top"]),
            int(params["every"]),
        )
        out = backtest(verification, fee=args.fee, **params)
        rows.append(
            {
                **params,
                "stima_%": row["rendimento_%"],
                "stima_Sharpe": row["Sharpe"],
                "verifica_%": out["rendimento_%"],
                "verifica_Sharpe": out["Sharpe"],
                "verifica_DD_%": out["drawdown_%"],
            }
        )
    table = pd.DataFrame(rows)
    print("\n===== fuori campione: scelta 2021-2023, resa 2024-oggi =====")
    print(
        f"  stima     BTC {bench_in['BTC comprare e tenere']['rendimento_%']:.1f}%  "
        f"universo {bench_in['universo a peso uguale']['rendimento_%']:.1f}%"
    )
    print(
        f"  verifica  BTC {bench_out['BTC comprare e tenere']['rendimento_%']:.1f}%  "
        f"universo {bench_out['universo a peso uguale']['rendimento_%']:.1f}%"
    )
    print(table.to_string(index=False))
    # La correlazione fra le due colonne dice se la **scelta dei parametri** trasferisce; il livello
    # dice se paga la **famiglia**. Sono due domande diverse e possono avere risposte opposte.
    if len(table) > 2:
        print(f"  correlazione stima/verifica sulle prime 10: {table['stima_%'].corr(table['verifica_%']):.2f}")

    # Tutta la griglia sulla finestra di verifica: e' la sola colonna che non ha selezione dentro.
    whole = run_grid(verification, args.fee)
    btc_out = bench_out["BTC comprare e tenere"]["rendimento_%"]
    equal_out = bench_out["universo a peso uguale"]["rendimento_%"]
    print(
        f"  tutta la griglia sul solo 2024-oggi: mediana {whole['rendimento_%'].median():.1f}%  |  "
        f"in utile {100 * (whole['rendimento_%'] > 0).mean():.0f}%  |  "
        f"sopra BTC {100 * (whole['rendimento_%'] > btc_out).mean():.0f}%  |  "
        f"sopra l'universo {100 * (whole['rendimento_%'] > equal_out).mean():.0f}%  |  "
        f"Sharpe mediano {whole['Sharpe'].median():.2f} contro {bench_out['universo a peso uguale']['Sharpe']:.2f}"
    )
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        table.to_csv(OUTPUT_DIR / f"{args.save}.csv", index=False)


def pairs_report(args) -> None:
    """La piu' forte fra due, su tutte le coppie dei cinque: la domanda sulle coppie, misurata."""
    rows = []
    for a, b in itertools.combinations(MAJORS, 2):
        closes = load_universe([a, b], args.interval, args.since, args.until)
        if closes.shape[1] < 2 or closes.isna().all().any():
            continue
        closes = closes.dropna()
        result = backtest(closes, args.lookback, 1, args.every, args.fee)
        first = closes.iloc[0]
        last = closes.iloc[-1]
        rows.append(
            {
                "coppia": f"{a[:-4]}/{b[:-4]}",
                "rotazione_%": result["rendimento_%"],
                "Sharpe": result["Sharpe"],
                "DD_%": result["drawdown_%"],
                "tieni_il_1o_%": round((last[a] / first[a] - 1) * 100, 1),
                "tieni_il_2o_%": round((last[b] / first[b] - 1) * 100, 1),
                "meta_e_meta_%": round(((last[a] / first[a] + last[b] / first[b]) / 2 - 1) * 100, 1),
            }
        )
    table = pd.DataFrame(rows)
    print(f"\n===== la piu' forte fra due, lookback {args.lookback} barre, ogni {args.every} =====")
    print(table.to_string(index=False))
    beats = (table["rotazione_%"] > table["meta_e_meta_%"]).mean() * 100
    print(f"  batte il meta'-e-meta' passivo in {beats:.0f}% delle coppie")
    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        table.to_csv(OUTPUT_DIR / f"{args.save}.csv", index=False)


if __name__ == "__main__":
    main()
