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

Nessun look-ahead: la classifica alla barra `t` usa solo chiusure fino a `t` incluse, la posizione
si prende alla chiusura di `t` e il rendimento e' quello da `t` al ribilanciamento successivo --
la stessa convenzione del resto del repository.

    python -m scripts.cross_section --universe majors --interval 1d
    python -m scripts.cross_section --grid --interval 1d
    python -m scripts.cross_section --pairs
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from cryptofarm.data.klines import load_klines
from cryptofarm.paths import PROJECT_ROOT
from scripts.strategy_sweep import _annualised, _drawdown

# I cinque ad alta capitalizzazione del mandato, e l'universo largo dello store.
MAJORS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]
WIDE = MAJORS + [
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
]
UNIVERSES = {"majors": MAJORS, "wide": WIDE, "btc_eth": ["BTCUSDT", "ETHUSDT"]}

SINCE = "2021-01-01"
WALLET = 100.0
# A pronti si paga il listino taker di Binance, non quello dei perpetui: 0,1% per gamba.
FEE_PERCENT = 0.1
OUTPUT_DIR = PROJECT_ROOT / "reports"


def load_universe(symbols: list[str], interval: str, since: str, until: str | None = None) -> pd.DataFrame:
    """Chiusure allineate su un indice comune. Le celle prima del listing restano NaN, non zero."""
    closes = {}
    for symbol in symbols:
        candles = load_klines(symbol, interval)
        if candles.empty:
            continue
        closes[symbol] = candles["Close"]
    frame = pd.DataFrame(closes).sort_index()
    frame = frame[frame.index >= since]
    if until:
        frame = frame[frame.index < until]
    return frame


def _rebalance_rows(index: pd.DatetimeIndex, every: int) -> np.ndarray:
    """Le righe su cui si ribilancia. `every` e' in barre, non in giorni."""
    return np.arange(0, len(index), every)


def backtest(
    closes: pd.DataFrame,
    lookback: int,
    top: int,
    every: int,
    fee: float = FEE_PERCENT,
    regime: str = "none",
    regime_window: int = 50,
    skip: int = 0,
) -> dict:
    """Rotazione a peso uguale sui primi `top` per forza relativa.

    `skip` salta le ultime barre nel calcolo del momento (il classico 12-1 delle azioni serve a
    evitare l'inversione di breve). `regime` spegne tutto il portafoglio, non i singoli asset.
    """
    prices = closes.to_numpy(dtype=float)
    n_bars, n_assets = prices.shape
    if n_bars <= lookback + skip + 2:
        raise SystemExit("finestra troppo corta per questo lookback")

    # Momento causale: rendimento fra `t-lookback-skip` e `t-skip`, entrambi noti alla barra t.
    momentum = np.full_like(prices, np.nan)
    start = prices[: n_bars - lookback - skip]
    end = prices[lookback : n_bars - skip]
    momentum[lookback + skip :] = end / start - 1.0

    gate = np.ones(n_bars, dtype=bool)
    if regime == "btc":
        btc = closes["BTCUSDT"].to_numpy(dtype=float)
        mean = pd.Series(btc).rolling(regime_window).mean().to_numpy()
        gate = btc > mean
        gate[np.isnan(mean)] = False

    rebalances = _rebalance_rows(closes.index, every)
    equity = np.full(n_bars, WALLET, dtype=float)
    # Si contabilizza in **valore per asset**, non in pesi normalizzati: il portafoglio puo' stare
    # parzialmente in contanti (meno di `top` asset con forza positiva), e una normalizzazione a
    # somma uno cancellerebbe quella quota invece di tenerla ferma.
    holdings = np.zeros(n_assets)
    cash = WALLET
    turnover_total = 0.0
    n_rebalances = 0
    holdings_log: list[tuple[pd.Timestamp, tuple[str, ...]]] = []

    for start_row, next_row in zip(rebalances, list(rebalances[1:]) + [n_bars - 1]):
        row = start_row
        value = cash + holdings.sum()
        scores = momentum[row]
        eligible = np.where(~np.isnan(scores) & ~np.isnan(prices[row]))[0]
        target = np.zeros(n_assets)
        if gate[row] and len(eligible) >= top:
            chosen = eligible[np.argsort(scores[eligible])[::-1][:top]]
            # Forza relativa **negativa non si compra**: essere il meno peggio di un mercato che
            # scende non e' un segnale. La quota resta 1/top, quindi cio' che si scarta va in
            # contanti invece di concentrarsi su chi resta -- degradare verso il contante, non
            # verso una scommessa piu' grossa.
            chosen = chosen[scores[chosen] > 0]
            target[chosen] = value / top

        traded = float(np.abs(target - holdings).sum())
        cost = traded * fee / 100.0
        turnover_total += traded / value if value > 0 else 0.0
        if traded > 0:
            n_rebalances += 1
            holdings_log.append((closes.index[row], tuple(closes.columns[i] for i in np.where(target > 0)[0])))
        holdings = target
        cash = value - holdings.sum() - cost

        # Segnatura a mercato barra per barra fino al prossimo ribilanciamento.
        held = np.where(holdings > 0)[0]
        base = prices[row]
        for step in range(row, next_row + 1):
            grown = float(np.nansum(holdings[held] * prices[step][held] / base[held])) if len(held) else 0.0
            equity[step] = cash + grown
        if len(held):
            holdings[held] = holdings[held] * prices[next_row][held] / base[held]

    cagr, volatility, sharpe = _annualised(equity, closes.index)
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    return {
        "rendimento_%": round((equity[-1] / WALLET - 1) * 100, 1),
        "CAGR_%": round(cagr, 1),
        "Sharpe": round(sharpe, 2),
        "drawdown_%": round(_drawdown(equity), 1),
        "ribilanciamenti": n_rebalances,
        "turnover_annuo": round(turnover_total / max(years, 1e-9), 1),
        "_equity": equity,
        "_holdings": holdings_log,
    }


def benchmarks(closes: pd.DataFrame) -> dict[str, dict]:
    """Possesso passivo di BTC e dell'universo a peso uguale ribilanciato mai (comprare e tenere)."""
    out = {}
    btc = closes["BTCUSDT"].to_numpy(dtype=float)
    equity = WALLET * btc / btc[0]
    cagr, _, sharpe = _annualised(equity, closes.index)
    out["BTC comprare e tenere"] = {
        "rendimento_%": round((equity[-1] / WALLET - 1) * 100, 1),
        "CAGR_%": round(cagr, 1),
        "Sharpe": round(sharpe, 2),
        "drawdown_%": round(_drawdown(equity), 1),
        "ribilanciamenti": 0,
        "turnover_annuo": 0.0,
    }
    # Peso uguale su cio' che esiste alla prima barra, poi lasciato correre.
    prices = closes.to_numpy(dtype=float)
    first = prices[0]
    live = ~np.isnan(first)
    weights = live / live.sum()
    equity = WALLET * np.nansum(weights * prices / first, axis=1)
    cagr, _, sharpe = _annualised(equity, closes.index)
    out["universo a peso uguale"] = {
        "rendimento_%": round((equity[-1] / WALLET - 1) * 100, 1),
        "CAGR_%": round(cagr, 1),
        "Sharpe": round(sharpe, 2),
        "drawdown_%": round(_drawdown(equity), 1),
        "ribilanciamenti": 0,
        "turnover_annuo": 0.0,
    }
    return out


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
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        selfcheck()
        return

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


def selfcheck() -> None:
    """Il minimo che fallisce se la contabilita' si rompe. `--selfcheck` per eseguirlo."""
    index = pd.date_range("2021-01-01", periods=120, freq="1D")

    # 1. Prezzi fermi, nessuna commissione: il capitale non si muove e non si compra nulla
    #    (momento zero non e' > 0).
    flat = pd.DataFrame({"BTCUSDT": 100.0, "ETHUSDT": 100.0}, index=index)
    result = backtest(flat, lookback=10, top=1, every=5, fee=0.0)
    assert abs(result["rendimento_%"]) < 1e-9, result["rendimento_%"]
    assert result["ribilanciamenti"] == 0, result["ribilanciamenti"]

    # 2. Un solo asset che sale del 100%, comprato appena il momento diventa positivo:
    #    il risultato e' il rendimento dal momento dell'acquisto, non da inizio serie.
    ramp = pd.DataFrame({"BTCUSDT": np.linspace(100.0, 200.0, 120)}, index=index)
    bought_at = ramp["BTCUSDT"].iloc[10]  # prima barra con momento definito e positivo
    expected = (200.0 / bought_at - 1) * 100
    result = backtest(ramp, lookback=10, top=1, every=5, fee=0.0)
    assert abs(result["rendimento_%"] - round(expected, 1)) < 0.2, (result["rendimento_%"], expected)

    # 3. La commissione toglie, e toglie in proporzione al giro d'affari.
    free = backtest(ramp, lookback=10, top=1, every=5, fee=0.0)["rendimento_%"]
    paid = backtest(ramp, lookback=10, top=1, every=5, fee=0.5)["rendimento_%"]
    assert paid < free, (paid, free)

    # 4. Contanti parziali: due asset, `top=2`, ma uno solo con momento positivo. Meta' capitale
    #    resta ferma invece di sparire -- e' il difetto che la prima versione aveva.
    mixed = pd.DataFrame(
        {"BTCUSDT": np.linspace(100.0, 200.0, 120), "ETHUSDT": np.linspace(100.0, 50.0, 120)},
        index=index,
    )
    half = backtest(mixed, lookback=10, top=2, every=5, fee=0.0)["rendimento_%"]
    full = backtest(mixed, lookback=10, top=1, every=5, fee=0.0)["rendimento_%"]
    # Con la contabilita' a pesi normalizzati la quota in contanti spariva a ogni ribilanciamento
    # e questo valeva −100%. Deve stare fra zero e l'investimento pieno, non sotto.
    assert 0 < half < full, (half, full)

    # 5. Nessun look-ahead: troncare la serie non cambia le rotazioni gia' decise.
    full = backtest(mixed, lookback=10, top=1, every=5, fee=0.0)["_holdings"]
    cut = backtest(mixed.iloc[:80], lookback=10, top=1, every=5, fee=0.0)["_holdings"]
    assert full[: len(cut)] == cut, (full[: len(cut)], cut)

    print("selfcheck: ok")


if __name__ == "__main__":
    main()
