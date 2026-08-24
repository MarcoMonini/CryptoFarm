"""Le tre verifiche che si fanno **dopo** aver scelto una configurazione.

Una griglia dice qual e' la configurazione migliore sull'intervallo e con le commissioni con cui
e' stata misurata. Non dice se quel risultato sopravvive a un cambio di timeframe, a una tariffa
diversa o a un periodo diverso -- e sono le tre cose che nella pratica cambiano di sicuro. Questo
script riprende le configurazioni migliori di ogni griglia (piu' quella con cui la pagina si apre,
come riferimento) e le rimisura:

- su tutti gli intervalli che il menu del simulatore offre, dal 5m al giorno;
- a commissioni da zero allo 0,2% per gamba, cioe' l'intervallo fra maker su Binance con sconto e
  taker senza;
- sul periodo intero contro i due sotto-periodi, gia' disponibili anno per anno dallo sweep.

Le configurazioni non vengono riottimizzate: sono le stesse righe, rieseguite altrove. Quando un
risultato regge solo dove e' stato scelto, e' li' che si vede.

    python -m scripts.strategy_focus --top 3
"""

from __future__ import annotations

import argparse

import pandas as pd

from cryptofarm.trading import config as page_config
from scripts.strategy_sweep import GRIDS, Indicators, run_cells
from scripts.sweep_report import REPORT_DIR, load_sweeps, operative

INTERVALS = ["5m", "15m", "30m", "1h", "4h", "1d"]
# Da maker con sconto BNB a taker pieno: le due estremita' realistiche del listino di Binance.
FEES = [0.0, 0.04, 0.075, 0.1, 0.2]


def _indicator_fields() -> list[str]:
    return list(Indicators.__dataclass_fields__)


def cells_from_rows(name: str, rows: pd.DataFrame) -> list[tuple[Indicators, dict]]:
    param_names = list(GRIDS[name]["params"])
    cells = []
    for _, row in rows.iterrows():
        indicators = Indicators(
            **{
                field: type(getattr(Indicators(), field))(row[field])
                for field in _indicator_fields()
                if field in row.index
            }
        )
        cells.append((indicators, {key: row[key] for key in param_names}))
    return cells


def default_cell(name: str) -> tuple[Indicators, dict]:
    """La configurazione con cui il simulatore si apre: il termine di paragone di ogni ottimizzazione."""
    indicators = Indicators(
        rsi_window=int(page_config.RSI_SHORT.value),
        rsi_window2=int(page_config.RSI_MEDIUM.value),
        rsi_window3=int(page_config.RSI_LONG.value),
        ema_window=int(page_config.EMA_SHORT.value),
        ema_window2=int(page_config.EMA_MEDIUM.value),
        ema_window3=int(page_config.EMA_LONG.value),
        atr_window=int(page_config.ATR_WINDOW.value),
        atr_multiplier=float(page_config.ATR_MULTIPLIER.value),
        kama_pow1=int(page_config.KAMA_POW1.value),
        kama_pow2=int(page_config.KAMA_POW2.value),
    )
    defaults = {
        "rsi_buy_limit": int(page_config.RSI_BUY_LIMIT.value),
        "rsi_sell_limit": int(page_config.RSI_SELL_LIMIT.value),
        "num_cond": int(page_config.NUM_CONDITIONS.value),
        "stop_loss": float(page_config.STOP_LOSS_PERCENT.value),
        "atr_window": indicators.atr_window,
        "atr_multiplier": indicators.atr_multiplier,
    }
    params = {key: defaults[key] for key in GRIDS[name]["params"]}
    return indicators, params


def selection(
    interval: str = "15m", top: int = 3, min_trades: int = 30, exclude: tuple[str, ...] = ()
) -> dict[str, list[tuple[Indicators, dict]]]:
    """Le configurazioni da riesaminare: le migliori di ogni griglia piu' il default della pagina.

    `exclude` serve a `atr_live_trade`, che simula trenta sotto-passi per candela: cinque minuti a
    configurazione sulle barre a 15m diventano un'ora e mezza su quelle a 5m, e sarebbero venti
    ore solo per il confronto fra intervalli.
    """
    chosen = {}
    for name, frame in load_sweeps(interval, fold=False).items():
        if name in exclude:
            continue
        vive = operative(frame, min_trades)
        if vive.empty:
            continue
        best = vive.sort_values("rendimento_%", ascending=False).head(top)
        cells = cells_from_rows(name, best)
        default = default_cell(name)
        if default not in cells:
            cells.append(default)
        chosen[name] = cells
    return chosen


def across_intervals(chosen: dict, workers: int = 4, fee: float = 0.1) -> pd.DataFrame:
    rows = []
    for interval in INTERVALS:
        for name, cells in chosen.items():
            results, _ = run_cells(
                GRIDS[name]["strategy"], cells, interval=interval, fee=fee, workers=workers, label=f"{name}"
            )
            rows.append(results.assign(griglia=name))
    return pd.concat(rows, ignore_index=True)


def across_fees(chosen: dict, interval: str = "15m", workers: int = 4) -> pd.DataFrame:
    rows = []
    for fee in FEES:
        for name, cells in chosen.items():
            results, _ = run_cells(
                GRIDS[name]["strategy"], cells, interval=interval, fee=fee, workers=workers, label=f"{name}"
            )
            rows.append(results.assign(griglia=name))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", default="15m", help="intervallo su cui sono state scelte le configurazioni")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-intervals", action="store_true")
    parser.add_argument("--skip-fees", action="store_true")
    parser.add_argument("--exclude", nargs="*", default=["atr_live_trade"])
    args = parser.parse_args()

    chosen = selection(args.interval, args.top, exclude=tuple(args.exclude))
    print(f"{sum(len(cells) for cells in chosen.values())} configurazioni da {len(chosen)} griglie")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_fees:
        fees = across_fees(chosen, args.interval, args.workers)
        fees.to_csv(REPORT_DIR / "commissioni.csv", index=False)
    if not args.skip_intervals:
        intervals = across_intervals(chosen, args.workers)
        intervals.to_csv(REPORT_DIR / "intervalli.csv", index=False)


if __name__ == "__main__":
    main()
