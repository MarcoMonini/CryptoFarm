"""Le viste sulle strategie a due versi: cosa regge, cosa e' rumore, cosa aggiunge lo short.

`scripts/sweep_report.py` fa lo stesso lavoro per le strategie storiche. Qui cambiano tre cose,
perche' cambiano le domande:

- **`effetto_short`** confronta la stessa configurazione con e senza il verso corto. E' l'unico
  modo di attribuire una differenza al verso invece che alla strategia.
- **`ablazioni`** spegne un filtro alla volta (regime, ADX, conferma di volume) tenendo fermo il
  resto: e' la misura di quanto vale ogni pezzo aggiunto, e la risposta alla domanda "serviva
  davvero questo indicatore".
- **`trasferimento`** sceglie i parametri su un dataset e li misura su un altro -- ciclo diverso o
  asset diverso. Con cinque anni di dati non c'e' abbastanza storia per un walk-forward annuale
  che significhi qualcosa; il passaggio da un ciclo all'altro invece e' esattamente la domanda che
  conta ("i parametri del 2017-2020 valgono nel 2021-2026?").

    python -m scripts.lab_report --interval 4h
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from cryptofarm.trading import strategies_ls as ls
from cryptofarm.trading.indicators_extra import ExtraCache
from scripts.strategy_lab import GRIDS, OUTPUT_DIR
from scripts.sweep_report import REPORT_DIR

MIN_TRADES = 10  # sotto questa soglia, su cinque anni, non si misura una strategia
# I parametri che devono restare interi quando si rilegge una riga di risultati (il parquet li
# riporta tutti come float).
_INTERI = {
    "channel",
    "regime_ema",
    "adx_window",
    "atr_window",
    "bb_window",
    "kc_window",
    "kc_atr_window",
    "obv_window",
    "kama_window",
    "stochrsi_window",
    "stochrsi_smooth",
    "fast",
    "slow",
    "span",
}
# I periodi con cui sono stati generati i file, per poter rieseguire la stessa configurazione.
SINCE_BY_SUFFIX = {"": "2021-01-01", "_ciclo2017": "2017-01-01"}
UNTIL_BY_SUFFIX = {"_ciclo2017": "2021-01-01"}


def load(symbol: str = "BTCUSD", interval: str | None = None, suffix: str = "") -> pd.DataFrame:
    """Le tabelle di uno stesso taglio (simbolo, intervallo, periodo).

    Il filtro sui nomi e' esplicito perche' i file annuali stanno nella stessa cartella e hanno un
    nome che comincia allo stesso modo: pescarli insieme ai risultati raddoppierebbe le righe e
    farebbe sparire le colonne che i due formati non condividono.
    """
    frames = []
    intervals = [interval] if interval else ["5m", "15m", "30m", "1h", "4h", "1d"]
    for name in GRIDS:
        for candidate in intervals:
            path = OUTPUT_DIR / f"{name}_{symbol}_{candidate}{suffix}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parameter_columns(frame: pd.DataFrame, strategy: str) -> list[str]:
    axes = list(GRIDS[strategy]["params"])
    return [column for column in axes if column in frame.columns]


def panoramica(frame: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    rows = []
    for (strategy, interval), group in frame.groupby(["strategia", "intervallo"]):
        vive = group[group["n_trade"] >= min_trades]
        if vive.empty:
            continue
        best = vive.loc[vive["rendimento_%"].idxmax()]
        rows.append(
            {
                "strategia": strategy,
                "intervallo": interval,
                "configurazioni": len(vive),
                "buy_hold_%": round(group["buy_hold_%"].iloc[0], 1),
                "migliore_%": round(best["rendimento_%"], 1),
                "mediana_%": round(vive["rendimento_%"].median(), 1),
                "in_utile_%": round((vive["rendimento_%"] > 0).mean() * 100, 1),
                "batte_bh_%": round((vive["rendimento_%"] > vive["buy_hold_%"]).mean() * 100, 1),
                "sharpe_migliore": round(vive["sharpe"].max(), 2),
                "dd_del_migliore_%": round(best["max_drawdown_%"], 1),
                "trade_anno_mediani": round(vive["trade_per_anno"].median(), 1),
            }
        )
    return pd.DataFrame(rows).sort_values(["strategia", "intervallo"])


def effetto_short(frame: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    rows = []
    for strategy, group in frame.groupby("strategia"):
        keys = ["simbolo", "intervallo"] + [c for c in parameter_columns(group, strategy) if c != "allow_short"]
        wide = group.pivot_table(index=keys, columns="allow_short", values="rendimento_%").dropna()
        sharpe = group.pivot_table(index=keys, columns="allow_short", values="sharpe").dropna()
        if wide.empty or True not in wide.columns or False not in wide.columns:
            continue
        rows.append(
            {
                "strategia": strategy,
                "coppie": len(wide),
                "mediana_solo_long_%": round(wide[False].median(), 1),
                "mediana_con_short_%": round(wide[True].median(), 1),
                "short_migliora_%": round((wide[True] > wide[False]).mean() * 100, 1),
                "sharpe_solo_long": round(sharpe[False].median(), 2),
                "sharpe_con_short": round(sharpe[True].median(), 2),
                "contributo_short_mediano_%": round(
                    group.loc[group["allow_short"] & (group["n_trade"] >= min_trades), "rendimento_short_%"].median(), 1
                ),
                "win_rate_short_%": round(
                    group.loc[group["allow_short"] & (group["n_trade"] >= min_trades), "win_rate_short_%"].median(), 1
                ),
            }
        )
    return pd.DataFrame(rows)


# Le ablazioni: per ogni strategia, il filtro da spegnere e il valore che lo spegne.
ABLAZIONI = {
    "donchian_breakout": [("regime_ema", 0, "filtro di trend (EMA lunga)"), ("adx_min", 0.0, "filtro ADX")],
    "trend_pullback": [("regime_ema", 0, "filtro di trend (EMA lunga)")],
    "squeeze_breakout": [("confirm_volume", False, "conferma di volume (OBV)")],
    "band_reversion_gated": [("adx_max", 100.0, "filtro di range (ADX)"), ("regime_ema", 0, "filtro di trend")],
    "ichimoku_trend": [("require_cloud", False, "conferma della nuvola")],
}


def ablazioni(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy, group in frame.groupby("strategia"):
        for column, off_value, label in ABLAZIONI.get(strategy, []):
            if column not in group.columns:
                continue
            spento = group[group[column] == off_value]
            acceso = group[group[column] != off_value]
            if spento.empty or acceso.empty:
                continue
            rows.append(
                {
                    "strategia": strategy,
                    "filtro": label,
                    "senza_mediana_%": round(spento["rendimento_%"].median(), 1),
                    "con_mediana_%": round(acceso["rendimento_%"].median(), 1),
                    "senza_in_utile_%": round((spento["rendimento_%"] > 0).mean() * 100, 1),
                    "con_in_utile_%": round((acceso["rendimento_%"] > 0).mean() * 100, 1),
                    "senza_trade_anno": round(spento["trade_per_anno"].median(), 1),
                    "con_trade_anno": round(acceso["trade_per_anno"].median(), 1),
                }
            )
    return pd.DataFrame(rows)


def trasferimento(stima: pd.DataFrame, verifica: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    """Sceglie la configurazione migliore su un dataset e la ritrova nell'altro."""
    rows = []
    for strategy, group in stima.groupby("strategia"):
        target = verifica[verifica["strategia"] == strategy]
        if target.empty:
            continue
        keys = [c for c in parameter_columns(group, strategy)]
        vive = group[group["n_trade"] >= min_trades]
        if vive.empty:
            continue
        best = vive.loc[vive["rendimento_%"].idxmax()]
        mask = pd.Series(True, index=target.index)
        for key in keys:
            mask &= target[key] == best[key]
        scelta = target[mask]
        if scelta.empty:
            continue
        resa = float(scelta["rendimento_%"].iloc[0])
        rows.append(
            {
                "strategia": strategy,
                "resa_in_stima_%": round(best["rendimento_%"], 1),
                "resa_in_verifica_%": round(resa, 1),
                "migliore_in_verifica_%": round(target["rendimento_%"].max(), 1),
                "mediana_in_verifica_%": round(target["rendimento_%"].median(), 1),
                "percentile": round((target["rendimento_%"] < resa).mean() * 100, 1),
                "buy_hold_verifica_%": round(target["buy_hold_%"].iloc[0], 1),
                "parametri": ", ".join(f"{key}={best[key]}" for key in keys),
            }
        )
    return pd.DataFrame(rows)


def _finestra(yearly: pd.DataFrame, keys: list[str], first: tuple[int, int], second: tuple[int, int], min_trades: int):
    records = []
    for values, group in yearly.groupby(keys, dropna=False):

        def composto(start: int, end: int) -> tuple[float, int]:
            window = group[(group["anno"] >= start) & (group["anno"] <= end)]
            return float(((1 + window["rendimento_%"] / 100).prod() - 1) * 100), int(window["n_trade"].sum())

        stima, trade = composto(*first)
        verifica, _ = composto(*second)
        records.append({"stima": stima, "trade": trade, "verifica": verifica})
    table = pd.DataFrame(records)
    return table[table["trade"] >= min_trades]


def fuori_campione(
    symbol: str = "BTCUSD",
    interval: str = "1d",
    suffix: str = "",
    first: tuple[int, int] = (2021, 2023),
    second: tuple[int, int] = (2024, 2026),
    min_trades: int = 6,
    storiche_suffix: str | None = "_2021_fee005",
) -> pd.DataFrame:
    """Scelta sulla prima meta' del ciclo, resa sulla seconda, per entrambe le famiglie.

    Con cinque anni e mezzo di dati non c'e' spazio per un walk-forward annuale: la divisione in
    due meta' -- 2021-2023, che contiene il crollo del 2022, e 2024-2026, che contiene l'ultima
    salita e la distribuzione -- e' il massimo che il campione regge. Il confronto vale perche' le
    due famiglie sono misurate sulle stesse due finestre.
    """
    from scripts.strategy_sweep import GRIDS as GRIDS_STORICHE
    from scripts.strategy_sweep import OUTPUT_DIR as SWEEP_DIR

    rows = []
    if storiche_suffix is not None:
        for name, grid in GRIDS_STORICHE.items():
            path = SWEEP_DIR / f"{name}_{interval}{storiche_suffix}_annuale.parquet"
            if not path.exists():
                continue
            yearly = pd.read_parquet(path)
            keys = [key for key in list(grid["indicators"]) + list(grid["params"]) if key in yearly.columns]
            keys += [key for key in ("ema_window", "ema_window2", "ema_window3") if key in yearly.columns]
            table = _finestra(yearly, list(dict.fromkeys(keys)), first, second, min_trades)
            if len(table) < 3:
                continue
            best = table.loc[table["stima"].idxmax()]
            prime = table.nlargest(min(5, len(table)), "stima")
            rows.append(
                {
                    "famiglia": "storica",
                    "strategia": name,
                    "configurazioni": len(table),
                    "scelta_in_stima_%": round(best["stima"], 1),
                    "resa_in_verifica_%": round(best["verifica"], 1),
                    "prime5_mediana_%": round(prime["verifica"].median(), 1),
                    "mediana_verifica_%": round(table["verifica"].median(), 1),
                    "rho": round(table["stima"].corr(table["verifica"], method="spearman"), 2),
                }
            )

    for name, grid in GRIDS.items():
        path = OUTPUT_DIR / f"{name}_{symbol}_{interval}{suffix}_annuale.parquet"
        if not path.exists():
            continue
        yearly = pd.read_parquet(path)
        keys = [key for key in list(grid["params"]) + ["overbought", "slow", "span"] if key in yearly.columns]
        table = _finestra(yearly, keys, first, second, min_trades)
        if len(table) < 3:
            continue
        best = table.loc[table["stima"].idxmax()]
        prime = table.nlargest(min(5, len(table)), "stima")
        rows.append(
            {
                "famiglia": "nuova",
                "strategia": name,
                "configurazioni": len(table),
                "scelta_in_stima_%": round(best["stima"], 1),
                "resa_in_verifica_%": round(best["verifica"], 1),
                "prime5_mediana_%": round(prime["verifica"].median(), 1),
                "mediana_verifica_%": round(table["verifica"].median(), 1),
                "rho": round(table["stima"].corr(table["verifica"], method="spearman"), 2),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("resa_in_verifica_%", ascending=False) if not table.empty else table


def per_anno(symbol: str, interval: str, strategy: str, params: dict, suffix: str = "") -> pd.DataFrame:
    path = OUTPUT_DIR / f"{strategy}_{symbol}_{interval}{suffix}_annuale.parquet"
    frame = pd.read_parquet(path)
    mask = pd.Series(True, index=frame.index)
    for key, value in params.items():
        if key in frame.columns:
            mask &= frame[key] == value
    return frame[mask].sort_values("anno")[["anno", "n_trade", "rendimento_%", "win_rate_%"]]


def classifica(
    interval: str = "1d", suffix_storiche: str = "_2021_fee005", min_trades: int = MIN_TRADES
) -> pd.DataFrame:
    """Le migliori di ogni famiglia, storiche e nuove, sullo stesso periodo e con lo stesso costo.

    Le due tabelle nascono da due motori diversi (`simulate_trading_with_commisions` per le
    storiche, `simulate_positions` per le nuove) e questa e' l'unica vista che le mette in fila.
    Perche' il confronto sia leale servono tre allineamenti: stesso periodo, stessa commissione --
    da cui il suffisso `_fee005`, cioe' la rigirata delle griglie storiche allo 0,05% -- e la
    consapevolezza che le nuove pagano anche il mantenimento della posizione, che sullo spot non
    esiste.
    """
    from scripts.strategy_sweep import GRIDS as GRIDS_STORICHE
    from scripts.strategy_sweep import OUTPUT_DIR as SWEEP_DIR

    rows = []
    for name in GRIDS_STORICHE:
        path = SWEEP_DIR / f"{name}_{interval}{suffix_storiche}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        vive = frame[frame["n_trade"] >= min_trades]
        if vive.empty:
            continue
        best = vive.loc[vive["rendimento_%"].idxmax()]
        rows.append(
            {
                "famiglia": "storica",
                "strategia": best["strategia"],
                "rendimento_%": round(best["rendimento_%"], 1),
                "sharpe": round(best["sharpe"], 2),
                "max_drawdown_%": round(best["max_drawdown_%"], 1),
                "trade_per_anno": round(best["trade_per_anno"], 1),
                "mediana_griglia_%": round(vive["rendimento_%"].median(), 1),
                "in_utile_%": round((vive["rendimento_%"] > 0).mean() * 100, 1),
                "verso": "solo long",
            }
        )

    lab = load("BTCUSD", interval, "")
    for name, group in lab.groupby("strategia"):
        vive = group[group["n_trade"] >= min_trades]
        if vive.empty:
            continue
        best = vive.loc[vive["rendimento_%"].idxmax()]
        rows.append(
            {
                "famiglia": "nuova",
                "strategia": name,
                "rendimento_%": round(best["rendimento_%"], 1),
                "sharpe": round(best["sharpe"], 2),
                "max_drawdown_%": round(best["max_drawdown_%"], 1),
                "trade_per_anno": round(best["trade_per_anno"], 1),
                "mediana_griglia_%": round(vive["rendimento_%"].median(), 1),
                "in_utile_%": round((vive["rendimento_%"] > 0).mean() * 100, 1),
                "verso": "long+short" if bool(best["allow_short"]) else "solo long",
            }
        )

    table = pd.DataFrame(rows).sort_values("rendimento_%", ascending=False)
    return table


LEVE = (1.0, 2.0, 3.0)
COSTI = ((0.02, 0.01), (0.05, 0.03), (0.10, 0.06))


def leva_e_costi(symbol: str = "BTCUSD", interval: str = "1d", suffix: str = "") -> pd.DataFrame:
    """Le configurazioni migliori rieseguite a leva 1, 2 e 3 e a tre livelli di costo.

    Serve a rendere confrontabili strategie con drawdown molto diversi. Un sistema che rende meta'
    del possesso passivo con un quarto del suo drawdown non e' peggiore: e' lo stesso rendimento a
    parita' di rischio, una volta portato alla leva che pareggia il drawdown. La colonna del costo
    dice quanto di quel vantaggio sopravvive alle commissioni e al funding.
    """
    from cryptofarm.trading.pnl import simulate_positions
    from scripts.strategy_lab import evaluate, prepare

    frame = load(symbol, interval, suffix)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for name, group in frame.groupby("strategia"):
        vive = group[group["n_trade"] >= MIN_TRADES]
        if vive.empty:
            continue
        best = vive.loc[vive["rendimento_%"].idxmax()]
        strategy = ls.STRATEGIES[name]
        signature = strategy.__code__.co_varnames[2 : strategy.__code__.co_argcount]
        params = {}
        for key in signature:
            if key not in best.index or pd.isna(best[key]):
                continue
            value = best[key]
            params[key] = bool(value) if isinstance(value, (bool, np.bool_)) else value
            if key in _INTERI:
                params[key] = int(value)
            elif not isinstance(params[key], bool):
                params[key] = float(value)
        candles = prepare(symbol, interval, SINCE_BY_SUFFIX.get(suffix, "2021-01-01"), UNTIL_BY_SUFFIX.get(suffix))
        events = strategy(candles, ExtraCache(candles), **params)
        for leverage in LEVE:
            for fee, carry in COSTI:
                operations = simulate_positions(events, 100.0, fee, carry, leverage)
                metrics = evaluate(candles, operations, 100.0, fee)
                rows.append(
                    {
                        "strategia": name,
                        "intervallo": interval,
                        "leva": leverage,
                        "fee_%": fee,
                        "carry_%": carry,
                        "rendimento_%": round(metrics["rendimento_%"], 1),
                        "sharpe": round(metrics["sharpe"], 2),
                        "max_drawdown_%": round(metrics["max_drawdown_%"], 1),
                        "n_trade": metrics["n_trade"],
                        "buy_hold_%": round(metrics["buy_hold_%"], 1),
                        "buy_hold_drawdown_%": round(metrics["buy_hold_drawdown_%"], 1),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default="BTCUSD")
    parser.add_argument("--interval", default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--stima", default="2021-2023", help="anni su cui si sceglie, es. 2017-2018")
    parser.add_argument("--verifica", default="2024-2026", help="anni su cui si misura la scelta")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    frame = load(args.symbol, args.interval, args.suffix)
    if frame.empty:
        raise SystemExit(f"nessun risultato per {args.symbol} {args.interval or ''}{args.suffix}")

    tabelle = {
        "lab_panoramica": panoramica(frame),
        "lab_effetto_short": effetto_short(frame),
        "lab_ablazioni": ablazioni(frame),
    }
    if args.interval:
        # Le griglie storiche esistono solo per il simbolo di riferimento: su un altro mercato la
        # classifica confronterebbe due asset diversi, e il confronto non direbbe niente.
        storiche = args.symbol == "BTCUSD" and not args.suffix
        if storiche:
            tabelle["lab_classifica"] = classifica(args.interval)
        finestre = [tuple(int(year) for year in window.split("-")) for window in (args.stima, args.verifica)]
        tabelle["lab_fuori_campione"] = fuori_campione(
            args.symbol,
            args.interval,
            args.suffix,
            first=finestre[0],
            second=finestre[1],
            storiche_suffix="_2021_fee005" if storiche else None,
        )
        tabelle["lab_leva_costi"] = leva_e_costi(args.symbol, args.interval, args.suffix)
    pd.set_option("display.width", 240)
    tabelle = {name: table for name, table in tabelle.items() if not table.empty}
    for name, table in tabelle.items():
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))

    if not args.no_save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        etichetta = f"{args.symbol}{('_' + args.interval) if args.interval else ''}{args.suffix}"
        for name, table in tabelle.items():
            table.to_csv(REPORT_DIR / f"{name}_{etichetta}.csv", index=False)
        print(f"\nTabelle in reports/ con etichetta {etichetta}")


if __name__ == "__main__":
    main()
