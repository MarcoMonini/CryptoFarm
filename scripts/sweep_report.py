"""Legge le tabelle di `strategy_sweep` e ne ricava le risposte, invece delle righe.

Uno sweep produce migliaia di configurazioni: guardarle ordinate per rendimento e' il modo piu'
rapido di scegliere il caso fortunato. Le viste qui sotto sono costruite per evitarlo.

- `panoramica` mette accanto **il migliore e la mediana** di ogni strategia. La distanza fra i due
  e' la misura di quanto il risultato dipenda dall'aver azzeccato i parametri.
- `sensibilita` guarda un parametro alla volta: se muovendolo la mediana non si sposta, quel
  widget non serve; se si ribalta, la strategia non e' utilizzabile senza sapere in anticipo il
  valore giusto.
- `stabilita` scompone il rendimento del migliore anno per anno. Una strategia che vive di un
  solo anno e' un accidente del campione.
- `fuori_campione` e' la sola misura che conta davvero: si sceglie il parametro migliore sui
  primi anni e si guarda cosa fa negli anni successivi, che il momento della scelta non ha visto.

    python -m scripts.sweep_report --interval 15m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cryptofarm.paths import PROJECT_ROOT
from scripts.strategy_sweep import GRIDS, OUTPUT_DIR, SYMBOL

REPORT_DIR = PROJECT_ROOT / "reports"
# La divisione in campione di stima e campione di verifica: cinque anni che contengono un ciclo
# completo (2017-2018) e uno di ripresa (2019-2021), poi tutto il resto.
IN_SAMPLE = (2017, 2021)
OUT_SAMPLE = (2022, 2026)

# Sotto una manciata di operazioni in nove anni non si sta misurando una strategia ma una singola
# posizione tenuta per anni: il rendimento e' quello del possesso passivo, con il fianco esposto a
# quale settimana e' capitata l'entrata. Le viste principali le tengono fuori e le contano a parte.
MIN_TRADES = 30

METRIC_COLUMNS = [
    "n_trade",
    "rendimento_%",
    "cagr_%",
    "sharpe",
    "max_drawdown_%",
    "win_rate_%",
    "profit_factor",
    "trade_medio_%",
    "esposizione_%",
    "trade_per_anno",
    "commissioni_%",
    "secondi",
]
KEY_COLUMNS = ["simbolo", "intervallo", "strategia"]


def _fold_triplet(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Le tre finestre EMA di alcune griglie si muovono insieme, non una alla volta.

    Trattarle come parametri indipendenti farebbe dire alle viste che nessuna delle tre sposta il
    risultato: fissate le altre due resta una riga sola, e una riga sola non ha escursione. Qui
    tornano a essere il parametro unico che sono, la terna.
    """
    if "ema_triplet" not in GRIDS.get(name, {}).get("indicators", {}):
        return frame
    columns = ["ema_window", "ema_window2", "ema_window3"]
    if not set(columns) <= set(frame.columns):
        return frame
    folded = frame.copy()
    folded["ema_triplet"] = ["/".join(str(int(value)) for value in row) for row in folded[columns].to_numpy()]
    return folded.drop(columns=columns)


def _stem(name: str, interval: str, symbol: str, suffix: str = "") -> str:
    """Ricostruisce il nome che `strategy_sweep.save` ha scritto, `--suffix` compreso.

    Il simbolo di riferimento non entra nel nome; il suffisso sì, sempre, ed e' il motivo per cui
    questo parametro esiste: senza, uno sweep lanciato con `--suffix` produceva file che questo
    modulo non trovava mai.
    """
    stem = f"{name}_{interval}" if symbol == SYMBOL else f"{name}_{interval}_{symbol}"
    return stem + suffix


def load_sweeps(
    interval: str = "15m",
    grids: list[str] | None = None,
    fold: bool = True,
    symbol: str = SYMBOL,
    suffix: str = "",
) -> dict[str, pd.DataFrame]:
    """`fold=False` restituisce le colonne come le ha scritte lo sweep: serve a chi deve
    ricostruire una configurazione per rieseguirla, non a chi la legge."""
    frames = {}
    for name in grids or GRIDS:
        path = OUTPUT_DIR / f"{_stem(name, interval, symbol, suffix)}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            frames[name] = _fold_triplet(name, frame) if fold else frame
    return frames


def load_yearly(
    interval: str = "15m", grids: list[str] | None = None, symbol: str = SYMBOL, suffix: str = ""
) -> dict[str, pd.DataFrame]:
    frames = {}
    for name in grids or GRIDS:
        path = OUTPUT_DIR / f"{_stem(name, interval, symbol, suffix)}_annuale.parquet"
        if path.exists():
            frames[name] = _fold_triplet(name, pd.read_parquet(path))
    return frames


def parameter_columns(frame: pd.DataFrame) -> list[str]:
    """Le colonne che nella tabella variano davvero: sono i parametri di quella griglia."""
    ignored = (
        set(METRIC_COLUMNS)
        | set(KEY_COLUMNS)
        | {"fee_%", "buy_hold_%", "buy_hold_drawdown_%", "volatilita_%", "trade_mediano_%", "durata_media_h"}
    )
    return [column for column in frame.columns if column not in ignored and frame[column].nunique() > 1]


def operative(frame: pd.DataFrame, min_trades: int = MIN_TRADES) -> pd.DataFrame:
    return frame[frame["n_trade"] >= min_trades]


def panoramica(sweeps: dict[str, pd.DataFrame], min_trades: int = MIN_TRADES) -> pd.DataFrame:
    rows = []
    for name, frame in sweeps.items():
        vive = operative(frame, min_trades)
        if vive.empty:
            vive = frame
        best = vive.loc[vive["rendimento_%"].idxmax()]
        rows.append(
            {
                "griglia": name,
                "strategia": frame["strategia"].iloc[0],
                "configurazioni": len(frame),
                f"con_meno_di_{min_trades}_trade": int((frame["n_trade"] < min_trades).sum()),
                "buy_hold_%": round(frame["buy_hold_%"].iloc[0], 1),
                "migliore_%": round(best["rendimento_%"], 1),
                "mediana_%": round(vive["rendimento_%"].median(), 1),
                "peggiore_%": round(vive["rendimento_%"].min(), 1),
                "in_utile_%": round((vive["rendimento_%"] > 0).mean() * 100, 1),
                "batte_bh_%": round((vive["rendimento_%"] > vive["buy_hold_%"]).mean() * 100, 1),
                "sharpe_migliore": round(vive["sharpe"].max(), 2),
                "trade_mediani_anno": int(vive["trade_per_anno"].median()),
                "dd_del_migliore_%": round(best["max_drawdown_%"], 1),
            }
        )
    return pd.DataFrame(rows).sort_values("migliore_%", ascending=False)


def sensibilita(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in sweeps.items():
        for column in parameter_columns(frame):
            for value, group in frame.groupby(column):
                rows.append(
                    {
                        "griglia": name,
                        "parametro": column,
                        "valore": value,
                        "configurazioni": len(group),
                        "mediana_%": round(group["rendimento_%"].median(), 1),
                        "migliore_%": round(group["rendimento_%"].max(), 1),
                        "in_utile_%": round((group["rendimento_%"] > 0).mean() * 100, 1),
                        "sharpe_mediano": round(group["sharpe"].median(), 2),
                        "trade_mediani_anno": int(group["trade_per_anno"].median()),
                    }
                )
    return pd.DataFrame(rows)


def escursione(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Quanto sposta il rendimento ogni singolo parametro, a parita' di tutti gli altri.

    E' la sensibilita' letta al contrario: invece della mediana per valore, la differenza fra il
    valore migliore e il peggiore. Un parametro che sposta poco e' un parametro che si puo'
    lasciare al default; uno che sposta molto e' quello su cui la strategia sta in piedi o cade.
    """
    rows = []
    for name, frame in sweeps.items():
        columns = parameter_columns(frame)
        for column in columns:
            others = [other for other in columns if other != column]
            if others:
                spread = frame.groupby(others)["rendimento_%"].agg(lambda values: values.max() - values.min())
                mediana = float(spread.median())
            else:
                mediana = float(frame["rendimento_%"].max() - frame["rendimento_%"].min())
            rows.append(
                {
                    "griglia": name,
                    "parametro": column,
                    "valori_provati": frame[column].nunique(),
                    "escursione_mediana_pp": round(mediana, 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["griglia", "escursione_mediana_pp"], ascending=[True, False])


def frequenza(sweeps: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tutte le configurazioni di tutte le griglie, raggruppate per quanto operano.

    E' la vista che mette in fila il risultato piu' ripetuto di questo lavoro: il numero di
    operazioni all'anno predice il rendimento meglio di qualunque parametro, e lo predice al
    contrario. Le commissioni sono la meta' della spiegazione; l'altra meta' e' che il margine
    medio per operazione, sui timeframe brevi, e' dello stesso ordine del costo di transazione.
    """
    tutte = pd.concat(sweeps.values(), ignore_index=True)
    fasce = pd.cut(
        tutte["trade_per_anno"],
        [0, 10, 30, 100, 300, 1000, float("inf")],
        labels=["<10", "10-30", "30-100", "100-300", "300-1000", ">1000"],
    )
    return (
        tutte.assign(fascia=fasce)
        .groupby("fascia", observed=True)
        .agg(
            configurazioni=("rendimento_%", "size"),
            mediana_rendimento=("rendimento_%", "median"),
            migliore=("rendimento_%", "max"),
            in_utile_pct=("rendimento_%", lambda values: (values > 0).mean() * 100),
            mediana_trade_medio=("trade_medio_%", "median"),
            mediana_commissioni=("commissioni_%", "median"),
            mediana_drawdown=("max_drawdown_%", "median"),
        )
        .round(2)
        .reset_index()
    )


def _config_key(frame: pd.DataFrame) -> list[str]:
    metric_like = set(METRIC_COLUMNS) | {"anno", "rendimento_%", "win_rate_%"}
    return [column for column in frame.columns if column not in metric_like]


def _compound(group: pd.DataFrame, first: int, last: int) -> float:
    window = group[(group["anno"] >= first) & (group["anno"] <= last)]
    return float(((1 + window["rendimento_%"] / 100).prod() - 1) * 100)


def fuori_campione(yearly: dict[str, pd.DataFrame], min_trades: int = MIN_TRADES) -> pd.DataFrame:
    """Sceglie i parametri sul periodo di stima e li misura su quello di verifica.

    Le tre righe che contano per ogni strategia: cosa avrebbe reso la configurazione scelta
    guardando solo i primi anni, cosa avrebbe reso la migliore scelta col senno di poi, e quanto
    rende la mediana di tutte -- che e' cio' che si ottiene scegliendo a caso.
    """
    rows = []
    for name, frame in yearly.items():
        keys = _config_key(frame)
        records = []
        for values, group in frame.groupby(keys, dropna=False):
            records.append(
                {
                    **dict(zip(keys, values if isinstance(values, tuple) else (values,))),
                    "stima_%": _compound(group, *IN_SAMPLE),
                    "verifica_%": _compound(group, *OUT_SAMPLE),
                    "trade": int(group["n_trade"].sum()),
                }
            )
        table = pd.DataFrame(records)
        table = table[table["trade"] >= min_trades]
        if table.empty:
            continue
        scelta = table.loc[table["stima_%"].idxmax()]
        senno = table.loc[table["verifica_%"].idxmax()]
        # La prima in classifica puo' essere fortunata: le prime dieci dicono se a trasferirsi e'
        # la regione di parametri o solo quella riga.
        prime = table.nlargest(min(10, len(table)), "stima_%")
        rows.append(
            {
                "griglia": name,
                "configurazioni": len(table),
                "scelta_su_stima_%": round(scelta["stima_%"], 1),
                "resa_in_verifica_%": round(scelta["verifica_%"], 1),
                "prime10_mediana_verifica_%": round(prime["verifica_%"].median(), 1),
                "prime10_in_utile_%": round((prime["verifica_%"] > 0).mean() * 100, 1),
                "migliore_in_verifica_%": round(senno["verifica_%"], 1),
                "mediana_in_verifica_%": round(table["verifica_%"].median(), 1),
                "percentile_della_scelta": round((table["verifica_%"] < scelta["verifica_%"]).mean() * 100, 1),
                "correlazione_stima_verifica": round(table["stima_%"].corr(table["verifica_%"], method="spearman"), 2),
                "parametri_scelti": ", ".join(
                    f"{key}={scelta[key]}" for key in _config_key(frame) if key not in KEY_COLUMNS
                ),
            }
        )
    return pd.DataFrame(rows)


def stabilita(
    sweeps: dict[str, pd.DataFrame], yearly: dict[str, pd.DataFrame], min_trades: int = MIN_TRADES
) -> pd.DataFrame:
    """Il rendimento annuo della configurazione migliore di ogni griglia, anno per anno."""
    rows = []
    for name, frame in sweeps.items():
        if name not in yearly or yearly[name].empty:
            continue
        vive = operative(frame, min_trades)
        best = (vive if not vive.empty else frame).loc[(vive if not vive.empty else frame)["rendimento_%"].idxmax()]
        table = yearly[name]
        keys = [key for key in _config_key(table) if key in best.index]
        mask = pd.Series(True, index=table.index)
        for key in keys:
            mask &= table[key] == best[key]
        for _, year in table[mask].sort_values("anno").iterrows():
            rows.append(
                {
                    "griglia": name,
                    "anno": int(year["anno"]),
                    "rendimento_%": round(year["rendimento_%"], 1),
                    "n_trade": int(year["n_trade"]),
                    "win_rate_%": round(year["win_rate_%"], 1),
                }
            )
    return pd.DataFrame(rows)


def _pivot(frame: pd.DataFrame, values: str) -> pd.DataFrame:
    keys = [key for key in _config_key(frame) if key not in KEY_COLUMNS]
    return frame.pivot_table(index=keys, columns="anno", values=values, aggfunc="sum").fillna(0)


def walk_forward(
    yearly: dict[str, pd.DataFrame], first_year: int = 2019, min_trades: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Riottimizza ogni anno sui soli anni gia' visti, e incassa quello che viene.

    E' la simulazione di come i parametri verrebbero scelti davvero: a fine anno si guarda cosa ha
    funzionato finora, si tiene quella configurazione per l'anno seguente e si ripete. A differenza
    di una divisione fissa in stima e verifica non c'e' una sola scelta fortunata che puo' reggere
    tutto il risultato, e a differenza del migliore assoluto non usa mai un dato che a quel momento
    non esisteva.

    Un anno in cui la configurazione scelta non apre operazioni vale zero, non e' un buco: e'
    esattamente cio' che succede a chi la sta seguendo.
    """
    riga, dettaglio = [], []
    for name, frame in yearly.items():
        returns = _pivot(frame, "rendimento_%") / 100
        trades = _pivot(frame, "n_trade")
        years = [year for year in sorted(returns.columns) if year >= first_year]
        capitale = 1.0
        for year in years:
            storico = [column for column in returns.columns if column < year]
            eleggibili = trades[storico].sum(axis=1) >= min_trades
            if not eleggibili.any():
                continue
            punteggio = (1 + returns.loc[eleggibili, storico]).prod(axis=1)
            scelta = punteggio.idxmax()
            resa = float(returns.loc[scelta, year])
            capitale *= 1 + resa
            dettaglio.append(
                {
                    "griglia": name,
                    "anno": int(year),
                    "rendimento_%": round(resa * 100, 1),
                    "n_trade": int(trades.loc[scelta, year]),
                    "capitale": round(capitale, 3),
                    "configurazione": ", ".join(
                        f"{key}={value}"
                        for key, value in zip(returns.index.names, scelta if isinstance(scelta, tuple) else (scelta,))
                    ),
                }
            )
        if dettaglio and dettaglio[-1]["griglia"] == name:
            anni = [row for row in dettaglio if row["griglia"] == name]
            riga.append(
                {
                    "griglia": name,
                    "anni": len(anni),
                    "rendimento_%": round((capitale - 1) * 100, 1),
                    "anni_in_utile_%": round(sum(1 for row in anni if row["rendimento_%"] > 0) / len(anni) * 100, 1),
                    "anno_peggiore_%": min(row["rendimento_%"] for row in anni),
                    "anno_migliore_%": max(row["rendimento_%"] for row in anni),
                    "cambi_di_configurazione": len({row["configurazione"] for row in anni}),
                }
            )
    return pd.DataFrame(riga), pd.DataFrame(dettaglio)


def riferimento(interval: str = "15m", symbol: str = SYMBOL) -> pd.DataFrame:
    """Il possesso passivo, anno per anno e sui due sotto-periodi: il metro di ogni riga sopra."""
    from scripts.strategy_sweep import load_interval

    close = load_interval(interval, symbol=symbol)["Close"]
    if close.empty:
        return pd.DataFrame()
    rows = []
    for label, first, last in [
        ("intero", 2017, 2026),
        (f"stima {IN_SAMPLE[0]}-{IN_SAMPLE[1]}", *IN_SAMPLE),
        (f"verifica {OUT_SAMPLE[0]}-{OUT_SAMPLE[1]}", *OUT_SAMPLE),
        ("walk-forward 2019-2026", 2019, 2026),
        *[(str(year), year, year) for year in range(2017, 2027)],
    ]:
        window = close[(close.index.year >= first) & (close.index.year <= last)]
        if window.empty:
            continue
        peak = window.cummax()
        rows.append(
            {
                "periodo": label,
                "buy_hold_%": round((window.iloc[-1] / window.iloc[0] - 1) * 100, 1),
                "max_drawdown_%": round(float((1 - window / peak).max() * 100), 1),
            }
        )
    return pd.DataFrame(rows)


def salva(tabelle: dict[str, pd.DataFrame], suffix: str = "") -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tabelle.items():
        frame.to_csv(REPORT_DIR / f"{name}{suffix}.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    sweeps = load_sweeps(args.interval, symbol=args.symbol, suffix=args.suffix)
    yearly = load_yearly(args.interval, symbol=args.symbol, suffix=args.suffix)
    if not sweeps:
        raise SystemExit(f"nessuno sweep in {OUTPUT_DIR} per {args.symbol} {args.interval}{args.suffix or ''}")
    etichetta = args.interval if args.symbol == SYMBOL else f"{args.interval}_{args.symbol}"

    tabelle = {
        f"riferimento_{etichetta}": riferimento(args.interval, args.symbol),
        f"panoramica_{etichetta}": panoramica(sweeps),
        f"frequenza_{etichetta}": frequenza(sweeps),
        f"sensibilita_{etichetta}": sensibilita(sweeps),
        f"escursione_{etichetta}": escursione(sweeps),
        f"stabilita_{etichetta}": stabilita(sweeps, yearly),
        f"fuori_campione_{etichetta}": fuori_campione(yearly),
    }
    avanti, avanti_dettaglio = walk_forward(yearly)
    tabelle[f"walk_forward_{etichetta}"] = avanti
    tabelle[f"walk_forward_dettaglio_{etichetta}"] = avanti_dettaglio
    pd.set_option("display.width", 220)
    for name, frame in tabelle.items():
        print(f"\n=== {name} ===")
        print(frame.to_string(index=False))
    if not args.no_save:
        salva(tabelle, args.suffix)
        print(f"\nTabelle in {Path(REPORT_DIR).relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
