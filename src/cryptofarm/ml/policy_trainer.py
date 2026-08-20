"""Addestramento della politica a tre azioni: dataset, DAgger, valutazione CPCV economica.

    python -m cryptofarm.ml.policy_trainer                       # configurazione di default
    python -m cryptofarm.ml.policy_trainer --symbols BTCUSDT ETHUSDT --dagger-rounds 0
    python -m cryptofarm.ml.policy_trainer --no-cpcv             # solo addestramento, senza CPCV

Due scelte che distinguono questo file da `trainer.py`, che resta quello della strategia
precedente (meta-labeling su triple-barrier) e non va toccato finche' il simulatore lo carica.

**La metrica primaria e' il rendimento di una traiettoria simulata, non la precision.** In una
politica sequenziale le due divergono sistematicamente: la precision si misura sugli stati
dell'esperto, il rendimento su quelli in cui la politica finisce da sola. Un modello puo' avere
ottima precision e perdere soldi, ed e' successo abbastanza spesso in questo progetto da non
volerlo scoprire di nuovo alla fine.

**L'etichetta ha orizzonte variabile**, perche' `t_exit` e' il pivot successivo confermato e non
un numero fisso di barre. L'embargo del CPCV va quindi dimensionato sulla **coda** di quella
distribuzione e non sulla mediana, o due fold a cavallo del taglio condividono futuro.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines
from cryptofarm.ml.dagger import DEFAULT_EPISODE_BARS, episode_bounds, rollout, state_coverage
from cryptofarm.ml.dataset import build_design_matrix
from cryptofarm.ml.directional_change import (
    BUY,
    HOLD,
    LABEL_NAMES,
    SELL,
    directional_change_pivots,
    soft_labels,
    tune_threshold,
)
from cryptofarm.ml.features import build_feature_frame
from cryptofarm.ml.models import build_model, fit_model, save_model
from cryptofarm.ml.policy import (
    FLAT,
    LONG,
    POSITION_FEATURES,
    expert_actions,
    position_features,
    randomised_states,
    simulate_expert_states,
)
from cryptofarm.ml.validation import CombinatorialPurgedCV
from cryptofarm.paths import MODELS_DIR

INTERVAL = "5m"
SINCE = "2022-01-01"
CAPTURE = 0.30
STRIDE = 6
DAGGER_ROUNDS = 2
# Estremi al giorno a cui si tara la soglia di ogni simbolo. La banda alta (8-12) e' il punto di
# lavoro di §10.3; quella bassa produce piu' occasioni con meno margine per trade, e quale delle
# due sopravviva a un modello imperfetto e' una domanda a cui risponde il CPCV, non la teoria.
EXTREMES_PER_DAY = (8.0, 12.0)
DECISION_THRESHOLD = 0.5
MAKER_ROUND_TRIP = 0.0008
MODEL_NAME = "policy_model"
# Percentile della durata delle gambe su cui si dimensiona l'embargo. Non la mediana: con
# orizzonte variabile e' la coda a decidere quanto futuro due fold contigui condividono.
EMBARGO_PERCENTILE = 95


class SymbolData:
    """Tutto cio' che serve di un simbolo, gia' allineato per posizione di barra."""

    def __init__(self, symbol: str, market: pd.DataFrame, close: np.ndarray, signals: np.ndarray, exits: np.ndarray):
        self.symbol = symbol
        self.market = market
        self.close = close
        self.signals = signals
        self.exits = exits  # timestamp di fine della gamba a cui la barra appartiene

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.market.index


def prepare_symbol(
    symbol: str,
    interval: str,
    since: str,
    capture: float,
    extremes_per_day: tuple[float, float] = EXTREMES_PER_DAY,
) -> SymbolData | None:
    """Feature, etichette e orizzonti di un simbolo, con la soglia tarata sul simbolo stesso."""
    candles = load_klines(symbol, interval)
    candles = candles[candles.index >= since]
    if len(candles) < 5000:
        return None

    features = build_feature_frame(candles, interval)
    market = build_design_matrix(features)
    usable = market.notna().all(axis=1).to_numpy()
    market = market[usable]
    if market.empty:
        return None

    aligned = features.loc[market.index]
    high, low, close = (aligned[column].to_numpy(float) for column in ("High", "Low", "Close"))
    days = (market.index[-1] - market.index[0]).total_seconds() / 86400
    threshold, _ = tune_threshold(high, low, days, target_per_day=extremes_per_day)
    pivots = directional_change_pivots(high, low, threshold)
    signals = soft_labels(close, pivots, capture)

    # Fine della gamba a cui ogni barra appartiene: e' la vita dell'etichetta, cio' su cui il
    # purging del CPCV deve ragionare. Le barre fuori da ogni gamba scadono con se' stesse.
    exits = market.index.to_numpy().copy()
    extremes = pivots["extreme_bar"].to_numpy()
    confirms = pivots["confirm_bar"].to_numpy()
    stamps = market.index.to_numpy()
    for position in range(len(pivots) - 1):
        first, last = int(confirms[position]), int(extremes[position + 1])
        if last >= first:
            exits[first : last + 1] = stamps[last]

    return SymbolData(symbol, market, close, signals, exits)


def training_rows(data: SymbolData, rng: np.random.Generator, stride: int) -> pd.DataFrame:
    """Righe di addestramento con stato di posizione campionato ed azione dell'esperto.

    Lo stato e' randomizzato, non dedotto: seguendo l'esperto il modello non vedrebbe mai una
    posizione aperta nel posto sbagliato, che e' l'unico stato in cui si trovera' quando avra'
    sbagliato (`policy.py`).
    """
    rows = np.arange(0, len(data.close), stride)
    rows = rows[rows > 0]
    state, entry, bars = randomised_states(data.close, rows, rng)

    block = position_features(data.close[rows], state, entry, bars)
    frame = data.market.iloc[rows].reset_index(drop=True)
    frame[list(POSITION_FEATURES)] = block.to_numpy()
    frame["__action"] = expert_actions(data.signals[rows], state)
    frame["__start"] = data.index[rows]
    frame["__exit"] = data.exits[rows]
    frame["__symbol"] = data.symbol
    return frame


class Panel:
    """I simboli concatenati in un unico blocco, con i confini che gli episodi non attraversano.

    Serve solo al rollout, ed e' l'unica ragione per cui il DAgger e' sostenibile: quindici serie
    fatte avanzare insieme costano quanto la piu' lunga, non quanto la loro somma.
    """

    def __init__(self, prepared: list[SymbolData], episode_bars: int = DEFAULT_EPISODE_BARS):
        self.prepared = prepared
        self.lengths = [len(data.close) for data in prepared]
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        self.market = np.vstack([data.market.to_numpy() for data in prepared])
        self.close = np.concatenate([data.close for data in prepared])
        self.signals = np.concatenate([data.signals for data in prepared])
        self.bounds = episode_bounds(self.lengths, episode_bars)

    def owner(self, rows: np.ndarray) -> np.ndarray:
        """Indice del simbolo a cui appartiene ciascuna riga globale."""
        return np.searchsorted(self.offsets, rows, side="right") - 1

    def local(self, rows: np.ndarray) -> np.ndarray:
        return rows - self.offsets[self.owner(rows)]


def dagger_rows(panel: Panel, model, stride: int, decision_threshold: float) -> tuple[pd.DataFrame, dict]:
    """Righe raccolte facendo girare la politica corrente su tutti i simboli, etichettate dall'esperto.

    La copertura degli stati si calcola sullo **stesso** rollout: e' la parte cara della procedura,
    e farne due -- per giunta a soglie diverse -- rendeva le due misure incoerenti oltre che lente.
    """
    full = rollout(model, panel.market, panel.close, panel.signals, panel.bounds, decision_threshold)
    coverage = state_coverage(full)
    visited = full.iloc[::stride]
    rows = visited["row"].to_numpy()

    block = position_features(
        panel.close[rows],
        visited["state"].to_numpy(),
        visited["entry_price"].to_numpy(),
        visited["bars_in_position"].to_numpy(),
    )
    frame = pd.DataFrame(panel.market[rows], columns=panel.prepared[0].market.columns)
    frame[list(POSITION_FEATURES)] = block.to_numpy()
    frame["__action"] = visited["expert"].to_numpy()

    owners, locals_ = panel.owner(rows), panel.local(rows)
    frame["__start"] = [panel.prepared[o].index[i] for o, i in zip(owners, locals_)]
    frame["__exit"] = [panel.prepared[o].exits[i] for o, i in zip(owners, locals_)]
    frame["__symbol"] = [panel.prepared[o].symbol for o in owners]
    return frame, coverage


def _split_columns(frame: pd.DataFrame):
    meta = [column for column in frame.columns if column.startswith("__")]
    return frame.drop(columns=meta), frame["__action"].to_numpy(), frame[meta]


def backtest(
    panel: Panel,
    model,
    decision_threshold: float,
    cost: float,
    rows: np.ndarray | None = None,
) -> pd.DataFrame:
    """Traiettoria simulata della politica: un'operazione per coppia ingresso/uscita, al netto del costo.

    `rows` limita la simulazione a una finestra (un blocco di test del CPCV). Le posizioni ancora
    aperte alla fine di un episodio **non** producono un'operazione: chiuderle al prezzo corrente
    regalerebbe alla politica un'uscita che non ha deciso lei.
    """
    if rows is None:
        market, close, signals, bounds = panel.market, panel.close, panel.signals, panel.bounds
        translate = np.arange(len(panel.close))
    else:
        rows = np.asarray(rows)
        market, close, signals = panel.market[rows], panel.close[rows], panel.signals[rows]
        # Gli episodi si ricostruiscono sui tratti contigui della selezione: una finestra di test
        # puo' spezzare un simbolo, e un episodio a cavallo del taglio non esiste.
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        lengths = np.diff(np.concatenate([[0], breaks, [len(rows)]]))
        bounds = episode_bounds(lengths)
        translate = rows

    visited = rollout(model, market, close, signals, bounds, decision_threshold)

    entries = visited[(visited["state"] == FLAT) & (visited["action"] == BUY)]
    exits = visited[(visited["state"] == LONG) & (visited["action"] == SELL)]
    if entries.empty or exits.empty:
        return pd.DataFrame(columns=["symbol", "entrata", "uscita", "barre", "lordo", "netto"])

    # Dentro un episodio ingressi e uscite si alternano per costruzione (il mascheramento lo
    # garantisce), quindi accoppiarli in ordine dentro ciascun episodio e' esatto.
    trades = []
    for episode, opened in entries.groupby("episode", sort=False):
        closed = exits[exits["episode"] == episode]
        pairs = min(len(opened), len(closed))
        if pairs == 0:
            continue
        open_rows = opened["row"].to_numpy()[:pairs]
        close_rows = closed["row"].to_numpy()[:pairs]
        global_open = translate[open_rows]
        owners = panel.owner(global_open)
        locals_open, locals_close = panel.local(global_open), panel.local(translate[close_rows])
        gross = close[close_rows] / close[open_rows] - 1.0
        for position in range(pairs):
            owner = panel.prepared[owners[position]]
            trades.append(
                {
                    "symbol": owner.symbol,
                    "entrata": owner.index[locals_open[position]],
                    "uscita": owner.index[locals_close[position]],
                    "barre": int(close_rows[position] - open_rows[position]),
                    "lordo": float(gross[position]),
                    "netto": float(gross[position] - cost),
                }
            )
    return pd.DataFrame(trades)


def summarise(trades: pd.DataFrame, days: float) -> dict[str, float]:
    if trades.empty:
        return {"operazioni": 0, "trade_giorno": 0.0, "netto_medio": 0.0, "win_rate": 0.0, "netto_giorno": 0.0}
    return {
        "operazioni": int(len(trades)),
        "trade_giorno": len(trades) / days,
        "netto_medio": float(trades["netto"].mean()),
        "win_rate": float((trades["netto"] > 0).mean()),
        "netto_giorno": float(trades["netto"].sum() / days),
        "barre_mediane": float(trades["barre"].median()),
    }


def train(
    symbols: list[str],
    interval: str = INTERVAL,
    since: str = SINCE,
    capture: float = CAPTURE,
    stride: int = STRIDE,
    extremes_per_day: tuple[float, float] = EXTREMES_PER_DAY,
    dagger_rounds: int = DAGGER_ROUNDS,
    decision_threshold: float = DECISION_THRESHOLD,
    cost: float = MAKER_ROUND_TRIP,
    run_cpcv: bool = True,
    seed: int = 7,
    name: str = MODEL_NAME,
) -> dict:
    started = time.time()
    rng = np.random.default_rng(seed)

    print(f"Preparazione di {len(symbols)} simboli su {interval} da {since}\n", flush=True)
    prepared: list[SymbolData] = []
    for symbol in symbols:
        data = prepare_symbol(symbol, interval, since, capture, extremes_per_day)
        if data is None:
            print(f"[{symbol}] dati insufficienti, saltato")
            continue
        prepared.append(data)
        distribution = {name: float((data.signals == code).mean()) for code, name in LABEL_NAMES.items()}
        print(
            f"[{symbol}] {len(data.market):,} barre | "
            f"hold {distribution['hold']:.1%} buy {distribution['buy']:.1%} sell {distribution['sell']:.1%}",
            flush=True,
        )
    if not prepared:
        raise RuntimeError("Nessun simbolo utilizzabile: popolare lo store con `cryptofarm.data.klines --update`")

    panel = Panel(prepared)
    dataset = pd.concat([training_rows(data, rng, stride) for data in prepared], ignore_index=True)
    X, y, meta = _split_columns(dataset)
    print(f"\nMatrice: {X.shape[0]:,} righe x {X.shape[1]} feature")
    print(_distribution_line("esperto, stato randomizzato", y))
    print(_distribution_line("esperto, stato dedotto", _expert_only_actions(prepared)))

    model = build_model("gbdt")
    fit_model(model, X.to_numpy(), y)
    print(f"\nAddestrato in {time.time() - started:.1f}s", flush=True)

    dagger_report = []
    for round_number in range(1, dagger_rounds + 1):
        print(f"\n--- DAgger, iterazione {round_number} ---", flush=True)
        added, coverage = dagger_rows(panel, model, stride, decision_threshold)
        dataset = pd.concat([dataset, added], ignore_index=True)
        X, y, meta = _split_columns(dataset)

        print(
            f"{len(added):,} righe aggiunte | disaccordo {coverage['disaccordo']:.2%} | "
            f"ingressi mancati {coverage['ingressi_mancati']:.2%} | uscite mancate {coverage['uscite_mancate']:.2%}"
        )
        dagger_report.append({"iterazione": round_number, "righe_aggiunte": int(len(added)), **coverage})

        model = build_model("gbdt")
        fit_model(model, X.to_numpy(), y)

    print("\n=== Backtest in-sample della politica (tutto il periodo) ===", flush=True)
    all_trades = backtest(panel, model, decision_threshold, cost)
    days = sum((data.index[-1] - data.index[0]).total_seconds() / 86400 for data in prepared)
    in_sample = summarise(all_trades, days)
    print(_summary_line(in_sample))

    cpcv_report = []
    if run_cpcv:
        cpcv_report = _cpcv(panel, meta, decision_threshold, cost, stride, rng)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{name}.joblib"
    save_model(model, model_path)
    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "features": list(X.columns),
        "labeling": {
            "method": "directional_change",
            "capture": capture,
            "interval": interval,
            "since": since,
            "extremes_per_day": list(extremes_per_day),
        },
        "symbols": [data.symbol for data in prepared],
        "rows": int(len(X)),
        "decision_threshold": decision_threshold,
        "round_trip_cost": cost,
        "dagger": dagger_report,
        "in_sample": in_sample,
        "cpcv": cpcv_report,
        "seconds": round(time.time() - started, 1),
    }
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nModello in {model_path}, rapporto in {MODELS_DIR / (name + '.json')}")
    return report


def _cpcv(panel: Panel, meta, decision_threshold, cost, stride, rng) -> list[dict]:
    """Valutazione out-of-sample: si riaddestra su ogni split e si simula sul blocco di test.

    Il modello addestrato prima **non** viene valutato qui: sarebbe in-sample. Ogni split
    riaddestra da zero sulle sole righe di training ammesse dopo purging ed embargo.
    """
    prepared = panel.prepared
    legs = np.concatenate([(data.exits - data.index.to_numpy()) for data in prepared])
    embargo = pd.Timedelta(np.percentile(legs.astype("timedelta64[m]").astype(float), EMBARGO_PERCENTILE), unit="m")
    print(f"\n=== CPCV (embargo {embargo}, p{EMBARGO_PERCENTILE} della durata delle gambe) ===", flush=True)

    splitter = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=embargo)
    t_start = pd.Series(meta["__start"].to_numpy())
    t_exit = pd.Series(meta["__exit"].to_numpy())
    dataset = pd.concat([training_rows(data, rng, stride) for data in prepared], ignore_index=True)
    X, y, _ = _split_columns(dataset)
    values = X.to_numpy()

    rows = []
    for split, (train_index, test_index) in enumerate(splitter.split(t_start, t_exit), start=1):
        fold = build_model("gbdt")
        fit_model(fold, values[train_index], y[train_index])

        window = (t_start.iloc[test_index].min(), t_start.iloc[test_index].max())
        selected, days = [], 0.0
        for position, data in enumerate(prepared):
            inside = np.flatnonzero((data.index >= window[0]) & (data.index <= window[1]))
            if len(inside) < 500:
                continue
            selected.append(inside + panel.offsets[position])
            days += (data.index[inside[-1]] - data.index[inside[0]]).total_seconds() / 86400
        trades = (
            backtest(panel, fold, decision_threshold, cost, rows=np.concatenate(selected))
            if selected
            else pd.DataFrame()
        )
        summary = summarise(trades, max(days, 1e-9))
        summary["split"] = split
        summary["da"] = window[0]
        summary["a"] = window[1]
        rows.append(summary)
        print(f"  split {split:>2}: {_summary_line(summary)}", flush=True)

    frame = pd.DataFrame(rows)
    print(
        f"\nSu {len(frame)} split: netto/giorno mediano {frame['netto_giorno'].median():+.4%}, "
        f"quota di split in utile {(frame['netto_giorno'] > 0).mean():.0%}, "
        f"trade/giorno mediano {frame['trade_giorno'].median():.2f}"
    )
    return rows


def _summary_line(summary: dict) -> str:
    return (
        f"{summary['operazioni']:>6,} operazioni | {summary['trade_giorno']:.2f}/giorno | "
        f"netto medio {summary['netto_medio']:+.3%} | win rate {summary['win_rate']:.1%} | "
        f"netto/giorno {summary['netto_giorno']:+.4%}"
    )


def _distribution_line(name: str, actions: np.ndarray) -> str:
    counts = {label: float((actions == code).mean()) for code, label in ((HOLD, "hold"), (BUY, "buy"), (SELL, "sell"))}
    return f"  {name:<32} hold {counts['hold']:.1%}  buy {counts['buy']:.1%}  sell {counts['sell']:.1%}"


def _expert_only_actions(prepared: list[SymbolData]) -> np.ndarray:
    """Le azioni che si otterrebbero **senza** randomizzare, per il confronto stampato."""
    return np.concatenate([expert_actions(data.signals, simulate_expert_states(data.signals)) for data in prepared])


def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra la politica a tre azioni.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default=INTERVAL)
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--capture", type=float, default=CAPTURE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument(
        "--extremes-per-day",
        nargs=2,
        type=float,
        default=list(EXTREMES_PER_DAY),
        metavar=("MIN", "MAX"),
        help="banda a cui tarare la soglia di ogni simbolo (default: 8 12)",
    )
    parser.add_argument("--dagger-rounds", type=int, default=DAGGER_ROUNDS)
    parser.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--no-cpcv", action="store_true")
    parser.add_argument("--name", default=MODEL_NAME, help="nome dei file di modello e rapporto")
    args = parser.parse_args()

    train(
        args.symbols,
        interval=args.interval,
        since=args.since,
        capture=args.capture,
        stride=args.stride,
        extremes_per_day=tuple(args.extremes_per_day),
        dagger_rounds=args.dagger_rounds,
        decision_threshold=args.threshold,
        run_cpcv=not args.no_cpcv,
        name=args.name,
    )


if __name__ == "__main__":
    main()
