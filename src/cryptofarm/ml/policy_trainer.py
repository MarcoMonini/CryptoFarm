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
    confirmed_reversal_rows,
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

    def __init__(
        self,
        symbol: str,
        market: pd.DataFrame,
        close: np.ndarray,
        signals: np.ndarray,
        exits: np.ndarray,
        exit_rows: np.ndarray,
        reversal_rows: np.ndarray,
    ):
        self.symbol = symbol
        self.market = market
        self.close = close
        self.signals = signals
        self.exits = exits  # timestamp di fine della gamba a cui la barra appartiene
        self.exit_rows = exit_rows  # e la sua riga, che serve per il prezzo
        self.reversal_rows = reversal_rows  # prima conferma di massimo, uscita causale

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
    exit_rows = np.arange(len(close))
    extremes = pivots["extreme_bar"].to_numpy()
    confirms = pivots["confirm_bar"].to_numpy()
    stamps = market.index.to_numpy()
    for position in range(len(pivots) - 1):
        first, last = int(confirms[position]), int(extremes[position + 1])
        if last >= first:
            exits[first : last + 1] = stamps[last]
            exit_rows[first : last + 1] = last

    return SymbolData(symbol, market, close, signals, exits, exit_rows, confirmed_reversal_rows(pivots, len(close)))


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
        return {
            "operazioni": 0,
            "trade_giorno": 0.0,
            "lordo_medio": 0.0,
            "netto_medio": 0.0,
            "win_rate": 0.0,
            "netto_giorno": 0.0,
        }
    return {
        "operazioni": int(len(trades)),
        "trade_giorno": len(trades) / days,
        # L'edge **lordo** e' il numero diagnostico: dice se il modello ha imparato qualcosa,
        # separatamente dal fatto che il costo se lo mangi. Il netto da solo confonde le due cose.
        "lordo_medio": float(trades["lordo"].mean()),
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
    run_holdout: bool = True,
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
    # Il dataset base resta separato da quello arricchito col DAgger: e' l'unico su cui si puo'
    # fare cross-validation senza portarsi dentro informazione della finestra di test.
    base = pd.concat([training_rows(data, rng, stride) for data in prepared], ignore_index=True)
    dataset = base
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

    cpcv_report, holdout_report = [], {}
    if run_cpcv:
        cpcv_report = _cpcv(panel, base, decision_threshold, cost)
    if run_holdout:
        holdout_report = _holdout(panel, base, dagger_rounds, stride, decision_threshold, cost)

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
        "holdout": holdout_report,
        "seconds": round(time.time() - started, 1),
    }
    (MODELS_DIR / f"{name}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nModello in {model_path}, rapporto in {MODELS_DIR / (name + '.json')}")
    return report


def embargo_width(prepared: list[SymbolData]) -> pd.Timedelta:
    """Embargo dimensionato sulla coda della durata delle gambe, non sulla mediana.

    `t_exit` qui e' il pivot successivo confermato, quindi l'orizzonte e' variabile: e' la coda a
    decidere quanto futuro condividono due blocchi contigui.
    """
    legs = np.concatenate([(data.exits - data.index.to_numpy()) for data in prepared])
    minutes = np.percentile(legs.astype("timedelta64[m]").astype(float), EMBARGO_PERCENTILE)
    return pd.Timedelta(minutes, unit="m")


def _test_windows(starts: np.ndarray, test_index: np.ndarray) -> list[tuple]:
    """Gli intervalli temporali coperti dai gruppi di test, uno per blocco contiguo.

    `CombinatorialPurgedCV` sceglie combinazioni di gruppi che in generale **non** sono adiacenti:
    nello split (0, 5) i blocchi di test sono il primo e l'ultimo, e in mezzo c'e' il training.
    Ridurre il test a un solo intervallo `[min, max]` significherebbe rimisurare la politica
    proprio sui dati su cui e' stata addestrata.

    I gruppi sono blocchi contigui nell'ordinamento per `t_start`, quindi si ricostruiscono dai
    salti fra le posizioni ordinate delle righe di test, senza riscrivere qui la logica dello
    splitter.
    """
    order = np.argsort(starts, kind="stable")
    rank = np.empty(len(starts), dtype=np.int64)
    rank[order] = np.arange(len(starts))
    ranks = np.sort(rank[test_index])
    if not len(ranks):
        return []
    breaks = np.flatnonzero(np.diff(ranks) != 1) + 1
    return [(starts[order[block[0]]], starts[order[block[-1]]]) for block in np.split(ranks, breaks)]


def _cpcv(panel: Panel, dataset: pd.DataFrame, decision_threshold, cost) -> list[dict]:
    """Valutazione out-of-sample della politica **base**, senza righe DAgger.

    Le righe DAgger sono deliberatamente escluse, e non per risparmiare tempo: sono prodotte
    facendo girare un modello addestrato su *tutti* i dati, quindi ognuna porta con se' un po' di
    informazione della finestra che poi fa da test. Metterle in un fold di training sarebbe
    leakage, e per giunta del tipo che non si vede in nessuna metrica.

    Il contributo del DAgger si misura invece in `_holdout`, dove le sue righe vengono raccolte
    dalla sola finestra di training.
    """
    prepared = panel.prepared
    embargo = embargo_width(prepared)
    print(f"\n=== CPCV della politica base (embargo {embargo}, p{EMBARGO_PERCENTILE} delle gambe) ===", flush=True)

    splitter = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=embargo)
    X, y, meta = _split_columns(dataset)
    t_start = pd.Series(meta["__start"].to_numpy())
    t_exit = pd.Series(meta["__exit"].to_numpy())
    values = X.to_numpy()

    rows = []
    for split, (train_index, test_index) in enumerate(splitter.split(t_start, t_exit), start=1):
        fold = build_model("gbdt")
        fit_model(fold, values[train_index], y[train_index])

        windows = _test_windows(t_start.to_numpy(), test_index)
        selected, days = [], 0.0
        for position, data in enumerate(prepared):
            index = data.index.to_numpy()
            spans = [(index >= start) & (index <= end) for start, end in windows]
            inside = np.flatnonzero(np.logical_or.reduce(spans))
            if len(inside) < 500:
                continue
            selected.append(inside + panel.offsets[position])
            # La durata e' la somma delle finestre, non la distanza fra la prima e l'ultima:
            # su gruppi non adiacenti quest'ultima includerebbe il training che sta in mezzo e
            # gonfierebbe il denominatore di `netto_giorno`.
            for span in spans:
                covered = np.flatnonzero(span)
                if len(covered):
                    days += (index[covered[-1]] - index[covered[0]]) / np.timedelta64(1, "s") / 86400
        trades = (
            backtest(panel, fold, decision_threshold, cost, rows=np.concatenate(selected))
            if selected
            else pd.DataFrame()
        )
        summary = summarise(trades, max(days, 1e-9))
        summary["split"] = split
        summary["finestre"] = ", ".join(f"{start:%Y-%m-%d}->{end:%Y-%m-%d}" for start, end in windows)
        rows.append(summary)
        print(f"  split {split:>2}: {_summary_line(summary)}", flush=True)

    frame = pd.DataFrame(rows)
    print(
        f"\nSu {len(frame)} split: netto/giorno mediano {frame['netto_giorno'].median():+.4%}, "
        f"quota di split in utile {(frame['netto_giorno'] > 0).mean():.0%}, "
        f"trade/giorno mediano {frame['trade_giorno'].median():.2f}"
    )
    return rows


def _holdout(
    panel: Panel,
    dataset: pd.DataFrame,
    dagger_rounds: int,
    stride: int,
    decision_threshold: float,
    cost: float,
    train_fraction: float = 0.75,
) -> dict:
    """Un solo taglio temporale, ma con l'intera procedura -- DAgger incluso -- dentro il training.

    E' la misura che dice se il DAgger serve davvero. Le sue righe si raccolgono facendo girare
    un modello che ha visto **solo** la finestra di training, e si tengono solo quelle datate
    prima del taglio: il rollout attraversa anche il futuro, ma quelle righe si buttano.
    """
    prepared = panel.prepared
    embargo = embargo_width(prepared)
    stamps = np.sort(dataset["__start"].unique())
    cut = pd.Timestamp(stamps[int(len(stamps) * train_fraction)])
    print(f"\n=== Holdout con DAgger (taglio {cut:%Y-%m-%d}, embargo {embargo}) ===", flush=True)

    def before_cut(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["__start"] < cut - embargo]

    training = before_cut(dataset)
    X, y, _ = _split_columns(training)
    model = build_model("gbdt")
    fit_model(model, X.to_numpy(), y)

    for round_number in range(1, dagger_rounds + 1):
        added, coverage = dagger_rows(panel, model, stride, decision_threshold)
        added = before_cut(added)
        training = pd.concat([training, added], ignore_index=True)
        X, y, _ = _split_columns(training)
        model = build_model("gbdt")
        fit_model(model, X.to_numpy(), y)
        print(
            f"  DAgger {round_number}: {len(added):,} righe dalla finestra di training | "
            f"disaccordo {coverage['disaccordo']:.2%}",
            flush=True,
        )

    selected, days = [], 0.0
    for position, data in enumerate(prepared):
        inside = np.flatnonzero(data.index >= cut + embargo)
        if len(inside) < 500:
            continue
        selected.append(inside + panel.offsets[position])
        days += (data.index[inside[-1]] - data.index[inside[0]]).total_seconds() / 86400
    rows = np.concatenate(selected)
    summary = summarise(backtest(panel, model, decision_threshold, cost, rows=rows), max(days, 1e-9))
    summary["taglio"] = cut
    print(f"  fuori campione: {_summary_line(summary)}")

    # Sweep della soglia: non richiede riaddestramento, solo un backtest per valore. Alzarla
    # significa operare solo con piu' convinzione, ed e' l'unica leva disponibile a modello fermo.
    print("  soglia di decisione:")
    sweep = []
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9):
        trades = backtest(panel, model, threshold, cost, rows=rows)
        entry = summarise(trades, max(days, 1e-9))
        entry["soglia"] = threshold
        sweep.append(entry)
        print(f"    {threshold:.1f}: {_summary_line(entry)}", flush=True)
    summary["sweep"] = sweep

    attribution = entry_attribution(panel, model, rows, cost, decision_threshold)
    if attribution.get("operazioni"):
        print(
            f"  attribuzione (stessi ingressi, uscite diverse):\n"
            f"    politica              {attribution['lordo_politica']:+.3%}\n"
            f"    conferma (causale)    {attribution['lordo_uscita_conferma']:+.3%}  "
            f"netto {attribution['netto_uscita_conferma']:+.3%}\n"
            f"    esperto (lookahead)   {attribution['lordo_uscita_esperto']:+.3%}  "
            f"netto {attribution['netto_uscita_esperto']:+.3%}\n"
            f"    perfetta (irraggiungibile) {attribution['lordo_uscita_perfetta']:+.3%}\n"
            f"  controllo, ingressi a caso: perfetta {attribution['lordo_ingresso_casuale']:+.3%} | "
            f"conferma {attribution['casuale_uscita_conferma']:+.3%}\n"
            f"  vantaggio dell'ingresso: {attribution['vantaggio_sul_caso']:+.3%} con uscita perfetta, "
            f"{attribution['vantaggio_causale']:+.3%} con uscita eseguibile"
        )
    summary["attribuzione"] = attribution
    return summary


def entry_attribution(panel: Panel, model, rows: np.ndarray, cost: float, decision_threshold: float) -> dict:
    """Separa la qualita' degli ingressi da quella delle uscite, con tre uscite a confronto.

    Un edge lordo prossimo a zero ha due cause diverse e opposte, e il numero aggregato non le
    distingue: o gli ingressi sono casuali, e allora nessuna uscita li salva; oppure gli ingressi
    valgono e l'uscita restituisce quello che avevano preso.

    Le uscite confrontate, tutte a partire **dagli ingressi della politica**:

    - *politica*: quando il modello dice SELL. E' il risultato reale.
    - *esperto*: alla prima barra etichettata SELL. E' il tetto **raggiungibile**, perche'
      l'etichetta e' costruita su pivot confermati.
    - *perfetta*: all'estremo della gamba. Non e' raggiungibile da nessuno -- l'estremo si conosce
      solo dopo -- e serve solo a dire quanto vale l'ingresso in assoluto.

    Il confronto che conta e' politica contro **esperto**: se anche l'esperto restituisce quello
    che l'ingresso aveva preso, il problema e' l'etichetta e non il modello.
    """
    trades = backtest(panel, model, decision_threshold, cost, rows=rows)
    if trades.empty:
        return {"operazioni": 0}

    generator = np.random.default_rng(11)
    perfect, expert, causal, chance, chance_causal = [], [], [], [], []
    lookup = {data.symbol: position for position, data in enumerate(panel.prepared)}
    for symbol, group in trades.groupby("symbol"):
        data = panel.prepared[lookup[symbol]]
        entry_rows = data.index.get_indexer(pd.DatetimeIndex(group["entrata"]))
        perfect.append(_exit_returns(data, entry_rows, data.exit_rows[entry_rows]))
        expert.append(_exit_returns(data, entry_rows, _first_sell_after(data.signals, entry_rows)))
        causal.append(_exit_returns(data, entry_rows, data.reversal_rows[entry_rows]))
        # Controllo: stessi ingressi in numero, ma presi a caso nella stessa finestra. Senza,
        # "+0,20% con uscita perfetta" non significa niente -- meta' delle barre sta in una gamba
        # al rialzo per costruzione, e uscire al suo estremo paga anche entrando a caso.
        drawn = generator.choice(np.arange(entry_rows.min(), entry_rows.max() + 1), size=len(entry_rows))
        chance.append(_exit_returns(data, drawn, data.exit_rows[drawn]))
        chance_causal.append(_exit_returns(data, drawn, data.reversal_rows[drawn]))
    perfect, expert, causal, chance, chance_causal = (
        np.concatenate(x) for x in (perfect, expert, causal, chance, chance_causal)
    )

    policy_gross = float(trades["lordo"].mean())
    return {
        "operazioni": int(len(trades)),
        "lordo_politica": policy_gross,
        "lordo_uscita_conferma": float(causal.mean()),
        "netto_uscita_conferma": float(causal.mean() - cost),
        "lordo_uscita_esperto": float(expert.mean()),
        "netto_uscita_esperto": float(expert.mean() - cost),
        "lordo_uscita_perfetta": float(perfect.mean()),
        "lordo_ingresso_casuale": float(chance.mean()),
        "casuale_uscita_conferma": float(chance_causal.mean()),
        "vantaggio_sul_caso": float(perfect.mean() - chance.mean()),
        # E' questo il numero sfruttabile: quanto vale l'ingresso del modello con una regola di
        # uscita che si puo' davvero eseguire, al netto di cosa renderebbe entrando a caso.
        "vantaggio_causale": float(causal.mean() - chance_causal.mean()),
    }


def _exit_returns(data: SymbolData, entry_rows: np.ndarray, exit_rows: np.ndarray) -> np.ndarray:
    return data.close[exit_rows] / data.close[entry_rows] - 1.0


def _first_sell_after(signals: np.ndarray, entry_rows: np.ndarray) -> np.ndarray:
    """Prima barra etichettata SELL a partire da ciascun ingresso; l'ultima barra se non arriva.

    E' l'uscita che l'esperto stesso puo' eseguire: l'etichetta SELL vive nella gamba discendente,
    quindi arriva **dopo** il massimo, in ritardo esattamente di quanto costa la conferma.
    """
    sells = np.flatnonzero(signals == SELL)
    if len(sells) == 0:
        return np.full(len(entry_rows), len(signals) - 1)
    position = np.searchsorted(sells, entry_rows, side="left")
    position = np.minimum(position, len(sells) - 1)
    return np.maximum(sells[position], entry_rows)


def _summary_line(summary: dict) -> str:
    return (
        f"{summary['operazioni']:>6,} operazioni | {summary['trade_giorno']:.2f}/giorno | "
        f"lordo {summary['lordo_medio']:+.3%} | netto {summary['netto_medio']:+.3%} | "
        f"win rate {summary['win_rate']:.1%} | netto/giorno {summary['netto_giorno']:+.4%}"
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
    parser.add_argument("--no-cpcv", action="store_true", help="salta la cross-validation, non l'holdout")
    parser.add_argument("--no-holdout", action="store_true")
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
        run_holdout=not args.no_holdout,
        name=args.name,
    )


if __name__ == "__main__":
    main()
