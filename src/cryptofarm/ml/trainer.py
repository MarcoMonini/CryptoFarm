"""Orchestrazione dell'addestramento: dallo store di candele al modello salvato.

Questo file non contiene logica propria. Le feature stanno in `features.py`, le etichette in
`labeling.py`, la costruzione della matrice in `dataset.py`, i modelli in `models.py`, le
metriche in `evaluate.py`. Qui c'e' solo l'ordine in cui vengono usati e la configurazione.

    python -m cryptofarm.ml.trainer                    # addestra con la configurazione di default
    python -m cryptofarm.ml.trainer --model gru        # con un modello sequenziale
    python -m cryptofarm.ml.trainer --symbols BTCUSDT --intervals 15m

Prerequisito: lo store di candele. Si popola con `python -m cryptofarm.data.klines --update`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, interval_to_minutes, load_klines
from cryptofarm.ml import evaluate
from cryptofarm.ml.dataset import DEFAULT_STRIDE, LAGS, build_samples, time_split
from cryptofarm.ml.features import build_feature_frame
from cryptofarm.ml.labeling import (
    BUY,
    FEE_FLOOR_MULTIPLE,
    HORIZON_BARS,
    ROUND_TRIP_FEE,
    SL_ATR_MULTIPLE,
    TP_ATR_MULTIPLE,
    barrier_widths,
    format_distribution,
    triple_barrier_labels,
)
from cryptofarm.ml.models import build_model, fit_model, load_model, predict_proba, save_model
from cryptofarm.ml.signals import buy_probabilities
from cryptofarm.paths import MODELS_DIR

INTERVALS = ["5m", "15m", "30m", "1h"]
TRAIN_FRACTION = 0.8
# Sotto questa probabilita' di "buy" il modello non opera. Il valore di default e' un punto di
# partenza: quello giusto lo stabilisce lo sweep sull'aspettativa a fine addestramento.
DEFAULT_DECISION_THRESHOLD = 0.5
MODEL_NAME = "signal_model"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_dataset(
    symbols: list[str],
    intervals: list[str],
    horizon: int = HORIZON_BARS,
    stride: int = DEFAULT_STRIDE,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Assembla la matrice di addestramento su tutte le coppie simbolo/timeframe.

    Restituisce anche un frame di contesto (simbolo, timeframe, barriere effettive) che non
    entra nel modello ma serve alla valutazione economica: l'aspettativa per operazione dipende
    dall'ampiezza delle barriere delle candele effettivamente selezionate.
    """
    matrices, labels, contexts = [], [], []

    for symbol in symbols:
        base = load_klines(symbol)
        if base.empty:
            print(f"[{symbol}] assente dallo store, saltato")
            continue

        for interval in intervals:
            candles = load_klines(symbol, interval)
            features = build_feature_frame(candles, interval)
            if len(features) < horizon * 4:
                continue

            label_series = triple_barrier_labels(features, horizon=horizon)
            matrix, selected_labels = build_samples(
                features,
                label_series,
                expected_minutes=interval_to_minutes(interval),
                horizon=horizon,
                stride=stride,
            )
            if matrix.empty:
                continue

            take_profit, stop_loss = barrier_widths(features["ATR"])
            positions = features.index.get_indexer(matrix.index)
            contexts.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "interval": interval,
                        "take_profit": take_profit[positions],
                        "stop_loss": stop_loss[positions],
                    },
                    index=matrix.index,
                )
            )
            matrices.append(matrix)
            labels.append(selected_labels)
            if verbose:
                print(
                    f"[{symbol} {interval}] {len(matrix)} righe "
                    f"({matrix.index[0]:%Y-%m-%d} .. {matrix.index[-1]:%Y-%m-%d})",
                    flush=True,
                )

    if not matrices:
        raise RuntimeError("Nessun dato utilizzabile: popolare lo store con `cryptofarm.data.klines --update`")

    return pd.concat(matrices), pd.concat(labels), pd.concat(contexts)


def train(
    symbols: list[str],
    intervals: list[str],
    model_kind: str = "gbdt",
    horizon: int = HORIZON_BARS,
    stride: int = DEFAULT_STRIDE,
) -> dict:
    started = time.time()
    print(f"Costruzione del dataset da {len(symbols)} simboli x {len(intervals)} timeframe\n")
    X, y, context = build_dataset(symbols, intervals, horizon, stride)

    print(f"\n{format_distribution(y.to_numpy(), 'dataset completo')}")
    print(f"Matrice: {X.shape[0]:,} righe x {X.shape[1]} feature = {X.memory_usage(deep=True).sum() / 1e6:.0f} MB")

    # L'embargo copre l'orizzonte dell'etichetta sul timeframe piu' lungo: e' li' che due righe a
    # cavallo del taglio condividono piu' futuro.
    longest = max(interval_to_minutes(interval) for interval in intervals)
    embargo = pd.Timedelta(minutes=longest * horizon)
    train_mask, validation_mask = time_split(X.index, TRAIN_FRACTION, embargo)
    print(
        f"\nSplit temporale globale con embargo di {embargo}: "
        f"train {train_mask.sum():,} righe, validation {validation_mask.sum():,} righe"
    )
    print(f"  train fino al   {X.index[train_mask].max():%Y-%m-%d}")
    print(f"  validation dal  {X.index[validation_mask].min():%Y-%m-%d}")
    print(format_distribution(y.to_numpy()[train_mask], "training"))
    print(format_distribution(y.to_numpy()[validation_mask], "validation"))

    # Una sola conversione: `to_numpy()` copia, e su milioni di righe ripeterla tre volte
    # significa tre gigabyte invece di uno.
    values = X.to_numpy()
    all_labels = y.to_numpy()
    X_train = values[train_mask]
    y_train = all_labels[train_mask]
    X_validation = values[validation_mask]
    y_validation = all_labels[validation_mask]
    del values

    print(f"\nAddestramento del modello '{model_kind}'...", flush=True)
    fit_started = time.time()
    model = build_model(model_kind, input_shape=X_train.shape[1:])
    fit_model(model, X_train, y_train)
    fit_seconds = time.time() - fit_started
    print(f"Addestrato in {fit_seconds:.1f}s")

    probabilities = predict_proba(model, X_validation)
    predictions = np.argmax(probabilities, axis=1)

    print("\n" + evaluate.classification_summary(y_validation, predictions))

    auc = evaluate.ranking_auc(y_validation, probabilities)
    print(f"AUC di P(buy): {auc:.4f}  (0,50 = nessun segnale; sopra 0,55 = segnale sfruttabile)")

    take_profit = context["take_profit"].to_numpy()[validation_mask]
    stop_loss = context["stop_loss"].to_numpy()[validation_mask]
    sweep = evaluate.quantile_sweep(y_validation, probabilities, take_profit, stop_loss, ROUND_TRIP_FEE)

    print("\nAspettativa per quota di candele selezionate (le migliori per P(buy)):")
    print("(atteso_lordo e' prima delle commissioni, atteso_per_trade dopo)")
    print(evaluate.format_sweep(sweep))

    # La quota migliore si sceglie sull'aspettativa **lorda**: e' quella che misura la capacita'
    # del modello. Quanto ne sopravvive dipende dai costi di esecuzione, che sono una decisione
    # separata e vengono riportati sotto.
    eligible = sweep[sweep["operazioni"] >= 200]
    chosen = None if eligible.empty else eligible.loc[eligible["atteso_lordo"].idxmax()].to_dict()
    if chosen is None:
        print("\nNessuna quota raggiunge abbastanza operazioni per essere valutata.")
        threshold = DEFAULT_DECISION_THRESHOLD
    else:
        threshold = float(chosen["soglia"])
        print(
            f"\nQuota migliore: il {chosen['quota']:.2%} di candele piu' promettenti "
            f"(P(buy) >= {threshold:.3f}) -> {int(chosen['operazioni']):,} operazioni, "
            f"win rate {chosen['win_rate']:.2%} contro un break-even di {chosen['break_even']:.2%}"
        )
        print(f"Barriera media {chosen['barriera']:.2%} | edge lordo {chosen['atteso_lordo']:+.3%} " f"per operazione")
        print("\nSensibilita' al costo di esecuzione:")
        print(
            evaluate.format_fee_sensitivity(evaluate.fee_sensitivity(chosen["atteso_lordo"], int(chosen["operazioni"])))
        )
    print(evaluate.signal_summary(probabilities, threshold))
    print(f"Lift sul base rate: {evaluate.lift_over_base_rate(y_validation, probabilities, threshold):.2f}x")

    # Risultato per timeframe: dice se addestrare un modello unico su tutti sta aiutando o
    # diluendo, che e' una domanda a cui una metrica aggregata non risponde.
    print("\nPer timeframe (alla soglia scelta):")
    intervals_validation = context["interval"].to_numpy()[validation_mask]
    for interval in intervals:
        mask = intervals_validation == interval
        if mask.sum() == 0:
            continue
        selected = mask & (probabilities[:, BUY] >= threshold)
        win_rate = (y_validation[selected] == BUY).mean() if selected.sum() else float("nan")
        print(
            f"  {interval:>4}: {int(selected.sum()):>7,} operazioni su {int(mask.sum()):>8,} candele, "
            f"win rate {win_rate:.2%}"
        )

    extension = ".joblib" if model_kind == "gbdt" else ".keras"
    model_path = MODELS_DIR / f"{MODEL_NAME}{extension}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, model_path)

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model_kind": model_kind,
        "model_path": model_path.name,
        "features": list(X.columns),
        "lags": list(LAGS),
        "labeling": {
            "method": "triple_barrier",
            "horizon_bars": horizon,
            "tp_atr_multiple": TP_ATR_MULTIPLE,
            "sl_atr_multiple": SL_ATR_MULTIPLE,
            "round_trip_fee": ROUND_TRIP_FEE,
            "fee_floor_multiple": FEE_FLOOR_MULTIPLE,
        },
        "data": {
            "symbols": symbols,
            "intervals": intervals,
            "stride": stride,
            "rows": int(len(X)),
            "first": str(X.index.min()),
            "last": str(X.index.max()),
            "train_rows": int(train_mask.sum()),
            "validation_rows": int(validation_mask.sum()),
        },
        "decision_threshold": threshold,
        "auc": round(auc, 4),
        "fit_seconds": round(fit_seconds, 1),
        "sweep": sweep.to_dict(orient="records"),
    }
    metadata_path = MODELS_DIR / f"{MODEL_NAME}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"\nModello salvato in {model_path}")
    print(f"Metadata in {metadata_path}")
    print(f"Totale: {(time.time() - started) / 60:.1f} minuti")
    return metadata


def get_model_predictions(df: pd.DataFrame, model, threshold: float | None = None) -> pd.DataFrame:
    """Applica un modello a un DataFrame di mercato: punto di ingresso usato dal simulatore.

    Ricostruisce le feature dai soli OHLCV con le costanti di questo pacchetto, invece di
    riusare le colonne che il chiamante ha gia' in tabella: `trading/simulator.py` le calcola
    con i periodi scelti dagli slider della dashboard, e un modello alimentato con feature
    calcolate diversamente da come e' stato addestrato sbaglia in silenzio.

    Aggiunge due colonne: `P_buy`, la probabilita' grezza, e `Prediction` con 1 sulle candele
    che superano la soglia e 0 altrove.

    **Non esiste una predizione "sell" per candela.** Il modello stima se un ingresso in quel
    punto raggiunge il take-profit prima dello stop-loss: la classe opposta significa "brutto
    momento per comprare" e copre circa il 60% delle candele, quindi emetterla come segnale di
    vendita ne produce a valanga. L'uscita da una posizione e' governata dalle barriere, in
    `signals.barrier_signals`.
    """
    threshold = threshold if threshold is not None else stored_decision_threshold()

    scores = buy_probabilities(df, model)
    result = df.copy()
    result["P_buy"] = scores.reindex(result.index)
    result["Prediction"] = (result["P_buy"] >= threshold).astype(int)
    return result


# Precedenza: modello delle gambe, poi meta-labeling, poi il classificatore di segnale
# originale. Il modello della strategia piu' recente e' quello che si vuole vedere sul grafico;
# per tornare al precedente basta spostarne l'artefatto altrove.
#
# **`policy_model` non e' piu' in catena.** Restava per primo, quindi bastava che l'artefatto
# esistesse in `models/` perche' la voce «AI Model» eseguisse la politica a tre azioni -- il
# disegno chiuso in negativo da `strategy.md` §12-13, con la causa misurata in §13.1 (entrare e
# uscire alla conferma cattura zero in media, su ogni simbolo e a ogni soglia, prima dei costi).
# L'artefatto non viene cancellato: se serve rivederlo, si rimette il nome in questa tupla.
# `leg_model` e' **fuori dalla catena di proposito**, pur avendo l'artefatto su disco. Il ciclo di
# dubbio del 2026-08-28 ha misurato che il suo netto medio per ingresso e' negativo a tutte e sei
# le soglie provate, e che il verdetto «PASSA» si accontentava di battere un p95 anch'esso
# negativo. Finche' non e' rivalidato non deve servire la voce «AI Model» della pagina: bastava
# spostare l'artefatto, ma cosi' la ragione resta scritta dove qualcuno la rilegge.
MODEL_PRECEDENCE = ("meta_model", MODEL_NAME)


def stored_decision_threshold() -> float:
    """Soglia scelta durante l'addestramento, letta dai metadata del modello."""
    for name in MODEL_PRECEDENCE:
        metadata_path = MODELS_DIR / f"{name}.json"
        if metadata_path.exists():
            try:
                return float(json.loads(metadata_path.read_text())["decision_threshold"])
            except Exception:
                continue
    return DEFAULT_DECISION_THRESHOLD


DEFAULT_EXIT_THRESHOLD = 0.90


def stored_exit_threshold() -> float:
    """Soglia d'**uscita**, che vale su `P(giu)` e non su `P(su)`.

    Non e' la stessa di `stored_decision_threshold`, e confonderle e' un difetto gia' capitato:
    le due teste hanno distribuzioni diverse, quindi un valore che sulla prima seleziona l'8%
    delle barre sulla seconda ne seleziona l'80%, e ogni posizione si chiude alla barra dopo
    averla aperta. Il default alto e' deliberato: senza una calibrazione l'uscita a modello deve
    quasi non scattare, perche' l'ablazione la misura dannosa (vedi `ml/signals.leg_signals`).
    """
    for name in MODEL_PRECEDENCE:
        metadata_path = MODELS_DIR / f"{name}.json"
        if metadata_path.exists():
            try:
                return float(json.loads(metadata_path.read_text())["exit_threshold"])
            except Exception:
                continue
    return DEFAULT_EXIT_THRESHOLD


def _meta_metadata() -> dict | None:
    """Metadata del modello meta, se ne esiste uno addestrato."""
    path = MODELS_DIR / "meta_model.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _model_path(name: str) -> Path | None:
    """L'artefatto addestrato per questa famiglia, in qualunque formato sia stato salvato."""
    return next((p for ext in (".joblib", ".keras") if (p := MODELS_DIR / f"{name}{ext}").exists()), None)


def active_model_name() -> str | None:
    """Quale famiglia di modello e' addestrata, e quindi quale strategia governa i segnali.

    Unica fonte di verita' per la precedenza: `load_signal_model` carica questo modello e
    `ai_model_simulation` sceglie in base a questo nome, quindi i due non possono divergere.
    """
    return next((name for name in MODEL_PRECEDENCE if _model_path(name)), None)


def meta_parameters() -> dict:
    """Parametri della catena di segnale, letti dai metadata del modello.

    Non sono duplicati qui come costanti di proposito: barriere, soglia CUSUM e parametri di
    esecuzione devono essere **esattamente** quelli con cui il modello e' stato addestrato, e
    l'unico modo per garantirlo e' leggerli dall'artefatto invece che riscriverli.
    """
    metadata = _meta_metadata() or {}
    labeling = metadata.get("labeling", {})
    execution = metadata.get("execution", {})
    primary = metadata.get("primary", {})
    return {
        "horizon_hours": labeling.get("horizon_hours", 24.0),
        "tp_multiple": labeling.get("tp_multiple", 1.5),
        "sl_multiple": labeling.get("sl_multiple", 1.0),
        "round_trip_fee": labeling.get("round_trip_fee", 0.0012),
        "fee_floor_multiple": labeling.get("fee_floor_multiple", 5.0),
        "cusum_sigma": primary.get("threshold_sigma", 3.0),
        "limit_offset_atr": execution.get("limit_offset_atr", 0.5),
        "limit_patience": execution.get("patience_bars", 12),
    }


def load_signal_model():
    """Carica il modello di segnale addestrato, qualunque formato abbia."""
    name = active_model_name()
    if name is None:
        raise FileNotFoundError(f"Nessun modello in {MODELS_DIR}. Addestrarne uno con `cryptofarm.ml.trainer`.")
    return load_model(_model_path(name))


def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra il modello di segnale.")
    parser.add_argument("--model", default="gbdt", help="gbdt (default), gru, cnn, lstm")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--horizon", type=int, default=HORIZON_BARS)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    args = parser.parse_args()

    train(
        symbols=args.symbols or DEFAULT_SYMBOLS,
        intervals=args.intervals or INTERVALS,
        model_kind=args.model,
        horizon=args.horizon,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
