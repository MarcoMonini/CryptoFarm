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

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, interval_to_minutes, load_klines
from cryptofarm.ml import evaluate
from cryptofarm.ml.dataset import DEFAULT_STRIDE, LAGS, build_design_matrix, build_samples, time_split
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

    take_profit = context["take_profit"].to_numpy()[validation_mask]
    stop_loss = context["stop_loss"].to_numpy()[validation_mask]
    sweep = evaluate.threshold_sweep(y_validation, probabilities, take_profit, stop_loss, ROUND_TRIP_FEE)

    print("\nAspettativa economica per soglia di decisione su P(buy):")
    print("(atteso_per_trade e' il rendimento medio per operazione al netto delle commissioni)")
    print(evaluate.format_sweep(sweep))

    chosen = evaluate.best_threshold(sweep)
    if chosen is None:
        print("\nNessuna soglia raggiunge abbastanza operazioni per essere valutata.")
        threshold = DEFAULT_DECISION_THRESHOLD
    else:
        threshold = float(chosen["soglia"])
        print(
            f"\nSoglia migliore: {threshold:.2f} -> {int(chosen['operazioni']):,} operazioni, "
            f"win rate {chosen['win_rate']:.2%} contro un break-even di {chosen['break_even']:.2%}, "
            f"atteso {chosen['atteso_per_trade']:+.3%} per operazione"
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

    La colonna `Prediction` usa la convenzione del simulatore: 0 = hold, 1 = buy, 2 = sell.
    """
    threshold = threshold if threshold is not None else _stored_threshold()
    minutes = _infer_interval_minutes(df.index)
    interval = f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"

    features = build_feature_frame(df, interval)
    matrix = build_design_matrix(features)
    usable = matrix.notna().all(axis=1)
    matrix = matrix[usable]
    if matrix.empty:
        result = df.copy()
        result["Prediction"] = 0
        return result

    probabilities = predict_proba(model, matrix.to_numpy())

    predictions = np.zeros(len(matrix), dtype=int)
    predictions[probabilities[:, BUY] >= threshold] = 1
    # Vendere quando il modello e' altrettanto convinto della direzione opposta.
    predictions[probabilities[:, 2] >= threshold] = 2

    result = df.loc[matrix.index].copy()
    result["Prediction"] = predictions
    return result


def _infer_interval_minutes(index: pd.DatetimeIndex) -> int:
    if len(index) < 2:
        return 15
    return int(round(np.median(np.diff(index.to_numpy()).astype("timedelta64[m]").astype(float))))


def _stored_threshold() -> float:
    """Soglia scelta durante l'addestramento, letta dai metadata del modello."""
    metadata_path = MODELS_DIR / f"{MODEL_NAME}.json"
    if metadata_path.exists():
        try:
            return float(json.loads(metadata_path.read_text())["decision_threshold"])
        except Exception:
            pass
    return DEFAULT_DECISION_THRESHOLD


def load_signal_model():
    """Carica il modello di segnale addestrato, qualunque formato abbia."""
    for extension in (".joblib", ".keras"):
        path = MODELS_DIR / f"{MODEL_NAME}{extension}"
        if path.exists():
            return load_model(path)
    raise FileNotFoundError(f"Nessun modello in {MODELS_DIR}. Addestrarne uno con `cryptofarm.ml.trainer`.")


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
