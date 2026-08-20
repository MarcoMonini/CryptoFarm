"""Addestramento del secondario di meta-labeling, validato in CPCV.

Differenze rispetto all'approccio precedente, tutte conseguenza delle misure in `strategy.md`:

- **Campionamento a eventi (CUSUM)** invece che a orologio: si valuta solo quando il prezzo ha
  accumulato un movimento rilevante, ~30 volte al giorno per simbolo.
- **Etichette gia' nette**: incorporano commissioni per lato corretto e la possibilita' che
  l'ordine limite non si riempia. La precision del modello *e'* il win rate netto.
- **CPCV con purging ed embargo** invece di uno split singolo: produce una distribuzione di
  performance, non un punto, ed e' cio' che rende calcolabile il PBO.
- **Pesi di unicita'**: le osservazioni le cui vite si sovrappongono portano meno informazione, e
  pesarle uguali gonfia la fiducia in cio' che si e' misurato.
- **Il criterio di selezione e' l'aspettativa netta**, non una metrica statistica.

    python -m cryptofarm.ml.meta_trainer
    python -m cryptofarm.ml.meta_trainer --symbols BTCUSDT ETHUSDT --intervals 5m
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
from cryptofarm.ml.dataset import build_design_matrix, cusum_events
from cryptofarm.ml.features import build_feature_frame
from cryptofarm.ml.meta import build_meta_labels, expectancy_by_quantile
from cryptofarm.ml.models import build_model, fit_model, predict_proba, save_model
from cryptofarm.ml.validation import (
    CombinatorialPurgedCV,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    sample_uniqueness,
)
from cryptofarm.paths import MODELS_DIR

# --- Campionamento -------------------------------------------------------------------------
INTERVALS = ["5m", "15m"]
CUSUM_SIGMA = 3.0  # ~30 eventi/giorno per simbolo, misurato su tutti e 15

# --- Barriere ------------------------------------------------------------------------------
HORIZON_HOURS = 24  # in tempo, non in barre: i timeframe devono restare confrontabili
TP_MULTIPLE = 1.5
SL_MULTIPLE = 1.0
FEE_FLOOR_MULTIPLE = 5.0  # pavimento allo 0,6% sullo stop, 0,9% sul target
ROUND_TRIP_FEE = 0.0012  # maker in ingresso + taker in uscita

# --- Esecuzione ----------------------------------------------------------------------------
# Offset e pazienza scelti misurando l'aspettativa non condizionata: un limite piu' profondo
# migliora il prezzo di ingresso piu' di quanto costi in occasioni mancate, fino a ~0,5 ATR.
LIMIT_OFFSET_ATR = 0.5
LIMIT_PATIENCE_BARS = 12

# --- Validazione ---------------------------------------------------------------------------
CV_GROUPS = 8
CV_TEST_GROUPS = 2  # C(8,2) = 28 split
# Quote di eventi su cui operare. 0,13 corrisponde al target di 4 trade/giorno/simbolo dati
# ~30 eventi al giorno. Ognuna e' una configurazione, e va contata nel Deflated Sharpe Ratio.
QUANTILES = (0.05, 0.10, 0.13, 0.20, 0.30)
MODEL_NAME = "meta_model"


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build_meta_dataset(symbols: list[str], intervals: list[str], verbose: bool = True):
    """Matrice, etichette nette e vite delle osservazioni, su tutte le coppie simbolo/timeframe."""
    matrices, labels, contexts = [], [], []

    for symbol in symbols:
        for interval in intervals:
            candles = load_klines(symbol, interval)
            if candles.empty:
                continue
            features = build_feature_frame(candles, interval)
            horizon_bars = int(HORIZON_HOURS * 60 / interval_to_minutes(interval))
            if len(features) < horizon_bars * 4:
                continue

            events = cusum_events(features["Close"], CUSUM_SIGMA)
            # Il primario non puo' segnalare dove la matrice di progetto non e' calcolabile,
            # ne' cosi' vicino alla fine da non avere futuro osservabile.
            matrix = build_design_matrix(features)
            usable = matrix.notna().all(axis=1).to_numpy()
            usable[-horizon_bars:] = False
            events = events[usable[events]]
            if len(events) == 0:
                continue

            meta = build_meta_labels(
                features,
                events,
                horizon_bars=horizon_bars,
                tp_multiple=TP_MULTIPLE,
                sl_multiple=SL_MULTIPLE,
                round_trip_fee=ROUND_TRIP_FEE,
                fee_floor_multiple=FEE_FLOOR_MULTIPLE,
                offset_atr=LIMIT_OFFSET_ATR,
                patience=LIMIT_PATIENCE_BARS,
            )
            meta["symbol"] = symbol
            meta["interval"] = interval
            matrices.append(matrix.iloc[events])
            labels.append(meta)
            contexts.append(len(events))
            if verbose:
                days = (features.index[-1] - features.index[0]).total_seconds() / 86400
                print(
                    f"[{symbol} {interval}] {len(events):,} eventi ({len(events)/days:.1f}/gg), "
                    f"riempiti {meta['traded'].mean():.0%}, profittevoli {meta['meta_label'].mean():.1%}",
                    flush=True,
                )

    if not matrices:
        raise RuntimeError("Nessun dato: popolare lo store con `cryptofarm.data.klines --update`")
    return pd.concat(matrices), pd.concat(labels, ignore_index=True)


def train(
    symbols: list[str],
    intervals: list[str],
    model_kind: str = "gbdt",
    permutation_splits: int = 10,
) -> dict:
    started = time.time()
    print(f"Costruzione del dataset da {len(symbols)} simboli x {len(intervals)} timeframe\n")
    X, meta = build_meta_dataset(symbols, intervals)

    values = X.to_numpy()
    y = meta["meta_label"].to_numpy()
    outcome = meta["outcome"].to_numpy()
    net = meta["net_return"].to_numpy()
    filled = meta["traded"].to_numpy()
    t_start = pd.Series(pd.to_datetime(meta["t_start"].to_numpy()))
    t_exit = pd.Series(pd.to_datetime(meta["t_exit"].to_numpy()))

    print(f"\n{len(X):,} eventi x {X.shape[1]} feature")
    print(f"Riempimenti: {filled.mean():.1%} | profittevoli fra i riempiti: {y[filled].mean():.1%}")
    print(f"Aspettativa non condizionata: {net.mean():+.4%} per evento, {net[filled].mean():+.4%} per trade")

    uniqueness = sample_uniqueness(t_start, t_exit)
    print(
        f"Unicita' media dei campioni: {uniqueness.mean():.3f} "
        f"(le osservazioni si sovrappongono, quindi pesano meno di una ciascuna)"
    )

    # Peso per attribuzione di rendimento: un trade che vale +2% e uno che vale +0,05% non sono
    # la stessa osservazione, ma per un classificatore binario lo sono. Pesare per |rendimento|
    # allinea cio' che il modello ottimizza (l'errore di classificazione) a cio' che conta
    # davvero (l'aspettativa). Senza, il modello impara a indovinare il *segno* piu' spesso
    # possibile, che non e' la stessa cosa che guadagnare.
    magnitude = np.abs(net)
    attribution = uniqueness * magnitude
    positive = attribution > 0
    attribution = np.where(positive, attribution / attribution[positive].mean(), 0.0)

    embargo = pd.Timedelta(hours=HORIZON_HOURS)
    cv = CombinatorialPurgedCV(CV_GROUPS, CV_TEST_GROUPS, embargo=embargo)
    print(f"\nCPCV: {cv.get_n_splits()} split, {CV_GROUPS} gruppi, embargo {embargo}\n")

    in_sample = np.zeros((cv.get_n_splits(), len(QUANTILES)))
    out_sample = np.zeros_like(in_sample)
    per_split_returns: list[np.ndarray] = []
    selected_fill: list[float] = []
    selected_outcomes: list[np.ndarray] = []
    permutation_expectancy: list[float] = []
    rng = np.random.default_rng(42)

    for position, (train_idx, test_idx) in enumerate(cv.split(t_start, t_exit)):
        # Si addestra solo sugli eventi in cui un trade e' davvero avvenuto. Un ordine non
        # riempito ha rendimento nullo, che e' **migliore** di un trade in perdita: etichettarlo
        # come "non profittevole" lo accomuna ai perdenti e insegna al modello a evitare gli
        # ingressi che non si riempiono, il che non ha nulla a che vedere con la loro qualita'.
        fit_idx = train_idx[filled[train_idx]]
        if len(fit_idx) < 1000 or len(test_idx) < 200:
            in_sample[position] = np.nan
            out_sample[position] = np.nan
            continue
        model = build_model(model_kind)
        fit_model(model, values[fit_idx], y[fit_idx], sample_weight=attribution[fit_idx])

        train_scores = predict_proba(model, values[train_idx])[:, 1]
        test_scores = predict_proba(model, values[test_idx])[:, 1]
        in_frame = expectancy_by_quantile(train_scores, net[train_idx], QUANTILES)
        out_frame = expectancy_by_quantile(test_scores, net[test_idx], QUANTILES)
        in_sample[position] = in_frame["atteso_netto"].to_numpy()
        out_sample[position] = out_frame["atteso_netto"].to_numpy()

        target = QUANTILES.index(0.13)
        order = np.argsort(-test_scores)[: max(1, int(len(test_scores) * QUANTILES[target]))]
        selected = test_idx[order]
        per_split_returns.append(net[selected])
        # Controllo di sanita' decisivo: un ordine non riempito rende zero invece che negativo,
        # quindi un modello che imparasse a selezionare eventi *che non si riempiono* mostrerebbe
        # un'aspettativa per evento eccellente senza mai fare un trade. Se il fill rate fra gli
        # eventi scelti crolla sotto quello generale, il guadagno e' un artefatto.
        selected_fill.append(float(filled[selected].mean()))
        selected_outcomes.append(outcome[selected])

        # Controllo di permutazione: con le etichette di training mescolate il modello non puo'
        # imparare nulla, quindi la sua selezione deve valere quanto una a caso. Se restasse
        # redditizia, il guadagno verrebbe da una via che aggira le etichette -- cioe' da un
        # leakage -- e non dalla capacita' predittiva.
        if position < permutation_splits:
            shuffled = rng.permutation(y[fit_idx])
            control = build_model(model_kind)
            fit_model(control, values[fit_idx], shuffled, sample_weight=attribution[fit_idx])
            control_scores = predict_proba(control, values[test_idx])[:, 1]
            control_order = np.argsort(-control_scores)[: len(order)]
            permutation_expectancy.append(float(net[test_idx][control_order].mean()))
        print(
            f"  split {position + 1:>2}/{cv.get_n_splits()}: train {len(train_idx):>7,} "
            f"test {len(test_idx):>6,} | atteso out-of-sample al 13% "
            f"{out_frame['atteso_netto'].iloc[target]:+.4%}",
            flush=True,
        )

    valid = ~np.isnan(out_sample).any(axis=1)
    out_valid = out_sample[valid]
    print(f"\n{valid.sum()} split validi su {cv.get_n_splits()}")

    print("\nDistribuzione dell'aspettativa netta out-of-sample per quota di eventi:")
    print(f"{'quota':>7} {'mediana':>10} {'media':>10} {'quota di split positivi':>24}")
    print("-" * 56)
    summary = []
    for column, quantile in enumerate(QUANTILES):
        values_q = out_valid[:, column]
        summary.append(
            {
                "quota": quantile,
                "mediana": float(np.median(values_q)),
                "media": float(values_q.mean()),
                "split_positivi": float((values_q > 0).mean()),
            }
        )
        print(
            f"{quantile:>6.0%} {np.median(values_q):>+9.4%} {values_q.mean():>+9.4%} " f"{(values_q > 0).mean():>23.0%}"
        )

    pbo = probability_of_backtest_overfitting(in_sample[valid], out_valid)
    print(
        f"\nPBO (probabilita' di backtest overfitting): {pbo:.2f}  "
        f"({'ACCETTABILE' if pbo < 0.5 else 'LA SELEZIONE FA PEGGIO DEL CASO'})"
    )

    print(
        f"\nFill rate fra gli eventi selezionati: {np.mean(selected_fill):.1%} "
        f"(generale {filled.mean():.1%}) -- se fosse molto piu' basso, il guadagno sarebbe "
        f"il non aver operato, non l'aver operato bene"
    )

    mix = np.concatenate(selected_outcomes)
    print("\nEsiti dei trade selezionati (0 timeout, 1 take-profit, 2 stop-loss):")
    for code, name in ((1, "take-profit"), (2, "stop-loss"), (0, "timeout")):
        share = float((mix == code).mean())
        print(f"  {name:<12} {share:>6.1%}")

    if permutation_expectancy:
        control = float(np.mean(permutation_expectancy))
        real = float(np.median(out_valid[:, QUANTILES.index(0.13)]))
        print(
            f"\nTest di permutazione su {len(permutation_expectancy)} split: "
            f"con etichette mescolate {control:+.4%} contro {real:+.4%} del modello reale"
        )
        adjusted = real - control
        print(
            f"  **Edge corretto per permutazione: {adjusted:+.4%}** -- e' il numero da usare. "
            f"Il controllo cattura la parte di guadagno che viene dal selezionare comunque un "
            f"sottoinsieme (eventi ad alta volatilita' si riempiono meno spesso, e un ordine non "
            f"riempito rende zero invece che negativo), e va sottratta."
        )
        if control > real * 0.5:
            print("  ATTENZIONE: il controllo guadagna piu' di meta' del modello -> sospetto leakage")

    pooled = np.concatenate(per_split_returns) if per_split_returns else np.array([])
    dsr = deflated_sharpe_ratio(pooled, trials=len(QUANTILES) * 4) if len(pooled) else float("nan")
    print(f"Deflated Sharpe Ratio (su {len(QUANTILES) * 4} configurazioni contate): {dsr:.3f}")

    best = max(summary, key=lambda row: row["mediana"])
    print(
        f"\nQuota migliore per mediana out-of-sample: {best['quota']:.0%} "
        f"-> {best['mediana']:+.4%} per operazione, positiva su {best['split_positivi']:.0%} degli split"
    )

    # Modello finale su tutti i dati, con la stessa pesatura per unicita'.
    print("\nAddestramento del modello finale su tutto il dataset...", flush=True)
    final = build_model(model_kind)
    fit_model(final, values[filled], y[filled], sample_weight=attribution[filled])
    scores = predict_proba(final, values)[:, 1]
    threshold = float(np.quantile(scores, 1 - best["quota"]))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    save_model(final, model_path)

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "kind": "meta_label_secondary",
        "model_kind": model_kind,
        "features": list(X.columns),
        "primary": {"filter": "cusum", "threshold_sigma": CUSUM_SIGMA},
        "labeling": {
            "method": "triple_barrier_net_of_costs",
            "horizon_hours": HORIZON_HOURS,
            "tp_multiple": TP_MULTIPLE,
            "sl_multiple": SL_MULTIPLE,
            "fee_floor_multiple": FEE_FLOOR_MULTIPLE,
            "round_trip_fee": ROUND_TRIP_FEE,
        },
        "execution": {
            "limit_offset_atr": LIMIT_OFFSET_ATR,
            "patience_bars": LIMIT_PATIENCE_BARS,
            "entry_mode": "maker",
            "exit_mode": "taker",
            "fill_rate": float(meta["traded"].mean()),
            "fill_rate_selected": float(np.mean(selected_fill)),
        },
        "data": {
            "symbols": symbols,
            "intervals": intervals,
            "events": int(len(X)),
            "first": str(t_start.min()),
            "last": str(t_exit.max()),
            "mean_uniqueness": float(uniqueness.mean()),
        },
        "validation": {
            "scheme": "combinatorial_purged_cv",
            "groups": CV_GROUPS,
            "test_groups": CV_TEST_GROUPS,
            "splits": int(valid.sum()),
            "embargo_hours": HORIZON_HOURS,
            "pbo": None if np.isnan(pbo) else round(float(pbo), 4),
            "deflated_sharpe": None if np.isnan(dsr) else round(float(dsr), 4),
            "quantile_summary": summary,
            "permutation_control": (float(np.mean(permutation_expectancy)) if permutation_expectancy else None),
        },
        "decision_quantile": best["quota"],
        "decision_threshold": threshold,
        "expected_net_per_trade": best["mediana"],
        "expected_net_permutation_adjusted": (
            best["mediana"] - float(np.mean(permutation_expectancy)) if permutation_expectancy else None
        ),
    }
    (MODELS_DIR / f"{MODEL_NAME}.json").write_text(json.dumps(metadata, indent=2, default=str))

    print(f"\nModello salvato in {model_path}")
    print(f"Soglia di decisione: {threshold:.4f} (quota {best['quota']:.0%})")
    print(f"Totale: {(time.time() - started) / 60:.1f} minuti")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Addestra il secondario di meta-labeling.")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--intervals", nargs="+", default=None)
    parser.add_argument("--model", default="gbdt")
    args = parser.parse_args()
    train(args.symbols or DEFAULT_SYMBOLS, args.intervals or INTERVALS, args.model)


if __name__ == "__main__":
    main()
