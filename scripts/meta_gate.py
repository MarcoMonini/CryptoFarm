"""Meta-etichettatura sopra una strategia primaria vera, messa in comune su tutto l'universo.

La pipeline ML del progetto e' chiusa in negativo (`.claude/docs/strategy.md` §10-13), ma la
formulazione che ha fallito era una sola: **prevedere il mercato barra per barra su un asset**, con
etichette da directional change che si e' misurato costare due soglie e valerne meno di due.

Questa e' l'altra formulazione, quella che `strategy.md` §2.3 raccomandava e che non e' mai stata
provata cosi': il modello **non decide quando comprare**. Lo decide la strategia a indicatori. Il
modello decide soltanto *se lasciar passare* il segnale che quella ha gia' prodotto. Tre differenze
che cambiano il problema, non i dettagli:

- **il vincolo economico e' dentro il target.** L'etichetta e' "questa operazione, come la
  strategia la eseguirebbe davvero, chiude sopra i costi", non "il prezzo sale". Non c'e' modo di
  avere ragione sul segno e perdere soldi;
- **il campione e' per operazione, non per barra.** Sono migliaia di righe invece di milioni, ed e'
  un bene: le barre di un trend sono quasi tutte la stessa osservazione ripetuta;
- **gli asset si mettono in comune.** Il modello vede tutte le operazioni di tutti i simboli, con
  feature scale-free piu' il contesto trasversale (forza relativa, ampiezza del mercato). E' la
  ragione per cui `qlib` misura IC su trecento titoli invece che su uno: un vantaggio da IC 0,03
  non e' sfruttabile su un asset alla volta, lo diventa su una sezione larga.

Validazione con `PurgedKFold` ed embargo (`ml/validation.py`): le operazioni si sovrappongono nel
tempo, quindi il k-fold ordinario qui misura sul futuro gia' visto.

Il verdetto non e' l'AUC. E' se il portafoglio filtrato batte quello non filtrato **fuori
campione**, sulle stesse operazioni e con gli stessi costi.

    python -m scripts.meta_gate --strategy trend_pullback --interval 4h
    python -m scripts.meta_gate --strategy trend_pullback --interval 4h --oos 2024-01-01
    python -m scripts.meta_gate --selfcheck
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from cryptofarm.data.klines import load_klines
from cryptofarm.ml.bar_features import ASSET_COLUMNS, CROSS_COLUMNS, asset_features, cross_features
from cryptofarm.ml.validation import PurgedKFold
from cryptofarm.paths import PROJECT_ROOT
from cryptofarm.trading import strategies_ls as ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions
from cryptofarm.trading.rotation import MAJORS, WIDE

SINCE = "2021-01-01"
WALLET = 100.0
FEE_PERCENT = 0.05
CARRY_PERCENT = 0.0  # a pronti e solo lunghi non c'e' funding da pagare
OUTPUT_DIR = PROJECT_ROOT / "reports"

# I parametri della primaria non si ottimizzano qui: si prende una configurazione centrale e si
# misura cosa aggiunge il filtro. Ottimizzare primaria e filtro insieme e' il modo classico di
# leggere il rumore due volte.
PRIMARY_PARAMS = {
    "trend_pullback": {
        "regime_ema": 200,
        "stochrsi_window": 14,
        "oversold": 0.2,
        "overbought": 0.8,
        "atr_window": 14,
        "atr_multiplier": 2.0,
        "allow_short": False,
    },
    "donchian_breakout": {
        "channel": 20,
        "regime_ema": 200,
        "adx_window": 14,
        "adx_min": 20,
        "atr_window": 14,
        "atr_multiplier": 3.0,
        "allow_short": False,
    },
    "band_reversion_gated": {"allow_short": False},
    "squeeze_breakout": {"allow_short": False},
    "ichimoku_trend": {"allow_short": False},
}


# ---------------------------------------------------------------------------------------------
# Feature: tutte scale-free, tutte note alla barra di ingresso
# ---------------------------------------------------------------------------------------------


# `features_frame` e `cross_features` **vivono nel pacchetto**, non qui: le usano anche
# `ml/bar_features.py` e il modello nuovo, e due copie della stessa costruzione sono il modo in cui
# addestramento e inferenza divergono senza dare segno. Qui restano i nomi con cui questo script e
# `scripts/ai_voter.py` le hanno sempre chiamate.
features_frame = asset_features


# ---------------------------------------------------------------------------------------------
# Costruzione del campione
# ---------------------------------------------------------------------------------------------


def build_samples(
    symbols: list[str],
    strategy: str,
    interval: str,
    since: str,
    until: str | None,
    fee: float,
) -> pd.DataFrame:
    """Una riga per operazione della primaria, con le feature della barra di ingresso.

    L'etichetta e' `Profit > 0` dell'operazione **gia' al netto di commissioni**: il vincolo
    economico e' dentro il target invece di essere scoperto dopo.
    """
    closes = {}
    per_symbol = {}
    for symbol in symbols:
        candles = load_klines(symbol, interval)
        candles = candles[candles.index >= since]
        if until:
            candles = candles[candles.index < until]
        if len(candles) < 300:
            continue
        per_symbol[symbol] = candles
        closes[symbol] = candles["Close"]
    closes = pd.DataFrame(closes).sort_index()
    cross = cross_features(closes)

    rows = []
    for symbol, candles in per_symbol.items():
        cache = ExtraCache(candles)
        events = ls.STRATEGIES[strategy](candles, cache, **PRIMARY_PARAMS[strategy])
        operations = simulate_positions(events, WALLET, fee, CARRY_PERCENT)
        if not operations:
            continue
        frame = features_frame(candles, cache)
        frame["rango_forza"] = cross["rango_forza"][symbol].reindex(frame.index)
        frame["ampiezza_mercato"] = cross["ampiezza_mercato"].reindex(frame.index)
        frame["forza_su_btc"] = cross["forza_su_btc"][symbol].reindex(frame.index)

        for operation in operations:
            entry = operation["Buy_Time"]
            if entry not in frame.index:
                continue
            # Il rendimento per unita' di capitale, non il profitto assoluto: `simulate_positions`
            # compone il portafoglio, quindi il profitto della centesima operazione non e'
            # confrontabile con quello della prima.
            gross = (operation["Sell_Price"] / operation["Buy_Price"] - 1.0) * 100
            net = gross - fee * (1 + operation["Sell_Price"] / operation["Buy_Price"])
            rows.append(
                {
                    "simbolo": symbol,
                    "t_start": entry,
                    "t_exit": operation["Sell_Time"],
                    "netto_%": net,
                    "y": int(net > 0),
                    **frame.loc[entry].to_dict(),
                }
            )

    samples = pd.DataFrame(rows).sort_values("t_start").reset_index(drop=True)
    return samples.dropna(subset=["netto_%"])


# Le 16 di questo script: struttura dell'asset piu' contesto trasversale. Il modello nuovo ne usa
# tre in piu' (posizionamento e timeframe) e per questo tiene il proprio elenco in `bar_features`.
FEATURE_COLUMNS = [*ASSET_COLUMNS, *CROSS_COLUMNS]


# ---------------------------------------------------------------------------------------------
# Valutazione
# ---------------------------------------------------------------------------------------------


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC per ranghi, senza dipendere da sklearn.metrics per una riga."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    positives, negatives = y.sum(), (1 - y).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def sequential_equity(net_percent: np.ndarray) -> float:
    """Composizione sequenziale dei rendimenti per operazione.

    **Non e' un portafoglio.** Le operazioni di quindici simboli si sovrappongono nel tempo, e
    questo conto le tratta come scommesse consecutive sull'intero capitale. Serve solo a rendere
    visibile il peso delle code -- una manciata di perdite grosse la schiaccia anche quando la
    mediana e' positiva, ed e' un'informazione che il netto medio nasconde. Le colonne su cui
    ragionare restano precisione e netto per operazione.
    """
    return float(np.prod(1.0 + net_percent / 100.0))


def gate_report(samples: pd.DataFrame, folds: int = 6, embargo_bars: int = 24, seed: int = 0) -> pd.DataFrame:
    """Previsioni fuori campione con purging ed embargo, poi il conto economico a piu' soglie."""
    X = samples[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = samples["y"].to_numpy(dtype=int)
    embargo = pd.Timedelta(hours=embargo_bars)
    splitter = PurgedKFold(n_splits=folds, embargo=embargo)

    predictions = np.full(len(samples), np.nan)
    for train_idx, test_idx in splitter.split(samples["t_start"], samples["t_exit"]):
        if len(train_idx) < 100 or len(np.unique(y[train_idx])) < 2:
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        predictions[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    scored = samples.assign(p=predictions).dropna(subset=["p"])
    net = scored["netto_%"].to_numpy()
    base_rate = float(scored["y"].mean())
    auc = _auc(scored["y"].to_numpy(), scored["p"].to_numpy())

    rows = []
    for threshold in [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        keep = scored["p"].to_numpy() >= threshold if threshold else np.ones(len(scored), bool)
        if keep.sum() < 10:
            continue
        rows.append(
            {
                "soglia": threshold or "nessuna",
                "operazioni": int(keep.sum()),
                "quota_tenuta_%": round(100 * keep.mean(), 1),
                "precisione_%": round(100 * scored["y"].to_numpy()[keep].mean(), 1),
                "netto_medio_%": round(float(net[keep].mean()), 3),
                "netto_mediano_%": round(float(np.median(net[keep])), 3),
                "composto_seq": round(sequential_equity(net[keep]), 2),
            }
        )
    table = pd.DataFrame(rows)
    table.attrs["auc"] = auc
    table.attrs["base_rate"] = base_rate
    table.attrs["n"] = len(scored)
    return table


def selfcheck() -> None:
    """Un segnale piantato dentro le feature deve essere trovato; rumore puro non deve esserlo."""
    rng = np.random.default_rng(0)
    n = 1200
    t0 = pd.date_range("2021-01-01", periods=n, freq="4h")
    signal = rng.normal(size=n)
    net = signal * 2.0 + rng.normal(scale=0.5, size=n)  # il segnale determina l'esito
    frame = pd.DataFrame({"t_start": t0, "t_exit": t0 + pd.Timedelta("8h"), "netto_%": net, "y": (net > 0).astype(int)})
    for column in FEATURE_COLUMNS:
        frame[column] = rng.normal(size=n)
    frame["adx"] = signal  # una feature porta tutto il segnale

    table = gate_report(frame, folds=4, embargo_bars=8)
    assert table.attrs["auc"] > 0.75, table.attrs["auc"]
    filtered = table[table["soglia"] == 0.6]
    unfiltered = table[table["soglia"] == "nessuna"]
    assert filtered["netto_medio_%"].iloc[0] > unfiltered["netto_medio_%"].iloc[0], table

    # Rumore puro: l'AUC deve stare attorno a 0,5 e il filtro non deve inventare vantaggio.
    frame["adx"] = rng.normal(size=n)
    table = gate_report(frame, folds=4, embargo_bars=8)
    assert 0.40 < table.attrs["auc"] < 0.60, table.attrs["auc"]
    print("selfcheck: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strategy", default="trend_pullback", choices=list(ls.STRATEGIES))
    parser.add_argument("--universe", default="wide", choices=["majors", "wide"])
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--until", default=None)
    parser.add_argument("--oos", default=None, help="data da cui parte la finestra di verifica")
    parser.add_argument("--fee", type=float, default=FEE_PERCENT)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--save", default="")
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        selfcheck()
        return

    symbols = MAJORS if args.universe == "majors" else WIDE
    samples = build_samples(symbols, args.strategy, args.interval, args.since, args.until, args.fee)
    print(
        f"{args.strategy} [{args.universe} {args.interval}]: {len(samples)} operazioni su "
        f"{samples['simbolo'].nunique()} simboli, {samples['t_start'].min().date()} ->"
        f"{samples['t_start'].max().date()}"
    )
    print(
        f"  senza filtro: precisione {100 * samples['y'].mean():.1f}%, netto medio "
        f"{samples['netto_%'].mean():.3f}%, composto sequenziale"
        f"{sequential_equity(samples['netto_%'].to_numpy()):.2f}x"
    )

    table = gate_report(samples, folds=args.folds)
    print(f"\n===== filtro meta, {table.attrs['n']} operazioni valutate fuori campione =====")
    print(f"  AUC {table.attrs['auc']:.3f}   tasso di base {100 * table.attrs['base_rate']:.1f}%")
    print(table.to_string(index=False))

    if args.oos:
        estimation = samples[samples["t_start"] < args.oos]
        verification = samples[samples["t_start"] >= args.oos]
        print(f"\n===== verifica temporale: addestrato fino al {args.oos}, misurato dopo =====")
        print(f"  {len(estimation)} operazioni in stima, {len(verification)} in verifica")
        model = HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=0,
        )
        model.fit(estimation[FEATURE_COLUMNS].to_numpy(dtype=float), estimation["y"].to_numpy())
        p = model.predict_proba(verification[FEATURE_COLUMNS].to_numpy(dtype=float))[:, 1]
        net = verification["netto_%"].to_numpy()
        rows = []
        for threshold in [0.0, 0.45, 0.50, 0.55, 0.60]:
            keep = p >= threshold if threshold else np.ones(len(p), bool)
            if keep.sum() < 5:
                continue
            # Controllo con selezione casuale: la stessa quantita' di operazioni, scelte a caso.
            # Con una primaria a coda lunga bastano poche operazioni fortunate per far salire il
            # netto medio, quindi il numero da battere non e' zero, e' il percentile alto del caso.
            rng = np.random.default_rng(12345)
            draws = np.array(
                [net[rng.choice(len(net), size=int(keep.sum()), replace=False)].mean() for _ in range(500)]
            )
            rows.append(
                {
                    "soglia": threshold or "nessuna",
                    "operazioni": int(keep.sum()),
                    "precisione_%": round(100 * verification["y"].to_numpy()[keep].mean(), 1),
                    "netto_medio_%": round(float(net[keep].mean()), 3),
                    "caso_p95_%": round(float(np.percentile(draws, 95)), 3),
                    "percentile_nel_caso": round(float((draws < net[keep].mean()).mean() * 100), 1),
                    "composto_seq": round(sequential_equity(net[keep]), 2),
                }
            )
        oos_table = pd.DataFrame(rows)
        print(f"  AUC in verifica: {_auc(verification['y'].to_numpy(), p):.3f}")
        print(oos_table.to_string(index=False))
        if args.save:
            OUTPUT_DIR.mkdir(exist_ok=True)
            oos_table.to_csv(OUTPUT_DIR / f"{args.save}_oos.csv", index=False)

    if args.save:
        OUTPUT_DIR.mkdir(exist_ok=True)
        table.to_csv(OUTPUT_DIR / f"{args.save}.csv", index=False)


if __name__ == "__main__":
    main()
