"""Misure che stanno dietro a `strategy.md`, in forma riutilizzabile.

Ogni funzione restituisce un DataFrame invece di stampare: cosi' le stesse misure alimentano sia
la riga di comando sia la pagina Streamlit (`app/analysis_dashboard.py`), senza duplicare la
logica in due posti che poi divergono.

I risultati vengono messi in cache su disco in `analysis_cache/`, perche' alcune misure
richiedono minuti su undici milioni di candele e una dashboard che le ricalcola a ogni
interazione e' inutilizzabile.

    python -m scripts.analysis --all          # calcola tutto e riempie la cache
    python -m scripts.analysis --capacity     # solo una misura
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cryptofarm.data.klines import DEFAULT_SYMBOLS, load_klines, store_manifest
from cryptofarm.paths import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "analysis_cache"
BAR_MINUTES = 5
SINCE = "2022-01-01"
CHUNK = 100_000
DEFAULT_PANEL = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT"]

# Configurazioni di barriere usate in tutto il documento: (take-profit, stop-loss, orizzonte in ore).
BARRIER_CONFIGS = [(0.006, 0.003, 8), (0.008, 0.004, 8), (0.010, 0.005, 12), (0.012, 0.006, 24)]
FEE_SCENARIOS = [("taker 0,20%", 0.0020), ("BNB 0,15%", 0.0015), ("maker 0,04%", 0.0004)]


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def cached(name: str, builder, refresh: bool = False) -> pd.DataFrame:
    """Legge la misura dalla cache, o la calcola e la salva."""
    path = _cache_path(name)
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    frame = builder()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def _panel(symbols=None, interval="5m", since=SINCE) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in symbols or DEFAULT_PANEL:
        data = load_klines(symbol, interval)
        data = data[data.index >= since]
        if len(data) > 1000:
            frames[symbol] = data
    return frames


# ---------------------------------------------------------------------------------------------
# Copertura dello store e regimi di mercato
# ---------------------------------------------------------------------------------------------


def store_coverage() -> pd.DataFrame:
    return store_manifest()


def market_regimes(symbol: str = "BTCUSDT", window_days: int = 30, threshold: float = 0.10) -> pd.DataFrame:
    """Quota di giorni per anno in regime bull, bear o laterale.

    La classificazione e' sul rendimento su finestra mobile: serve a verificare che il dataset
    contenga tutti i regimi e che siano concentrati in periodi distinti, condizione senza la
    quale la distribuzione della CPCV e' l'artefatto di un solo ciclo.
    """
    candles = load_klines(symbol, "1h")
    daily = candles["Close"].resample("1D").last().dropna()
    trailing = daily.pct_change(window_days)
    regime = pd.cut(trailing, [-np.inf, -threshold, threshold, np.inf], labels=["bear", "sideways", "bull"])
    table = pd.DataFrame({"anno": daily.index.year, "regime": regime.values}).dropna()
    counts = table.groupby(["anno", "regime"], observed=True).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1), axis=0).reset_index()
    return shares


# ---------------------------------------------------------------------------------------------
# Tempo al target, con e senza censura
# ---------------------------------------------------------------------------------------------


def _kaplan_meier_median(bars: np.ndarray, hit: np.ndarray) -> float:
    order = np.argsort(bars)
    ordered_bars, ordered_hit = bars[order], hit[order]
    at_risk = len(ordered_bars)
    survival = 1.0
    for value in np.unique(ordered_bars):
        mask = ordered_bars == value
        events = int(ordered_hit[mask].sum())
        if at_risk > 0 and events > 0:
            survival *= 1 - events / at_risk
            if survival <= 0.5:
                return float(value)
        at_risk -= int(mask.sum())
    return float("inf")


def time_to_target(targets=(0.003, 0.004, 0.006, 0.010), horizon_hours: int = 24) -> pd.DataFrame:
    """Tempo perche' il prezzo salga di una data percentuale, condizionato e non.

    La versione **condizionata** (mediana sui soli casi che il target lo raggiungono) e' quella
    che compariva nella prima stesura di `strategy.md` ed e' distorta verso il basso: sovrastima
    la capacita' di fare trade frequenti. La versione non condizionata usa Kaplan-Meier con i
    casi non raggiunti trattati come censurati.
    """
    horizon = int(horizon_hours * 60 / BAR_MINUTES)
    frames = _panel()
    rows = []
    for target in targets:
        all_bars, all_hit = [], []
        for data in frames.values():
            high, close = data["High"].to_numpy(), data["Close"].to_numpy()
            usable = len(close) - horizon
            windows = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:usable]
            upper = close[:usable] * (1 + target)
            for start in range(0, usable, CHUNK):
                stop = min(start + CHUNK, usable)
                touched = windows[start:stop] >= upper[start:stop, None]
                reached = touched.any(axis=1)
                all_bars.append(np.where(reached, touched.argmax(axis=1) + 1, horizon))
                all_hit.append(reached)
        bars = np.concatenate(all_bars)
        hit = np.concatenate(all_hit)
        km = _kaplan_meier_median(bars, hit)
        rows.append(
            {
                "target": target,
                "p_raggiunto": float(hit.mean()),
                "mediana_condizionata_h": float(np.median(bars[hit])) * BAR_MINUTES / 60,
                "mediana_reale_h": km * BAR_MINUTES / 60 if np.isfinite(km) else np.inf,
            }
        )
    frame = pd.DataFrame(rows)
    frame["errore"] = frame["mediana_reale_h"] / frame["mediana_condizionata_h"] - 1
    return frame


# ---------------------------------------------------------------------------------------------
# Capacita': holding time reale ed economia
# ---------------------------------------------------------------------------------------------


def first_touch(high, low, close, take_profit, stop_loss, horizon):
    """Primo contatto: esito, barre fino all'uscita, rendimento realizzato."""
    usable = len(close) - horizon
    if usable <= 0:
        return np.empty(0, np.int8), np.empty(0), np.empty(0)
    future_high = np.lib.stride_tricks.sliding_window_view(high[1:], horizon)[:usable]
    future_low = np.lib.stride_tricks.sliding_window_view(low[1:], horizon)[:usable]
    future_close = np.lib.stride_tricks.sliding_window_view(close[1:], horizon)[:usable]
    entry = close[:usable]
    upper, lower = entry * (1 + take_profit), entry * (1 - stop_loss)

    outcome = np.zeros(usable, np.int8)
    bars = np.full(usable, horizon, np.int32)
    realised = np.zeros(usable)

    for start in range(0, usable, CHUNK):
        stop = min(start + CHUNK, usable)
        hit_up = future_high[start:stop] >= upper[start:stop, None]
        hit_down = future_low[start:stop] <= lower[start:stop, None]
        never = horizon + 1
        first_up = np.where(hit_up.any(1), hit_up.argmax(1), never)
        first_down = np.where(hit_down.any(1), hit_down.argmax(1), never)

        block_outcome = np.zeros(stop - start, np.int8)
        block_bars = np.full(stop - start, horizon, np.int32)
        block_return = future_close[start:stop, -1] / entry[start:stop] - 1.0

        won = first_up < first_down
        lost = (first_down <= first_up) & (first_down != never)
        block_outcome[won], block_bars[won], block_return[won] = 1, first_up[won] + 1, take_profit
        block_outcome[lost], block_bars[lost], block_return[lost] = -1, first_down[lost] + 1, -stop_loss

        outcome[start:stop], bars[start:stop], realised[start:stop] = block_outcome, block_bars, block_return
    return outcome, bars, realised


def barrier_capacity(target_trades_per_day: int = 4) -> pd.DataFrame:
    """Per ogni configurazione di barriere: esiti, holding time reale e capacita'.

    E' la tabella che vincola tutto il resto. Il tempo di detenzione non e' il tempo al target:
    con barriere 2:1 la maggior parte dei trade chiude sullo stop, che e' piu' vicino.
    """
    frames = _panel()
    rows = []
    for take_profit, stop_loss, hours in BARRIER_CONFIGS:
        horizon = int(hours * 60 / BAR_MINUTES)
        outcomes, bars, realised = [], [], []
        for data in frames.values():
            o, b, r = first_touch(
                data["High"].to_numpy(),
                data["Low"].to_numpy(),
                data["Close"].to_numpy(),
                take_profit,
                stop_loss,
                horizon,
            )
            outcomes.append(o), bars.append(b), realised.append(r)
        outcome = np.concatenate(outcomes)
        holding_hours = np.concatenate(bars).mean() * BAR_MINUTES / 60
        timeout_mask = outcome == 0
        rows.append(
            {
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "orizzonte_h": hours,
                "p_tp": float((outcome == 1).mean()),
                "p_sl": float((outcome == -1).mean()),
                "p_timeout": float(timeout_mask.mean()),
                "return_timeout": float(np.concatenate(realised)[timeout_mask].mean()) if timeout_mask.any() else 0.0,
                "holding_h": holding_hours,
                "tetto_trade_giorno": 24 / holding_hours,
                "in_mercato_per_target": min(1.0, target_trades_per_day * holding_hours / 24),
            }
        )
    return pd.DataFrame(rows)


def break_even_table(capacity: pd.DataFrame | None = None) -> pd.DataFrame:
    """Break-even ed expectancy sulla distribuzione completa degli esiti, per regime di fee.

    La formula analitica `(sl+f)/((tp-f)+(sl+f))` assume che ogni trade chiuda su una barriera di
    prezzo. Qui i timeout entrano al loro rendimento reale, che risulta positivo.
    """
    capacity = capacity if capacity is not None else barrier_capacity()
    rows = []
    for _, config in capacity.iterrows():
        take_profit, stop_loss = config["take_profit"], config["stop_loss"]
        p_tp, p_sl, p_to = config["p_tp"], config["p_sl"], config["p_timeout"]
        e_to = config["return_timeout"]
        resolved = p_tp + p_sl
        win_rate = p_tp / resolved
        for name, fee in FEE_SCENARIOS:
            break_even = (resolved * (stop_loss + fee) - p_to * (e_to - fee)) / (
                resolved * ((take_profit - fee) + (stop_loss + fee))
            )
            expectancy = p_tp * (take_profit - fee) + p_sl * (-stop_loss - fee) + p_to * (e_to - fee)
            rows.append(
                {
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "regime_fee": name,
                    "fee": fee,
                    "win_rate_misurato": win_rate,
                    "break_even": break_even,
                    "divario_punti": (break_even - win_rate) * 100,
                    "expectancy": expectancy,
                }
            )
    return pd.DataFrame(rows)


def random_walk_comparison(
    capacity: pd.DataFrame | None = None, sigma: float = 0.00121, paths: int = 40_000
) -> pd.DataFrame:
    """Confronto con una random walk driftless della stessa volatilita'.

    Serve a distinguere cio' che e' struttura di mercato da cio' che e' semplice diffusione: se i
    tempi misurati coincidono con quelli della random walk, nella *tempistica* non c'e' edge da
    estrarre e l'edge, se c'e', e' tutto nella direzione.
    """
    capacity = capacity if capacity is not None else barrier_capacity()
    generator = np.random.default_rng(0)
    rows = []
    for _, config in capacity.iterrows():
        horizon = int(config["orizzonte_h"] * 60 / BAR_MINUTES)
        path = np.cumsum(generator.normal(0, sigma, (paths, horizon)), axis=1)
        hit_up = path >= np.log1p(config["take_profit"])
        hit_down = path <= np.log1p(-config["stop_loss"])
        never = horizon + 1
        first_up = np.where(hit_up.any(1), hit_up.argmax(1), never)
        first_down = np.where(hit_down.any(1), hit_down.argmax(1), never)
        bars = np.minimum(
            np.where(first_up == never, horizon, first_up + 1),
            np.where(first_down == never, horizon, first_down + 1),
        )
        rows.append(
            {
                "take_profit": config["take_profit"],
                "stop_loss": config["stop_loss"],
                "holding_random_walk_h": bars.mean() * BAR_MINUTES / 60,
                "holding_reale_h": config["holding_h"],
                "p_tp_random_walk": float((first_up < first_down).mean()),
                "p_tp_reale": config["p_tp"],
            }
        )
    frame = pd.DataFrame(rows)
    frame["rapporto_holding"] = frame["holding_reale_h"] / frame["holding_random_walk_h"]
    return frame


# ---------------------------------------------------------------------------------------------
# Campionamento a eventi
# ---------------------------------------------------------------------------------------------


def cusum_rates(multiples=(2.0, 3.0, 4.0, 5.0)) -> pd.DataFrame:
    """Eventi CUSUM al giorno per simbolo e soglia.

    La soglia e' in multipli della volatilita' locale: e' cio' che rende il campionamento
    confrontabile fra asset senza calibrazione per simbolo.
    """
    from cryptofarm.ml.dataset import cusum_events

    rows = []
    for symbol, data in _panel(DEFAULT_SYMBOLS).items():
        days = (data.index[-1] - data.index[0]).total_seconds() / 86400
        returns = np.diff(np.log(data["Close"].to_numpy()))
        sigma = float(np.nanmedian(pd.Series(returns).rolling(288).std()))
        for multiple in multiples:
            events = cusum_events(data["Close"], multiple)
            rows.append(
                {"symbol": symbol, "sigma": sigma, "soglia_sigma": multiple, "eventi_giorno": len(events) / days}
            )
    return pd.DataFrame(rows)


def portfolio_concurrency(capacity: pd.DataFrame | None = None, symbols: int = 15, per_day: int = 4) -> pd.DataFrame:
    """Posizioni contemporaneamente aperte a portafoglio: media e picco.

    Il picco e' cio' che vincola il capitale, e la simulazione lo **sottostima** perche' assume
    arrivi indipendenti fra simboli, mentre le criptovalute si muovono insieme.
    """
    capacity = capacity if capacity is not None else barrier_capacity()
    generator = np.random.default_rng(1)
    rows = []
    for _, config in capacity.iterrows():
        holding = config["holding_h"]
        peaks = []
        for _ in range(120):
            starts = np.concatenate(
                [generator.uniform(0, 2000 * 24, generator.poisson(per_day * 2000)) for _ in range(symbols)]
            )
            edges = np.concatenate(
                [np.stack([starts, np.ones_like(starts)]), np.stack([starts + holding, -np.ones_like(starts)])],
                axis=1,
            )
            edges = edges[:, np.argsort(edges[0])]
            peaks.append(np.cumsum(edges[1]).max())
        rows.append(
            {
                "take_profit": config["take_profit"],
                "stop_loss": config["stop_loss"],
                "posizioni_medie": symbols * per_day * holding / 24,
                "picco_mediano": float(np.median(peaks)),
                "picco_p99": float(np.percentile(peaks, 99)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Directional change: ritardo di conferma, soglia per simbolo, distribuzione delle classi
# ---------------------------------------------------------------------------------------------


PIVOT_THRESHOLDS = (0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015)


def pivot_delays(thresholds=PIVOT_THRESHOLDS) -> pd.DataFrame:
    """Ritardo di conferma dei pivot, per simbolo e per soglia.

    Il ritardo e' la distanza fra l'estremo e la barra in cui diventa conoscibile. E' il numero
    che decide se una feature costruita sui pivot e' utilizzabile: se il p90 e' lungo, il modello
    non vede la struttura quando servirebbe. Insieme al ritardo si misura quanto della gamba
    resta da prendere alla conferma, che e' il tetto economico della strategia.
    """
    from cryptofarm.ml.directional_change import capturable_fraction, directional_change_pivots, leg_table

    rows = []
    for symbol, data in _panel(DEFAULT_SYMBOLS).items():
        high, low, close = (data[c].to_numpy(float) for c in ("High", "Low", "Close"))
        days = (data.index[-1] - data.index[0]).total_seconds() / 86400
        for threshold in thresholds:
            pivots = directional_change_pivots(high, low, threshold)
            if len(pivots) < 3:
                continue
            delay = (pivots["confirm_bar"] - pivots["extreme_bar"]).to_numpy()
            legs = capturable_fraction(leg_table(pivots), close)
            rows.append(
                {
                    "symbol": symbol,
                    "soglia": threshold,
                    "estremi_giorno": len(pivots) / days,
                    "ritardo_mediano": float(np.median(delay)),
                    "ritardo_p90": float(np.percentile(delay, 90)),
                    "ritardo_p99": float(np.percentile(delay, 99)),
                    "gamba_mediana": float(legs["size"].median()),
                    "catturabile_mediana": float(legs["capturable_at_confirm"].median()),
                }
            )
    return pd.DataFrame(rows)


def pivot_labels(capture: float = 0.60, delays: pd.DataFrame | None = None) -> pd.DataFrame:
    """Soglia tarata per simbolo e distribuzione delle classi con l'etichetta morbida.

    La soglia si sceglie per portare gli estremi a 8-12 al giorno (4 trade/giorno per lato con
    margine di scarto), e va tarata **per simbolo**: la stessa percentuale su asset con
    volatilita' diverse produce tassi molto diversi.
    """
    from cryptofarm.ml.directional_change import (
        capturable_fraction,
        directional_change_pivots,
        label_distribution,
        leg_table,
        soft_labels,
        tune_threshold,
    )

    del delays  # firma mantenuta per la dashboard; la soglia si ritara qui per simbolo
    rows = []
    for symbol, data in _panel(DEFAULT_SYMBOLS).items():
        high, low, close = (data[c].to_numpy(float) for c in ("High", "Low", "Close"))
        days = (data.index[-1] - data.index[0]).total_seconds() / 86400
        threshold, rate = tune_threshold(high, low, days)
        pivots = directional_change_pivots(high, low, threshold)
        distribution = label_distribution(soft_labels(close, pivots, capture))
        delay = (pivots["confirm_bar"] - pivots["extreme_bar"]).to_numpy()
        legs = capturable_fraction(leg_table(pivots), close)
        # Cio' che si porta a casa entrando alla conferma, prima dei costi. La media pesata sulla
        # dimensione conta piu' della mediana: il rendimento viene dalle gambe lunghe.
        take = legs["size"] * legs["capturable_at_confirm"]
        weights = legs["size"] / legs["size"].sum()
        rows.append(
            {
                "symbol": symbol,
                "soglia_tarata": threshold,
                "estremi_giorno": rate,
                "ritardo_mediano": float(np.median(delay)),
                "ritardo_p90": float(np.percentile(delay, 90)),
                "gamba_mediana": float(legs["size"].median()),
                "catturabile_mediana": float(legs["capturable_at_confirm"].median()),
                "catturabile_pesata": float((legs["capturable_at_confirm"] * weights).sum()),
                "presa_mediana": float(take.median()),
                "quota_oltre_maker": float((take > 0.0008).mean()),
                "quota_oltre_taker": float((take > 0.0040).mean()),
                "hold": distribution["hold"],
                "buy": distribution["buy"],
                "sell": distribution["sell"],
                "positivi": distribution["buy"] + distribution["sell"],
            }
        )
    return pd.DataFrame(rows)


MEASURES = {
    "store_coverage": store_coverage,
    "market_regimes": market_regimes,
    "time_to_target": time_to_target,
    "barrier_capacity": barrier_capacity,
    "break_even": break_even_table,
    "random_walk": random_walk_comparison,
    "cusum_rates": cusum_rates,
    "concurrency": portfolio_concurrency,
    "pivot_delays": pivot_delays,
    "pivot_labels": pivot_labels,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcola le misure che stanno dietro a strategy.md.")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="ricalcola anche se la cache esiste")
    for name in MEASURES:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true")
    args = parser.parse_args()

    selected = [name for name in MEASURES if args.all or getattr(args, name)]
    if not selected:
        parser.error("indicare almeno una misura, o --all")

    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        frame = cached(name, MEASURES[name], refresh=args.refresh)
        print(frame.to_string(index=False))
    print(f"\nCache in {CACHE_DIR}")


if __name__ == "__main__":
    main()
