"""Costruisce lo store 5m a partire dal dataset pubblico Bitstamp BTC/USD a un minuto.

`data/klines.py` prende le candele dai dump di `data.binance.vision`. Dove quell'host non e'
raggiungibile (rete d'ufficio, CI chiusa, sessioni remote con egress filtrato) lo store resta
vuoto e ogni misura di lungo periodo diventa impossibile. Questo script riempie lo stesso store
da una fonte alternativa, cosi' `load_klines` e tutto cio' che ci sta sopra funzionano senza
sapere da dove arrivano i dati.

Fonte: <https://github.com/ff137/bitstamp-btcusd-minute-data>, candele BTC/USD di Bitstamp a un
minuto dal 2012-01-01 a oggi (file storico + file di aggiornamento giornaliero). Non e' lo stesso
mercato di BTCUSDT/BTCUSDC su Binance: prezzi e volumi differiscono di frazioni di punto e le
commissioni non sono le stesse, quindi va bene per misurare **come si comportano le strategie**,
non per replicare il P&L di un conto Binance al centesimo.

Il minuto e' la granularita' nativa; il 5m si ottiene per aggregazione con le stesse convenzioni
di `resample_klines` (etichetta a sinistra, estremo sinistro incluso).

Attenzione a una proprieta' della fonte: il dataset non ha minuti mancanti perche' i buchi sono
riempiti con candele piatte a volume zero. Sono innocue una volta aggregate finche' restano rare,
ma nei primi anni (2012-2014, Bitstamp ancora sottile) sono la maggioranza: `--since` esiste per
tagliarle via, e `--report` stampa la quota di barre a volume nullo anno per anno.

    python -m scripts.import_bitstamp --source /percorso/al/clone --since 2015-01-01
    python -m scripts.import_bitstamp --report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cryptofarm.data.klines import _RESAMPLE_AGG, BASE_INTERVAL, COLUMNS, store_path
from cryptofarm.paths import MARKET_DATA_DIR

SYMBOL = "BTCUSD"
HISTORICAL = "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
UPDATES = "data/updates/btcusd_bitstamp_1min_latest.csv"


def _read_minutes(source: Path) -> pd.DataFrame:
    frames = []
    for relative in (HISTORICAL, UPDATES):
        path = source / relative
        if not path.exists():
            raise FileNotFoundError(f"manca {path}: il clone non e' quello atteso")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
        frame = frame.set_index("timestamp").rename(columns=str.capitalize)
        frames.append(frame[COLUMNS].astype(float))
    minutes = pd.concat(frames)
    minutes = minutes[~minutes.index.duplicated(keep="last")].sort_index()
    minutes.index.name = "Open time"
    return minutes


def build_store(source: Path, since: str | None = None, store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    minutes = _read_minutes(source)
    if since:
        minutes = minutes[minutes.index >= since]
    candles = minutes.resample(f"{BASE_INTERVAL[:-1]}min").agg(_RESAMPLE_AGG).dropna()
    path = store_path(SYMBOL, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(path)
    return candles


def flat_bar_report(store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    """Quota di barre 5m senza scambi, anno per anno: la misura della qualita' della fonte."""
    candles = pd.read_parquet(store_path(SYMBOL, store_dir))
    flat = (candles["Volume"] == 0) | (candles["High"] == candles["Low"])
    return (
        pd.DataFrame({"anno": candles.index.year, "flat": flat.to_numpy()})
        .groupby("anno")
        .agg(barre=("flat", "size"), piatte=("flat", "sum"))
        .assign(quota=lambda frame: (frame["piatte"] / frame["barre"] * 100).round(3))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="/home/user/data-bitstamp", help="clone del repository dei dati")
    parser.add_argument("--since", default=None, help="scarta le candele precedenti a questa data")
    parser.add_argument("--store-dir", default=str(MARKET_DATA_DIR))
    parser.add_argument("--report", action="store_true", help="solo la quota di barre piatte")
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    if not args.report:
        candles = build_store(Path(args.source), args.since, store_dir)
        print(f"{len(candles):,} candele {BASE_INTERVAL} ({candles.index[0]:%Y-%m-%d} .. {candles.index[-1]:%Y-%m-%d})")
        print(f"scritte in {store_path(SYMBOL, store_dir)}")
    print(flat_bar_report(store_dir).to_string())


if __name__ == "__main__":
    main()
