"""Costruisce lo store 5m da dataset pubblici di candele a un minuto.

`data/klines.py` prende le candele dai dump di `data.binance.vision`. Dove quell'host non e'
raggiungibile (rete d'ufficio, CI chiusa, sessioni remote con egress filtrato) lo store resta
vuoto e ogni misura di lungo periodo diventa impossibile. Questo script lo riempie da fonti
alternative con la stessa struttura, cosi' `load_klines` e tutto cio' che ci sta sopra funzionano
senza sapere da dove arrivano i dati.

Due fonti, scelte per ragioni diverse:

- **bitstamp** — <https://github.com/ff137/bitstamp-btcusd-minute-data>, BTC/USD dal 2012 a oggi.
  E' la serie principale: lunga, continua, aggiornata ogni giorno.
- **bitfinex** — <https://github.com/Zombie-3000/Bitfinex-historical-data>, piu' coppie ma solo
  2016-2019, in file annuali. Serve a rifare le stesse misure su un secondo mercato e un secondo
  exchange: e' un controllo, non la serie di riferimento. Attenzione all'ordine delle colonne,
  che e' `timestamp, open, close, high, low, volume` -- **close prima di high**, al contrario di
  ogni altra fonte.

Nessuna delle due e' Binance: prezzi e volumi differiscono di frazioni di punto e le commissioni
non sono le stesse. Va bene per misurare **come si comportano le strategie**, non per replicare il
P&L di un conto Binance al centesimo.

Il minuto e' la granularita' nativa di entrambe; il 5m si ottiene per aggregazione con le stesse
convenzioni di `resample_klines` (etichetta a sinistra, estremo sinistro incluso).

Attenzione a una proprieta' della prima fonte: il dataset non ha minuti mancanti perche' i buchi
sono riempiti con candele piatte a volume zero. Sono innocue una volta aggregate finche' restano
rare, ma nei primi anni (2012-2014, Bitstamp ancora sottile) sono la maggioranza: `--since` esiste
per tagliarle via, e `--report` stampa la quota di barre piatte anno per anno.

    python -m scripts.import_candles --source /percorso/al/clone
    python -m scripts.import_candles --format bitfinex --symbol ETHUSD \\
        --source /percorso/al/clone/ETHUSD/Candles_1m
    python -m scripts.import_candles --report --symbol ETHUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cryptofarm.data.klines import _RESAMPLE_AGG, BASE_INTERVAL, COLUMNS, store_path
from cryptofarm.paths import MARKET_DATA_DIR

DEFAULT_SYMBOL = "BTCUSD"
BITSTAMP_FILES = (
    "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz",
    "data/updates/btcusd_bitstamp_1min_latest.csv",
)
# L'ordine dei dump di Bitfinex, che non e' OHLC.
BITFINEX_COLUMNS = ["timestamp", "Open", "Close", "High", "Low", "Volume"]


def _read_bitstamp(source: Path) -> pd.DataFrame:
    frames = []
    for relative in BITSTAMP_FILES:
        path = source / relative
        if not path.exists():
            raise FileNotFoundError(f"manca {path}: il clone non e' quello atteso")
        frame = pd.read_csv(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s")
        frames.append(frame.set_index("timestamp").rename(columns=str.capitalize)[COLUMNS].astype(float))
    return pd.concat(frames)


def _read_bitfinex(source: Path) -> pd.DataFrame:
    """Un `merged.csv` per anno, senza intestazione, timestamp in millisecondi."""
    paths = sorted(source.glob("*/merged.csv"))
    if not paths:
        raise FileNotFoundError(f"nessun merged.csv sotto {source}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, header=None, names=BITFINEX_COLUMNS)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
        frames.append(frame.set_index("timestamp")[COLUMNS].astype(float))
    return pd.concat(frames)


READERS = {"bitstamp": _read_bitstamp, "bitfinex": _read_bitfinex}


def read_minutes(source: Path, source_format: str = "bitstamp") -> pd.DataFrame:
    minutes = READERS[source_format](source)
    minutes = minutes[~minutes.index.duplicated(keep="last")].sort_index()
    minutes.index.name = "Open time"
    return minutes


def build_store(
    source: Path,
    symbol: str = DEFAULT_SYMBOL,
    source_format: str = "bitstamp",
    since: str | None = None,
    store_dir: Path = MARKET_DATA_DIR,
) -> pd.DataFrame:
    minutes = read_minutes(source, source_format)
    if since:
        minutes = minutes[minutes.index >= since]
    candles = minutes.resample(f"{BASE_INTERVAL[:-1]}min").agg(_RESAMPLE_AGG).dropna()
    path = store_path(symbol, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    candles.to_parquet(path)
    return candles


def flat_bar_report(symbol: str = DEFAULT_SYMBOL, store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    """Quota di barre 5m senza scambi, anno per anno: la misura della qualita' della fonte."""
    candles = pd.read_parquet(store_path(symbol, store_dir))
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
    parser.add_argument("--format", default="bitstamp", choices=sorted(READERS), dest="source_format")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--since", default=None, help="scarta le candele precedenti a questa data")
    parser.add_argument("--store-dir", default=str(MARKET_DATA_DIR))
    parser.add_argument("--report", action="store_true", help="solo la quota di barre piatte")
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    if not args.report:
        candles = build_store(Path(args.source), args.symbol, args.source_format, args.since, store_dir)
        print(f"{len(candles):,} candele {BASE_INTERVAL} ({candles.index[0]:%Y-%m-%d} .. {candles.index[-1]:%Y-%m-%d})")
        print(f"scritte in {store_path(args.symbol, store_dir)}")
    print(flat_bar_report(args.symbol, store_dir).to_string())


if __name__ == "__main__":
    main()
