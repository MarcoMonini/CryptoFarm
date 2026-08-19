"""Download e archiviazione locale delle candele Binance.

Lo store tiene **un solo intervallo per simbolo**, il 5m, e deriva 15m/30m/1h aggregandolo.
L'aggregazione e' esatta (verificata contro i dump ufficiali: differenza nulla su OHLC), quindi
una sola fonte di verita' invece di quattro archivi da tenere allineati, e un quarto delle
richieste di rete.

I dati arrivano dai dump mensili di `data.binance.vision`, non dalla REST API. La differenza e'
decisiva su questa scala: il costo del download e' interamente **latenza per richiesta** (~2,7 s
misurati, identici su REST e su CDN), non banda. La REST API restituisce al massimo 1000 candele
per richiesta ed e' soggetta a rate limit, quindi ~18.700 richieste sequenziali per lo storico
completo, circa 14 ore. Un dump mensile contiene un mese intero (~8.900 candele su 5m) ed e' un
file statico su CDN senza rate limit, quindi parallelizzabile: ~1.350 file con 32 worker, sotto i
10 minuti. La REST API resta usata solo per la coda (le candele di oggi, che nei dump non ci sono
ancora).

Solo dati pubblici: non serve nessuna credenziale.

Uso da riga di comando:

    python -m cryptofarm.data.klines --update            # scarica/aggiorna lo store di default
    python -m cryptofarm.data.klines --manifest          # mostra cosa c'e' nello store
    python -m cryptofarm.data.klines --update --symbols BTCUSDT ETHUSDT
"""

from __future__ import annotations

import argparse
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from binance import Client

from cryptofarm.paths import MARKET_DATA_DIR

# Selezione di riferimento: liquidita' alta, storico lungo e profili di volatilita' diversi
# (store of value, L1 ad alto throughput, meme, DeFi). I simboli senza dati vengono saltati.
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "LTCUSDT",
    "TRXUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
]

# Intervallo effettivamente archiviato. Tutti gli altri sono derivati da questo per aggregazione.
BASE_INTERVAL = "5m"
DERIVED_INTERVALS = ["15m", "30m", "1h"]

DUMP_BASE = "https://data.binance.vision/data/spot"
FIRST_MONTH = "2017-08"  # prima del listing di qualunque coppia: i mesi vuoti danno 404 e si saltano
DOWNLOAD_WORKERS = 32  # il download e' latency-bound: la concorrenza e' l'unica leva che conta
COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

_DUMP_COLUMNS = [
    "Open time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Close time",
    "Quote asset volume",
    "Number of trades",
    "Taker buy base asset volume",
    "Taker buy quote asset volume",
    "Ignore",
]

_RESAMPLE_AGG = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

_thread_local = threading.local()


def interval_to_minutes(interval: str) -> int:
    """Converte un intervallo Binance ("5m", "1h", "1d") nel numero di minuti."""
    unit, value = interval[-1], interval[:-1]
    factors = {"m": 1, "h": 60, "d": 60 * 24, "w": 60 * 24 * 7}
    if unit not in factors or not value.isdigit():
        raise ValueError(f"Intervallo non supportato: {interval!r}")
    return int(value) * factors[unit]


def store_path(symbol: str, store_dir: Path = MARKET_DATA_DIR) -> Path:
    return Path(store_dir) / f"{symbol}-{BASE_INTERVAL}.parquet"


def resample_klines(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggrega le candele base in un intervallo piu' lungo.

    Esatto rispetto alle candele native dell'exchange: `Open time` e' l'inizio della barra,
    quindi la convenzione di default di `resample` (etichetta a sinistra, estremo sinistro
    incluso) coincide con quella di Binance.
    """
    if interval == BASE_INTERVAL:
        return df
    minutes = interval_to_minutes(interval)
    if minutes % interval_to_minutes(BASE_INTERVAL) != 0:
        raise ValueError(f"{interval} non e' un multiplo di {BASE_INTERVAL}")
    return df.resample(f"{minutes}min").agg(_RESAMPLE_AGG).dropna()


def _open_times_to_datetime(values: pd.Series) -> pd.Series:
    """Converte la colonna dei timestamp gestendo il cambio di unita' nei dump.

    Fino a dicembre 2024 i dump esprimono `Open time` in millisecondi; da gennaio 2025 in
    microsecondi, senza nessun segnale nel formato del file. Distinguerli dall'ordine di
    grandezza e' l'unico modo affidabile: un timestamp in millisecondi resta sotto 1e14 per
    secoli, uno in microsecondi lo supera gia' nel 1973.
    """
    numeric = pd.to_numeric(values)
    unit = "us" if numeric.iloc[0] > 1e14 else "ms"
    return pd.to_datetime(numeric, unit=unit)


def _parse_dump(content: bytes) -> pd.DataFrame:
    archive = zipfile.ZipFile(io.BytesIO(content))
    raw = pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])), header=None, names=_DUMP_COLUMNS)
    # I dump piu' recenti includono una riga di intestazione, quelli storici no.
    if not str(raw.iloc[0]["Open time"]).replace(".", "", 1).isdigit():
        raw = raw.iloc[1:]
    raw["Open time"] = _open_times_to_datetime(raw["Open time"])
    raw.set_index("Open time", inplace=True)
    return raw[COLUMNS].astype(float)


class _DumpMissing(Exception):
    """Il dump non esiste: coppia non ancora quotata in quel periodo. Non e' un errore."""


def _fetch_dump(symbol: str, kind: str, stamp: str, attempts: int = 3) -> pd.DataFrame:
    """Scarica e decodifica un dump ("monthly"/"daily").

    Solleva `_DumpMissing` se il file non esiste (404), e propaga qualunque altro errore dopo
    aver ritentato. La distinzione e' essenziale: trattare un errore di rete o di parsing come
    "mese inesistente" tronca l'archivio in silenzio, che e' esattamente il modo in cui
    l'unita' di misura cambiata nei dump del 2025 e' passata inosservata.
    """
    url = f"{DUMP_BASE}/{kind}/klines/{symbol}/{BASE_INTERVAL}/{symbol}-{BASE_INTERVAL}-{stamp}.zip"
    session = _thread_session()
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=120)
            if response.status_code == 404:
                raise _DumpMissing(stamp)
            response.raise_for_status()
            return _parse_dump(response.content)
        except _DumpMissing:
            raise
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"{symbol} {kind} {stamp}: {last_error}")


def _thread_session() -> requests.Session:
    """Una `requests.Session` per thread: le sessioni non sono thread-safe."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=DOWNLOAD_WORKERS)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [stamp.strftime("%Y-%m") for stamp in pd.date_range(start, end, freq="MS")]


def _fetch_recent_from_api(symbol: str, start: pd.Timestamp) -> pd.DataFrame:
    """Coda del download: le candele che nei dump non sono ancora pubblicate.

    Sono al massimo poche ore, quindi bastano una o due richieste REST.
    """
    client = Client(api_key="<api_key>", api_secret="<api_secret>")
    frames = []
    cursor = int(start.timestamp() * 1000)
    step = interval_to_minutes(BASE_INTERVAL) * 60_000
    for _ in range(10):  # limite di sicurezza: 10.000 candele bastano ampiamente per la coda
        chunk = client.get_klines(symbol=symbol, interval=BASE_INTERVAL, limit=1000, startTime=cursor)
        if not chunk:
            break
        frame = pd.DataFrame(chunk, columns=_DUMP_COLUMNS)
        frame["Open time"] = pd.to_datetime(frame["Open time"], unit="ms")
        frame.set_index("Open time", inplace=True)
        frames.append(frame[COLUMNS].astype(float))
        if len(chunk) < 1000:
            break
        cursor = chunk[-1][0] + step
    if not frames:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="Open time"))
    return pd.concat(frames)


def load_klines(symbol: str, interval: str = BASE_INTERVAL, store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    """Legge le candele di un simbolo, aggregandole se l'intervallo richiesto e' derivato."""
    path = store_path(symbol, store_dir)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="Open time"))
    return resample_klines(pd.read_parquet(path), interval)


def update_symbol(
    symbol: str,
    store_dir: Path = MARKET_DATA_DIR,
    workers: int = DOWNLOAD_WORKERS,
) -> pd.DataFrame:
    """Scarica o aggiorna l'archivio 5m di un simbolo, saltando i mesi gia' presenti."""
    existing = load_klines(symbol, BASE_INTERVAL, store_dir)
    now = pd.Timestamp.utcnow().tz_localize(None)

    if existing.empty:
        first_month = pd.Timestamp(FIRST_MONTH + "-01")
    else:
        # Si riparte dal mese dell'ultima candela: quel mese va riscaricato perche' quando e'
        # stato salvato poteva essere incompleto.
        first_month = existing.index[-1].normalize().replace(day=1)

    last_complete_month = (now.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
    months = _months_between(first_month, last_complete_month)
    days = [
        stamp.strftime("%Y-%m-%d")
        for stamp in pd.date_range(now.replace(day=1).normalize(), now.normalize() - pd.Timedelta(days=1), freq="D")
    ]

    def fetch(job):
        kind, stamp = job
        try:
            return _fetch_dump(symbol, kind, stamp)
        except _DumpMissing:
            return None

    jobs = [("monthly", stamp) for stamp in months] + [("daily", stamp) for stamp in days]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # `map` propaga le eccezioni: un errore di rete o di formato fa fallire l'aggiornamento
        # del simbolo invece di produrre un archivio troncato che sembra completo.
        downloaded = list(pool.map(fetch, jobs))

    frames = [frame for frame in downloaded if frame is not None and not frame.empty]
    if not existing.empty:
        frames.append(existing)
    if not frames:
        return existing

    step = pd.Timedelta(minutes=interval_to_minutes(BASE_INTERVAL))

    def consolidate(parts: list[pd.DataFrame]) -> pd.DataFrame:
        frame = pd.concat(parts)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        # L'ultima candela puo' essere ancora in formazione: archiviarla significherebbe salvare
        # un massimo/minimo destinato a cambiare.
        return frame[frame.index + step <= now]

    path = store_path(symbol, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    combined = consolidate(frames)
    combined.to_parquet(path)

    # La coda REST copre solo le ore che i dump non hanno ancora pubblicato. E' best-effort e
    # arriva dopo il salvataggio: un timeout dell'API non deve far perdere il download appena
    # completato, che puo' valere un'ora di lavoro.
    try:
        tail = _fetch_recent_from_api(symbol, combined.index[-1] + step)
    except Exception as exc:
        print(f"(coda REST non recuperata: {exc}) ", end="")
        return combined

    if not tail.empty:
        combined = consolidate([combined, tail])
        combined.to_parquet(path)
    return combined


def update_store(
    symbols: list[str] | None = None,
    store_dir: Path = MARKET_DATA_DIR,
    workers: int = DOWNLOAD_WORKERS,
) -> pd.DataFrame:
    """Aggiorna tutti i simboli richiesti. Un simbolo che fallisce non blocca gli altri."""
    symbols = symbols or DEFAULT_SYMBOLS
    started = time.time()
    for position, symbol in enumerate(symbols, start=1):
        before = len(load_klines(symbol, BASE_INTERVAL, store_dir))
        print(f"[{position}/{len(symbols)}] {symbol}: {before} candele in archivio...", end=" ", flush=True)
        try:
            frame = update_symbol(symbol, store_dir, workers)
        except Exception as exc:
            print(f"SALTATO ({exc})", flush=True)
            continue
        if frame.empty:
            print("nessun dato disponibile", flush=True)
        else:
            print(
                f"+{len(frame) - before} -> {len(frame)} ({frame.index[0]:%Y-%m-%d} .. {frame.index[-1]:%Y-%m-%d})",
                flush=True,
            )
    print(f"\nAggiornamento completato in {(time.time() - started) / 60:.1f} minuti")
    return store_manifest(store_dir)


def store_manifest(store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    """Copertura dello store: righe, periodo e dimensione per ogni simbolo."""
    rows = []
    for path in sorted(Path(store_dir).glob(f"*-{BASE_INTERVAL}.parquet")):
        frame = pd.read_parquet(path)
        rows.append(
            {
                "symbol": path.stem.rsplit("-", 1)[0],
                "rows": len(frame),
                "first": frame.index[0] if len(frame) else pd.NaT,
                "last": frame.index[-1] if len(frame) else pd.NaT,
                "MB": round(path.stat().st_size / 1e6, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gestisce lo store locale di candele Binance.")
    parser.add_argument("--update", action="store_true", help="scarica/aggiorna i simboli richiesti")
    parser.add_argument("--manifest", action="store_true", help="stampa la copertura dello store")
    parser.add_argument("--symbols", nargs="+", default=None, help=f"default: {len(DEFAULT_SYMBOLS)} simboli")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    parser.add_argument("--store-dir", default=str(MARKET_DATA_DIR))
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    if args.update:
        update_store(args.symbols, store_dir, args.workers)

    manifest = store_manifest(store_dir)
    if manifest.empty:
        print(f"Store vuoto: {store_dir}")
        return
    print(manifest.to_string(index=False))
    print(f"\n{manifest['rows'].sum():,} candele 5m, {manifest['MB'].sum():.0f} MB in {store_dir}")
    print(f"Intervalli derivabili per aggregazione: {', '.join(DERIVED_INTERVALS)}")


if __name__ == "__main__":
    main()
