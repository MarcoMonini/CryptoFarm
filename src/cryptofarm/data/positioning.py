"""Store locale del **posizionamento**: chi sta comprando, quanto è affollato, a che prezzo.

Gemello di `data/klines.py`, stessa infrastruttura di dump bulk, stesso store parquet. Cambia
cosa contiene: non i prezzi, ma le sole serie che questo progetto non aveva mai avuto.

## Perché esiste

Ogni misura fatta finora in questo repository parte da OHLCV, e `strategy.md` §14 registra come
non soddisfatto il punto «dati di microstruttura, l'unica informazione che il modello non ha mai
avuto». `data.binance.vision` pubblica per i perpetui, **a cinque minuti**, quattro serie che
descrivono il posizionamento aggregato — fra cui il rapporto fra volume aggressivo comprato e
venduto, cioè la stessa informazione che si estrarrebbe da `aggTrades` ma già aggregata e senza
scaricare centinaia di gigabyte. Più il funding rate, ogni otto ore.

## Quali colonne servono davvero, misurato prima di scriverle

Pannello su 5 asset × 2 finestre (2022-2024 e 2024-2026), IC di Spearman contro il rendimento a
cinque giorni a 4h. Su dodici derivate ne sopravvivono **due**, e dicono la stessa cosa:

| serie | \\|IC\\| medio | celle con segno concorde |
|---|---|---|
| `retail_accounts_ratio` | 0,061 | **10 su 10**, sempre negativo |
| `top_positions_ratio` | 0,065 | 8 su 10, negativo |
| `funding_rate`, `open_interest`, `taker_buy_sell_ratio` e derivate | 0,01-0,07 | 0-2 su 10 |

Cioè: **quando i lunghi sono affollati, i cinque giorni successivi rendono meno**. Il funding, che
era l'ipotesi ovvia e l'unico caso di studio cripto di ML4T, non passa il controllo di segno.

Lo store le conserva **tutte** lo stesso: arrivano nello stesso file, non costano niente in più, e
il giorno in cui si vuole rifare il pannello su un'altra finestra i dati ci sono già. Chi sceglie
le feature è `ml/features.py`, non questo modulo.

## L'allineamento temporale, che è il punto in cui si può barare

Un'istantanea marcata `00:00:00` è lo stato **all'inizio** della barra che apre alle 00:00, non
alla sua chiusura. Allinearla a quella barra usa quindi informazione più vecchia della decisione
che ci si prende sopra, mai più nuova: è conservativo per costruzione, ed è la ragione per cui
`load_positioning` fa `reindex(..., method="ffill")` e non un'interpolazione.

Il funding rate è una funzione a gradini: vale dal momento in cui viene applicato fino al
successivo, quindi il `ffill` è la sua rappresentazione esatta, non un'approssimazione.

## Copertura

I `metrics` esistono solo come dump **giornalieri** e cominciano nel 2021 per BTC/ETH, nel 2022
per il resto. Chi ne fa feature perde quindi il 2021: è il prezzo dichiarato di questa fonte.

    python -m cryptofarm.data.positioning --update
    python -m cryptofarm.data.positioning --manifest
"""

from __future__ import annotations

import argparse
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from cryptofarm.data.klines import (
    DEFAULT_SYMBOLS,
    DOWNLOAD_WORKERS,
    _DumpMissing,
    _months_between,
    _thread_session,
)
from cryptofarm.paths import MARKET_DATA_DIR

DUMP_BASE = "https://data.binance.vision/data/futures/um"
# I metrics non esistono prima di questa data su nessun simbolo: i mesi precedenti danno 404.
FIRST_DAY = "2021-01-01"
BASE_MINUTES = 5

# Nomi nostri, non quelli di Binance: `sum_toptrader_long_short_ratio` e
# `count_toptrader_long_short_ratio` differiscono di una parola e significano cose diverse
# (posizioni contro conti), ed e' il genere di coppia che si scambia leggendo in fretta.
_METRICS_RENAME = {
    "sum_open_interest": "open_interest",
    "sum_open_interest_value": "open_interest_value",
    "count_toptrader_long_short_ratio": "top_accounts_ratio",
    "sum_toptrader_long_short_ratio": "top_positions_ratio",
    "count_long_short_ratio": "retail_accounts_ratio",
    "sum_taker_long_short_vol_ratio": "taker_buy_sell_ratio",
}
COLUMNS = [*_METRICS_RENAME.values(), "funding_rate"]


def store_path(symbol: str, store_dir: Path = MARKET_DATA_DIR) -> Path:
    return Path(store_dir) / f"{symbol}-positioning.parquet"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="Open time"), dtype=float)


def _get(url: str, attempts: int = 3) -> bytes:
    """Scarica un dump. `_DumpMissing` sul 404, che qui significa «giorno non quotato»."""
    session = _thread_session()
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=120)
            if response.status_code == 404:
                raise _DumpMissing(url)
            response.raise_for_status()
            return response.content
        except _DumpMissing:
            raise
        except Exception as exc:  # rete o formato: si ritenta, non si tronca l'archivio
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"{url}: {last}")


def _read_zipped_csv(content: bytes) -> pd.DataFrame:
    archive = zipfile.ZipFile(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(archive.read(archive.namelist()[0])))


def fetch_metrics_day(symbol: str, day: str) -> pd.DataFrame | None:
    """Le sei serie a 5 minuti di un giorno, o `None` se quel giorno non esiste."""
    try:
        raw = _read_zipped_csv(_get(f"{DUMP_BASE}/daily/metrics/{symbol}/{symbol}-metrics-{day}.zip"))
    except _DumpMissing:
        return None
    raw["create_time"] = pd.to_datetime(raw["create_time"])
    frame = raw.set_index("create_time").rename(columns=_METRICS_RENAME)
    frame.index.name = "Open time"
    return frame[list(_METRICS_RENAME.values())].astype(float)


def fetch_funding_month(symbol: str, month: str) -> pd.Series | None:
    """Il funding rate di un mese, indicizzato al momento in cui viene applicato."""
    try:
        raw = _read_zipped_csv(_get(f"{DUMP_BASE}/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"))
    except _DumpMissing:
        return None
    # I dump vecchi usano `fundingTime`/`fundingRate`, i recenti `calc_time`/`last_funding_rate`.
    when = "calc_time" if "calc_time" in raw.columns else "fundingTime"
    rate = "last_funding_rate" if "last_funding_rate" in raw.columns else "fundingRate"
    index = (
        pd.to_datetime(raw[when], unit="ms") if pd.api.types.is_numeric_dtype(raw[when]) else pd.to_datetime(raw[when])
    )
    series = pd.Series(raw[rate].astype(float).to_numpy(), index=index, name="funding_rate")
    series.index.name = "Open time"
    return series.sort_index()


def load_positioning(
    symbol: str,
    index: pd.DatetimeIndex | None = None,
    store_dir: Path = MARKET_DATA_DIR,
) -> pd.DataFrame:
    """Legge lo store; con `index`, riporta le serie su quell'indice **senza guardare avanti**.

    `method="ffill"` prende l'ultima istantanea non successiva a ogni timestamp richiesto. Su un
    indice piu' lento della base a 5 minuti questo significa il valore all'**inizio** della barra,
    che e' informazione piu' vecchia della decisione presa alla sua chiusura -- mai piu' nuova.
    """
    path = store_path(symbol, store_dir)
    if not path.exists():
        return _empty() if index is None else pd.DataFrame(columns=COLUMNS, index=index, dtype=float)
    frame = pd.read_parquet(path)
    if index is None:
        return frame
    return frame.reindex(index, method="ffill")


def update_symbol(
    symbol: str,
    store_dir: Path = MARKET_DATA_DIR,
    workers: int = DOWNLOAD_WORKERS,
) -> pd.DataFrame:
    """Scarica o aggiorna un simbolo, saltando i giorni gia' in archivio."""
    existing = load_positioning(symbol, store_dir=store_dir)
    now = pd.Timestamp.utcnow().tz_localize(None)
    ieri = now.normalize() - pd.Timedelta(days=1)

    # Si riparte dall'ultimo giorno presente e non dal successivo: quel giorno poteva essere
    # incompleto quando e' stato salvato.
    inizio = pd.Timestamp(FIRST_DAY) if existing.empty else existing.index[-1].normalize()
    giorni = [d.strftime("%Y-%m-%d") for d in pd.date_range(inizio, ieri, freq="D")]
    mesi = _months_between(inizio.replace(day=1), ieri)
    if not giorni:
        return existing

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # `map` propaga: un errore di rete fa fallire il simbolo invece di produrre un archivio
        # troncato che sembra completo. E' la stessa scelta di `klines.update_symbol`.
        parti = [p for p in pool.map(lambda d: fetch_metrics_day(symbol, d), giorni) if p is not None]
        fondi = [s for s in pool.map(lambda m: fetch_funding_month(symbol, m), mesi) if s is not None]

    if not parti and existing.empty:
        return existing

    metriche = pd.concat(parti) if parti else pd.DataFrame(columns=list(_METRICS_RENAME.values()))
    if not existing.empty:
        metriche = pd.concat([existing[list(_METRICS_RENAME.values())], metriche])
    metriche = metriche[~metriche.index.duplicated(keep="last")].sort_index()

    funding = pd.concat(fondi) if fondi else pd.Series(dtype=float, name="funding_rate")
    if not existing.empty and existing["funding_rate"].notna().any():
        funding = pd.concat([existing["funding_rate"].dropna(), funding])
    funding = funding[~funding.index.duplicated(keep="last")].sort_index()

    frame = metriche.copy()
    # Il funding e' una funzione a gradini fra un'applicazione e la successiva: `ffill` la
    # rappresenta esattamente. Le righe prima del primo funding restano NaN, che e' il vero.
    frame["funding_rate"] = funding.reindex(frame.index, method="ffill") if len(funding) else float("nan")
    frame = frame[COLUMNS].astype(float)
    frame.index.name = "Open time"

    path = store_path(symbol, store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return frame


def update_store(
    symbols: list[str] | None = None,
    store_dir: Path = MARKET_DATA_DIR,
    workers: int = DOWNLOAD_WORKERS,
) -> pd.DataFrame:
    """Aggiorna tutti i simboli richiesti. Un simbolo che fallisce non blocca gli altri."""
    symbols = symbols or DEFAULT_SYMBOLS
    started = time.time()
    for position, symbol in enumerate(symbols, start=1):
        prima = len(load_positioning(symbol, store_dir=store_dir))
        print(f"[{position}/{len(symbols)}] {symbol}: {prima} righe in archivio...", end=" ", flush=True)
        try:
            frame = update_symbol(symbol, store_dir, workers)
        except Exception as exc:
            print(f"SALTATO ({exc})", flush=True)
            continue
        if frame.empty:
            print("nessun dato disponibile (perpetuo non quotato?)", flush=True)
        else:
            print(
                f"+{len(frame) - prima} -> {len(frame)} ({frame.index[0]:%Y-%m-%d} .. {frame.index[-1]:%Y-%m-%d})",
                flush=True,
            )
    print(f"\nAggiornamento completato in {(time.time() - started) / 60:.1f} minuti")
    return store_manifest(store_dir)


def store_manifest(store_dir: Path = MARKET_DATA_DIR) -> pd.DataFrame:
    """Copertura dello store: righe, periodo, colonne piene e dimensione per simbolo."""
    rows = []
    for path in sorted(Path(store_dir).glob("*-positioning.parquet")):
        frame = pd.read_parquet(path)
        rows.append(
            {
                "symbol": path.stem.rsplit("-", 1)[0],
                "rows": len(frame),
                "first": frame.index[0] if len(frame) else pd.NaT,
                "last": frame.index[-1] if len(frame) else pd.NaT,
                "funding_%": round(100 * frame["funding_rate"].notna().mean(), 1) if len(frame) else 0.0,
                "MB": round(path.stat().st_size / 1e6, 1),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Store locale del posizionamento sui perpetui Binance.")
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--symbols", nargs="+", default=None, help=f"default: {len(DEFAULT_SYMBOLS)} simboli")
    parser.add_argument("--workers", type=int, default=DOWNLOAD_WORKERS)
    parser.add_argument("--store-dir", default=str(MARKET_DATA_DIR))
    args = parser.parse_args()

    store_dir = Path(args.store_dir)
    if args.update:
        update_store(args.symbols, store_dir, args.workers)

    manifest = store_manifest(store_dir)
    if manifest.empty:
        print(f"Nessuno store di posizionamento in {store_dir}")
        return
    print(manifest.to_string(index=False))
    print(f"\n{manifest['rows'].sum():,} righe a {BASE_MINUTES}m, {manifest['MB'].sum():.0f} MB in {store_dir}")


if __name__ == "__main__":
    main()
