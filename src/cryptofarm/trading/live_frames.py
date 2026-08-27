"""Barre lunghe **in formazione**, ricostruite a ogni barra breve come le vedrebbe il bot live.

Il bot live, alle 10:00, non aspetta la mezzanotte per sapere qualcosa della giornata: vede una
barra 1D aperta all'apertura, con massimo e minimo correnti e chiusura provvisoria uguale
all'ultimo prezzo. Quella barra parziale **non e' look-ahead**, perche' e' costruita solo con dati
fino alle 10:00 -- ed e' una cosa diversa dalla barra 1D *completa* di quel giorno, che invece lo
sarebbe (`mtf.align_to_lower` serve a quello, e resta il modo giusto di leggere le barre chiuse).

La differenza fra i due conta anche in quantita': aspettare la chiusura giornaliera vuol dire
reagire fino a ventiquattro ore dopo, e la maggior parte dei segnali muore in quell'attesa.

## Il costo, e perche' non serve un ciclo

La ricostruzione ingenua rifa' l'aggregazione e gli indicatori a ogni barra breve: su cinque anni
a 15 minuti sono ~175.000 passi, ognuno che ripercorre tutta la storia. E' quadratico e non si
esegue.

Non serve, perche' la barra in formazione ha forma chiusa e **vettoriale**: dentro il proprio
periodo l'apertura e' la prima, il massimo e' il massimo corrente, il minimo il minimo corrente,
la chiusura e' il prezzo di adesso e il volume la somma corrente. `groupby` + `cummax`/`cummin`/
`cumsum` le producono tutte in O(N) senza nessun ciclo Python.

E la parte cara **non dipende da nessun parametro di strategia**: si calcola una volta per
(simbolo, intervallo) e si riusa su tutta la griglia. Sopra ci vanno gli indicatori, che invece i
parametri li hanno, e che si sollevano a valore provvisorio in O(1) ciascuno tenendo lo stato
ricorsivo all'**ultima chiusura** e combinandolo con la barra parziale -- senza mai committarlo
finche' il periodo non chiude davvero (`provisional_ema` e' l'esempio del motivo).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes, resample_klines
from cryptofarm.trading.mtf import align_to_lower


class FormingBars(NamedTuple):
    """OHLCV della barra lunga in formazione, un valore per ogni barra breve.

    `closes_here` marca le barre brevi che **chiudono** il periodo lungo: e' li' che uno stato
    ricorsivo va committato, e solo li'. L'ultima barra della serie la marca solo se il periodo e'
    davvero completo, cosi' una storia che finisce a meta' giornata non finalizza una barra finta.
    """

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    bar_id: np.ndarray
    closes_here: np.ndarray


def forming_bars(candles: pd.DataFrame, interval: str) -> FormingBars:
    """Ricostruisce la barra in formazione di `interval` a ogni barra di `candles`.

    L'`bar_id` e' il numero di periodi dall'epoca, la stessa ancora che usa `resample_klines`:
    la barra in formazione all'ultima barra breve del periodo coincide **esattamente** con la
    barra aggregata, ed e' cio' che il selfcheck verifica.
    """
    minuti = interval_to_minutes(interval)
    minuti_epoca = candles.index.values.astype("datetime64[m]").astype("int64")
    bar_id = minuti_epoca // minuti
    gruppi = candles.groupby(bar_id)

    passo = int(np.median(np.diff(minuti_epoca))) if len(minuti_epoca) > 1 else minuti
    # Chiude il periodo chi ha un successore in un altro periodo; l'ultima barra solo se il suo
    # successore *cadrebbe* fuori, cioe' se il periodo e' completo davvero.
    closes_here = np.empty(len(candles), dtype=bool)
    closes_here[:-1] = bar_id[1:] != bar_id[:-1]
    closes_here[-1] = (minuti_epoca[-1] + passo) >= (bar_id[-1] + 1) * minuti

    return FormingBars(
        open=gruppi["Open"].transform("first").to_numpy(),
        high=gruppi["High"].cummax().to_numpy(),
        low=gruppi["Low"].cummin().to_numpy(),
        close=candles["Close"].to_numpy(),
        volume=gruppi["Volume"].cumsum().to_numpy(),
        bar_id=bar_id,
        closes_here=closes_here,
    )


def provisional_ema(
    closed_ema: np.ndarray,
    closed_index: pd.DatetimeIndex,
    interval: str,
    lower_index: pd.DatetimeIndex,
    close_now: np.ndarray,
    span: int,
) -> np.ndarray:
    """EMA lunga aggiornata con la barra in formazione, in O(1) per barra breve.

    E' il modello di tutti gli indicatori ricorsivi (EMA, ATR di Wilder, KAMA, ADX): lo stato
    resta fermo all'ultima chiusura, e il valore provvisorio si ricava combinandolo con la barra
    parziale. Nessuno stato viene aggiornato prima che il periodo chiuda, quindi la barra
    provvisoria non puo' contaminare la storia.
    """
    alpha = 2.0 / (span + 1)
    base = align_to_lower(closed_ema, closed_index, interval, lower_index)
    return alpha * np.asarray(close_now, dtype=float) + (1 - alpha) * base


def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-01", periods=6 * 96, freq="15min", name="Open time")
    passeggiata = 100 + np.cumsum(rng.normal(0, 0.5, len(idx)))
    candele = pd.DataFrame(
        {
            "Open": passeggiata,
            "High": passeggiata + rng.random(len(idx)),
            "Low": passeggiata - rng.random(len(idx)),
            "Close": passeggiata + rng.normal(0, 0.2, len(idx)),
            "Volume": rng.random(len(idx)) * 10,
        },
        index=idx,
    )

    for interval in ("1h", "4h", "1d"):
        f = forming_bars(candele, interval)
        aggregate = resample_klines(candele, interval)

        # 1. Alla chiusura del periodo la barra in formazione E' la barra aggregata, esattamente.
        alla_chiusura = pd.DataFrame(
            {"Open": f.open, "High": f.high, "Low": f.low, "Close": f.close, "Volume": f.volume},
            index=idx,
        )[f.closes_here]
        alla_chiusura.index = pd.to_datetime(f.bar_id[f.closes_here] * interval_to_minutes(interval), unit="m")
        atteso = aggregate.loc[alla_chiusura.index]
        assert np.allclose(alla_chiusura.to_numpy(), atteso.to_numpy()), interval

        # 2. Dentro il periodo la barra parziale e' contenuta in quella finale: mai piu' estrema.
        finale_high = align_to_lower(aggregate["High"], aggregate.index, interval, idx)
        visti = ~np.isnan(finale_high)
        # confronta la parziale con la finale *dello stesso* periodo, presa dall'aggregato
        per_id = aggregate.set_index(
            aggregate.index.values.astype("datetime64[m]").astype("int64") // interval_to_minutes(interval)
        )
        assert (f.high <= per_id["High"].reindex(f.bar_id).to_numpy() + 1e-9).all(), interval
        assert (f.low >= per_id["Low"].reindex(f.bar_id).to_numpy() - 1e-9).all(), interval
        assert visti.any()

        # 3. La chiusura provvisoria e' sempre il prezzo corrente: e' cio' che la rende reattiva.
        assert np.array_equal(f.close, candele["Close"].to_numpy())

    # 4. Troncare la storia non cambia niente di gia' emesso: nessuna barra futura entra.
    meta = len(idx) // 2
    intero = forming_bars(candele, "4h")
    troncato = forming_bars(candele.iloc[:meta], "4h")
    assert np.allclose(troncato.high, intero.high[:meta])
    assert np.allclose(troncato.volume, intero.volume[:meta])

    # 5. Una storia che finisce a meta' periodo non finalizza una barra incompleta.
    parziale = forming_bars(candele.iloc[: meta + 3], "1d")
    assert not parziale.closes_here[-1]

    print("live_frames selfcheck: 5 controlli passati")


if __name__ == "__main__":
    _selfcheck()
