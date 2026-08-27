"""Allineamento fra intervalli, senza look-ahead.

Una strategia che legge il quadro macro a un giorno e decide su barre da quindici minuti ha un
solo modo di sbagliare in maniera invisibile: usare una barra lunga **prima che sia chiusa**.

`data.klines.resample_klines` etichetta le barre a sinistra, come Binance: la barra 1D del
2024-03-05 copre l'intera giornata e il suo Close si conosce solo alle 00:00 del 2024-03-06. Alle
10:00 del 2024-03-05 quella barra **non esiste ancora**. Il modo naturale di scrivere
l'allineamento --

    daily_series.reindex(index_15m, method="ffill")

-- restituisce proprio quella barra, cioe' inietta nella decisione delle 10:00 il risultato del
resto della giornata. Il backtest ne esce spettacolare e completamente falso, e nessuna delle
protezioni gia' in casa lo vede: il test che tronca la serie fra le barre non lo intercetta,
perche' la barra lunga incriminata resta identica in entrambe le troncature.

`align_to_lower` sposta la serie sul proprio **istante di disponibilita'** prima di propagarla.
E' l'unico posto in cui questa aritmetica va scritta: chi allinea a mano la rifa' e prima o poi
la sbaglia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes


def align_to_lower(
    higher: pd.Series | np.ndarray,
    higher_index: pd.DatetimeIndex,
    higher_interval: str,
    lower_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Porta una serie calcolata su `higher_interval` sull'indice di barre piu' corte.

    Il valore della barra lunga etichettata `d` diventa disponibile a `d + durata`, e da li' in
    poi resta il valore corrente fino alla chiusura della successiva. Le barre corte precedenti
    alla prima chiusura lunga restano **NaN**, ed e' voluto: prima di quel momento il quadro
    macro non e' noto e chi chiama deve astenersi, non indovinare.
    """
    values = higher.to_numpy() if isinstance(higher, pd.Series) else np.asarray(higher)
    if len(values) != len(higher_index):
        raise ValueError(f"serie di {len(values)} valori contro un indice di {len(higher_index)}")
    disponibile = pd.Series(
        values,
        index=pd.DatetimeIndex(higher_index) + pd.Timedelta(minutes=interval_to_minutes(higher_interval)),
    )
    return disponibile.reindex(pd.DatetimeIndex(lower_index), method="ffill").to_numpy()


def _selfcheck() -> None:
    """Il controllo che distingue l'allineamento corretto da quello ingenuo."""
    giorni = pd.date_range("2024-03-01", periods=5, freq="1D")
    # Ogni giorno vale il proprio numero: cosi' si legge subito *quale* barra e' stata usata.
    valori = np.arange(len(giorni), dtype=float)
    quarti = pd.date_range("2024-03-01", periods=5 * 96, freq="15min")

    allineato = align_to_lower(valori, giorni, "1d", quarti)
    serie = pd.Series(allineato, index=quarti)

    # 1. Il primo giorno intero non ha nessuna barra 1d chiusa alle spalle: deve restare NaN.
    assert serie.loc["2024-03-01"].isna().all(), "il primo giorno non puo' avere un valore"

    # 2. Alle 10:00 del giorno 2 il valore disponibile e' quello del giorno 1, non del giorno 2.
    assert serie.loc["2024-03-02 10:00"] == 0.0, serie.loc["2024-03-02 10:00"]

    # 3. Esattamente a mezzanotte la barra appena chiusa e' disponibile.
    assert serie.loc["2024-03-02 00:00"] == 0.0
    assert serie.loc["2024-03-03 00:00"] == 1.0

    # 4. L'allineamento ingenuo sbaglia proprio dove conta, ed e' il motivo di questo modulo.
    ingenuo = pd.Series(valori, index=giorni).reindex(quarti, method="ffill")
    assert ingenuo.loc["2024-03-02 10:00"] == 1.0, "l'ingenuo dovrebbe usare la barra non chiusa"
    assert ingenuo.loc["2024-03-02 10:00"] != serie.loc["2024-03-02 10:00"]

    # 5. Troncare le barre corte non cambia i valori gia' emessi.
    meta = quarti[: len(quarti) // 2]
    assert np.array_equal(align_to_lower(valori, giorni, "1d", meta), allineato[: len(meta)], equal_nan=True)

    print("mtf selfcheck: 5 controlli passati")


if __name__ == "__main__":
    _selfcheck()
