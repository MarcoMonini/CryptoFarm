"""La politica a due azioni: allineamento, costo e memoria della posizione.

Tre proprieta' che si rompono in silenzio e che nessuna misura di rendimento segnalerebbe:
la transizione deve leggere lo stato **all'inizio** del passo e il rendimento **del** passo; il
costo deve essere pagato per ogni cambio e una volta sola; e la posizione precedente deve stare
davvero nello stato, altrimenti l'agente e' un classificatore per barra travestito.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptofarm.ml.rl import (
    FEE,
    fitted_q,
    posizioni,
    rendimento,
    transizioni_simbolo,
    unisci,
)


@pytest.fixture()
def serie():
    rng = np.random.default_rng(1)
    n = 900
    close = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    features = np.column_stack([np.arange(n, dtype=float), rng.normal(size=n)])
    return features, close


def test_la_transizione_legge_lo_stato_prima_del_passo(serie):
    """`stato[k]` e' la barra di decisione, `logret[k]` il tratto che viene dopo. Se scivolassero
    di uno la politica leggerebbe il futuro, e il rendimento sembrerebbe ottimo."""
    features, close = serie
    t = transizioni_simbolo(features, close, cadenza=10, fasi=1)
    d = np.arange(0, len(close) - 10, 10)
    assert np.array_equal(t.stato[:, 0], features[d, 0])
    assert np.allclose(t.logret, np.log(close[d + 10] / close[d]))
    assert np.array_equal(t.successivo[:, 0], features[d + 10, 0])


def test_le_sfasature_moltiplicano_senza_uscire_dalla_serie(serie):
    features, close = serie
    una = transizioni_simbolo(features, close, cadenza=12, fasi=1)
    otto = transizioni_simbolo(features, close, cadenza=12, fasi=4)
    assert 3 * len(una) < len(otto) <= 4 * len(una) + 4
    assert np.isfinite(otto.logret).all()


def test_una_colonna_mancante_non_butta_la_riga(serie):
    """HistGradientBoosting tratta i NaN da solo: scartare la riga intera toglieva l'85% del
    campione, perche' le colonne di posizionamento mancano per interi anni."""
    features, close = serie
    bucato = features.copy()
    bucato[:, 1] = np.nan
    assert len(transizioni_simbolo(bucato, close, cadenza=10, fasi=1)) == len(
        transizioni_simbolo(features, close, cadenza=10, fasi=1)
    )
    # Una riga senza nessun valore invece non e' uno stato, e va via.
    tutto_nan = features.copy()
    tutto_nan[0] = np.nan
    assert (
        len(transizioni_simbolo(tutto_nan, close, cadenza=10, fasi=1))
        == len(transizioni_simbolo(features, close, cadenza=10, fasi=1)) - 1
    )


def test_il_rendimento_paga_un_lato_per_cambio():
    logret = np.log(np.array([1.10, 1.0, 1.05]))
    # dentro, dentro, fuori: un ingresso all'inizio e un'uscita alla fine, due lati in tutto.
    azioni = np.array([1, 1, 0], dtype=np.int8)
    atteso = np.exp(logret[0] + logret[1] - 2 * FEE) - 1
    assert rendimento(azioni, logret) == pytest.approx(atteso * 100)
    # Il possesso passivo paga un lato solo, perche' non chiude.
    assert rendimento(np.ones(3, dtype=np.int8), logret) == pytest.approx((np.exp(logret.sum() - FEE) - 1) * 100)


def test_il_costo_allarga_la_banda_di_non_fare():
    """L'unica prova che la posizione stia nello stato: se lo stato la ignorasse, il costo non
    potrebbe cambiare il numero di cambi -- deciderebbe barra per barra."""
    rng = np.random.default_rng(0)
    n = 4000
    segnale = rng.normal(size=n)
    close = np.exp(np.cumsum(np.concatenate([[0.0], 0.02 * segnale + 0.01 * rng.normal(size=n)])))
    feat = np.column_stack([segnale, rng.normal(size=n)])
    batch = transizioni_simbolo(np.vstack([feat, feat[-1]]), close, cadenza=1)

    gratis = posizioni(fitted_q(batch, giri=1, costo=0.0, max_iter=50), batch.stato)
    caro = posizioni(fitted_q(batch, giri=1, costo=0.05, max_iter=50), batch.stato)
    assert np.abs(np.diff(caro)).sum() < np.abs(np.diff(gratis)).sum()
    # E il segnale resta seguito quando e' gratis muoversi.
    assert gratis[batch.stato[:, 0] > 0.5].mean() > 0.8


def test_unisci_rifiuta_un_batch_vuoto():
    with pytest.raises(ValueError):
        unisci([])
