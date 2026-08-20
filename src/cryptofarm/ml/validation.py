"""Cross-validation per serie temporali con etichette che si sovrappongono.

Il k-fold standard qui non e' una semplificazione, e' un errore: le etichette triple-barrier
vivono da `t` a `t_exit`, quindi due osservazioni le cui vite si intersecano condividono futuro.
Metterne una in training e l'altra in test significa valutare su informazione gia' vista, e il
risultato e' una stima ottimista che non sopravvive al mercato reale.

Tre difese, tutte necessarie e nessuna sufficiente da sola:

- **Purging**: dal training si rimuove ogni osservazione la cui vita interseca quella di una
  qualunque osservazione di test.
- **Embargo**: si scarta anche una finestra temporale subito dopo il test, perche' oltre alla
  sovrapposizione esplicita resta l'autocorrelazione seriale.
- **Pesi di unicita'**: un'osservazione che condivide la propria vita con altre venti porta molta
  meno informazione di una isolata, e pesarle uguali gonfia la fiducia in cio' che si e' misurato.

`CombinatorialPurgedCV` estende il purged k-fold a piu' gruppi di test per split, restituendo una
**distribuzione** di performance out-of-sample invece di un punto. E' quella distribuzione che
rende calcolabile la probabilita' di backtest overfitting.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _as_arrays(t_start: pd.Series, t_exit: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    return t_start.to_numpy(), t_exit.to_numpy()


def purge_train_indices(
    train: np.ndarray,
    test: np.ndarray,
    t_start: np.ndarray,
    t_exit: np.ndarray,
    embargo: np.timedelta64,
) -> np.ndarray:
    """Rimuove dal training le osservazioni che si sovrappongono al test, piu' l'embargo.

    Un'osservazione di training sopravvive solo se la sua vita `[t_start, t_exit]` non interseca
    l'intervallo coperto dal test, allargato dell'embargo dal lato successivo.
    """
    if len(test) == 0 or len(train) == 0:
        return train
    test_from = t_start[test].min()
    test_to = t_exit[test].max() + embargo
    overlaps = (t_exit[train] >= test_from) & (t_start[train] <= test_to)
    return train[~overlaps]


class PurgedKFold:
    """K-fold cronologico con purging ed embargo.

    I fold sono blocchi temporali contigui, non campioni casuali: mescolare le righe di una serie
    temporale distrugge proprio la struttura che si vuole validare.
    """

    def __init__(self, n_splits: int = 5, embargo: pd.Timedelta = pd.Timedelta(0)):
        if n_splits < 2:
            raise ValueError("servono almeno 2 fold")
        self.n_splits = n_splits
        self.embargo = np.timedelta64(embargo)

    def split(self, t_start: pd.Series, t_exit: pd.Series):
        starts, exits = _as_arrays(t_start, t_exit)
        order = np.argsort(starts, kind="stable")
        for block in np.array_split(order, self.n_splits):
            test = np.sort(block)
            train = np.setdiff1d(np.arange(len(starts)), test, assume_unique=False)
            yield purge_train_indices(train, test, starts, exits, self.embargo), test

    def get_n_splits(self) -> int:
        return self.n_splits


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation.

    La serie viene divisa in `n_groups` blocchi temporali; ogni split usa come test una
    combinazione di `n_test_groups` blocchi, e come training tutto il resto dopo purging ed
    embargo. Gli split sono quindi `C(n_groups, n_test_groups)`.

    Il punto non e' avere piu' stime, e' averne una **distribuzione**: con un solo split non si
    puo' distinguere un modello che funziona da uno fortunato, e non si puo' calcolare il PBO.
    Con blocchi da circa un anno ogni combinazione vede inoltre una miscela di regimi diversa,
    che e' esattamente cio' che rende la distribuzione informativa e non un artefatto di un
    singolo ciclo di mercato.
    """

    def __init__(
        self,
        n_groups: int = 8,
        n_test_groups: int = 2,
        embargo: pd.Timedelta = pd.Timedelta(0),
    ):
        if n_test_groups >= n_groups:
            raise ValueError("i gruppi di test devono essere meno del totale")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = np.timedelta64(embargo)

    def get_n_splits(self) -> int:
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(self, t_start: pd.Series, t_exit: pd.Series):
        starts, exits = _as_arrays(t_start, t_exit)
        order = np.argsort(starts, kind="stable")
        groups = [np.sort(block) for block in np.array_split(order, self.n_groups)]
        everything = np.arange(len(starts))

        for chosen in combinations(range(self.n_groups), self.n_test_groups):
            test = np.sort(np.concatenate([groups[g] for g in chosen]))
            train = np.setdiff1d(everything, test, assume_unique=False)
            # Purging per ogni blocco di test separatamente: blocchi non contigui coprono
            # intervalli temporali disgiunti, e trattarli come uno solo scarterebbe tutto cio'
            # che sta in mezzo.
            for g in chosen:
                train = purge_train_indices(train, groups[g], starts, exits, self.embargo)
            yield train, test


def sample_uniqueness(t_start: pd.Series, t_exit: pd.Series) -> np.ndarray:
    """Quanto e' unica ogni osservazione: 1 = vive da sola, 0.1 = condivide con altre nove.

    Calcolata come media, sulla vita dell'osservazione, del reciproco del numero di osservazioni
    contemporaneamente attive. E' il peso corretto da dare in addestramento: senza, le zone dense
    di eventi sovrapposti contano molte volte la stessa informazione.

    Implementazione a somme cumulate: il conteggio di concorrenza si ottiene da un array di
    incrementi (+1 all'inizio, -1 dopo la fine) e la media sulla vita da una seconda cumulata,
    quindi il costo e' lineare invece che quadratico -- necessario su milioni di righe.
    """
    starts = t_start.to_numpy()
    exits = t_exit.to_numpy()
    timeline = np.unique(np.concatenate([starts, exits]))
    begin = np.searchsorted(timeline, starts, side="left")
    end = np.searchsorted(timeline, exits, side="right")

    concurrency = np.zeros(len(timeline) + 1, dtype=np.int64)
    np.add.at(concurrency, begin, 1)
    np.add.at(concurrency, end, -1)
    concurrency = np.cumsum(concurrency)[: len(timeline)]

    inverse = np.zeros(len(timeline), dtype=float)
    active = concurrency > 0
    inverse[active] = 1.0 / concurrency[active]
    cumulative = np.concatenate([[0.0], np.cumsum(inverse)])

    span = np.maximum(end - begin, 1)
    return (cumulative[end] - cumulative[begin]) / span


def probability_of_backtest_overfitting(
    in_sample: np.ndarray,
    out_of_sample: np.ndarray,
) -> float:
    """PBO: quanto spesso la configurazione migliore in-sample finisce sotto la mediana fuori.

    `in_sample` e `out_of_sample` sono matrici (split x configurazioni) della stessa metrica.
    Per ogni split si sceglie la configurazione migliore in-sample e si guarda il suo rango
    out-of-sample.

    **Un PBO sopra 0,5 significa che la procedura di selezione fa peggio di una scelta a caso**,
    ed e' un risultato che va riportato accanto a qualunque performance, non tenuto da parte.
    """
    in_sample = np.asarray(in_sample, dtype=float)
    out_of_sample = np.asarray(out_of_sample, dtype=float)
    if in_sample.shape != out_of_sample.shape:
        raise ValueError("in-sample e out-of-sample devono avere la stessa forma")
    if in_sample.shape[1] < 2:
        return float("nan")

    below = 0
    for split in range(in_sample.shape[0]):
        best = int(np.nanargmax(in_sample[split]))
        rank = np.mean(out_of_sample[split] <= out_of_sample[split][best])
        if rank <= 0.5:
            below += 1
    return below / in_sample.shape[0]


def deflated_sharpe_ratio(
    returns: np.ndarray,
    trials: int,
    benchmark: float = 0.0,
) -> float:
    """Probabilita' che lo Sharpe osservato sia reale e non il massimo di `trials` tentativi.

    Testando molte configurazioni il massimo Sharpe osservato e' distorto verso l'alto anche se
    nessuna ha edge. Il DSR corregge per il numero di prove, per asimmetria e curtosi dei
    rendimenti -- entrambe rilevanti qui, perche' con barriere 2:1 i rendimenti sono asimmetrici
    per costruzione -- e per la lunghezza della serie.

    `trials` va contato **onestamente**, includendo le configurazioni provate e scartate: e'
    l'errore piu' comune nell'applicare questa correzione.
    """
    from scipy.stats import kurtosis, norm, skew

    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]
    n = len(returns)
    if n < 10 or returns.std(ddof=1) == 0:
        return float("nan")

    observed = returns.mean() / returns.std(ddof=1)
    skewness = float(skew(returns))
    excess_kurtosis = float(kurtosis(returns, fisher=True))

    # Sharpe atteso come massimo di `trials` estrazioni indipendenti a edge nullo.
    euler = 0.5772156649
    if trials > 1:
        expected_max = (1 - euler) * norm.ppf(1 - 1 / trials) + euler * norm.ppf(1 - 1 / (trials * np.e))
    else:
        expected_max = 0.0
    threshold = benchmark + expected_max / np.sqrt(n - 1)

    denominator = np.sqrt(1 - skewness * observed + (excess_kurtosis) / 4 * observed**2)
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan")
    return float(norm.cdf((observed - threshold) * np.sqrt(n - 1) / denominator))
