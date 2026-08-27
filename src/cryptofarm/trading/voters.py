"""Da cambi di posizione a **voto per barra**, con memoria e decadimento.

Le strategie di `strategies_ls` emettono eventi: «alle 14:00 vai lungo». Un insieme a confluenza
ha invece bisogno di sapere, **a ogni barra**, cosa dice ciascun votante e quanto ci crede. Sono
due domande diverse e questo modulo fa il passaggio, in due pezzi separati apposta:

1. `held_state` -- lo stato tenuto, in `{-1, 0, +1}`, propagando gli eventi in avanti. E' la
   posizione che quella strategia avrebbe adesso, niente di piu';
2. `decayed_vote` -- il voto, in `[-1, +1]`, che vale 1 sulla barra in cui il votante *scatta* e
   poi sfuma.

## Perche' il voto sfuma invece di restare acceso

Un segnale non vale solo sulla barra in cui scatta -- ma non vale nemmeno per sempre. Senza
memoria, un voto a 4H e uno a 1H non cadono quasi mai sulla stessa barra da quindici minuti e la
confluenza **non innesca mai**: la memoria converte «conferme simultanee», rare, in «conferme
entro una finestra», frequenti. E' il meccanismo che fa *aumentare* le occasioni, non un filtro.

Il decadimento e' l'altra meta': senza, una strategia che tiene una posizione per mesi voterebbe
a piena forza per mesi, e l'insieme diventerebbe quella strategia con delle decorazioni.

```
v(t) = stato(t)            se lo stato cambia a t verso un verso (il votante scatta)
v(t) = v(t-1) * lambda     altrimenti,  e 0 sotto epsilon
```

Un'inversione diretta scatta di nuovo, quindi riparte a forza piena col verso nuovo. Un ritorno a
flat **non** azzera: il segnale precedente sfuma come gli altri, perche' uscire non e' un segnale
contrario, e' un'assenza.

`half_life_bars` e' l'unico parametro, e nel disegno e' **globale**: si esprime in barre del
timeframe del votante e si converte in barre dell'indice passato moltiplicando per il rapporto
degli intervalli -- `emivita * interval_to_minutes("4h") / interval_to_minutes("15m")` per un
votante a 4H letto su un indice a 15 minuti. Cosi' un segnale giornaliero resta vivo per giorni e
uno a quindici minuti per ore, con un numero solo invece di sei.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAI = -2  # sentinella: su questa barra non c'e' nessun cambio di posizione


def held_state(events: list, index: pd.DatetimeIndex) -> np.ndarray:
    """Lo stato tenuto a ogni barra di `index`, dai cambi di posizione `events`.

    `events` e' il formato di `strategies_ls`: `(timestamp, prezzo, obiettivo)` con obiettivo in
    `{+1, 0, -1}`. La decisione si prende alla chiusura della barra, quindi vale **da quella barra
    stessa** -- la stessa convenzione con cui `pnl.simulate_positions` la esegue.

    Prima del primo evento lo stato e' 0. Un timestamp che non cade esattamente su una barra
    dell'indice e' un errore e viene segnalato: quasi sempre vuol dire che il votante gira su un
    indice diverso da quello su cui lo si sta leggendo, ed e' un disallineamento che passerebbe
    silenzioso rovinando ogni misura a valle.
    """
    index = pd.DatetimeIndex(index)
    stato = np.zeros(len(index), dtype=np.int8)
    if not events or len(index) == 0:
        return stato

    quando = pd.DatetimeIndex([e[0] for e in events])
    dove = index.searchsorted(quando)
    if (dove >= len(index)).any() or (index[np.minimum(dove, len(index) - 1)] != quando).any():
        raise ValueError("un evento non cade su nessuna barra dell'indice: votante disallineato")

    # L'ultimo evento di una stessa barra vince: e' l'ordine in cui li ha emessi la strategia.
    cambio = np.full(len(index), _MAI, dtype=np.int8)
    cambio[dove] = np.fromiter((e[2] for e in events), dtype=np.int8, count=len(events))

    noto = cambio != _MAI
    ultimo = np.maximum.accumulate(np.where(noto, np.arange(len(index)), -1))
    return np.where(ultimo >= 0, cambio[np.maximum(ultimo, 0)], 0).astype(np.int8)


def decayed_vote(
    state: np.ndarray,
    half_life_bars: float,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Il voto in `[-1, +1]`: pieno quando il votante scatta, poi in decadimento esponenziale.

    Ricorsione in forma chiusa -- fra due scatti il voto e' `verso * lambda**eta` -- quindi O(N)
    senza ciclo. `epsilon` taglia la coda a zero esatto: sotto quella soglia il voto non sposta
    nessuna soglia e tenerlo acceso costerebbe solo confusione nelle diagnosi.
    """
    state = np.asarray(state, dtype=np.int8)
    n = len(state)
    voto = np.zeros(n, dtype=float)
    if n == 0:
        return voto
    if half_life_bars <= 0:
        raise ValueError(f"emivita non positiva: {half_life_bars}")

    precedente = np.empty(n, dtype=np.int8)
    precedente[0] = 0
    precedente[1:] = state[:-1]
    scatta = (state != precedente) & (state != 0)

    posizione = np.arange(n)
    ultimo_scatto = np.maximum.accumulate(np.where(scatta, posizione, -1))
    vivo = ultimo_scatto >= 0

    lam = 0.5 ** (1.0 / half_life_bars)
    voto[vivo] = state[ultimo_scatto[vivo]] * lam ** (posizione[vivo] - ultimo_scatto[vivo])
    voto[np.abs(voto) < epsilon] = 0.0
    return voto


def _selfcheck() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="15min", name="Open time")
    eventi = [(idx[3], 100.0, 1), (idx[10], 105.0, 0), (idx[20], 99.0, -1)]
    stato = held_state(eventi, idx)

    # 1. Lo stato e' quello tenuto: 0 prima del primo evento, poi propagato in avanti.
    assert stato[2] == 0 and stato[3] == 1 and stato[9] == 1
    assert stato[10] == 0 and stato[19] == 0
    assert stato[20] == -1 and stato[-1] == -1

    # 2. Il voto e' pieno dove il votante scatta, e solo li'.
    voto = decayed_vote(stato, half_life_bars=4)
    assert voto[3] == 1.0 and voto[20] == -1.0
    assert abs(voto[4]) < 1.0

    # 3. Dopo un'emivita il voto e' meta'. E' la definizione, e la verifica che lambda sia giusto.
    assert np.isclose(voto[3 + 4], 0.5)

    # 4. Uscire non azzera il voto di colpo: sfuma. Un'uscita non e' un segnale contrario.
    assert voto[10] > 0 and voto[10] < voto[9]

    # 5. Sotto epsilon il voto e' zero esatto, non un residuo che sporca le diagnosi.
    #    A sedici barre da uno scatto con emivita quattro il voto vale 0,0625: vivo sopra 0,05,
    #    spento sopra 0,1. La coda si taglia dove dice epsilon, non dove capita.
    assert np.isclose(voto[19], 0.0625)
    assert decayed_vote(stato, half_life_bars=4, epsilon=0.1)[19] == 0.0

    # 6. Causalita': troncare la storia non cambia niente di gia' emesso.
    meta = 15
    troncato = decayed_vote(held_state([e for e in eventi if e[0] <= idx[meta - 1]], idx[:meta]), 4)
    assert np.allclose(troncato, voto[:meta])

    # 7. Un votante su un indice sbagliato si fa notare invece di allinearsi da solo.
    try:
        held_state([(idx[3] + pd.Timedelta(minutes=7), 100.0, 1)], idx)
    except ValueError:
        pass
    else:
        raise AssertionError("un evento fuori griglia doveva sollevare")

    print("voters selfcheck: 7 controlli passati")


if __name__ == "__main__":
    _selfcheck()
