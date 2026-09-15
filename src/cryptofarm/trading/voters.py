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
v(t) = 0                                          se stato(t) == 0
v(t) = stato(t) * (pavimento + (1-pavimento) * lambda**eta)   altrimenti
```

dove `eta` sono le barre trascorse dall'ultimo scatto. Il voto e' quindi **l'opinione moltiplicata
per la recenza**, e non la recenza da sola.

## Perche' un pavimento, e perche' lo zero secco all'uscita

La prima versione era `v(t) = v(t-1) * lambda`, senza pavimento e senza azzeramento: il voto
seguiva solo la recenza dell'ultimo scatto, e l'opinione del votante non entrava piu' dopo la
prima barra. Misurato su 400 giorni sintetici, due difetti opposti e tutti e due grossi:

- **votanti muti mentre erano convinti.** `zone_regime` era in posizione sul 74,5% delle barre e
  con voto gia' spento sul **91,3%** di quelle. La macrostruttura, che per disegno deve «sostenere
  il punteggio per tutta la durata di un trend», contribuiva solo nelle ore attorno all'incrocio,
  cioe' esattamente quando e' piu' soggetta a whipsaw. `zone_struttura` 67,1%, `bande_conferma`
  84,4%;
- **voti fantasma.** Al contrario, `pullback` aveva un voto acceso con la posizione gia' chiusa sul
  **49,0%** delle barre. Il caso peggiore erano le bande, che entrano sulla banda inferiore ed
  escono su quella **opposta**: il voto +1 sopravviveva alla chiusura e continuava a dire «lungo»
  dal massimo in giu'.

Il pavimento risolve il primo, l'azzeramento il secondo, e nessuno dei due tocca la memoria che
serve alla confluenza multi-piano: lo stato **e' tenuto** (`held_state` lo propaga in avanti),
quindi un votante a 4H continua a votare su ogni barra da quindici minuti finche' la sua posizione
e' aperta. Cio' che sfuma e' la *forza*, da 1 al pavimento, non la presenza.

Il pavimento va tenuto sotto la soglia **minima raggiungibile**, e la ragione sta in
`PAVIMENTO_DEL_VOTO`: un collegio interamente d'accordo ma tutto vecchio vale esattamente
`pavimento`, e sopra quella riga aprirebbe da solo, senza che nessuno abbia scattato.

`half_life_bars` si esprime in barre dell'indice su cui il voto viene letto. Chi chiama lo ricava
dal timeframe del votante moltiplicando per il rapporto degli intervalli, **con un tetto**: senza,
un votante di regime a 1D su un indice a 15 minuti arriva a 576 barre di emivita, cioe' resta a
forza quasi piena per settimane, e la recenza smette di voler dire qualcosa. Il tetto sta in
`confluence.TETTO_EMIVITA_MINUTI` perche' e' li' che si conosce l'intervallo di base.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MAI = -2  # sentinella: su questa barra non c'e' nessun cambio di posizione

# La frazione di forza che un votante conserva finche' tiene la posizione, comunque vecchio sia il
# suo scatto.
#
# **Il vincolo che lo sceglie**: a pesi a somma 1, un collegio tutto d'accordo e tutto vecchio vale
# esattamente questo numero. Perche' un consenso fermo non possa aprire da solo, deve stare sotto
# la soglia **minima raggiungibile**, che non e' `theta_base` ma `theta_base - theta_macro` --
# 0,20 con i default, perche' un macro a favore sconta la soglia. Sotto quella riga la confluenza
# resta un incontro di *eventi* e continua a decidere **quando**; sopra diventa un rilevatore di
# stato che apre perche' tutti sono dentro, che e' un'altra strategia.
#
# A 0,15 un collegio fermo vale 0,15 contro una soglia che non scende sotto 0,20, e **un solo**
# scatto recente aggiunge (1-0,15)/7 = 0,121 e porta a 0,271: basta con il macro a favore, ne
# servono circa due a macro neutro. Chi muove `theta_base` o `theta_macro` deve rifare questo
# conto: il vincolo e' una relazione fra tre numeri, non un valore.
PAVIMENTO_DEL_VOTO = 0.15


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
    pavimento: float = PAVIMENTO_DEL_VOTO,
    epsilon: float = 0.05,
) -> np.ndarray:
    """Il voto in `[-1, +1]`: **l'opinione moltiplicata per la recenza**, zero quando non c'e'.

    Pieno sulla barra in cui il votante scatta, poi sfuma verso `pavimento` -- non verso zero --
    finche' la posizione resta aperta, e zero esatto appena lo stato torna a flat. Le ragioni di
    tutte e tre le scelte, con le misure, stanno nella docstring del modulo.

    Forma chiusa, quindi O(N) senza ciclo. `epsilon` taglia la coda a zero esatto; con un pavimento
    positivo non morde mai, e resta per chi chiama con `pavimento=0` -- che e' il comportamento
    vecchio, tenuto raggiungibile perche' e' l'ablazione con cui si misura quanto vale il
    pavimento.
    """
    state = np.asarray(state, dtype=np.int8)
    n = len(state)
    voto = np.zeros(n, dtype=float)
    if n == 0:
        return voto
    if half_life_bars <= 0:
        raise ValueError(f"emivita non positiva: {half_life_bars}")
    if not 0.0 <= pavimento <= 1.0:
        raise ValueError(f"pavimento fuori da [0, 1]: {pavimento}")

    precedente = np.empty(n, dtype=np.int8)
    precedente[0] = 0
    precedente[1:] = state[:-1]
    scatta = (state != precedente) & (state != 0)

    posizione = np.arange(n)
    ultimo_scatto = np.maximum.accumulate(np.where(scatta, posizione, -1))
    # `state != 0` e' la meta' nuova della condizione: fuori posizione non si vota. Senza, il voto
    # sopravviveva alla chiusura e diceva «lungo» mentre il votante era gia' uscito -- il 49% delle
    # barre per `pullback`, e per le bande dal massimo in giu', visto che escono sulla banda opposta.
    vivo = (ultimo_scatto >= 0) & (state != 0)

    lam = 0.5 ** (1.0 / half_life_bars)
    eta = posizione[vivo] - ultimo_scatto[vivo]
    # Il segno e' quello dello stato **di adesso**, non quello letto allo scatto: sono lo stesso
    # numero (dentro una corsa di stato costante l'unico modo di cambiare verso e' un altro
    # scatto), ma scriverlo cosi' dice che il voto e' l'opinione corrente e non un ricordo.
    voto[vivo] = state[vivo] * (pavimento + (1.0 - pavimento) * lam**eta)
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
    voto = decayed_vote(stato, half_life_bars=4, pavimento=0.0)
    assert voto[3] == 1.0 and voto[20] == -1.0
    assert abs(voto[4]) < 1.0

    # 3. Dopo un'emivita il voto e' meta'. E' la definizione, e la verifica che lambda sia giusto.
    assert np.isclose(voto[3 + 4], 0.5)

    # 4. Fuori posizione non si vota: appena lo stato torna a flat il voto e' zero esatto.
    #    E' la meta' che toglie i voti fantasma -- il votante che dice «lungo» dopo aver chiuso.
    assert voto[10] == 0.0 and voto[9] > 0

    # 5. Con il pavimento il voto non scende mai sotto quella frazione finche' la posizione tiene.
    #    E' la meta' che toglie i votanti muti: prima, a sedici barre dallo scatto, `zone_regime`
    #    era spento pur essendo convinto.
    tenuto = decayed_vote(held_state([(idx[3], 100.0, 1)], idx), half_life_bars=4)
    assert (tenuto[3:] >= PAVIMENTO_DEL_VOTO).all(), "chi tiene la posizione non deve ammutolire"
    assert tenuto[-1] < tenuto[3], "ma la forza sfuma lo stesso: il voto e' opinione per recenza"
    assert np.isclose(tenuto[3], 1.0)

    # 6. `epsilon` taglia la coda solo quando non c'e' pavimento: e' l'ablazione, non il default.
    assert np.isclose(voto[7], 0.5)
    assert decayed_vote(stato, half_life_bars=4, pavimento=0.0, epsilon=0.6)[7] == 0.0

    # 7. Causalita': troncare la storia non cambia niente di gia' emesso.
    meta = 15
    troncato = decayed_vote(held_state([e for e in eventi if e[0] <= idx[meta - 1]], idx[:meta]), 4, pavimento=0.0)
    assert np.allclose(troncato, voto[:meta])

    # 8. Un votante su un indice sbagliato si fa notare invece di allinearsi da solo.
    try:
        held_state([(idx[3] + pd.Timedelta(minutes=7), 100.0, 1)], idx)
    except ValueError:
        pass
    else:
        raise AssertionError("un evento fuori griglia doveva sollevare")

    print("voters selfcheck: 8 controlli passati")


if __name__ == "__main__":
    _selfcheck()
