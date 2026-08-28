"""Politica a due azioni con il costo dentro la ricompensa: fitted Q-iteration.

## Perche' questa forma, e non uno stop

Tre misure, su 15 simboli e la regola a esposizione gia' cablata (`.claude/docs/modello-swing.md`
§5.2), dicono da dove viene la perdita fuori campione:

| domanda | misura | esito |
|---|---|---|
| gli ingressi cadono prima dei crolli? | drawdown a 3 giorni dopo un ingresso: mediana −3,88%, p05 −15,82%; dopo una barra giornaliera qualunque: −3,94%, −15,72% | no, sono indistinguibili dal caso |
| uno stop taglia la perdita? | netto OOS: senza −201%, stop 3% −229%, 5% −314%, 8% −327%, trailing 12% −418% | no, la peggiora sempre |
| dove va il denaro? | lordo **+401%**, netto **−201%**, 3.009 operazioni × 0,2% = **602%** | nella commissione |

Il segnale esiste al lordo. A mangiarlo e' il **numero di giri**: passare la stessa regola da una
decisione al giorno a una ogni due giorni porta il netto da −201% a +306%, senza toccare il modello.

Ne segue la forma dell'agente. Non un filtro di rischio -- e' misurato dannoso -- ma una politica
che sceglie la **posizione** sapendo quanto costa cambiarla:

    r(s, p, a) = a · log(P'/P) − costo · |a − p|

La posizione precedente `p` sta **nello stato**, quindi la banda di non-fare non e' un'isteresi
scritta a mano: e' cio' che emerge quando il costo e' dentro l'obiettivo. E la classe di politiche
contiene il possesso passivo (`a ≡ 1`), che e' il riferimento da battere: l'agente puo' solo
aggiungere qualcosa a una politica che sa gia' rappresentare.

## Cosa la distingue dalla politica a tre azioni gia' chiusa in negativo

`strategy.md` §11-13: quella entrava alla conferma di un minimo e usciva alla conferma di un
massimo, ed e' **somma nulla per costruzione** prima dei costi (la conferma si paga due volte, la
gamba mediana vale 1,76-2,05 soglie). Il vincolo economico era scoperto dopo. Qui e' dentro il
target, che e' l'unica riformulazione che §13.4 lasciava aperta.

## L'algoritmo

Fitted Q-iteration, offline, su un batch fisso: nessuna interazione con l'ambiente, quindi nessuno
shift di distribuzione da politica che esplora. Due regressori, uno per azione, sullo stato
`[feature, posizione]`. A ogni giro il bersaglio e' `r + γ · max_a' Q(s', a')`, con `s'` che porta
come posizione **l'azione appena presa** -- e' quel collegamento che rende il costo un investimento
e non una tassa istantanea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# Costo di **un** cambio di posizione (un lato). Il giro completo ne paga due, uno all'ingresso e
# uno all'uscita, e li paga in due transizioni diverse: e' cosi' che l'agente vede il conto vero.
COSTO = 0.001
# 0,95 a cadenza giornaliera e' un orizzonte di circa venti giorni. Sotto, l'agente non aspetta
# mai una gamba; sopra, il costo di un cambio sparisce dentro il valore terminale.
GAMMA = 0.95
GIRI = 5


@dataclass(frozen=True)
class Transizioni:
    """Il batch: stato, stato successivo e rendimento logaritmico del passo."""

    stato: np.ndarray  # (m, k) le feature al momento della decisione
    successivo: np.ndarray  # (m, k) le feature alla decisione dopo
    logret: np.ndarray  # (m,) log(P'/P) fra le due

    def __len__(self) -> int:
        return len(self.logret)


def transizioni_simbolo(
    features: np.ndarray, close: np.ndarray, cadenza: int, fasi: int = 1
) -> Transizioni:
    """Le transizioni di un simbolo, campionate ogni `cadenza` barre a partire da `fasi` sfasature.

    Le sfasature moltiplicano il campione senza cambiare il problema: la decisione resta una ogni
    `cadenza` barre, ma la si osserva partendo da momenti diversi della giornata. Righe cosi'
    ottenute si sovrappongono, quindi contano meno di quante sono -- vanno bene per stimare, non
    per dichiarare una significativita'.
    """
    passo = max(cadenza // max(fasi, 1), 1)
    pezzi = []
    for offset in range(0, cadenza, passo):
        d = np.arange(offset, len(close) - cadenza, cadenza)
        if len(d) < 2:
            continue
        x, y = features[d], features[d + cadenza]
        buoni = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
        buoni &= (close[d] > 0) & (close[d + cadenza] > 0)
        pezzi.append((x[buoni], y[buoni], np.log(close[d + cadenza][buoni] / close[d][buoni])))
    if not pezzi:
        vuoto = np.empty((0, features.shape[1]))
        return Transizioni(vuoto, vuoto, np.empty(0))
    return Transizioni(*(np.concatenate(p) for p in zip(*pezzi)))


def unisci(pezzi: list[Transizioni]) -> Transizioni:
    pieni = [p for p in pezzi if len(p)]
    if not pieni:
        raise ValueError("nessuna transizione")
    return Transizioni(*(np.concatenate(a) for a in zip(*((p.stato, p.successivo, p.logret) for p in pieni))))


def _con_posizione(stato: np.ndarray, posizione: float) -> np.ndarray:
    return np.hstack([stato, np.full((len(stato), 1), posizione)])


def fitted_q(
    batch: Transizioni,
    giri: int = GIRI,
    gamma: float = GAMMA,
    costo: float = COSTO,
    seme: int = 0,
    max_iter: int = 200,
    verboso: bool = False,
) -> list[HistGradientBoostingRegressor]:
    """Due regressori, `Q[0]` e `Q[1]`, sullo stato `[feature, posizione]`.

    Al primo giro il bersaglio e' la sola ricompensa immediata: e' la politica miope, che con questo
    costo e' gia' una linea di base sensata. Ogni giro dopo aggiunge un passo di orizzonte.
    """
    # Ogni transizione genera due righe, una per valore della posizione precedente. Lo stato
    # successivo invece non dipende da `p`: porta l'azione, non la posizione da cui si veniva.
    stati = np.vstack([_con_posizione(batch.stato, 0.0), _con_posizione(batch.stato, 1.0)])
    precedente = np.concatenate([np.zeros(len(batch)), np.ones(len(batch))])
    logret = np.tile(batch.logret, 2)

    Q: list[HistGradientBoostingRegressor] = []
    for giro in range(giri):
        if Q:
            valore = {
                a: np.maximum(*(q.predict(_con_posizione(batch.successivo, float(a))) for q in Q))
                for a in (0, 1)
            }
        else:
            valore = {0: np.zeros(len(batch)), 1: np.zeros(len(batch))}
        nuovi = []
        for azione in (0, 1):
            ricompensa = azione * logret - costo * np.abs(azione - precedente)
            bersaglio = ricompensa + gamma * np.tile(valore[azione], 2)
            modello = HistGradientBoostingRegressor(
                max_iter=max_iter, learning_rate=0.06, random_state=seme + azione
            )
            nuovi.append(modello.fit(stati, bersaglio))
        Q = nuovi
        if verboso:
            scarto = np.mean(Q[1].predict(stati) > Q[0].predict(stati))
            print(f"  giro {giro + 1}/{giri}: quota lunga {scarto * 100:.1f}%", flush=True)
    return Q


def posizioni(Q: list[HistGradientBoostingRegressor], stato: np.ndarray, iniziale: int = 0) -> np.ndarray:
    """La politica srotolata: `a_t = argmax_a Q(s_t, a_{t-1}, a)`, un passo alla volta.

    Il ciclo e' obbligato -- lo stato porta la propria uscita precedente -- ma le quattro chiamate
    a `predict` stanno fuori: dentro resta un confronto fra numeri gia' calcolati.
    """
    if not len(stato):
        return np.zeros(0, dtype=np.int8)
    q = {(p, a): Q[a].predict(_con_posizione(stato, float(p))) for p in (0, 1) for a in (0, 1)}
    fuori = np.empty(len(stato), dtype=np.int8)
    p = int(iniziale)
    for i in range(len(stato)):
        p = int(q[(p, 1)][i] > q[(p, 0)][i])
        fuori[i] = p
    return fuori


def rendimento(azioni: np.ndarray, logret: np.ndarray, costo: float = COSTO, iniziale: int = 0) -> float:
    """Rendimento composto netto in percento della sequenza di posizioni, costi dei cambi inclusi."""
    cambi = np.abs(np.diff(azioni.astype(float), prepend=float(iniziale)))
    return float(np.exp(np.sum(azioni * logret - costo * cambi)) - 1.0) * 100


def _selfcheck() -> None:
    """Un mondo in cui la risposta giusta e' nota: una feature che dice il rendimento di domani.

    Verifica due cose, e sono le due che possono rompersi in silenzio: che la politica segua il
    segnale, e che **alzare il costo allarghi la banda di non-fare**. La seconda e' l'unica prova
    che `p` stia davvero nello stato: se lo stato la ignorasse, il costo non potrebbe cambiare
    niente sul numero di cambi.
    """
    rng = np.random.default_rng(0)
    n = 6000
    segnale = rng.normal(size=n)
    # Rendimento giornaliero pilotato dal segnale della vigilia, piu' rumore.
    passo = 0.02 * segnale + 0.01 * rng.normal(size=n)
    close = np.exp(np.cumsum(np.concatenate([[0.0], passo])))
    feat = np.column_stack([segnale, rng.normal(size=n)])
    # `transizioni_simbolo` legge a cadenza 1: qui una barra e' gia' un giorno.
    batch = transizioni_simbolo(np.vstack([feat, feat[-1]]), close, cadenza=1)

    Q = fitted_q(batch, giri=2, costo=0.0, max_iter=60)
    a = posizioni(Q, batch.stato)
    alto = batch.stato[:, 0] > 0.5
    assert a[alto].mean() > 0.8, f"con segnale alto deve stare lungo: {a[alto].mean():.2f}"
    assert a[batch.stato[:, 0] < -0.5].mean() < 0.2, "con segnale basso deve stare fuori"

    cambi_gratis = np.abs(np.diff(a)).sum()
    Q_caro = fitted_q(batch, giri=2, costo=0.02, max_iter=60)
    cambi_cari = np.abs(np.diff(posizioni(Q_caro, batch.stato))).sum()
    assert cambi_cari < cambi_gratis, f"il costo deve ridurre i cambi: {cambi_cari} vs {cambi_gratis}"

    # Il possesso passivo e' rappresentabile: e' il riferimento, e deve stare dentro la classe.
    sempre = np.ones(len(batch), dtype=np.int8)
    assert rendimento(sempre, batch.logret, costo=0.0) > 0
    print(f"selfcheck ok: cambi {cambi_gratis} -> {cambi_cari} alzando il costo")


if __name__ == "__main__":
    _selfcheck()
