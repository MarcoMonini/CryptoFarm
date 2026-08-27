"""La strategia a confluenza: quattro piani, sei votanti, una soglia che si muove.

Il disegno completo, con le ragioni di ogni scelta, sta in `.claude/docs/strategia-confluenza.md`.
Qui c'e' l'attuazione degli stadi S3-S5, e questa docstring dice **solo** le cose che servono a
chi legge il codice o ne cambia un pezzo.

## Che problema risolve, in una riga

Il quadro macro (piano lungo) decide *se*, i piani intermedi decidono *se davvero*, il piano breve
decide *quando*. Ogni piano risponde a una domanda diversa: quattro voti sulla stessa domanda
sarebbero **una sola opinione contata quattro volte**, e il punteggio che ne uscirebbe sembrerebbe
continuo mentre e' binario travestito.

## I piani sono multipli dell'intervallo di base, non intervalli fissi

`FATTORI` li esprime in multipli delle candele che si passano: su barre da 15 minuti sono
esattamente la scala 1h / 4h / 1d del disegno, ma la stessa strategia gira su qualunque base. E'
anche il motivo per cui i parametri dei votanti si cercano nei `tuned_defaults` dell'intervallo
*risultante*: a base 15m i votanti girano a 1h e 4h, che sono misurati; a base diversa possono
cadere su intervalli mai misurati, e allora restano i default scritti a mano delle loro funzioni.

## Cosa e' *davvero* in formazione, e cosa no

Il bot live alle 10:00 non aspetta la mezzanotte: vede una barra 1D parziale, e quella non e'
look-ahead (`live_frames.py`). Ma sollevare a valore provvisorio una *strategia* qualunque non e'
generico -- ogni indicatore ricorsivo va sollevato a mano -- e il disegno **congela i votanti**,
cioe' vieta di riscriverli. Quindi:

- **i sei votanti decidono alla chiusura della propria barra lunga** e il loro stato entra
  nell'indice breve con `mtf.align_to_lower`, cioe' un periodo dopo. Nessun look-ahead, ma
  nemmeno reattivita' intra-periodo;
- **il cancello e la struttura leggono il prezzo di adesso** contro la media del piano lungo
  chiusa. E' li' che sta la reattivita' intra-periodo: meta' del confronto e' gia' l'ultimo
  prezzo, ed e' il cancello che il disegno vuole reattivo, perche' quando si chiude manda a flat
  senza discutere.

`barre_in_formazione=False` fa aspettare anche a quei due la chiusura del piano lungo: e'
l'**ablazione** che misura quanto vale reagire prima. La differenza fra le due e' un numero.

Una cosa che questa attuazione ha tolto invece di aggiungerla: sollevare la media a valore
provvisorio con `provisional_ema` **non cambia il segno del confronto**, per algebra, quindi qui
non e' un meccanismo ma un parametro finto. La spiegazione sta in `_sign_su_media`. Il
sollevamento serve dove conta il valore -- una distanza, una banda, uno stop -- non il lato.

## Il difetto che questo file puo' avere

Un disallineamento fra i piani non solleva niente e produce risultati *migliori* del vero. Le due
difese sono `align_to_lower` (che sposta lo stato lungo di un periodo intero prima di leggerlo) e
il fatto che `held_state` rifiuta un evento che non cade su una barra dell'indice. Chi tocca il
passaggio fra i piani rilegga `tests/test_confluence.py::test_nessun_piano_lungo_anticipa`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes, resample_klines
from cryptofarm.trading import strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.mtf import align_to_lower
from cryptofarm.trading.voters import decayed_vote, held_state

# I quattro piani, in multipli dell'intervallo di base. Su 15m: innesco 15m, conferma 1h,
# struttura 4h, regime 1d -- la scala del disegno, ottenuta senza scriverci dentro nessun
# intervallo fisso.
FATTORI = {"innesco": 1, "conferma": 4, "struttura": 16, "regime": 96}


# Il piano di regime deve durare quanto un regime: fra mezza giornata e una settimana. Sotto, la
# «macro» e' rumore con un nome altisonante; sopra, una media di cinquanta barre chiede decenni di
# storia. La scala x1/x4/x16/x96 e' nata su barre da quindici minuti e li' cade esatta su
# 15m/1h/4h/1d, ma il menu ne offre nove e non c'e' niente che impedisca di scegliere le altre.
REGIME_MIN_MINUTI = 12 * 60
REGIME_MAX_MINUTI = 7 * 24 * 60


def piani(interval: str) -> dict[str, str]:
    """I quattro piani effettivi per un intervallo di base, come stringhe leggibili."""
    minuti = interval_to_minutes(interval)
    return {nome: _intervallo(minuti * fattore) for nome, fattore in FATTORI.items()}


def ore_richieste(interval: str, regime_ema: int) -> int:
    """Quante ore di storia servono perche' il cancello di regime possa aprirsi.

    E' il numero che manca a chi apre la pagina: il default sono 240 ore, e a quindici minuti la
    media di regime ne chiede piu' di mille. Sotto quella soglia non e' che la strategia non trovi
    occasioni -- e' che non puo' trovarne, e le due cose si leggono uguali.
    """
    return int(interval_to_minutes(interval) * FATTORI["regime"] * regime_ema / 60)


def scala_fuori_misura(interval: str) -> str:
    """Un avviso quando l'intervallo scelto porta il piano di regime fuori scala, altrimenti "".

    Restituisce testo perche' e' un avviso da mostrare, non una condizione da gestire: la
    strategia gira lo stesso, e sta a chi guarda decidere se quel che ne esce significhi qualcosa.
    """
    minuti_regime = interval_to_minutes(interval) * FATTORI["regime"]
    if minuti_regime < REGIME_MIN_MINUTI:
        return f"regime plane is {_intervallo(minuti_regime)}: too short to be a regime"
    if minuti_regime > REGIME_MAX_MINUTI:
        return f"regime plane is {_intervallo(minuti_regime)}: its moving average needs decades of history"
    return ""


@dataclass(frozen=True)
class Votante:
    """Un votante: chi e', su che piano guarda, e come si esegue.

    `famiglia` non e' decorativa: l'ampiezza minima si conta in **famiglie distinte**, non in
    votanti, ed e' cio' che impedisce a un peso grande di aprire una posizione da solo. Oggi i sei
    sono uno per famiglia; il votante AI entrera' nella famiglia "trasversale" senza cambiare il
    conteggio.

    `menu` e' il nome con cui la strategia compare in `tuned_defaults`, vuoto per chi non e' mai
    stato misurato: quei votanti restano ai default della propria funzione.
    """

    nome: str
    famiglia: str
    piano: str
    esegui: Callable[[pd.DataFrame, ExtraCache, dict], list]
    menu: str = ""


def _ichimoku(df, cache, v):
    return strategies_ls.ichimoku_trend(
        df,
        cache,
        fast=int(v.get("ICHIMOKU_FAST", 9)),
        slow=int(v.get("ICHIMOKU_SLOW", 26)),
        span=int(v.get("ICHIMOKU_SPAN", 52)),
        require_cloud=bool(v.get("REQUIRE_CLOUD", True)),
    )


def _donchian(df, cache, v):
    return strategies_ls.donchian_breakout(
        df,
        cache,
        channel=int(v.get("DONCHIAN_CHANNEL", 20)),
        adx_min=float(v.get("ADX_MIN", 20.0)),
        atr_multiplier=float(v.get("DONCHIAN_ATR_MULT", 3.0)),
        regime_ema=int(v.get("REGIME_EMA", 200)),
    )


def _squeeze(df, cache, v):
    return strategies_ls.squeeze_breakout(
        df,
        cache,
        bb_dev=float(v.get("BB_DEV", 2.0)),
        kc_multiplier=float(v.get("KC_MULTIPLIER", 1.5)),
        atr_multiplier=float(v.get("SQUEEZE_ATR_MULT", 2.5)),
        confirm_volume=bool(v.get("CONFIRM_VOLUME", True)),
    )


def _pullback(df, cache, v):
    return strategies_ls.trend_pullback(df, cache)


def _reversione(df, cache, v):
    return strategies_ls.band_reversion_gated(df, cache)


def _flusso(df, cache, v):
    """Il votante di flusso: l'unico che non legge il prezzo, e per questo l'unico che decorrela.

    Non e' una strategia di `strategies_ls` -- non ne esiste una di questa famiglia -- ma le due
    serie ci sono gia' in `ExtraCache`. Vota lungo quando l'OBV sale **e** l'MFI non e' in
    ipercomprato, corto allo specchio: la pendenza dice la direzione del flusso, l'MFI dice se il
    flusso e' gia' stato tutto speso.
    """
    finestra = int(v.get("OBV_WINDOW", 20))
    pendenza = cache.obv_slope(finestra)
    mfi = cache.mfi(finestra)
    lungo = (pendenza > 0) & (mfi < 80)
    corto = (pendenza < 0) & (mfi > 20)
    stato = np.where(lungo, 1, np.where(corto, -1, 0)).astype(np.int8)
    stato[np.isnan(pendenza) | np.isnan(mfi)] = 0
    # Il formato comune e' quello degli eventi: si emettono i soli cambi.
    cambi = np.flatnonzero(np.diff(stato, prepend=np.int8(0)))
    chiusure = df["Close"].to_numpy()
    return [(df.index[i], float(chiusure[i]), int(stato[i])) for i in cambi]


VOTANTI: tuple[Votante, ...] = (
    Votante("ichimoku", "inseguimento", "struttura", _ichimoku, "Ichimoku Trend"),
    Votante("donchian", "rottura", "struttura", _donchian, "Donchian Breakout"),
    Votante("flusso", "volume", "struttura", _flusso, "Squeeze Breakout"),
    Votante("squeeze", "volatilita", "conferma", _squeeze, "Squeeze Breakout"),
    Votante("pullback", "rientro", "conferma", _pullback),
    Votante("reversione", "ritorno_media", "innesco", _reversione),
)


@dataclass
class Confluenza:
    """Il risultato: gli eventi da eseguire, e tutto cio' che serve a capire **chi** li ha fatti.

    Le diagnosi non sono un extra da guardare a fine analisi: `necessarieta` va riportata accanto
    a ogni risultato. Se un votante e' necessario in piu' del 60% degli ingressi, l'insieme e'
    quel votante travestito, e il numero lo dice prima che lo dica il mercato.
    """

    eventi: list
    indice: pd.DatetimeIndex
    voti: dict[str, np.ndarray]
    pesi: dict[str, float]
    punteggio: np.ndarray
    soglia: np.ndarray
    regime: np.ndarray
    struttura: np.ndarray
    famiglie_concordi: np.ndarray
    concordi_lungo: np.ndarray | None = None
    k_famiglie: int = 2
    barre_del_regime: int = 0
    barre_chieste_dal_regime: int = 0
    necessarieta: dict[str, float] = field(default_factory=dict)
    ingressi: int = 0

    def perche_non_entra(self) -> str:
        """La prima condizione che non si e' mai verificata, in inglese e con i numeri.

        Zero operazioni non e' un risultato: e' una domanda. Le condizioni d'ingresso sono quattro
        in `and`, e senza sapere **quale** non si e' mai avverata non si sa se la strategia sia
        prudente, mal tarata o senza abbastanza storia -- che sono tre cose diverse e chiedono tre
        rimedi diversi. Restituisce "" quando le operazioni ci sono.
        """
        if self.ingressi:
            return ""
        if self.barre_del_regime < self.barre_chieste_dal_regime:
            return (
                f"not enough history: the regime plane has {self.barre_del_regime} bars and its "
                f"moving average needs {self.barre_chieste_dal_regime}. The gate stays shut, so no "
                "long entry is possible. Load a longer window."
            )
        aperto = self.regime > 0
        if not aperto.any():
            return "the regime gate never opened: price stayed below the regime plane average the whole window."
        sopra = aperto & (self.punteggio >= self.soglia)
        if not sopra.any():
            picco = float(self.punteggio[aperto].max())
            minima = float(self.soglia[aperto].min())
            return (
                f"the gate opened but the score never reached the threshold: peak {picco:+.2f} "
                f"against a threshold that never fell below {minima:.2f}. Lower «Entry threshold»."
            )
        if self.concordi_lungo is not None and not (sopra & (self.concordi_lungo >= self.k_famiglie)).any():
            return (
                f"score and gate agreed, but never with {self.k_famiglie} families at once "
                f"(at most {int(self.concordi_lungo[sopra].max())}). Lower «Families required to agree»."
            )
        return "everything agreed but the trigger never fired: lower «Trigger breakout window» or set it to 0."

    def eventi_con_priorita(self) -> list:
        """Gli stessi eventi con il **margine sopra la soglia** come quarto elemento.

        Serve al paniere a capitale condiviso (`trading/portfolio.py`): quando due asset parlano
        sulla stessa barra vince il segnale piu' netto, non il primo dell'ordine alfabetico. Sulle
        uscite il margine e' zero, perche' una chiusura non compete con niente.
        """
        posizioni = self.indice.get_indexer([e[0] for e in self.eventi])
        return [
            (*evento[:3], abs(self.punteggio[i]) - self.soglia[i] if evento[2] != 0 else 0.0)
            for evento, i in zip(self.eventi, posizioni)
        ]

    def spiega(self, quando) -> str:
        """Chi ha generato il segnale su quella barra, e con che contributo. Una riga."""
        i = self.indice.get_indexer([pd.Timestamp(quando)], method="pad")[0]
        if i < 0:
            return ""
        parti = [f"{nome} {self.pesi[nome] * voto[i]:+.2f}" for nome, voto in self.voti.items() if abs(voto[i]) > 1e-9]
        return (
            f"punteggio {self.punteggio[i]:+.2f} / soglia {self.soglia[i]:.2f} · "
            f"famiglie {int(self.famiglie_concordi[i])} · " + ", ".join(parti or ["nessun votante attivo"])
        )


def _intervallo(minuti: int) -> str:
    """Da minuti alla stringa che `resample_klines` capisce, senza tabelle di corrispondenza."""
    if minuti % 1440 == 0:
        return f"{minuti // 1440}d"
    if minuti % 60 == 0:
        return f"{minuti // 60}h"
    return f"{minuti}m"


def _parametri_congelati(menu: str, intervallo: str) -> dict:
    """I parametri misurati per quella strategia a quell'intervallo, o niente.

    **Congelare i votanti e' il vincolo portante del disegno**: ritararli dentro l'insieme porta
    il conto dei parametri liberi da nove a oltre venticinque, e a quel punto niente di cio' che
    esce e' distinguibile dalla fortuna. Questa funzione e' l'unico posto da cui i loro valori
    arrivano, e non prende niente dalla griglia dell'insieme.
    """
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    return dict(PER_INTERVALLO.get(intervallo, {}).get(menu, {})) if menu else {}


def _pesi(nomi: list[str], w_max: float, pesi: dict[str, float] | None = None) -> dict[str, float]:
    """Pesi normalizzati a somma 1 con tetto `w_max`, applicato ripetutamente fino a tenuta.

    A pesi uguali il tetto non morde mai -- con sei votanti ognuno vale 0,167 contro un tetto di
    0,30 -- e questo e' voluto: il tetto e' li' per la versione tarata (S7), dove serve a impedire
    che l'insieme diventi *un* segnale con delle decorazioni.
    """
    valori = {n: 1.0 for n in nomi} if pesi is None else {n: float(pesi.get(n, 0.0)) for n in nomi}
    totale = sum(valori.values())
    if totale <= 0:
        raise ValueError("pesi tutti nulli")
    valori = {n: p / totale for n, p in valori.items()}
    for _ in range(len(nomi)):
        eccedenti = {n for n, p in valori.items() if p > w_max + 1e-12}
        if not eccedenti:
            break
        residuo = 1.0 - w_max * len(eccedenti)
        liberi = sum(p for n, p in valori.items() if n not in eccedenti)
        valori = {
            n: (w_max if n in eccedenti else (p / liberi * residuo if liberi > 0 else 0.0)) for n, p in valori.items()
        }
    return valori


def stati_dei_votanti(
    candles: pd.DataFrame, interval: str, votanti: tuple[Votante, ...] = VOTANTI
) -> dict[str, np.ndarray]:
    """Lo stato per barra dei votanti, che e' **la parte cara e l'unica che non dipende dalla griglia**.

    I parametri dei votanti sono congelati, quindi il loro stato dipende solo da (simbolo,
    intervallo): si calcola una volta e si riusa su tutte le configurazioni. Sulla griglia larga
    di `scripts/confluence_lab.py` questo e' la differenza fra ore e minuti, e non costa niente in
    correttezza -- e' lo stesso valore, calcolato una volta invece che mille.
    """
    minuti_base = interval_to_minutes(interval)
    return {v.nome: _stato_del_votante(v, candles, minuti_base) for v in votanti}


def _stato_del_votante(votante, candele, minuti_base) -> np.ndarray:
    """Lo stato di un votante, calcolato sul suo piano e riportato sull'indice di base.

    Il passaggio critico e' `align_to_lower`: sposta lo stato in avanti di **un periodo intero**,
    perche' la barra lunga di adesso chiude nel futuro e leggerla adesso inietterebbe il futuro
    nella decisione. E' l'unico punto in cui questo modulo puo' produrre risultati falsi positivi.
    """
    fattore = FATTORI[votante.piano]
    if fattore == 1:
        lungo = candele
        intervallo = _intervallo(minuti_base)
    else:
        intervallo = _intervallo(minuti_base * fattore)
        lungo = resample_klines(candele, intervallo)
    if len(lungo) < 3:
        return np.zeros(len(candele), dtype=np.int8)

    valori = _parametri_congelati(votante.menu, intervallo)
    eventi = votante.esegui(lungo, ExtraCache(lungo), valori)
    stato_lungo = held_state(eventi, lungo.index)
    if fattore == 1:
        return stato_lungo
    allineato = align_to_lower(stato_lungo, lungo.index, intervallo, candele.index)
    return np.nan_to_num(allineato).astype(np.int8)


def evaluate(
    candles: pd.DataFrame,
    interval: str,
    *,
    theta_base: float = 0.35,
    theta_macro: float = 0.15,
    isteresi: float = 0.10,
    emivita: float = 6.0,
    w_max: float = 0.30,
    k_famiglie: int = 2,
    innesco: int = 0,
    atr_window: int = 14,
    atr_multiplier: float = 3.0,
    regime_ema: int = 50,
    struttura_ema: int = 50,
    barre_in_formazione: bool = True,
    allow_short: bool = False,
    pesi: dict[str, float] | None = None,
    votanti: tuple[Votante, ...] = VOTANTI,
    stati: dict[str, np.ndarray] | None = None,
) -> Confluenza:
    """Esegue la confluenza sulle candele date e restituisce eventi e diagnosi.

    `interval` e' l'intervallo **delle candele passate**, non quello di un piano: i piani si
    ricavano da li' con `FATTORI`.

    L'ingresso vuole quattro cose insieme sulla stessa barra: cancello aperto nel verso,
    `punteggio` oltre la `soglia`, almeno `k_famiglie` famiglie concordi, e l'innesco se e'
    acceso. L'uscita e' la prima delle tre che arriva -- punteggio sotto `soglia - isteresi`,
    stop a trailing ATR, cancello che si chiude.

    `stati` sono gli stati dei votanti gia' calcolati (`stati_dei_votanti`): non cambiano niente
    del risultato, servono solo a non ricalcolare su ogni cella di una griglia la sola parte che
    dalla griglia non dipende.

    L'**isteresi non e' un dettaglio**: senza, si entra e si esce sulla stessa barra ogni volta
    che il punteggio oscilla attorno alla soglia, e il conto delle commissioni mangia tutto.
    """
    if len(candles) < 3:
        raise ValueError("servono almeno tre barre")
    minuti_base = interval_to_minutes(interval)

    voti: dict[str, np.ndarray] = {}
    famiglie: dict[str, str] = {}
    for votante in votanti:
        stato = stati[votante.nome] if stati else _stato_del_votante(votante, candles, minuti_base)
        # L'emivita e' una sola, espressa in barre del timeframe del votante: qui si converte in
        # barre di base. Un segnale di struttura resta vivo giorni, uno di innesco ore.
        voti[votante.nome] = decayed_vote(stato, emivita * FATTORI[votante.piano])
        famiglie[votante.nome] = votante.famiglia

    w = _pesi([v.nome for v in votanti], w_max, pesi)
    punteggio = sum(w[n] * voti[n] for n in voti)

    regime = _sign_su_media(candles, minuti_base, "regime", regime_ema, barre_in_formazione)
    struttura = _sign_su_media(candles, minuti_base, "struttura", struttura_ema, barre_in_formazione)
    soglia = theta_base - theta_macro * (regime + struttura) / 2

    concordi_lungo = _famiglie_concordi(voti, famiglie, +1)
    concordi_corto = _famiglie_concordi(voti, famiglie, -1)
    famiglie_concordi = np.where(punteggio >= 0, concordi_lungo, concordi_corto)

    eventi, ingressi = _percorri(
        candles,
        punteggio=punteggio,
        soglia=soglia,
        regime=regime,
        concordi_lungo=concordi_lungo,
        concordi_corto=concordi_corto,
        isteresi=isteresi,
        k_famiglie=k_famiglie,
        innesco=innesco,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier,
        allow_short=allow_short,
    )

    risultato = Confluenza(
        eventi=eventi,
        indice=candles.index,
        voti=voti,
        pesi=w,
        punteggio=punteggio,
        soglia=soglia,
        regime=regime,
        struttura=struttura,
        famiglie_concordi=famiglie_concordi,
        concordi_lungo=concordi_lungo,
        k_famiglie=k_famiglie,
        barre_del_regime=len(resample_klines(candles, _intervallo(minuti_base * FATTORI["regime"]))),
        barre_chieste_dal_regime=regime_ema,
        ingressi=len(ingressi),
    )
    risultato.necessarieta = _necessarieta(voti, famiglie, w, soglia, ingressi, k_famiglie)
    return risultato


def _sign_su_media(candele, minuti_base, piano, span, in_formazione) -> np.ndarray:
    """Il verso del prezzo rispetto alla EMA del piano: +1, 0 o -1, su ogni barra di base.

    `in_formazione` decide **quale prezzo** si confronta con la media: quello di adesso, che e'
    cio' che vede il bot live a meta' giornata, oppure quello dell'ultima chiusura del piano
    lungo, che e' l'attesa fino a ventiquattro ore. La differenza fra i due e' l'ablazione che
    misura quanto vale reagire prima della chiusura.

    ## Perche' la media resta quella chiusa in tutti e due i casi

    Sollevare la EMA a valore provvisorio qui non servirebbe: `provisional_ema` restituisce
    `a*prezzo + (1-a)*chiusa`, quindi

        segno(prezzo - provvisoria) = segno((1-a) * (prezzo - chiusa)) = segno(prezzo - chiusa)

    per qualunque `a` in (0,1). **Su un confronto di segno il sollevamento e' algebricamente un
    non-fare**, e tenerlo sarebbe un parametro che sembra collegato a qualcosa e non lo e'. Il
    sollevamento serve dove il valore conta, non il lato: una distanza, una banda, uno stop.
    """
    fattore = FATTORI[piano]
    intervallo = _intervallo(minuti_base * fattore)
    lungo = resample_klines(candele, intervallo) if fattore > 1 else candele
    if len(lungo) < 2:
        return np.zeros(len(candele))

    media = ExtraCache(lungo).ema(span)
    prezzo = candele["Close"].to_numpy()
    if fattore > 1:
        media = align_to_lower(media, lungo.index, intervallo, candele.index)
        if not in_formazione:
            prezzo = align_to_lower(lungo["Close"], lungo.index, intervallo, candele.index)
    return np.sign(np.nan_to_num(prezzo - media, nan=0.0))


def _famiglie_concordi(voti, famiglie, verso) -> np.ndarray:
    """Quante **famiglie distinte** votano in quel verso, barra per barra.

    Famiglie e non votanti: e' il freno che impedisce a un peso grande, da solo, di aprire una
    posizione. Oggi la distinzione non morde -- i sei sono uno per famiglia -- ma mordera' appena
    si aggiunge un secondo votante di prezzo, ed e' allora che serve gia' scritta.
    """
    per_famiglia: dict[str, np.ndarray] = {}
    for nome, voto in voti.items():
        attivo = (voto * verso) > 0
        famiglia = famiglie[nome]
        per_famiglia[famiglia] = attivo if famiglia not in per_famiglia else (per_famiglia[famiglia] | attivo)
    return np.sum(list(per_famiglia.values()), axis=0).astype(float)


def _percorri(
    candele,
    *,
    punteggio,
    soglia,
    regime,
    concordi_lungo,
    concordi_corto,
    isteresi,
    k_famiglie,
    innesco,
    atr_window,
    atr_multiplier,
    allow_short,
):
    """Il ciclo di posizione: ingressi, stop a trailing e uscite, nell'ordine del disegno.

    Convenzioni identiche a `strategies_ls`, e non per gusto dell'uniformita': lo stop usa l'ATR e
    l'estremo a `i-1`, mai quelli della barra su cui viene testato, altrimenti la barra che fa
    scattare lo stop e' anche quella che decide dove stava.
    """
    indice = candele.index
    chiusure = candele["Close"].to_numpy()
    massimi = candele["High"].to_numpy()
    minimi = candele["Low"].to_numpy()
    atr = ExtraCache(candele).atr(atr_window)

    # L'innesco a 15 minuti: il prezzo deve rompere l'estremo delle ultime barre **precedenti**.
    # Con `innesco=0` e' spento, ed e' il default: e' un parametro in piu' e va acceso misurando.
    if innesco > 0:
        alto = candele["High"].rolling(innesco).max().shift(1).to_numpy()
        basso = candele["Low"].rolling(innesco).min().shift(1).to_numpy()
    else:
        alto = np.full(len(candele), -np.inf)
        basso = np.full(len(candele), np.inf)

    eventi: list = []
    ingressi: list[int] = []
    posizione = 0
    estremo = 0.0

    for i in range(1, len(candele)):
        prezzo = chiusure[i]
        uscito_ora = False

        if posizione != 0 and not np.isnan(atr[i - 1]):
            if posizione > 0:
                stop = estremo - atr_multiplier * atr[i - 1]
                if minimi[i] <= stop:
                    eventi.append((indice[i], float(stop), 0))
                    posizione, uscito_ora = 0, True
                else:
                    estremo = max(estremo, massimi[i])
            else:
                stop = estremo + atr_multiplier * atr[i - 1]
                if massimi[i] >= stop:
                    eventi.append((indice[i], float(stop), 0))
                    posizione, uscito_ora = 0, True
                else:
                    estremo = min(estremo, minimi[i])

        uscita = soglia[i] - isteresi
        if posizione > 0 and (punteggio[i] < uscita or regime[i] < 0):
            eventi.append((indice[i], float(prezzo), 0))
            posizione, uscito_ora = 0, True
        elif posizione < 0 and (punteggio[i] > -uscita or regime[i] > 0):
            eventi.append((indice[i], float(prezzo), 0))
            posizione, uscito_ora = 0, True

        # Chi e' appena uscito non rientra sulla stessa barra. L'isteresi frena il punteggio che
        # oscilla attorno alla soglia, ma non questo: uno stop scattato dentro la barra lascia il
        # punteggio dov'era, e senza il freno si ricomprerebbe subito pagando due commissioni per
        # tornare esattamente dov'eravamo.
        if posizione == 0 and not uscito_ora:
            lungo = regime[i] > 0 and punteggio[i] >= soglia[i] and concordi_lungo[i] >= k_famiglie and prezzo > alto[i]
            corto = (
                allow_short
                and regime[i] < 0
                and punteggio[i] <= -soglia[i]
                and concordi_corto[i] >= k_famiglie
                and prezzo < basso[i]
            )
            if lungo or corto:
                posizione = 1 if lungo else -1
                eventi.append((indice[i], float(prezzo), posizione))
                ingressi.append(i)
                estremo = prezzo

    return eventi, ingressi


def _necessarieta(voti, famiglie, pesi, soglia, ingressi, k_famiglie) -> dict[str, float]:
    """In che frazione degli ingressi ciascun votante era **indispensabile**.

    Indispensabile vuol dire: azzerandolo, quell'ingresso non sarebbe avvenuto -- perche' il
    punteggio scende sotto la soglia, o perche' l'ampiezza scende sotto il minimo. Sopra il 60%
    l'insieme e' quel votante travestito.
    """
    if not ingressi:
        return {nome: 0.0 for nome in voti}
    barre = np.array(ingressi)
    verso = np.sign(sum(pesi[n] * voti[n] for n in voti)[barre])
    verso[verso == 0] = 1

    conteggi = {}
    for nome in voti:
        restanti = {n: v for n, v in voti.items() if n != nome}
        punteggio = sum(pesi[n] * restanti[n] for n in restanti)[barre] if restanti else np.zeros(len(barre))
        sotto_soglia = punteggio * verso < soglia[barre]
        ampiezza = np.array([_famiglie_concordi(restanti, famiglie, int(v))[b] for b, v in zip(barre, verso)])
        conteggi[nome] = float(np.mean(sotto_soglia | (ampiezza < k_famiglie)))
    return conteggi
