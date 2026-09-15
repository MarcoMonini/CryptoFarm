"""La strategia a confluenza: quattro piani, sette votanti, una soglia che si muove.

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
look-ahead. Ma sollevare a valore provvisorio una *strategia* qualunque non e'
generico -- ogni indicatore ricorsivo va sollevato a mano -- e il disegno **congela i votanti**,
cioe' vieta di riscriverli. Quindi:

- **i sette votanti decidono alla chiusura della propria barra lunga** e il loro stato entra
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
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd

from cryptofarm.data.klines import interval_to_minutes, resample_klines
from cryptofarm.ml import signals
from cryptofarm.trading import strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.mtf import align_to_lower
from cryptofarm.trading.voters import PAVIMENTO_DEL_VOTO, decayed_vote, held_state

# I quattro piani, in multipli dell'intervallo di base. Su 15m: innesco 15m, conferma 1h,
# struttura 4h, regime 1d -- la scala del disegno, ottenuta senza scriverci dentro nessun
# intervallo fisso.
FATTORI = {"innesco": 1, "conferma": 4, "struttura": 16, "regime": 96}


# Il piano di regime deve durare quanto un regime: fra mezza giornata e una settimana. Sotto, la
# «macro» e' rumore con un nome altisonante; sopra, una media di cinquanta barre chiede decenni di
# storia. La scala x1/x4/x16/x96 e' nata su barre da quindici minuti e li' cade esatta su
# 15m/1h/4h/1d, ma il menu ne offre nove e non c'e' niente che impedisca di scegliere le altre.
# La finestra dell'ATR con cui si normalizza la distanza dalla media di un piano. E' un'unita' di
# misura, non un parametro: la tangente iperbolica rende il risultato insensibile al suo valore
# esatto, e renderlo tarabile aggiungerebbe un grado di liberta' che non compra niente.
ATR_DI_NORMALIZZAZIONE = 14

# Il tetto all'emivita di un voto, in **minuti di calendario** e non in barre, perche' e' una
# durata e deve voler dire la stessa cosa su ogni intervallo di base.
#
# L'emivita e' una sola e si converte in barre di base moltiplicando per `FATTORI[piano]`. Senza
# tetto il piano di regime arriva a `6 x 96 = 576` barre: su base 15m sono sei giorni di emivita,
# cioe' un votante che resta a forza quasi piena per settimane sulla scorta di un solo incrocio.
#
# Va detto per quel che vale: con il pavimento in casa questo tetto compra poco. Il segno del voto
# adesso e' l'opinione di adesso, non un ricordo, quindi la persistenza lunga di `zone_regime` non
# e' piu' un difetto del voto -- e' la sua opinione vera, il suo incrocio a 1d non e' avvenuto.
# Misurato, l'effetto e' sulla sola **forza**: 0,175 contro 0,154 di voto medio in posizione, e
# 388 ingressi contro 380. Resta perche' 576 barre sono comunque fuori scala e morderebbero su un
# votante che cambia idea piu' spesso, non perche' sposti i numeri di oggi.
TETTO_EMIVITA_MINUTI = 24 * 60

# Le due macchine di esecuzione. Non sono due tarature: rispondono a due domande diverse.
#
# `cancello` e' quella misurata su quindici asset e sette anni: sta fuori dal mercato per
# definizione e ci entra quando quattro condizioni cadono insieme, uscendo sullo stop,
# sull'isteresi o sul cancello. E' il default, e resta il default finche' l'altra non e' misurata.
#
# `inversione` sta **sempre a mercato**, lunga o corta, e la posizione e' il segno dell'ultimo
# attraversamento di soglia. Niente cancello, niente ampiezza minima, niente innesco, niente
# isteresi: la durata di un'operazione e' la distanza fra due attraversamenti opposti.
MODALITA = ("cancello", "inversione")

# Il moltiplicatore dello stop **per modalita'**, perche' in una delle due lo stop non e' un
# freno: e' il pilota.
#
# In `cancello` lo stop chiude e la posizione va a flat: sbagliarlo costa un'uscita anticipata.
# In `inversione` lo stop **ribalta**, quindi ogni volta che salta apre l'operazione successiva,
# e a 3 ATR salta di continuo. Misurato su BTCUSDT a 15m dal 2024 (92.321 barre), con la soglia
# a 0,35 e il collegio a sette: 5.328 ribaltamenti di cui **5.056 (95%) decisi dallo stop**,
# tenuta mediana 3,5 ore, capitale da 100 a 0,61. Con lo stop spento: 84 ribaltamenti, tutti dal
# punteggio, tenuta mediana 180 ore, capitale 182. Cioe' col default ereditato l'«always in» non
# seguiva i voti, seguiva un canale a 3 ATR -- che e' il difetto per cui questa costante esiste.
#
# Il valore 0 non e' un parametro nuovo: e' quello che `atr_multiplier` gia' aveva senza
# significato. Chi vuole lo stop nella modalita' a inversione lo accende sapendo che decide lui.
STOP_PREDEFINITO: dict[str, float] = {"cancello": 3.0, "inversione": 0.0}

# L'orientamento dei **voti e del punteggio**: -1 vuol dire lungo, +1 vuol dire corto.
#
# E' l'asse dell'etichetta a swing (`ml/labeling.swing_leg_target`: -1 su un minimo locale, +1 su
# un massimo), non quello delle posizioni. I due convivevano nel repo e sulla stessa pagina i due
# riquadri giravano su assi opposti: il *Voters* con +1 = lungo, lo *Swing target* con -1 = zona
# d'acquisto. Letti insieme sembravano darsi torto mentre dicevano la stessa cosa.
#
# **Gli eventi emessi restano nella convenzione di posizione** (+1 = lungo), che e' quella di
# `strategies_ls`, `pnl.simulate_positions`, `portfolio` e del bot live. Qui si rietichetta
# l'opinione, non l'ordine: nessuna operazione si sposta di una barra, e
# `test_l_inversione_del_segno_non_sposta_nessun_ordine` e' li' per tenerlo fermo.
VERSO_DEL_VOTO = -1


def convinzione(punteggio, verso):
    """Quanto il punteggio sostiene un'operazione in direzione `verso`, come numero **positivo**.

    `verso` e' nella convenzione di posizione (+1 lungo, -1 corto) perche' e' quella in cui e'
    scritto il resto del ciclo; il punteggio e' nell'orientamento dei voti. Questa funzione e'
    l'unico posto in cui i due si incontrano, ed e' voluto: sparse nelle disuguaglianze, le
    conversioni di segno sono esattamente il difetto che questo modulo ha gia' avuto due volte.
    Chi un domani rigira l'asse cambia `VERSO_DEL_VOTO` e nient'altro.
    """
    return punteggio * verso * VERSO_DEL_VOTO


REGIME_MIN_MINUTI = 12 * 60
REGIME_MAX_MINUTI = 7 * 24 * 60


def emivita_in_barre(emivita: float, piano: str, minuti_base: int, tetto_minuti: float | None = None) -> float:
    """L'emivita di un voto in barre dell'indice di base, tosata al tetto.

    `emivita` e' in barre del **piano** del votante; moltiplicarla per `FATTORI[piano]` la porta in
    barre di base. Il tetto e' in minuti di calendario perche' e' quel che si vuole limitare: sei
    giorni di memoria restano sei giorni tanto a 15m quanto a 1h, mentre «576 barre» vuol dire due
    durate diverse. `tetto_minuti=None` toglie il tetto, ed e' l'ablazione con cui si misura
    quanto vale averlo.
    """
    barre = emivita * FATTORI[piano]
    if tetto_minuti is None:
        return barre
    return min(barre, max(tetto_minuti / minuti_base, 1.0))


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


class Par(NamedTuple):
    """Un parametro di un votante: come si chiama nei tre posti in cui vive.

    Tre nomi e non uno perche' i tre posti sono davvero diversi e farli coincidere a forza
    costringerebbe a rinominare cose che appartengono ad altri: `config` e' lo spazio dei widget
    della pagina, dove serve un prefisso per non collidere con la strategia omonima del menu;
    `kwarg` e' l'argomento della funzione in `strategies_ls`, che non sa niente della pagina;
    `misurato` e' la chiave in `tuned_defaults`, che e' un file **generato** e non si rinomina.
    """

    config: str
    kwarg: str
    misurato: str = ""


@dataclass(frozen=True)
class Votante:
    """Un votante: chi e', su che piano guarda, come si esegue e quali manopole ha.

    `famiglia` non e' decorativa: l'ampiezza minima si conta in **famiglie distinte**, non in
    votanti, ed e' cio' che impedisce a un peso grande di aprire una posizione da solo.

    `menu` e' il nome con cui la strategia compare in `tuned_defaults`, vuoto per chi non e' mai
    stato misurato: quei votanti restano ai default scritti in `config`.

    ## Aggiungere o togliere un votante

    Si scrive la funzione, si costruisce un `Votante` e lo si passa a `registra`. Da li' in poi
    tutto il resto si adatta da solo: il conteggio delle famiglie, i pesi, la necessarieta', i
    widget della barra laterale, la griglia del banco. Toglierlo e' `selezione(...)` senza il suo
    nome. **Non c'e' nessun elenco da tenere allineato a mano**, ed e' l'unica forma di modularita'
    che conta: quella in cui dimenticarsi un posto non e' possibile.
    """

    nome: str
    famiglia: str
    piano: str
    esegui: Callable[[pd.DataFrame, ExtraCache, dict], list]
    parametri: tuple[Par, ...] = ()
    menu: str = ""


REGISTRO: dict[str, Votante] = {}


def registra(votante: Votante) -> Votante:
    """Mette un votante nel registro. Sostituisce quello con lo stesso nome, se c'e'."""
    REGISTRO[votante.nome] = votante
    return votante


def selezione(*nomi: str) -> tuple[Votante, ...]:
    """I votanti indicati, nell'ordine dato. Senza argomenti, tutti quelli registrati."""
    if not nomi:
        return tuple(REGISTRO.values())
    mancanti = [nome for nome in nomi if nome not in REGISTRO]
    if mancanti:
        raise KeyError(f"votanti sconosciuti: {mancanti}. Registrati: {sorted(REGISTRO)}")
    return tuple(REGISTRO[nome] for nome in nomi)


def _ichimoku(df, cache, p):
    return strategies_ls.ichimoku_trend(
        df,
        cache,
        fast=int(p["fast"]),
        slow=int(p["slow"]),
        span=int(p["span"]),
        require_cloud=bool(p["require_cloud"]),
    )


def _donchian(df, cache, p):
    return strategies_ls.donchian_breakout(
        df,
        cache,
        channel=int(p["channel"]),
        adx_window=int(p["adx_window"]),
        adx_min=float(p["adx_min"]),
        atr_window=int(p["atr_window"]),
        atr_multiplier=float(p["atr_multiplier"]),
        regime_ema=int(p["regime_ema"]),
    )


def _squeeze(df, cache, p):
    return strategies_ls.squeeze_breakout(
        df,
        cache,
        bb_window=int(p["bb_window"]),
        bb_dev=float(p["bb_dev"]),
        kc_window=int(p["kc_window"]),
        kc_multiplier=float(p["kc_multiplier"]),
        atr_window=int(p["atr_window"]),
        atr_multiplier=float(p["atr_multiplier"]),
        confirm_volume=bool(p["confirm_volume"]),
        obv_window=int(p["obv_window"]),
    )


def _pullback(df, cache, p):
    return strategies_ls.trend_pullback(
        df,
        cache,
        regime_ema=int(p["regime_ema"]),
        stochrsi_window=int(p["stochrsi_window"]),
        stochrsi_smooth=int(p["stochrsi_smooth"]),
        oversold=float(p["oversold"]),
        overbought=float(p["overbought"]),
        atr_multiplier=float(p["atr_multiplier"]),
    )


def _reversione(df, cache, p):
    return strategies_ls.band_reversion_gated(
        df,
        cache,
        kama_window=int(p["kama_window"]),
        band_multiplier=float(p["band_multiplier"]),
        adx_max=float(p["adx_max"]),
        stop_multiplier=float(p["stop_multiplier"]),
    )


def _bande(df, cache, p):
    """Le bande ATR come votante, e **con il verso corto acceso, esplicitamente**.

    `atr_band_bounce` e' l'unica di `strategies_ls` che ha `allow_short=False` per default, e
    quel default e' giusto **per una strategia**: aperta da sola su un asset con deriva positiva,
    la gamba corta paga quattro anni su cinque di essere dalla parte sbagliata della deriva.

    Un votante non e' una strategia: non apre niente, dice un'opinione. Prendere il default
    silenziosamente rendeva questa famiglia -- due votanti su sette, i due della famiglia
    `bande` -- **strutturalmente incapace di votare corto**, e la conseguenza non era una
    prudenza: era che `concordi_corto` non poteva mai arrivare a tutte le famiglie, e che sul
    punteggio l'estensione *sopra* la media non pesava mentre quella *sotto* pesava. Un'asimmetria
    fra i due versi che nessuno aveva deciso e che nessun test vedeva.

    E' anche l'unica famiglia per cui la misura sostiene il verso corto: sul ritorno alla media il
    corto ha il 52,3% di operazioni in utile e un contributo mediano di −3,6%, cioe' costa quasi
    niente (`.claude/docs/strategie-nuove.md` §4.5). Vendere un'estensione sopra la media in un
    mercato laterale e' simmetrico a comprarne una sotto, ed e' esattamente cio' che questo
    votante deve poter dire.
    """
    return strategies_ls.atr_band_bounce(
        df,
        cache,
        kama_window=int(p["kama_window"]),
        band_multiplier=float(p["band_multiplier"]),
        stop_multiplier=float(p["stop_multiplier"]),
        allow_short=True,
    )


def _zone(df, cache, p):
    return strategies_ls.trend_zone(df, cache, fast=int(p["fast"]), slow=int(p["slow"]))


def _flusso(df, cache, p):
    """Il votante di flusso: l'unico che non legge il prezzo, e per questo l'unico che decorrela.

    Non e' una strategia di `strategies_ls` -- non ne esiste una di questa famiglia -- ma le due
    serie ci sono gia' in `ExtraCache`. Vota lungo quando l'OBV sale **e** l'MFI non e' in
    ipercomprato, corto allo specchio: la pendenza dice la direzione del flusso, l'MFI dice se il
    flusso e' gia' stato tutto speso.
    """
    finestra = int(p["finestra"])
    pendenza, mfi = cache.obv_slope(finestra), cache.mfi(finestra)
    lungo = (pendenza > 0) & (mfi < float(p["mfi_alto"]))
    corto = (pendenza < 0) & (mfi > float(p["mfi_basso"]))
    stato = np.where(lungo, 1, np.where(corto, -1, 0)).astype(np.int8)
    stato[np.isnan(pendenza) | np.isnan(mfi)] = 0
    cambi = np.flatnonzero(np.diff(stato, prepend=np.int8(0)))
    chiusure = df["Close"].to_numpy()
    return [(df.index[i], float(chiusure[i]), int(stato[i])) for i in cambi]


def _modello(df, cache, p):
    """Il votante a modello: **solo lungo, o zitto**, e non e' una semplificazione.

    Il modello a swing prevede la prossimita' a un estremo locale, e la forma misurata di quel
    segnale e' a U: sia il polo -1 (vicino a un minimo) sia il polo +1 (tendenza forte in corso)
    precedono rendimenti sopra la media, il centro sta sotto
    (`.claude/docs/modello-swing.md` §5.1). Il segno **non** dice il verso, quindi far votare
    `sign(previsione)` darebbe alla confluenza un voto corto sulle barre che rendono di piu'.
    Cio' che il modello sa dire e' *quanto* stare esposti, e qui lo dice votando +1 quando
    `|previsione|` e' alta e 0 quando e' bassa. Non vota mai corto.

    Ne segue che il suo contributo e' asimmetrico: sui ribassi non aggiunge, si toglie di mezzo.
    E' l'unico votante che ha questa forma, ed e' anche il motivo per cui sta in una famiglia sua:
    l'ampiezza minima si conta in famiglie, e un votante che non sa dire il verso non deve poter
    completare un consenso corto.

    Senza artefatto restituisce zero eventi. Non e' un caso limite da tollerare, e' la condizione
    del servizio pubblico, dove `models/` e' vuoto.

    Le due colonne di posizionamento restano NaN: `esegui` riceve le candele, non il simbolo. La
    perdita e' misurata e vale un decimillesimo di IC (§4, «senza posizionamento» +0,0539 contro
    +0,0540), quindi non vale un parametro in piu' nella firma di tutti gli altri votanti.

    **Quale modello.** Quello in testa a `trainer.MODEL_PRECEDENCE`, non uno scelto qui: se
    c'e' il modello d'ingresso vota quello -- +1 mentre una sua operazione e' aperta -- altrimenti
    la politica RL, altrimenti il modello a swing. Nei primi due casi `entra`/`esci` non hanno
    effetto: la selettivita' del modello d'ingresso sta nei metadata del suo artefatto e la soglia
    della politica sta dentro l'obiettivo con cui e' stata addestrata. Un secondo votante a modello sarebbe stato
    la scelta comoda, ed e' sbagliata due volte: i due rispondono alla stessa domanda a partire
    dalle stesse 41 colonne, quindi voterebbero insieme, e l'ampiezza si conta in famiglie proprio
    per non far pesare due volte la stessa opinione.
    """
    # Fuori dalla sua scala il modello d'ingresso non e' piu' quello misurato (`entry_fuori_misura`):
    # li' si passa al successivo in `MODEL_PRECEDENCE` invece di votare con una soglia che a 1d
    # marcherebbe una barra su quattro. A base 15m il piano d'innesco e' 15m e il caso non si pone;
    # a base 1h si pone su tutti e quattro i piani.
    utilizzabili = () if signals.entry_fuori_misura(df.index) else (signals.ENTRY_VELOCE, signals.ENTRY_LENTO)
    nome = next((n for n in utilizzabili if signals.entry_model_disponibile(n)), "")
    politica = signals.rl_model()
    if nome:
        # Il modello d'ingresso vota +1 mentre una sua operazione e' aperta. Le due soglie non
        # hanno effetto: qui la selettivita' viene dai metadata dell'artefatto, ed e' l'unica cosa
        # da cui viene il vantaggio misurato.
        servizio = signals.entry_metadata(nome)
        frame = signals.swing_features(df)
        dentro = signals.entry_exposure(
            signals.entry_predictions(df, signals.entry_model(nome), frame=frame),
            float(servizio["soglia"]),
            signals.entry_tenuta(df.index, servizio),
            signals.entry_gate(df, frame=frame) if nome != signals.ENTRY_LENTO else None,
        )
    elif politica is not None:
        dentro = signals.rl_exposure(politica, df)
    else:
        modello = signals.swing_model()
        if modello is None:
            return []
        previsto = signals.swing_predictions(df, modello)
        dentro = signals.swing_exposure(previsto, float(p["entra"]), float(p["esci"]), signals.swing_cadenza(df.index))
    stato = dentro.astype(np.int8)
    cambi = np.flatnonzero(np.diff(stato, prepend=np.int8(0)))
    chiusure = df["Close"].to_numpy()
    return [(df.index[i], float(chiusure[i]), int(stato[i])) for i in cambi]


for _votante in (
    Votante(
        "ichimoku",
        "inseguimento",
        "struttura",
        _ichimoku,
        (
            Par("CONF_ICHIMOKU_FAST", "fast", "ICHIMOKU_FAST"),
            Par("CONF_ICHIMOKU_SLOW", "slow", "ICHIMOKU_SLOW"),
            Par("CONF_ICHIMOKU_SPAN", "span", "ICHIMOKU_SPAN"),
            Par("CONF_ICHIMOKU_CLOUD", "require_cloud", "REQUIRE_CLOUD"),
        ),
        "Ichimoku Trend",
    ),
    Votante(
        "flusso",
        "volume",
        "struttura",
        _flusso,
        (
            Par("CONF_FLOW_WINDOW", "finestra", "OBV_WINDOW"),
            Par("CONF_FLOW_MFI_ALTO", "mfi_alto"),
            Par("CONF_FLOW_MFI_BASSO", "mfi_basso"),
        ),
        "Squeeze Breakout",
    ),
    Votante(
        "pullback",
        "rientro",
        "conferma",
        _pullback,
        (
            Par("CONF_PULLBACK_REGIME_EMA", "regime_ema"),
            Par("CONF_PULLBACK_STOCH_WINDOW", "stochrsi_window"),
            Par("CONF_PULLBACK_STOCH_SMOOTH", "stochrsi_smooth"),
            Par("CONF_PULLBACK_OVERSOLD", "oversold"),
            Par("CONF_PULLBACK_OVERBOUGHT", "overbought"),
            Par("CONF_PULLBACK_ATR_MULT", "atr_multiplier"),
        ),
    ),
    Votante(
        "bande_conferma",
        "bande",
        "conferma",
        _bande,
        (
            Par("CONF_BANDE_KAMA", "kama_window"),
            Par("CONF_BANDE_BAND_MULT", "band_multiplier"),
            Par("CONF_BANDE_STOP_MULT", "stop_multiplier"),
        ),
    ),
    Votante(
        "bande_innesco",
        "bande",
        "innesco",
        _bande,
        (
            Par("CONF_BANDE_KAMA_VELOCE", "kama_window"),
            Par("CONF_BANDE_BAND_MULT_VELOCE", "band_multiplier"),
            Par("CONF_BANDE_STOP_MULT_VELOCE", "stop_multiplier"),
        ),
    ),
    Votante(
        "zone_regime",
        "macrostruttura",
        "regime",
        _zone,
        (Par("CONF_ZONE_FAST", "fast"), Par("CONF_ZONE_SLOW", "slow")),
    ),
    Votante(
        "zone_struttura",
        "macrostruttura",
        "struttura",
        _zone,
        (Par("CONF_ZONE_FAST_STRUTTURA", "fast"), Par("CONF_ZONE_SLOW_STRUTTURA", "slow")),
    ),
    Votante(
        "modello",
        "modello",
        # Sul piano di base, che e' il piu' vicino ai 5m su cui il modello e' addestrato. E' anche
        # l'unico piano su cui `_stato_del_votante` non passa da `align_to_lower`, e va bene:
        # `build_swing_features` allinea da se' le proprie scale lunghe, quindi il ritardo c'e'
        # gia' dove serve e aggiungerne un altro sposterebbe il segnale senza renderlo piu' onesto.
        "innesco",
        _modello,
        (Par("CONF_MODELLO_ENTRA", "entra"), Par("CONF_MODELLO_ESCI", "esci")),
    ),
):
    registra(_votante)


def votanti_predefiniti() -> tuple[Votante, ...]:
    """I sette votanti a indicatore. **Il votante a modello non e' nel default**, mai.

    Prima la condizione era l'artefatto su disco, e la domanda era sbagliata. Quel che rovina
    l'insieme non e' un votante assente: e' un votante **presente e muto**. I pesi sono fissi e
    si normalizzano sui votanti del collegio, quindi presuppongono che tutti parlino con frequenza
    confrontabile; chi tace non e' neutro, toglie il suo peso dal punteggio di tutti gli altri su
    ogni barra in cui non parla, cioe' **alza la soglia senza dirlo**.

    `modello` tace per disegno, non per caso. La selettivita' del modello d'ingresso sta nei
    metadata del suo artefatto ed e' tutto il suo vantaggio misurato (`modello-ingresso.md`): a
    cinque minuti marca una barra su millecinquecento. Misurato su quindici simboli a 15m dal 2021,
    197.371 barre ciascuno, la frazione di barre in cui ciascun votante tiene una posizione:

        modello        0,004 - 0,033      (mediana 0,020)
        pullback       0,253 - 0,282
        ichimoku       0,272 - 0,318
        bande_conferma 0,831 - 0,915
        bande_innesco  0,892 - 0,949
        zone_regime    0,950
        flusso         0,957 - 0,971
        zone_struttura 0,996

    Un ordine di grandezza sotto il penultimo, su tutti e quindici. Con un ottavo del peso e lo
    0,2% delle barre, su BTCUSDT il punteggio passava da deviazione 0,161 a 0,141 e le barre sopra
    0,35 da **2,22% a 0,82%**: un terzo delle occasioni degli altri sette, tolte da chi non stava
    votando.

    **Quel che questo non e'.** Non e' un miglioramento del rendimento, e va detto perche' e'
    misurato ed e' il contrario: a soglia ferma, toglierlo fa salire il punteggio, quindi si opera
    di piu' (BTCUSDT a cancello, 766 operazioni contro 1.088) e su questi dati operare di piu'
    peggiora sempre -- mediana sui quindici simboli da -48,6% a -57,5%. La diluizione stava
    facendo da rialzo occulto della soglia, e un rialzo occulto della soglia e' esattamente cio'
    che non si vuole: la soglia si sceglie e si misura, non si eredita da chi tace.
    **Ne segue che `theta_base` va misurato sul collegio corretto**, ed e' quel che fa la sezione
    sulla modalita' a inversione in `.claude/docs/strategia-confluenza.md`.

    Resta nel registro, cosi' `selezione("modello", ...)` lo raggiunge sempre: fuori dal default
    non vuol dire fuori dalla misura. Chi vuole rimetterlo nel collegio deve prima dargli un peso
    che tenga conto di quanto parla, che e' un disegno diverso da quello a pesi fissi.

    Il collegio non dipende piu' da cosa c'e' in `models/`: la pagina in locale e il servizio
    pubblico girano lo stesso insieme, e i test non cambiano esito con gli artefatti sul disco.
    """
    return tuple(v for v in selezione() if v.nome != "modello")


VOTANTI: tuple[Votante, ...] = votanti_predefiniti()


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
    stati: dict[str, np.ndarray]
    pesi: dict[str, float]
    punteggio: np.ndarray
    soglia: np.ndarray
    soglia_corta: np.ndarray
    regime: np.ndarray
    struttura: np.ndarray
    famiglie_concordi: np.ndarray
    stop: np.ndarray | None = None
    motivi: dict = field(default_factory=dict)
    modalita: str = "cancello"
    # Quante volte il punteggio ha riribaltato la posizione sulla barra **subito dopo** un
    # ribaltamento dello stop. E' il costo dichiarato dello stop che inverte invece di chiudere, e
    # sta qui perche' la decisione di mettergli un freno si prenda su un numero e non a sensazione.
    ping_pong: int = 0
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
        if self.modalita == "inversione":
            # Qui le condizioni non sono quattro in `and`: ce n'e' una sola, e zero operazioni
            # vuol dire che il punteggio non ha mai attraversato la soglia in nessuno dei due versi.
            picco = float(np.abs(self.punteggio).max()) if len(self.punteggio) else 0.0
            return (
                f"the score never crossed the threshold in either direction: it peaked at "
                f"{picco:.2f} against a threshold of {self.soglia[0]:.2f}. Lower «Entry threshold»."
            )
        if self.barre_del_regime < self.barre_chieste_dal_regime:
            return (
                f"not enough history: the regime plane has {self.barre_del_regime} bars and its "
                f"moving average needs {self.barre_chieste_dal_regime}. The gate stays shut, so no "
                "long entry is possible. Load a longer window."
            )
        aperto = self.regime > 0
        if not aperto.any():
            return "the regime gate never opened: price stayed below the regime plane average the whole window."
        sopra = aperto & (convinzione(self.punteggio, +1) >= self.soglia)
        if not sopra.any():
            # Il punteggio e' nell'orientamento dei voti, quindi il consenso lungo piu' forte e'
            # il suo **minimo**. Si riporta a numero positivo per stamparlo accanto alla soglia,
            # che e' una magnitudine: «picco +0,28 contro una soglia di 0,35» si legge, «-0,28
            # contro 0,35» fa sembrare che manchi il segno.
            picco = float(convinzione(self.punteggio[aperto], +1).max())
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
            (*evento[:3], abs(self.punteggio[i]) - self._soglia_del_verso(evento[2])[i] if evento[2] != 0 else 0.0)
            for evento, i in zip(self.eventi, posizioni)
        ]

    def _soglia_del_verso(self, verso: int) -> np.ndarray:
        """La soglia contro cui quel verso si misura. Non e' la stessa: il macro le muove
        all'opposto, perche' un quadro che da' ragione al lungo la toglie al corto."""
        return self.soglia if verso >= 0 else self.soglia_corta

    def spiega(self, quando) -> str:
        """Perche' quella barra ha operato. Una riga, e **diversa per ogni modalita'**.

        In `cancello` distingue ingressi e uscite, e non e' una rifinitura: quattro uscite su
        cinque sono lo stop a trailing, e su quelle il punteggio e i votanti non c'entrano niente.
        Mostrarli lo stesso -- com'era scritto la prima volta -- fa leggere «venduto mentre cinque
        votanti dicevano di comprare», che e' vero e del tutto fuorviante.

        In `inversione` **non esistono uscite**: ogni evento e' un ribaltamento, e la riga dice il
        verso in cui si e' finiti. Riusare il ramo di `cancello` scriveva «exit — trailing stop
        reversal» sopra un marcatore d'acquisto, che e' la stessa illeggibilita' al contrario.
        """
        quando = pd.Timestamp(quando)
        i = self.indice.get_indexer([quando], method="pad")[0]
        if i < 0:
            return ""
        motivo = self.motivi.get(quando)
        if self.modalita == "inversione":
            # Qui **non esistono uscite**: ogni evento e' un ribaltamento, e chiamarlo «exit» su un
            # marcatore d'acquisto e' esattamente il modo di rendere illeggibile un grafico. La riga
            # dice il verso in cui si e' finiti, poi cosa l'ha deciso.
            verso = next((e[2] for e in self.eventi if e[0] == quando), 0)
            if not verso:
                return ""
            testa = "long" if verso > 0 else "short"
            if motivo == "trailing stop reversal":
                livello = f" at {self.stop[i]:.2f}" if self.stop is not None and not np.isnan(self.stop[i]) else ""
                return f"{testa} — reversed by the trailing stop{livello}"
            parti = [
                f"{nome} {self.pesi[nome] * voto[i]:+.2f}" for nome, voto in self.voti.items() if abs(voto[i]) > 1e-9
            ]
            limite = VERSO_DEL_VOTO * verso * self.soglia[i]
            return f"{testa} — score {self.punteggio[i]:+.2f} crossed {limite:+.2f} · " + ", ".join(
                parti or ["no active voter"]
            )
        if motivo:
            coda = ""
            if motivo == "trailing stop" and self.stop is not None and not np.isnan(self.stop[i]):
                coda = f" at {self.stop[i]:.2f}"
            return f"exit — {motivo}{coda}"
        parti = [f"{nome} {self.pesi[nome] * voto[i]:+.2f}" for nome, voto in self.voti.items() if abs(voto[i]) > 1e-9]
        # La soglia si stampa **con il segno del verso in cui si e' entrati**, altrimenti accanto a
        # un punteggio di -0,41 comparirebbe una soglia di 0,35 e si leggerebbe come un ingresso
        # avvenuto sotto la soglia. Punteggio e soglia devono stare sullo stesso asse: -1 e' lungo.
        verso = 1 if convinzione(self.punteggio[i], +1) >= 0 else -1
        limite = VERSO_DEL_VOTO * verso * self._soglia_del_verso(verso)[i]
        return (
            f"entry — score {self.punteggio[i]:+.2f} / threshold {limite:+.2f} · "
            f"{int(self.famiglie_concordi[i])} families · " + ", ".join(parti or ["no active voter"])
        )


def _intervallo(minuti: int) -> str:
    """Da minuti alla stringa che `resample_klines` capisce, senza tabelle di corrispondenza."""
    if minuti % 1440 == 0:
        return f"{minuti // 1440}d"
    if minuti % 60 == 0:
        return f"{minuti // 60}h"
    return f"{minuti}m"


def valori_del_votante(votante: Votante, intervallo: str, override: dict | None = None) -> dict:
    """I parametri con cui girera' quel votante, in tre strati sovrapposti.

    1. il default della funzione, scritto in `config` come `CONF_*`;
    2. il valore **misurato** per l'intervallo del piano su cui il votante gira, se una misura
       esiste in `tuned_defaults`. Nota bene: l'intervallo del *piano*, non quello scelto nella
       pagina -- un votante di struttura a base 15m gira a 4h, e il suo valore misurato e' quello
       di 4h. Prendere quello di 15m sarebbe sbagliato in silenzio;
    3. l'override esplicito di chi chiama, che e' cio' che arriva dai widget e dalla griglia.

    Il terzo strato e' quello che il disegno originale non aveva: i votanti erano **congelati**
    perche' ritararli dentro l'insieme porta il conto dei parametri liberi da nove a oltre
    quaranta. Ora si possono muovere, ed e' una scelta esplicita di chi usa la pagina; il conto
    delle prove per la correzione di molteplicita' deve tenerne conto.
    """
    from cryptofarm.trading import config
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    misurati = PER_INTERVALLO.get(intervallo, {}).get(votante.menu, {}) if votante.menu else {}
    valori = {}
    for parametro in votante.parametri:
        default = getattr(config, parametro.config).value
        valori[parametro.kwarg] = misurati.get(parametro.misurato, default) if parametro.misurato else default
    valori.update((override or {}).get(votante.nome, {}))
    return valori


def _pesi(nomi: list[str], w_max: float, pesi: dict[str, float] | None = None) -> dict[str, float]:
    """Pesi normalizzati a somma 1 con tetto `w_max`, applicato ripetutamente fino a tenuta.

    A pesi uguali il tetto non morde mai -- con sette votanti ognuno vale 0,143 contro un tetto di
    0,30 -- e questo e' voluto: il tetto e' li' per la versione tarata (S7), dove serve a impedire
    che l'insieme diventi *un* segnale con delle decorazioni.
    """
    valori = {n: 1.0 for n in nomi} if pesi is None else {n: float(pesi.get(n, 0.0)) for n in nomi}
    totale = sum(valori.values())
    if totale <= 0:
        raise ValueError("pesi tutti nulli")
    valori = {n: p / totale for n, p in valori.items()}
    # Con `n` votanti non esiste nessuna assegnazione che sommi a 1 e stia tutta sotto `1/n`:
    # applicando il tetto alla lettera si capperebbero tutti e la somma verrebbe minore di uno,
    # cioe' il punteggio sarebbe sistematicamente piu' piccolo della soglia **senza che niente lo
    # dica**. Con sette votanti non si vedeva (0,143 sta sotto 0,30); con tre la somma faceva 0,90.
    w_max = max(w_max, 1.0 / len(nomi))
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
    candles: pd.DataFrame,
    interval: str,
    votanti: tuple[Votante, ...] = VOTANTI,
    parametri_votanti: dict | None = None,
) -> dict[str, np.ndarray]:
    """Lo stato per barra dei votanti, che e' **la parte cara e l'unica che non dipende dalla griglia**.

    I parametri dei votanti sono congelati, quindi il loro stato dipende solo da (simbolo,
    intervallo): si calcola una volta e si riusa su tutte le configurazioni. Sulla griglia larga
    di `scripts/confluence_lab.py` questo e' la differenza fra ore e minuti, e non costa niente in
    correttezza -- e' lo stesso valore, calcolato una volta invece che mille.
    """
    minuti_base = interval_to_minutes(interval)
    return {v.nome: _stato_del_votante(v, candles, minuti_base, parametri_votanti) for v in votanti}


def _stato_del_votante(votante, candele, minuti_base, parametri_votanti=None) -> np.ndarray:
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

    valori = valori_del_votante(votante, intervallo, parametri_votanti)
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
    barre_minime: int = 4,
    pazienza: int = 24,
    emivita: float = 6.0,
    pavimento: float = PAVIMENTO_DEL_VOTO,
    tetto_emivita: float | None = TETTO_EMIVITA_MINUTI,
    w_max: float = 0.30,
    k_famiglie: int = 2,
    innesco: int = 0,
    atr_window: int = 14,
    atr_multiplier: float | None = None,
    regime_ema: int = 50,
    struttura_ema: int = 50,
    barre_in_formazione: bool = True,
    allow_short: bool = False,
    modalita: str = "cancello",
    pesi: dict[str, float] | None = None,
    votanti: tuple[Votante, ...] = VOTANTI,
    parametri_votanti: dict | None = None,
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

    `pavimento` e `tetto_emivita` sono le due manopole della forma del voto, e stanno qui con i
    loro default e **senza un widget**: servono a misurare l'ablazione (`pavimento=0` e
    `tetto_emivita=None` riportano al voto che seguiva la sola recenza), non a essere girate a
    occhio sulla pagina. Il conto dei parametri liberi della confluenza e' gia' il problema che
    `scripts/multiplicity.py` misura.
    """
    if len(candles) < 3:
        raise ValueError("servono almeno tre barre")
    if modalita not in MODALITA:
        raise ValueError(f"modalita sconosciuta: {modalita!r}. Sono {MODALITA}")
    # Lo stop dipende dalla modalita' e non e' una taratura: le ragioni, con i numeri, stanno su
    # `STOP_PREDEFINITO`. Si risolve qui perche' qui passano tutti -- la pagina, il banco e chi
    # chiama da libreria -- e un default per modalita' scritto in tre posti sarebbe tre posti in
    # cui dimenticarsene. Un valore esplicito vince sempre, anche lo zero.
    if atr_multiplier is None:
        atr_multiplier = STOP_PREDEFINITO[modalita]
    if stati is not None and parametri_votanti:
        # Gli stati precalcolati valgono per i parametri con cui sono stati calcolati. Accettarli
        # insieme a un override darebbe un risultato sbagliato **senza dirlo**, che e' il modo in
        # cui una memoization rovina una misura.
        raise ValueError("`stati` precalcolati e `parametri_votanti` insieme: gli stati sarebbero vecchi")
    minuti_base = interval_to_minutes(interval)

    voti: dict[str, np.ndarray] = {}
    stati_usati: dict[str, np.ndarray] = {}
    famiglie: dict[str, str] = {}
    for votante in votanti:
        stato = stati[votante.nome] if stati else _stato_del_votante(votante, candles, minuti_base, parametri_votanti)
        # L'emivita e' una sola, espressa in barre del timeframe del votante: qui si converte in
        # barre di base e si tosa a `TETTO_EMIVITA_MINUTI`. Un segnale di struttura resta vivo
        # giorni, uno di innesco ore, e nessuno piu' di un giorno di calendario.
        #
        # `decayed_vote` lavora sullo stato tenuto, che e' una posizione: +1 e' lungo. Il voto
        # esce nell'orientamento dichiarato da `VERSO_DEL_VOTO`, e questa e' l'unica moltiplicazione
        # che lo produce -- da qui in giu' tutto legge voti e punteggio in quell'asse.
        stati_usati[votante.nome] = stato
        voti[votante.nome] = VERSO_DEL_VOTO * decayed_vote(
            stato, emivita_in_barre(emivita, votante.piano, minuti_base, tetto_emivita), pavimento
        )
        famiglie[votante.nome] = votante.famiglia

    w = _pesi([v.nome for v in votanti], w_max, pesi)
    punteggio = sum(w[n] * voti[n] for n in voti)

    regime = _forza_del_piano(candles, minuti_base, "regime", regime_ema, barre_in_formazione)
    struttura = _forza_del_piano(candles, minuti_base, "struttura", struttura_ema, barre_in_formazione)
    # I piani lunghi scontano la soglia **nel verso dell'operazione candidata**, e il verso in
    # questa formula e' cio' che mancava: `theta_base - theta_macro * macro` da sola abbassa la
    # soglia quando il macro sale, per tutti e due i versi. Sul lungo e' quel che si vuole; sul
    # corto e' il contrario esatto del disegno -- con regime e struttura entrambi a -1 la soglia
    # saliva a 0,50 proprio mentre il quadro macro dava ragione al corto, mentre il lungo con
    # macro a +1 ne chiedeva 0,20. Misurato sulle stesse candele: 0,201 contro 0,499.
    # La media sui piani **noti**: un piano che non si sa si astiene, non vota «neutro». Dove non
    # se ne sa nessuno dei due la soglia resta `theta_base`, che e' lo sconto nullo.
    # Scritta a mano invece che con `nanmean`, che su una barra senza nessun piano noto avverte e
    # restituisce NaN: qui quel caso non e' un'anomalia, e' la pagina appena aperta.
    noti = np.isfinite(regime).astype(float) + np.isfinite(struttura).astype(float)
    somma = np.nan_to_num(regime) + np.nan_to_num(struttura)
    macro = np.divide(somma, noti, out=np.zeros_like(somma), where=noti > 0)
    if modalita == "inversione":
        # Soglia **simmetrica**, e lo sconto macro spento: le ragioni, con i numeri, stanno in
        # `_percorri_inversione`. In breve: su una serie con deriva il piano di regime satura a
        # +1,000 con deviazione 0,000, quindi non modula niente e lascia solo un'asimmetria fissa
        # che rende la gamba corta irraggiungibile -- zero barre buone per un corto su 17.280.
        soglia = np.full(len(candles), float(theta_base))
        soglia_corta = soglia
    else:
        soglia = theta_base - theta_macro * macro
        soglia_corta = theta_base + theta_macro * macro

    # **Sullo stato, non sul voto.** L'ampiezza e' un conteggio di opinioni presenti, e un voto in
    # coda di decadimento non e' un'opinione: e' un ricordo. Con il pavimento e l'azzeramento
    # all'uscita i due conteggi coincidono per costruzione -- il voto e' zero se e solo se lo stato
    # lo e' -- ma dipendere da quell'invariante vorrebbe dire che cambiare la forma del voto
    # cambierebbe l'ampiezza di nascosto. Qui si chiede direttamente cio' che si vuole sapere.
    concordi_lungo = _famiglie_concordi(stati_usati, famiglie, +1)
    concordi_corto = _famiglie_concordi(stati_usati, famiglie, -1)
    famiglie_concordi = np.where(convinzione(punteggio, +1) >= 0, concordi_lungo, concordi_corto)

    ping_pong = 0
    if modalita == "inversione":
        eventi, ingressi, livello_stop, motivi, ping_pong = _percorri_inversione(
            candles,
            punteggio=punteggio,
            soglia=soglia,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier,
        )
    else:
        eventi, ingressi, livello_stop, motivi = _percorri(
            candles,
            punteggio=punteggio,
            soglia=soglia,
            soglia_corta=soglia_corta,
            regime=regime,
            concordi_lungo=concordi_lungo,
            concordi_corto=concordi_corto,
            isteresi=isteresi,
            barre_minime=barre_minime,
            pazienza=pazienza,
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
        stati=stati_usati,
        pesi=w,
        punteggio=punteggio,
        soglia=soglia,
        soglia_corta=soglia_corta,
        regime=regime,
        struttura=struttura,
        famiglie_concordi=famiglie_concordi,
        stop=livello_stop,
        motivi=motivi,
        concordi_lungo=concordi_lungo,
        k_famiglie=k_famiglie,
        barre_del_regime=len(resample_klines(candles, _intervallo(minuti_base * FATTORI["regime"]))),
        barre_chieste_dal_regime=regime_ema,
        ingressi=len(ingressi),
        modalita=modalita,
        ping_pong=ping_pong,
    )
    risultato.necessarieta = _necessarieta(voti, stati_usati, famiglie, w, soglia, soglia_corta, ingressi, k_famiglie)
    return risultato


def _forza_del_piano(candele, minuti_base, piano, span, in_formazione) -> np.ndarray:
    """Quanto il prezzo sta sopra (o sotto) la media del piano, in `[-1, +1]` e **con continuita'**.

    Prima era `np.sign(prezzo - media)`, cioe' -1, 0 o +1. Sembrava innocuo e non lo era: la
    soglia si ricava da qui, quindi prendeva cinque valori discreti e **saltava di 0,15 per volta**
    contro un punteggio la cui ampiezza totale e' 0,91. Misurato: una uscita per punteggio su
    quattro cadeva sulla barra esatta in cui la soglia era saltata, cioe' l'aveva decisa la soglia
    e non il punteggio.

    Ora la distanza dalla media si normalizza sull'ATR **dello stesso piano** -- cosi' il numero e'
    adimensionale e confrontabile fra asset e intervalli -- e si schiaccia con una tangente
    iperbolica. Il cancello resta lo stesso confronto di prima (`> 0` e' ancora «prezzo sopra la
    media»), ma la soglia si muove con continuita' invece che a scalini.

    L'ATR di normalizzazione ha finestra fissa a 14: **non e' un parametro libero**, e' l'unita' di
    misura. Renderlo tarabile aggiungerebbe un grado di liberta' per cambiare una scala che la
    tangente iperbolica gia' rende insensibile ai dettagli.

    `in_formazione` decide **quale prezzo** entra nel confronto: quello di adesso, che e' cio' che
    vede il bot live a meta' giornata, o quello dell'ultima chiusura del piano lungo, che e'
    l'attesa fino a un periodo intero. La differenza fra i due e' l'ablazione che misura quanto
    vale reagire prima della chiusura.

    ## Perche' la media resta quella chiusa in tutti e due i casi

    Sollevare la media a valore provvisorio qui non servirebbe: `provisional_ema` restituisce
    `a*prezzo + (1-a)*chiusa`, e per il **segno** vale `segno(prezzo - a*prezzo - (1-a)*chiusa) =
    segno(prezzo - chiusa)` per qualunque `a` in (0,1). Sul segno il sollevamento e'
    algebricamente un non-fare; sulla magnitudine cambia solo la scala, che la tangente iperbolica
    riassorbe. Il sollevamento serve dove conta il valore assoluto -- una banda, uno stop.
    """
    fattore = FATTORI[piano]
    intervallo = _intervallo(minuti_base * fattore)
    lungo = resample_klines(candele, intervallo) if fattore > 1 else candele
    if len(lungo) <= ATR_DI_NORMALIZZAZIONE:
        # Meno barre della finestra dell'ATR: `ta` solleverebbe un IndexError invece di dare NaN.
        # La risposta giusta e' «non lo so», che e' NaN e non zero -- degradare, non cadere, e'
        # la condizione in cui gira la pagina appena aperta.
        return np.full(len(candele), np.nan)

    cache = ExtraCache(lungo)
    media, ampiezza = cache.ema(span), cache.atr(ATR_DI_NORMALIZZAZIONE)
    prezzo = candele["Close"].to_numpy()
    if fattore > 1:
        media = align_to_lower(media, lungo.index, intervallo, candele.index)
        ampiezza = align_to_lower(ampiezza, lungo.index, intervallo, candele.index)
        if not in_formazione:
            prezzo = align_to_lower(lungo["Close"], lungo.index, intervallo, candele.index)

    with np.errstate(invalid="ignore", divide="ignore"):
        forza = np.tanh((prezzo - media) / (2.0 * ampiezza))
    # **NaN dove il piano non si sa, non zero.** Prima era `nan_to_num(..., nan=0.0)`, e uno zero
    # qui e' una bugia con due conseguenze. Sul grafico: un piano che chiede cinquanta barre e ne
    # ha venti si disegnava come una linea piatta a 0,0, indistinguibile da «prezzo esattamente
    # sulla sua media» -- il cancello risultava *neutro* mentre era *chiuso per ignoranza*, ed e'
    # esattamente il quadro in cui non si capisce perche' non arrivino segnali. Sulla soglia: uno
    # zero medializzato con il piano noto ne **dimezza** il contributo (misurato su venti giorni a
    # 15m: soglia media 0,371 contro 0,392), cioe' il piano ignoto vota, e vota «neutro».
    #
    # Il cancello non cambia comportamento: `NaN > 0` e' False, quindi resta chiuso come prima.
    # Cambia che adesso lo dice.
    forza[np.isinf(forza)] = np.nan
    return forza


def _famiglie_concordi(stati, famiglie, verso) -> np.ndarray:
    """Quante **famiglie distinte** tengono una posizione in quel verso, barra per barra.

    Famiglie e non votanti: e' il freno che impedisce a un peso grande, da solo, di aprire una
    posizione. Oggi la distinzione non morde -- i sei sono uno per famiglia -- ma mordera' appena
    si aggiunge un secondo votante di prezzo, ed e' allora che serve gia' scritta.

    **Prende gli stati, non i voti**, e `verso` e gli stati sono tutti e due nella convenzione di
    posizione (+1 lungo): qui non c'e' nessuna conversione d'asse da fare, perche' `held_state` non
    passa mai per l'orientamento dei voti. L'ampiezza e' un conteggio di opinioni **presenti** e un
    voto in coda di decadimento non e' un'opinione, e' un ricordo: con i voti, meta' delle
    «famiglie concordi» di `pullback` erano posizioni gia' chiuse.
    """
    per_famiglia: dict[str, np.ndarray] = {}
    for nome, stato in stati.items():
        attivo = (np.asarray(stato) * verso) > 0
        famiglia = famiglie[nome]
        per_famiglia[famiglia] = attivo if famiglia not in per_famiglia else (per_famiglia[famiglia] | attivo)
    return np.sum(list(per_famiglia.values()), axis=0).astype(float)


def _percorri(
    candele,
    *,
    punteggio,
    soglia,
    soglia_corta,
    regime,
    concordi_lungo,
    concordi_corto,
    isteresi,
    barre_minime,
    pazienza,
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

    ## L'isteresi ha un pavimento e un soffitto, e non sono simmetrici

    La banda di isteresi da sola sbagliava in tutte e due le direzioni. Verso il basso: si apriva e
    si chiudeva in due barre da quindici minuti, pagando due commissioni per tornare dov'eravamo.
    Verso l'alto: il punteggio decade piano, quindi restava appena sopra `soglia - isteresi` per
    ore -- mediana 14 barre oltre il primo calo sotto la soglia, coda a 84, cioe' ventun'ore.

    Quindi due limiti, e **valgono solo per l'uscita dal punteggio**:

    - `barre_minime` e' il pavimento: prima di quello il punteggio non puo' chiudere;
    - `pazienza` e' il soffitto: dopo tante barre consecutive sotto la soglia **semplice** si esce
      comunque, anche se il punteggio non e' mai caduto attraverso tutta la banda.

    **Lo stop e il cancello non sono soggetti al pavimento**, e la distinzione e' quella che conta:
    sono regole di rischio, non di opinione. Un pavimento che tiene aperta una posizione mentre lo
    stop e' saltato non e' pazienza, e' un difetto travestito da parametro.
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
    motivi: dict = {}
    barra_ingresso = -(10**9)
    barre_sotto = 0
    # Il livello dello stop, barra per barra, NaN quando si e' fuori. Non serve a decidere: serve
    # a **vederlo**. Quattro uscite su cinque sono lo stop, e senza questa serie il grafico mostra
    # una vendita mentre il punteggio e' tranquillamente sopra la soglia -- cioe' sembra incoerente
    # proprio dove e' piu' corretto.
    livello_stop = np.full(len(candele), np.nan)
    posizione = 0
    estremo = 0.0

    for i in range(1, len(candele)):
        prezzo = chiusure[i]
        uscito_ora = False

        if posizione != 0 and not np.isnan(atr[i - 1]):
            if posizione > 0:
                stop = estremo - atr_multiplier * atr[i - 1]
                livello_stop[i] = stop
                if minimi[i] <= stop:
                    eventi.append((indice[i], float(stop), 0))
                    motivi[indice[i]] = "trailing stop"
                    posizione, uscito_ora = 0, True
                else:
                    estremo = max(estremo, massimi[i])
            else:
                stop = estremo + atr_multiplier * atr[i - 1]
                livello_stop[i] = stop
                if massimi[i] >= stop:
                    eventi.append((indice[i], float(stop), 0))
                    motivi[indice[i]] = "trailing stop"
                    posizione, uscito_ora = 0, True
                else:
                    estremo = min(estremo, minimi[i])

        if posizione != 0:
            verso = 1 if posizione > 0 else -1
            # La soglia del verso in cui si e' dentro, non quella lunga per tutti e due: uscire
            # da un corto misurandosi contro la soglia lunga vuol dire che il macro favorevole al
            # corto *anticipa* l'uscita invece di ritardarla.
            attiva = soglia[i] if verso > 0 else soglia_corta[i]
            sostegno = convinzione(punteggio[i], verso)
            barre_sotto = barre_sotto + 1 if sostegno < attiva else 0
            per_isteresi = sostegno < attiva - isteresi
            per_pazienza = barre_sotto >= pazienza
            maturo = (i - barra_ingresso) >= barre_minime
            cancello_contro = regime[i] * verso < 0
            if cancello_contro or (maturo and (per_isteresi or per_pazienza)):
                eventi.append((indice[i], float(prezzo), 0))
                motivi[indice[i]] = (
                    "regime gate shut"
                    if cancello_contro
                    else (
                        "score below threshold for too long"
                        if per_pazienza and not per_isteresi
                        else "score fell through the hysteresis band"
                    )
                )
                posizione, uscito_ora = 0, True

        # Chi e' appena uscito non rientra sulla stessa barra. L'isteresi frena il punteggio che
        # oscilla attorno alla soglia, ma non questo: uno stop scattato dentro la barra lascia il
        # punteggio dov'era, e senza il freno si ricomprerebbe subito pagando due commissioni per
        # tornare esattamente dov'eravamo.
        if posizione == 0 and not uscito_ora:
            lungo = (
                regime[i] > 0
                and convinzione(punteggio[i], +1) >= soglia[i]
                and concordi_lungo[i] >= k_famiglie
                and prezzo > alto[i]
            )
            corto = (
                allow_short
                and regime[i] < 0
                and convinzione(punteggio[i], -1) >= soglia_corta[i]
                and concordi_corto[i] >= k_famiglie
                and prezzo < basso[i]
            )
            if lungo or corto:
                posizione = 1 if lungo else -1
                eventi.append((indice[i], float(prezzo), posizione))
                ingressi.append(i)
                estremo = prezzo
                barra_ingresso, barre_sotto = i, 0

    return eventi, ingressi, livello_stop, motivi


def _percorri_inversione(
    candele,
    *,
    punteggio,
    soglia,
    atr_window,
    atr_multiplier,
):
    """Il motore **sempre a mercato**: la posizione e' il segno dell'ultimo attraversamento.

    Una macchina diversa da `_percorri`, non una sua taratura. Le regole sono tre e basta:

    - **punteggio sotto `-soglia` -> lungo**, punteggio sopra `+soglia` -> corto. Sull'asse dei
      voti il lungo e' negativo (`VERSO_DEL_VOTO`), quindi «superata la soglia in negativo» e
      «compra» sono la stessa frase;
    - **fra i due attraversamenti non succede niente.** Nessuna isteresi, nessuna pazienza,
      nessun pavimento di barre: si tiene la posizione finche' il punteggio non attraversa la
      soglia *opposta*. E' da li' che viene la durata delle operazioni, e non da un parametro;
    - **non si va mai a flat.** Un attraversamento opposto non chiude e riapre: **ribalta**, in un
      evento solo, che e' cio' che `pnl.simulate_positions` sa eseguire e che le due liste
      separate di `simulate_trading_with_commisions` non saprebbero rappresentare.

    ## La soglia e' simmetrica, e qui non la sconta nessun macro

    `theta_macro` resta spento in questa modalita', e non per semplificare. Misurato su una serie
    con deriva positiva: il piano di regime satura a **+1,000 con deviazione standard 0,000** --
    il prezzo sta sopra la sua media a cinquanta giorni per tutta la finestra -- quindi lo sconto
    non modula niente, e' una costante. L'effetto e' solo di spostare in permanenza la soglia
    lunga a 0,208 e quella corta a 0,492: su 17.280 barre il punteggio bastava per un lungo in
    1.623 e per un corto in **zero**. In un sistema che deve stare sempre a mercato una asimmetria
    fissa non e' un'opinione sul macro, e' una gamba amputata.

    ## Lo stop ribalta, non chiude -- e con `atr_multiplier <= 0` non c'e'

    Stessa convenzione di `_percorri` -- ATR ed estremo a `i-1`, mai quelli della barra su cui lo
    stop viene testato -- ma l'esito e' diverso: la posizione si inverte invece di andare a flat,
    cosi' il sistema resta sempre a mercato anche quando lo stop taglia la gamba perdente. Il
    prezzo da pagare e' il **ping-pong**: lo stop ribalta contro il punteggio, e il punteggio puo'
    riribaltare subito dopo. Non c'e' nessun freno contro quello, di proposito -- un freno sarebbe
    un parametro in piu' -- e `Confluenza.ping_pong` lo conta.

    **Va acceso sapendo cosa fa, perche' qui domina.** Misurato su 300 giorni sintetici a 15m con
    soglia 0,35: con lo stop a 3 ATR le operazioni sono 1.167 e **il 97% delle inversioni le decide
    lo stop, non il punteggio**, con una durata mediana di 4,2 ore; senza stop sono 12, durata
    mediana **171 ore**. Cioe' con lo stop acceso questa non e' la macchina descritta -- «si compra
    e si tiene fino al segnale opposto» -- ma una macchina a stop con il punteggio come comparsa, e
    le operazioni restano corte esattamente come nella modalita' a cancello.

    `atr_multiplier <= 0` toglie lo stop del tutto, e non e' un parametro nuovo: e' il valore che
    quel parametro gia' aveva senza significato.
    """
    indice = candele.index
    chiusure = candele["Close"].to_numpy()
    massimi = candele["High"].to_numpy()
    minimi = candele["Low"].to_numpy()
    atr = ExtraCache(candele).atr(atr_window)

    eventi: list = []
    ingressi: list[int] = []
    motivi: dict = {}
    livello_stop = np.full(len(candele), np.nan)
    posizione = 0
    estremo = 0.0
    ping_pong = 0
    ribaltata_dallo_stop = -1

    for i in range(1, len(candele)):
        prezzo = chiusure[i]

        # 1. Lo stop, che ribalta. Va prima della soglia: e' una regola di rischio e non
        #    un'opinione, e sulla barra in cui salta decide lui.
        if posizione != 0 and atr_multiplier > 0 and not np.isnan(atr[i - 1]):
            if posizione > 0:
                stop = estremo - atr_multiplier * atr[i - 1]
                livello_stop[i] = stop
                saltato = minimi[i] <= stop
            else:
                stop = estremo + atr_multiplier * atr[i - 1]
                livello_stop[i] = stop
                saltato = massimi[i] >= stop
            if saltato:
                posizione = -posizione
                eventi.append((indice[i], float(stop), posizione))
                motivi[indice[i]] = "trailing stop reversal"
                ingressi.append(i)
                estremo = float(stop)
                ribaltata_dallo_stop = i
                continue
            estremo = max(estremo, massimi[i]) if posizione > 0 else min(estremo, minimi[i])
        elif posizione != 0:
            estremo = max(estremo, massimi[i]) if posizione > 0 else min(estremo, minimi[i])

        # 2. La soglia. Simmetrica, e l'unico modo di cambiare verso che non sia lo stop.
        if convinzione(punteggio[i], +1) >= soglia[i]:
            voluta = 1
        elif convinzione(punteggio[i], -1) >= soglia[i]:
            voluta = -1
        else:
            continue  # fra le due soglie non si fa niente: si tiene

        if voluta != posizione:
            # Un ribaltamento dal punteggio sulla barra **subito dopo** uno dallo stop e' il
            # ping-pong: si conta e si lascia avvenire, perche' il freno sarebbe un parametro.
            if i == ribaltata_dallo_stop + 1 and posizione != 0:
                ping_pong += 1
            posizione = voluta
            eventi.append((indice[i], float(prezzo), posizione))
            motivi[indice[i]] = "score crossed the threshold"
            ingressi.append(i)
            estremo = prezzo

    return eventi, ingressi, livello_stop, motivi, ping_pong


def _necessarieta(voti, stati, famiglie, pesi, soglia, soglia_corta, ingressi, k_famiglie) -> dict[str, float]:
    """In che frazione degli ingressi ciascun votante era **indispensabile**.

    Indispensabile vuol dire: azzerandolo, quell'ingresso non sarebbe avvenuto -- perche' il
    punteggio scende sotto la soglia, o perche' l'ampiezza scende sotto il minimo. Sopra il 60%
    l'insieme e' quel votante travestito.
    """
    if not ingressi:
        return {nome: 0.0 for nome in voti}
    barre = np.array(ingressi)
    # Si ritaglia **prima** di contare. La domanda riguarda solo le barre d'ingresso, ma la
    # versione precedente costruiva le serie intere una volta per ogni coppia (votante, ingresso)
    # per leggerne un elemento: su sette anni a quindici minuti erano 6.800 passate da 267.000
    # barre ciascuna, cioe' l'86% del tempo di `evaluate` speso su valori buttati via subito.
    # `_famiglie_concordi` lavora elemento per elemento, quindi accetta il `verso` come vettore e
    # una passata sola sostituisce le seimila.
    ai_bordi = {nome: voto[barre] for nome, voto in voti.items()}
    stati_ai_bordi = {nome: np.asarray(stato)[barre] for nome, stato in stati.items()}
    # Il verso dell'**operazione**, non il segno del punteggio: i due sono opposti, perche' un
    # consenso lungo produce un punteggio negativo.
    verso = np.sign(convinzione(sum(pesi[n] * ai_bordi[n] for n in ai_bordi), +1))
    verso[verso == 0] = 1
    # La soglia contro cui quell'ingresso si e' misurato davvero: quella lunga se e' un lungo,
    # quella corta se e' un corto. Con una sola soglia la diagnosi sui corti chiedeva a ogni
    # votante di superare una barriera che l'ingresso non aveva dovuto superare.
    attiva = np.where(verso > 0, soglia[barre], soglia_corta[barre])

    conteggi = {}
    for nome in voti:
        restanti = {n: v for n, v in ai_bordi.items() if n != nome}
        punteggio = sum(pesi[n] * restanti[n] for n in restanti) if restanti else np.zeros(len(barre))
        sotto_soglia = convinzione(punteggio, verso) < attiva
        # L'ampiezza si conta sugli stati, come nel motore: togliere un votante vuol dire togliere
        # la sua opinione, non la coda del suo voto.
        ampiezza = _famiglie_concordi({n: v for n, v in stati_ai_bordi.items() if n != nome}, famiglie, verso)
        conteggi[nome] = float(np.mean(sotto_soglia | (ampiezza < k_famiglie)))
    return conteggi
