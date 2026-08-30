"""La pagina come pagina, non come somma di funzioni.

Il guasto che ha tolto il simulatore dalla produzione non stava in un pezzo: stava
nell'assemblaggio. `load_signal_model()` chiamata senza condizione dentro `__main__` sollevava, e
la pagina non si apriva affatto -- mentre ogni funzione che ci girava dentro aveva i suoi test e
passavano tutti. `AppTest` esegue lo script come lo esegue Streamlit, quindi vede quel livello.

Non serve la rete: la vista a un asset parte senza candele e lo dice, e quella di rotazione legge
lo store locale, che qui viene svuotato di proposito in uno dei casi.
"""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from cryptofarm.ml import signals
from cryptofarm.ml.signals import entry_model_disponibile
from cryptofarm.trading import config, rotation

PAGINA = "src/cryptofarm/trading/simulator.py"
ROTAZIONE = config.ROTATION_MODES[1]


@pytest.fixture
def pagina() -> AppTest:
    prova = AppTest.from_file(PAGINA, default_timeout=120)
    prova.run()
    return prova


def test_la_pagina_si_apre(pagina: AppTest) -> None:
    """Senza candele non c'e' niente da simulare, ma la pagina deve esistere lo stesso."""
    assert not pagina.exception
    assert pagina.radio[0].options == config.ROTATION_MODES


def test_la_vista_a_un_asset_chiede_le_candele(pagina: AppTest) -> None:
    assert any("Fetch candles" in info.value for info in pagina.info)


def test_il_menu_offre_solo_le_strategie_del_registro(pagina: AppTest) -> None:
    """Il menu della pagina viva, non la lista in `config`: e' l'unico modo di vedere il filtro
    che toglie la voce AI quando il modello non c'e'."""
    menu = next(box for box in pagina.selectbox if box.label == "Strategy")
    assert set(menu.options) <= set(config.STRATEGIES)
    assert "Trend Pullback" not in menu.options  # tolta dal menu sulle misure fuori campione
    assert "Close RSI Reverse" in menu.options  # rimessa sulle stesse


@pytest.mark.skipif(
    rotation.load_universe(rotation.MAJORS, "1d", "2021-01-01").shape[1] < 2,
    reason="lo store delle candele non ha almeno due asset: la vista non ha cosa mostrare",
)
def test_la_vista_di_rotazione_confronta_con_i_due_riferimenti(pagina: AppTest) -> None:
    """Il numero da battere e' l'universo a peso uguale, non BTC: deve stare sotto gli occhi."""
    pagina.radio[0].set_value(ROTAZIONE).run()
    assert not pagina.exception
    etichette = {metrica.label for metrica in pagina.metric}
    assert {"Rotation", "Equal-weight universe", "BTC buy and hold", "Max drawdown"} <= etichette


def test_cambiare_intervallo_ricarica_i_valori_di_partenza(pagina: AppTest) -> None:
    """Il difetto che questa asserzione previene e' invisibile leggendo il codice.

    Streamlit conserva lo stato di un widget con la stessa chiave: senza l'intervallo dentro la
    chiave, cambiando timeframe i campi restavano fermi sui numeri del precedente e il default
    misurato non compariva mai. La pagina sembrava funzionare.
    """
    menu = next(box for box in pagina.selectbox if box.label == "Strategy")
    menu.set_value("Donchian Breakout").run()

    def canale(intervallo: str) -> tuple:
        """Valore e chiave letti **subito**: gli elementi di `AppTest` si rilegano al run corrente,
        quindi tenerne uno da parte e confrontarlo dopo un altro `run()` non misura niente."""
        next(b for b in pagina.selectbox if b.label == "Candle interval").set_value(intervallo).run()
        campo = next(n for n in pagina.number_input if n.label.startswith("Channel length"))
        return campo.value, campo.key

    (valore_ora, chiave_ora), (valore_giorno, chiave_giorno) = canale("1h"), canale("1d")
    assert valore_ora > valore_giorno
    # L'asserzione che conta e' questa, e va scritta sulla **chiave**: `AppTest` ricostruisce lo
    # stato a ogni `run()`, quindi il solo confronto fra i valori passa anche con la chiave
    # sbagliata -- verificato togliendo l'intervallo dalla chiave. In un browser vero no: li' la
    # sessione sopravvive, lo stato memorizzato vince sul valore iniziale, e i campi resterebbero
    # fermi sui numeri dell'intervallo precedente.
    assert chiave_ora != chiave_giorno, "la chiave del widget non porta l'intervallo"
    assert chiave_ora.endswith("_1h") and chiave_giorno.endswith("_1d")


def test_la_vista_di_rotazione_senza_store_avvisa_invece_di_rompersi(monkeypatch) -> None:
    """In produzione `market_data/` e' vuota: il piano non ha dischi persistenti.

    La vista legge lo store e non l'exchange, quindi li' non ha dati -- e deve dirlo, non
    sollevare. E' la stessa lezione del modello mancante, applicata alle candele.
    """
    # Si sostituisce `rotation.load_universe`, non la funzione decorata della pagina: `rotation`
    # e' un modulo vero e condiviso, mentre `AppTest` esegue lo script in un suo spazio dei nomi e
    # una toppa sulla funzione della pagina non lo raggiungerebbe.
    #
    # La cache si svuota **tutta**: `AppTest` riesegue lo script, quindi la funzione decorata che
    # la pagina usa non e' lo stesso oggetto di `simulator.universo_di_sessione`, e svuotare quella
    # lascerebbe in piedi l'universo pieno letto da un test precedente. Senza questa riga il test
    # passa da solo e fallisce in suite, che e' il modo peggiore di fallire.
    monkeypatch.setattr(rotation, "load_universe", lambda *args, **kwargs: pd.DataFrame())
    st.cache_data.clear()

    prova = AppTest.from_file(PAGINA, default_timeout=120)
    prova.run()
    prova.radio[0].set_value(ROTAZIONE).run()
    st.cache_data.clear()

    assert not prova.exception
    # Non basta "non e' esplosa": deve aver detto perche', altrimenti il test passerebbe anche
    # senza che la toppa arrivi alla pagina.
    assert any("candle store is empty" in avviso.value for avviso in prova.warning), [w.value for w in prova.warning]
    assert not prova.metric


def test_la_confluenza_offre_l_interruttore_delle_barre_in_formazione(pagina: AppTest) -> None:
    """`CONF_IN_FORMAZIONE` decide se il cancello legge il prezzo di adesso o aspetta la chiusura
    del piano lungo: e' l'ablazione che il banco misura, e la pagina deve poterla girare.

    Finche' non aveva un widget non entrava nel dizionario della barra laterale, e
    `panels.diagnosi_confluenza` cadeva con `KeyError` proprio quando serviva -- senza operazioni.
    La chiave porta l'intervallo per la ragione di `test_cambiare_intervallo_ricarica_i_valori_di_partenza`.
    """
    menu = next(box for box in pagina.selectbox if box.label == "Strategy")
    menu.set_value(config.CONFLUENCE_STRATEGY).run()

    intervallo = next(box for box in pagina.selectbox if box.label == "Candle interval").value
    interruttore = next(c for c in pagina.checkbox if c.label.startswith("React inside forming"))
    assert interruttore.value is config.CONF_IN_FORMAZIONE
    assert interruttore.key.endswith(f"_{intervallo}")


@pytest.mark.skipif(
    not all(entry_model_disponibile(n) for n in (signals.ENTRY_VELOCE, signals.ENTRY_LENTO)),
    reason="servono entrambi gli artefatti d'ingresso, che il repository non traccia",
)
def test_la_strategia_ai_lascia_scegliere_fra_i_due_modelli(pagina: AppTest) -> None:
    """I due artefatti sono due strategie, non due tarature: la pagina deve poterle separare.

    In servizio lavorano insieme -- il veloce opera, il lento fa da cancello -- e messi insieme
    non si vede in cosa differiscano. Il riquadro della previsione e' l'altra meta': mostra sullo
    stesso asse cio' che il modello prevede e cio' che gli e' stato insegnato, e per farlo deve
    aprirsi senza sollevare, che e' il livello da cui e' passato il guasto in produzione.
    """
    menu = next(box for box in pagina.selectbox if box.label == "Strategy")
    menu.set_value(config.AI_STRATEGY).run()

    scelta = next(r for r in pagina.radio if r.label == "Entry model")
    assert list(scelta.options) == ["Fast (trades)", "Slow (gates)"]

    next(c for c in pagina.checkbox if c.label.startswith("Show prediction")).set_value(True).run()
    assert not pagina.exception

    scelta = next(r for r in pagina.radio if r.label == "Entry model")
    scelta.set_value("Slow (gates)").run()
    assert not pagina.exception
