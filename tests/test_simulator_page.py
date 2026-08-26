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
