"""I valori di partenza misurati, e il loro innesto nella pagina.

`tuned_defaults.py` e' generato: nessuno lo rilegge riga per riga, quindi i controlli che qui
sembrano paranoia sono l'unica cosa che sta fra una rigenerazione sbagliata e una pagina che non si
apre. Il piu' importante e' il terzo: un valore fuori dai limiti del widget fa sollevare
`st.number_input`, e il generatore non conosce quei limiti.
"""

from __future__ import annotations

import pytest

from cryptofarm.trading import config, panels
from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

COPPIE = [
    (intervallo, voce, costante, valore)
    for intervallo, per_voce in PER_INTERVALLO.items()
    for voce, valori in per_voce.items()
    for costante, valore in valori.items()
]


def test_ci_sono_default_misurati() -> None:
    """Se la rigenerazione producesse un file vuoto, tutti i test sotto passerebbero a vuoto."""
    assert COPPIE, "nessun default misurato: `tuned_defaults.py` e' vuoto"


@pytest.mark.parametrize("intervallo", sorted(PER_INTERVALLO))
def test_gli_intervalli_misurati_sono_offerti_dalla_pagina(intervallo: str) -> None:
    assert intervallo in config.INTERVALS


@pytest.mark.parametrize("voce", sorted({voce for per_voce in PER_INTERVALLO.values() for voce in per_voce}))
def test_ogni_strategia_misurata_e_ancora_nel_menu(voce: str) -> None:
    """Potare il menu senza rigenerare lascerebbe qui default che non raggiungono nessuno."""
    assert voce in config.STRATEGIES
    assert voce in panels.STRATEGIE


@pytest.mark.parametrize("intervallo, voce, costante, valore", COPPIE)
def test_ogni_valore_sta_dentro_i_limiti_del_widget(intervallo: str, voce: str, costante: str, valore) -> None:
    """Il generatore non conosce i limiti dei widget: se li supera, la pagina solleva all'avvio."""
    campo = getattr(config, costante, None)
    if isinstance(campo, config.Param):
        assert campo.minimum <= valore <= campo.maximum, f"{intervallo}/{voce}: {costante}={valore} fuori scala"
    else:
        # Gli interruttori (CONFIRM_VOLUME, REQUIRE_CLOUD) non sono `Param` ma devono esistere.
        assert isinstance(getattr(config, costante), bool), f"{costante} non e' ne' Param ne' interruttore"


@pytest.mark.parametrize("intervallo, voce, costante, valore", COPPIE)
def test_ogni_parametro_misurato_e_usato_da_quella_strategia(intervallo: str, voce: str, costante: str, valore) -> None:
    """Un default che la strategia non legge non ha widget: si scriverebbe e non farebbe niente."""
    del valore
    if isinstance(getattr(config, costante, None), bool):
        return  # gli interruttori hanno una casella dedicata, non passano da `gruppi_di`
    assert costante in panels.parametri_di(voce), f"{intervallo}/{voce}: {costante} non e' fra i suoi parametri"


def test_ogni_intervallo_della_pagina_ha_un_ancora() -> None:
    """Nove intervalli nel menu, quattro misurati: la mappa deve coprirli tutti e non inventarne."""
    assert set(panels.ANCORA_MISURATA) == set(config.INTERVALS)
    assert set(panels.ANCORA_MISURATA.values()) <= set(config.INTERVALS)


def test_i_valori_predefiniti_cambiano_con_l_intervallo() -> None:
    """E' l'intera ragione di questo lavoro: la stessa strategia parte da numeri diversi."""
    a_giorno = panels.valori_predefiniti("Donchian Breakout", "1d")
    a_ora = panels.valori_predefiniti("Donchian Breakout", "1h")
    assert a_giorno["DONCHIAN_CHANNEL"] != a_ora["DONCHIAN_CHANNEL"]
    # La finestra si conta in **barre**: a barre piu' corte ne servono di piu' per coprire lo stesso
    # tratto di calendario. Il verso della disuguaglianza e' la lettura meccanica del risultato.
    assert a_ora["DONCHIAN_CHANNEL"] > a_giorno["DONCHIAN_CHANNEL"]


def test_senza_argomenti_restano_i_default_scritti_a_mano() -> None:
    """I test e chi calcola una serie fuori dalla pagina non devono vedere i valori misurati."""
    base = panels.valori_predefiniti()
    assert base["DONCHIAN_CHANNEL"] == config.DONCHIAN_CHANNEL.value


def test_un_intervallo_non_misurato_ricade_sull_ancora() -> None:
    """5m non e' stato misurato: prende i valori di 15m, e la pagina lo dichiara."""
    assert panels.ancora_di("5m") == "15m"
    assert panels.valori_misurati("Donchian Breakout", "5m") == panels.valori_misurati("Donchian Breakout", "15m")


def test_una_strategia_mai_misurata_non_ha_valori() -> None:
    assert panels.valori_misurati("AI Model", "1d") == {}
    assert panels.valori_misurati("Donchian Breakout", "intervallo inesistente") == {}
