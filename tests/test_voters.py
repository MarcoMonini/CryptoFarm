"""Lo stato per barra e il voto con memoria: `trading/voters.py`.

I due difetti che questo modulo puo' avere sono silenziosi. Il primo e' scambiare «tiene una
posizione» per «scatta»: un votante che resta lungo per mesi voterebbe a piena forza per mesi, e
l'insieme diventerebbe quella strategia sola. Il secondo e' un disallineamento di indice, che non
solleva niente e sposta ogni voto di qualche barra. I test qui sotto sono scritti per far cadere
entrambi.
"""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading.voters import decayed_vote, held_state


@pytest.fixture
def indice():
    return pd.date_range("2024-01-01", periods=40, freq="15min", name="Open time")


@pytest.fixture
def eventi(indice):
    return [(indice[3], 100.0, 1), (indice[10], 105.0, 0), (indice[20], 99.0, -1)]


def test_lo_stato_e_quello_tenuto_non_quello_dell_evento(indice, eventi):
    stato = held_state(eventi, indice)
    assert stato[2] == 0, "prima del primo evento non c'e' nessuna posizione"
    assert (stato[3:10] == 1).all()
    assert (stato[10:20] == 0).all()
    assert (stato[20:] == -1).all()


def test_senza_eventi_lo_stato_e_tutto_zero(indice):
    assert not held_state([], indice).any()


def test_il_voto_e_pieno_allo_scatto_e_dimezza_dopo_un_emivita(indice, eventi):
    voto = decayed_vote(held_state(eventi, indice), half_life_bars=4)
    assert voto[3] == 1.0
    assert np.isclose(voto[7], 0.5)
    assert np.isclose(voto[11], 0.25)


def test_tenere_la_posizione_non_e_scattare(indice):
    """Il difetto centrale: se «scatta» fosse «stato diverso da zero», il voto resterebbe 1."""
    stato = held_state([(indice[3], 100.0, 1)], indice)
    voto = decayed_vote(stato, half_life_bars=4)
    assert (stato[3:] == 1).all(), "lo stato e' tenuto..."
    assert voto[-1] < 0.1, "...ma il voto no: decade anche mentre la posizione resta aperta"


def test_un_inversione_diretta_riparte_a_forza_piena(indice):
    stato = held_state([(indice[3], 100.0, 1), (indice[9], 99.0, -1)], indice)
    voto = decayed_vote(stato, half_life_bars=4)
    assert voto[9] == -1.0, "un'inversione e' un segnale nuovo, non la coda del precedente"


def test_uscire_non_azzera_il_voto_di_colpo(indice, eventi):
    """Un ritorno a flat e' un'assenza di segnale, non un segnale contrario: sfuma come gli altri."""
    voto = decayed_vote(held_state(eventi, indice), half_life_bars=4)
    assert 0 < voto[10] < voto[9]


def test_epsilon_taglia_la_coda_a_zero_esatto(indice, eventi):
    stato = held_state(eventi, indice)
    assert np.isclose(decayed_vote(stato, 4, epsilon=0.01)[19], 0.0625)
    assert decayed_vote(stato, 4, epsilon=0.1)[19] == 0.0


def test_troncare_la_storia_non_cambia_niente_di_gia_emesso(indice, eventi):
    meta = 15
    intero = decayed_vote(held_state(eventi, indice), 4)
    passati = [e for e in eventi if e[0] <= indice[meta - 1]]
    assert np.allclose(decayed_vote(held_state(passati, indice[:meta]), 4), intero[:meta])


def test_un_evento_fuori_griglia_solleva(indice):
    """Un votante letto su un indice diverso dal proprio: si deve vedere, non allineare da solo."""
    with pytest.raises(ValueError, match="disallineato"):
        held_state([(indice[3] + pd.Timedelta(minutes=7), 100.0, 1)], indice)
    with pytest.raises(ValueError, match="disallineato"):
        held_state([(indice[-1] + pd.Timedelta(minutes=15), 100.0, 1)], indice)


def test_emivita_non_positiva_solleva(indice, eventi):
    with pytest.raises(ValueError):
        decayed_vote(held_state(eventi, indice), half_life_bars=0)
