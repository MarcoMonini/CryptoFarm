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

from cryptofarm.trading.voters import PAVIMENTO_DEL_VOTO, decayed_vote, held_state


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
    """Senza pavimento la forma e' quella di sempre: piena allo scatto, poi `lambda**eta`."""
    voto = decayed_vote(held_state(eventi, indice), half_life_bars=4, pavimento=0.0)
    assert voto[3] == 1.0
    assert np.isclose(voto[7], 0.5)


def test_tenere_la_posizione_non_e_scattare(indice):
    """Tenere non e' scattare: la **forza** sfuma anche mentre la posizione resta aperta.

    Era il difetto originale del modulo ed e' ancora vero, ma solo fino al pavimento: si veda
    `test_chi_tiene_la_posizione_non_ammutolisce` per l'altra meta'.
    """
    stato = held_state([(indice[3], 100.0, 1)], indice)
    voto = decayed_vote(stato, half_life_bars=4)
    assert (stato[3:] == 1).all(), "lo stato e' tenuto..."
    assert voto[-1] < voto[3], "...ma il voto no: la forza sfuma lo stesso"


def test_chi_tiene_la_posizione_non_ammutolisce(indice):
    """La meta' nuova, e il difetto che toglie.

    Col decadimento verso zero un votante spariva dal punteggio pur restando convinto:
    `zone_regime` era in posizione sul 74,5% delle barre e gia' muto sul **91,3%** di quelle, e la
    macrostruttura contribuiva solo nelle ore attorno al proprio incrocio -- cioe' quando e' piu'
    soggetta a whipsaw. Il pavimento e' il freno: la forza scende, la presenza no.
    """
    stato = held_state([(indice[3], 100.0, 1)], indice)
    voto = decayed_vote(stato, half_life_bars=2)  # emivita corta apposta: a 37 barre sarebbe zero
    assert (voto[3:] >= PAVIMENTO_DEL_VOTO).all(), "chi tiene la posizione deve continuare a votare"
    assert np.isclose(voto[-1], PAVIMENTO_DEL_VOTO, atol=1e-3), "e a regime vale esattamente il pavimento"
    # Il vincolo vero: sotto la soglia **minima raggiungibile**, non sotto `theta_base`. Con i
    # default un macro a favore sconta la soglia fino a 0,20, e un pavimento di 0,30 avrebbe fatto
    # aprire un collegio interamente fermo -- la confluenza avrebbe smesso di decidere *quando*.
    assert PAVIMENTO_DEL_VOTO < 0.35 - 0.15, "un collegio fermo deve restare sotto la soglia piu' bassa"


def test_un_inversione_diretta_riparte_a_forza_piena(indice):
    stato = held_state([(indice[3], 100.0, 1), (indice[9], 99.0, -1)], indice)
    voto = decayed_vote(stato, half_life_bars=4)
    assert voto[9] == -1.0, "un'inversione e' un segnale nuovo, non la coda del precedente"


def test_uscire_azzera_il_voto(indice, eventi):
    """Il difetto opposto, e il cambio deliberato rispetto alla prima versione del modulo.

    Prima un ritorno a flat lasciava sfumare il voto, con l'argomento che «uscire non e' un
    segnale contrario, e' un'assenza». Vero, ma il voto e' cio' che entra nel punteggio, e un
    votante fuori posizione che continua a dire «lungo» sposta la decisione con un'opinione che
    non ha piu'. Misurato: `pullback` era in quello stato sul **49,0%** delle barre, e le bande --
    che entrano sulla banda inferiore ed escono su quella **opposta** -- continuavano a votare
    lungo dal massimo in giu'. Un'assenza di segnale si scrive zero.
    """
    voto = decayed_vote(held_state(eventi, indice), half_life_bars=4)
    assert voto[9] > 0, "dentro la posizione si vota..."
    assert voto[10] == 0.0, "...e fuori no, dalla barra stessa dell'uscita"
    assert (voto[10:20] == 0.0).all(), "e per tutto il tempo in cui si resta fuori"


def test_epsilon_taglia_la_coda_solo_senza_pavimento(indice, eventi):
    """`epsilon` resta per l'ablazione `pavimento=0`; ai default non morde mai.

    L'invariante e' `epsilon <= pavimento`, non «col pavimento epsilon e' inerte»: un epsilon
    scelto **sopra** il pavimento taglia il voto di un votante che e' ancora in posizione, e non
    e' un caso da difendere -- e' una configurazione incoerente che qui si documenta invece di
    fingere che non esista.
    """
    stato = held_state(eventi, indice)
    assert np.isclose(decayed_vote(stato, 4, pavimento=0.0, epsilon=0.01)[7], 0.5)
    assert decayed_vote(stato, 4, pavimento=0.0, epsilon=0.6)[7] == 0.0

    # Ai default: il voto piu' debole possibile e' il pavimento, che sta sopra epsilon.
    assert 0.05 <= PAVIMENTO_DEL_VOTO, "epsilon di default deve stare sotto il pavimento"
    tenuto = decayed_vote(held_state([(indice[3], 100.0, 1)], indice), half_life_bars=2)
    assert (tenuto[3:] > 0).all(), "ai default nessun voto in posizione viene tagliato da epsilon"

    # E un epsilon incoerente (sopra il pavimento) taglia davvero: e' documentato, non difeso.
    assert decayed_vote(stato, 4, epsilon=0.6)[7] == 0.0


def test_un_pavimento_fuori_scala_solleva(indice, eventi):
    stato = held_state(eventi, indice)
    with pytest.raises(ValueError, match="pavimento"):
        decayed_vote(stato, 4, pavimento=1.5)


def test_troncare_la_storia_non_cambia_niente_di_gia_emesso(indice, eventi):
    meta = 15
    intero = decayed_vote(held_state(eventi, indice), 4)
    passati = [e for e in eventi if e[0] <= indice[meta - 1]]
    assert np.allclose(decayed_vote(held_state(passati, indice[:meta]), 4), intero[:meta])


def test_il_selfcheck_del_modulo_passa():
    from cryptofarm.trading.voters import _selfcheck

    _selfcheck()


def test_un_evento_fuori_griglia_solleva(indice):
    """Un votante letto su un indice diverso dal proprio: si deve vedere, non allineare da solo."""
    with pytest.raises(ValueError, match="disallineato"):
        held_state([(indice[3] + pd.Timedelta(minutes=7), 100.0, 1)], indice)
    with pytest.raises(ValueError, match="disallineato"):
        held_state([(indice[-1] + pd.Timedelta(minutes=15), 100.0, 1)], indice)


def test_emivita_non_positiva_solleva(indice, eventi):
    with pytest.raises(ValueError):
        decayed_vote(held_state(eventi, indice), half_life_bars=0)
