"""Un capitale solo su piu' asset: `trading/portfolio.py`.

Il controllo portante e' l'equivalenza: con **un** asset questo motore deve dare esattamente
quello di `pnl.simulate_positions`, centesimo per centesimo. Se i costi divergessero, il confronto
fra "un asset" e "cinque asset" -- che e' l'unica domanda per cui il modulo esiste -- misurerebbe
la differenza fra due contabilita' invece che la differenza fra due strategie.
"""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import portfolio
from cryptofarm.trading.pnl import simulate_positions
from cryptofarm.trading.portfolio import Portafoglio, curva_capitale, simulate_shared_capital


@pytest.fixture
def giorni():
    return pd.date_range("2024-01-01", periods=30, freq="1d")


def test_con_un_asset_solo_e_identico_a_simulate_positions(giorni):
    eventi = [
        (giorni[0], 100.0, 1),
        (giorni[5], 130.0, 0),
        (giorni[8], 120.0, -1),
        (giorni[12], 100.0, 0),
        (giorni[20], 90.0, 1),
        (giorni[25], 95.0, 0),
    ]
    atteso = simulate_positions(eventi, wallet=100.0)
    ottenuto = simulate_shared_capital({"BTC": eventi}, wallet=100.0).operazioni

    assert len(ottenuto) == len(atteso)
    for a, b in zip(atteso, ottenuto):
        assert a["Side"] == b["Side"] and a["Buy_Time"] == b["Buy_Time"]
        assert abs(a["Profit"] - b["Profit"]) < 1e-9, "i costi divergono da simulate_positions"
        assert abs(a["Wallet_After"] - b["Wallet_After"]) < 1e-9


def test_non_si_aprono_due_posizioni_insieme(giorni):
    """E' il vincolo del disegno: tutto il capitale su una posizione alla volta."""
    portafoglio = simulate_shared_capital(
        {
            "A": [(giorni[0], 100.0, 1), (giorni[10], 110.0, 0)],
            "B": [(giorni[2], 50.0, 1), (giorni[4], 60.0, 0)],
            "C": [(giorni[3], 20.0, 1), (giorni[6], 25.0, 0)],
        }
    )
    intervalli = [(o["Buy_Time"], o["Sell_Time"]) for o in portafoglio.operazioni]
    for (inizio_a, fine_a), (inizio_b, _) in zip(intervalli, intervalli[1:]):
        assert inizio_b >= fine_a, f"{inizio_b} si apre mentre {inizio_a}-{fine_a} e' ancora aperta"


def test_le_occasioni_perse_si_contano(giorni):
    """Se sono molte piu' delle operazioni fatte, il paniere non e' il rimedio alla scarsita'."""
    portafoglio = simulate_shared_capital(
        {
            "A": [(giorni[0], 100.0, 1), (giorni[20], 110.0, 0)],
            "B": [(giorni[2], 50.0, 1), (giorni[4], 60.0, 0)],
            "C": [(giorni[3], 20.0, 1), (giorni[6], 25.0, 0)],
        }
    )
    assert portafoglio.occasioni_perse == 2
    assert portafoglio.per_asset == {"A": 1, "B": 0, "C": 0}


def test_il_capitale_liberato_torna_disponibile(giorni):
    """Senza questo il paniere non farebbe piu' operazioni di un asset solo, che e' tutto il punto."""
    portafoglio = simulate_shared_capital(
        {
            "A": [(giorni[0], 100.0, 1), (giorni[3], 110.0, 0)],
            "B": [(giorni[5], 50.0, 1), (giorni[8], 60.0, 0)],
        },
        fee_percent=0.0,
        carry_daily_percent=0.0,
    )
    assert portafoglio.per_asset == {"A": 1, "B": 1}
    assert portafoglio.capitale_finale == pytest.approx(132.0)


def test_le_pari_merito_le_vince_la_priorita(giorni):
    """Su asset che si muovono insieme i segnali cadono sulla stessa barra: chi vince non puo'
    essere deciso dall'ordine del dizionario."""

    def portafoglio(priorita_a, priorita_b):
        return simulate_shared_capital(
            {
                "A": [(giorni[0], 100.0, 1, priorita_a), (giorni[2], 100.0, 0, 0.0)],
                "B": [(giorni[0], 50.0, 1, priorita_b), (giorni[2], 50.0, 0, 0.0)],
            }
        )

    assert portafoglio(0.9, 0.1).operazioni[0]["Asset"] == "A"
    assert portafoglio(0.1, 0.9).operazioni[0]["Asset"] == "B"


def test_la_concentrazione_denuncia_il_paniere_finto(giorni):
    finto = Portafoglio(operazioni=[], per_asset={"A": 9, "B": 1})
    vero = Portafoglio(operazioni=[], per_asset={"A": 5, "B": 5})
    assert finto.concentrazione == pytest.approx(0.9)
    assert vero.concentrazione == pytest.approx(0.5)


def test_la_curva_del_capitale_e_piatta_dentro_l_operazione(giorni):
    """Lo stesso limite di `simulate_positions`, tenuto uguale di proposito: il drawdown dentro
    l'operazione non si vede, e i due numeri restano confrontabili."""
    portafoglio = simulate_shared_capital(
        {"A": [(giorni[0], 100.0, 1), (giorni[4], 150.0, 0)]}, fee_percent=0.0, carry_daily_percent=0.0
    )
    curva = curva_capitale(portafoglio.operazioni, giorni)
    assert np.all(curva[:4] == 100.0)
    assert curva[4] == pytest.approx(150.0)
    assert curva[-1] == pytest.approx(150.0)


def test_un_paniere_vuoto_non_solleva(giorni):
    portafoglio = simulate_shared_capital({})
    assert portafoglio.operazioni == [] and np.isnan(portafoglio.capitale_finale)


# --- posizioni contemporanee -----------------------------------------------------------------


def test_uno_slot_e_esattamente_il_paniere_di_prima():
    """Il controllo che rende confrontabili i due simulatori.

    Se `n_slot=1` non riproducesse `simulate_shared_capital`, ogni differenza misurata fra uno e
    piu' slot conterrebbe anche le differenze fra i due motori, e non si saprebbe quale delle due
    si sta guardando.
    """
    idx = pd.date_range("2024-01-01", periods=12, freq="1d")
    eventi = {
        "A": [(idx[0], 100.0, 1), (idx[2], 110.0, 0), (idx[6], 100.0, 1), (idx[8], 90.0, 0)],
        "B": [(idx[1], 50.0, 1), (idx[3], 60.0, 0), (idx[9], 20.0, 1), (idx[10], 25.0, 0)],
    }
    uno = portfolio.simulate_shared_capital(eventi, fee_percent=0.05, carry_daily_percent=0.01)
    slot = portfolio.simulate_slots(eventi, n_slot=1, fee_percent=0.05, carry_daily_percent=0.01)

    assert slot.per_asset == uno.per_asset
    assert slot.occasioni_perse == uno.occasioni_perse
    assert slot.capitale_finale == pytest.approx(uno.capitale_finale)


def test_con_due_slot_il_secondo_asset_non_si_perde_piu():
    """E' tutto il punto: il segnale che prima veniva buttato adesso viene preso."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1d")
    eventi = {
        "A": [(idx[0], 100.0, 1), (idx[4], 110.0, 0)],
        "B": [(idx[1], 50.0, 1), (idx[3], 60.0, 0)],
    }
    uno = portfolio.simulate_slots(eventi, n_slot=1, fee_percent=0.0, carry_daily_percent=0.0)
    due = portfolio.simulate_slots(eventi, n_slot=2, fee_percent=0.0, carry_daily_percent=0.0)

    assert uno.occasioni_perse == 1 and uno.per_asset == {"A": 1, "B": 0}
    assert due.occasioni_perse == 0 and due.per_asset == {"A": 1, "B": 1}

    # A rende +10% su meta' capitale, B +20% sull'altra meta': 50*1,1 + 50*1,2 = 115.
    assert due.capitale_finale == pytest.approx(115.0)


def test_le_quote_non_mandano_il_portafoglio_in_leva_dopo_una_perdita():
    """La quota si prende dal **contante**, non dal capitale iniziale.

    Con `capitale/n_slot` fisso, dopo una perdita le quote successive resterebbero tarate sul
    capitale di partenza: la somma degli impegni supererebbe il contante e il portafoglio sarebbe
    in leva senza che nessuna riga lo dica.
    """
    idx = pd.date_range("2024-01-01", periods=12, freq="1d")
    eventi = {
        "A": [(idx[0], 100.0, 1), (idx[1], 50.0, 0)],  # -50% sulla prima quota
        "B": [(idx[2], 100.0, 1), (idx[3], 100.0, 0)],
        "C": [(idx[2], 100.0, 1), (idx[3], 100.0, 0)],
    }
    p = portfolio.simulate_slots(eventi, n_slot=2, fee_percent=0.0, carry_daily_percent=0.0)

    # Capitale dopo A: 50 di quota persa a meta' -> 100 - 50 + 25 = 75.
    # B e C aprono con 75/2 = 37,5 ciascuno: la somma impegnata e' 75, mai piu' del contante.
    impegnato = sum(o["Quantity"] * o["Buy_Price"] for o in p.operazioni if o["Asset"] in ("B", "C"))
    assert impegnato == pytest.approx(75.0)
    assert p.capitale_finale == pytest.approx(75.0)


def test_gli_slot_liberi_si_riusano():
    idx = pd.date_range("2024-01-01", periods=14, freq="1d")
    eventi = {
        "A": [(idx[0], 100.0, 1), (idx[1], 100.0, 0)],
        "B": [(idx[2], 100.0, 1), (idx[3], 100.0, 0)],
        "C": [(idx[4], 100.0, 1), (idx[5], 100.0, 0)],
    }
    p = portfolio.simulate_slots(eventi, n_slot=1, fee_percent=0.0, carry_daily_percent=0.0)
    assert p.per_asset == {"A": 1, "B": 1, "C": 1} and p.occasioni_perse == 0


def test_uno_slot_zero_non_si_accetta():
    with pytest.raises(ValueError, match="almeno uno slot"):
        portfolio.simulate_slots({}, n_slot=0)
