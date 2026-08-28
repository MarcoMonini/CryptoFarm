"""Il votante a modello: il controllo casuale, che e' il numero da battere.

Non si testa qui l'addestramento -- lo copre gia' `meta_gate.selfcheck`, che pianta un segnale
dentro le feature e verifica che venga trovato, e rumore puro e verifica che non lo sia. Qui si
tiene fermo il pezzo nuovo: il confronto contro una selezione casuale della stessa numerosita'.
"""

from __future__ import annotations

import numpy as np

from scripts.ai_voter import controllo_casuale


def test_il_caso_su_una_distribuzione_simmetrica_sta_attorno_a_zero():
    netto = np.concatenate([np.ones(500), -np.ones(500)])
    caso = controllo_casuale(netto, quanti=200, prove=300)
    assert abs(caso["caso_medio_%"]) < 0.15, caso
    assert caso["caso_p95_%"] > caso["caso_medio_%"]


def test_il_p95_si_stringe_quando_la_selezione_e_piu_grande():
    """Piu' operazioni si tengono, meno il caso puo' essere fortunato.

    E' la ragione per cui il controllo va rifatto **per ogni soglia** invece di calcolarne uno solo:
    una soglia severa tiene poche operazioni, e su poche operazioni il caso arriva molto piu' in
    alto. Confrontare un filtro severo col p95 di un filtro largo lo farebbe sembrare bravo.
    """
    rng = np.random.default_rng(0)
    netto = rng.normal(0.0, 3.0, 2000)
    stretto = controllo_casuale(netto, quanti=50, prove=400)
    largo = controllo_casuale(netto, quanti=1000, prove=400)
    assert stretto["caso_p95_%"] > largo["caso_p95_%"], (stretto, largo)


def test_le_estrazioni_servono_a_ricavare_il_percentile_dalla_stessa_distribuzione():
    """`_estrazioni` esce insieme al p95 apposta: due letture della stessa distribuzione.

    Calcolare il percentile da un secondo campionamento vorrebbe dire confrontare il risultato con
    un campione diverso da quello che ha prodotto il p95 riportato accanto.
    """
    netto = np.arange(-100.0, 100.0)
    caso = controllo_casuale(netto, quanti=40, prove=250)
    estrazioni = caso["_estrazioni"]
    assert len(estrazioni) == 250
    assert np.isclose(np.percentile(estrazioni, 95), caso["caso_p95_%"])
