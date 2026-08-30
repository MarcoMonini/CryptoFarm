"""Le due parti non banali del banco a swing: l'isteresi e il controllo casuale."""

import numpy as np

from scripts.swing_lab import collocazioni_casuali, indici_dei_segnali


def test_listeresi_evita_il_giro_a_vuoto_sulla_soglia():
    """Con una previsione che oscilla attorno alla soglia d'ingresso non si deve rientrare."""
    previsto = np.array([0.60, 0.44, 0.60, 0.44, 0.60, 0.10])
    coppie = indici_dei_segnali(previsto, entra=0.50, esci=0.40, cadenza=1)
    assert coppie == [(0, 5)], f"una sola operazione attesa, ottenute {coppie}"
    # senza isteresi (entra == esci) la stessa serie costa tre giri di commissioni
    assert len(indici_dei_segnali(previsto, entra=0.50, esci=0.50, cadenza=1)) == 3


def test_si_entra_anche_sui_minimi_previsti():
    """La regola guarda |previsione|: -0,6 e' «vicino a un minimo» ed e' un ingresso come +0,6."""
    assert indici_dei_segnali(np.array([-0.60, -0.10]), 0.50, 0.40, 1) == [(0, 1)]


def test_le_previsioni_mancanti_non_aprono_niente():
    coppie = indici_dei_segnali(np.array([np.nan, np.nan, 0.60, 0.10]), 0.50, 0.40, 1)
    assert coppie == [(2, 3)]


def test_le_collocazioni_casuali_non_si_sovrappongono():
    """Il motore di `pnl` tiene una posizione sola: coppie sovrapposte falserebbero il controllo."""
    for seme in range(20):
        coppie = collocazioni_casuali(n_barre=1000, durate=[50, 80, 30, 60], seme=seme)
        for (_, fine), (inizio_dopo, _) in zip(coppie, coppie[1:]):
            assert inizio_dopo > fine, f"seme {seme}: {coppie}"
        assert all(0 <= a < b < 1000 for a, b in coppie)


def test_le_durate_casuali_sono_quelle_vere():
    """Il controllo deve appaiare l'esposizione, non solo il numero di operazioni."""
    durate = [40, 90, 25]
    coppie = collocazioni_casuali(n_barre=5000, durate=durate, seme=1)
    assert [b - a for a, b in coppie] == durate
