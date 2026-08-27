"""Il banco della confluenza: `scripts/confluence_lab.py`.

Il banco non si puo' misurare qui -- lo store delle candele non c'e' -- ma si puo' verificare che
funzioni, ed e' il punto del suo `--selfcheck`: un banco che si scopre rotto solo sulla macchina
che ha i dati e' un banco che fa perdere il giro.
"""

import pytest

from scripts import confluence_lab as lab


def test_il_selfcheck_del_banco_passa():
    """Copre a valle: griglie, valutazione, paniere, riferimenti e riassunto, su dati finti."""
    lab._selfcheck()


@pytest.mark.parametrize("nome", lab.NOMI_GRIGLIA)
def test_ogni_griglia_e_eseguibile(nome):
    """Mezzo milione di celle non e' una griglia, e' una trappola: a 1,6 s l'una sono giorni.

    Il tetto non e' un'opinione sul gusto -- e' il punto oltre il quale nessuno la lancia davvero,
    e una griglia che nessuno lancia non misura niente.
    """
    configurazioni = lab.celle(nome)
    assert 0 < len(configurazioni) <= 5000, f"{nome}: {len(configurazioni)} celle"


def test_la_scansione_per_coordinata_copre_ogni_parametro():
    """La cartesiana congela cinque parametri su undici: se non li scandisse nessuno, il loro
    valore resterebbe una scelta mai messa alla prova."""
    configurazioni = lab.celle("coordinate")
    assert configurazioni[0] == lab.CENTRO
    for parametro, valori in lab.SCANSIONE.items():
        assert set(valori) <= {c[parametro] for c in configurazioni}, parametro


def test_il_centro_e_dentro_ogni_scansione():
    """Altrimenti la riga di partenza non sarebbe confrontabile con le sue variazioni."""
    for parametro, valori in lab.SCANSIONE.items():
        assert lab.CENTRO[parametro] in valori, f"{parametro}: il centro non e' fra i valori provati"


def test_il_riferimento_appaiato_si_sceglie_sulla_frequenza_non_sulla_resa():
    """Sceglierlo sulla resa lo renderebbe un secondo massimo di griglia, e il confronto fra la
    confluenza e il suo riferimento non direbbe piu' niente."""
    candele = lab._finte(giorni=60)
    obiettivo = 8.0
    righe = lab.riferimenti(candele, "15m", obiettivo)
    appaiato = righe[-1]
    assert appaiato["riferimento"] == "ichimoku (frequenza appaiata)"
    distanza = abs(appaiato["trade_anno"] - obiettivo)
    assert distanza <= abs(righe[1]["trade_anno"] - obiettivo) + 1e-9
