"""Il banco della confluenza: `scripts/confluence_lab.py`.

Il banco non si puo' misurare qui -- lo store delle candele non c'e' -- ma si puo' verificare che
funzioni, ed e' il punto del suo `--selfcheck`: un banco che si scopre rotto solo sulla macchina
che ha i dati e' un banco che fa perdere il giro.
"""

import pytest

from cryptofarm.trading import confluence
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
    # Lo stop non sta in `CENTRO` apposta: il suo valore dipende dalla modalita' -- 3 ATR dove
    # chiude, 0 dove ribalta -- e lo mette `celle` leggendo `confluence.STOP_PREDEFINITO`.
    assert configurazioni[0] == {**lab.CENTRO, "atr_multiplier": 3.0, "modalita": "cancello"}
    for parametro, valori in lab.SCANSIONE.items():
        assert set(valori) <= {c[parametro] for c in configurazioni}, parametro


def test_il_centro_e_dentro_ogni_scansione():
    """Altrimenti la riga di partenza non sarebbe confrontabile con le sue variazioni."""
    centro = lab.celle("coordinate")[0]
    for parametro, valori in lab.SCANSIONE.items():
        assert centro[parametro] in valori, f"{parametro}: il centro non e' fra i valori provati"


def test_in_inversione_i_parametri_ignorati_sono_inerti():
    """La lista si verifica a misura, non a lettura.

    `PARAMETRI_IGNORATI` e' un'ottimizzazione della griglia, e un'ottimizzazione basata su una lista
    scritta a mano e' una bugia che aspetta: basta collegare uno di quei parametri al motore a
    inversione perche' la griglia smetta in silenzio di misurare qualcosa che conta. Qui ciascuno
    gira ai **due estremi** del suo intervallo di scansione e si pretende che gli eventi siano
    identici -- se non lo sono, quel parametro e' vivo e va tolto dalla lista.
    """
    candele = lab._finte(giorni=120)
    stati = confluence.stati_dei_votanti(candele, "15m")

    def eventi(**kwargs):
        r = confluence.evaluate(candele, "15m", stati=stati, modalita="inversione", **kwargs)
        return [e[:3] for e in r.eventi]

    riferimento = eventi()
    assert len(riferimento) > 5, "senza operazioni il confronto non proverebbe niente"
    for parametro in lab.PARAMETRI_IGNORATI:
        valori = lab.SCANSIONE[parametro]
        assert eventi(**{parametro: valori[0]}) == riferimento, f"{parametro} e' vivo: toglilo dalla lista"
        assert eventi(**{parametro: valori[-1]}) == riferimento, f"{parametro} e' vivo: toglilo dalla lista"

    # E il controllo opposto, senza il quale il test passerebbe anche con la lista piena di tutto:
    # i parametri che quella macchina legge davvero devono spostare gli eventi.
    for parametro, estremi in (("theta_base", (0.15, 0.55)), ("emivita", (1.0, 24.0)), ("atr_multiplier", (0.0, 3.0))):
        assert eventi(**{parametro: estremi[0]}) != eventi(**{parametro: estremi[1]}), f"{parametro} doveva essere vivo"


def test_in_inversione_la_griglia_non_muove_quel_che_quella_macchina_non_legge():
    """`_percorri_inversione` non riceve isteresi, pazienza, barre minime, ampiezza ne' innesco.

    Muoverli in griglia costerebbe ore per righe **identiche**, e ogni riga identica entra nel
    conto delle prove che `scripts/multiplicity.py` corregge: prove che non sono prove rendono la
    correzione piu' severa senza aver misurato niente.
    """
    inversione = lab.celle("coordinate", "inversione")
    assert all(c["modalita"] == "inversione" for c in inversione)
    assert not {p for c in inversione for p in c} & set(lab.PARAMETRI_IGNORATI)
    assert inversione[0]["atr_multiplier"] == 0.0, "in inversione il centro ha lo stop spento"
    assert len(inversione) < len(lab.celle("coordinate")), "e la griglia e' piu' piccola"


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
