"""La strategia a confluenza: `trading/confluence.py`.

Il difetto che conta qui e' uno solo, e produce risultati **falsi positivi**: leggere una barra
lunga prima che sia chiusa. Non solleva niente, non si vede nel grafico, e migliora i numeri. Il
primo test e' scritto contro quello, e non e' il test ovvio: troncare la serie non lo vedrebbe,
perche' troncando fra le barre corte la barra lunga incriminata resta identica. Serve invece
**perturbare il futuro dentro una barra lunga gia' cominciata** e verificare che le decisioni
precedenti non si spostino di un capello.
"""

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import confluence


def _candele(giorni: int = 120, seme: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seme)
    idx = pd.date_range("2024-01-01", periods=96 * giorni, freq="15min", name="Open time")
    passo = 100 + np.cumsum(rng.normal(0.01, 0.4, len(idx)))
    return pd.DataFrame(
        {
            "Open": passo,
            "High": passo + abs(rng.normal(0, 0.5, len(idx))),
            "Low": passo - abs(rng.normal(0, 0.5, len(idx))),
            "Close": passo + rng.normal(0, 0.1, len(idx)),
            "Volume": rng.random(len(idx)) * 10,
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def candele():
    return _candele()


# Il taglio cade a mezzogiorno **e un quarto**: deliberatamente *dentro* una barra di ogni piano
# lungo, mai su un loro confine. Un taglio allineato ai confini non proverebbe niente -- le barre
# lunghe precedenti sarebbero interamente prima della scossa comunque -- ed e' l'errore che questo
# test aveva la prima volta che e' stato scritto: passava anche reintroducendo il difetto.
TAGLIO = 96 * 60 + 49


def test_nessun_piano_lungo_anticipa(candele):
    """Lo stato dei votanti prima del taglio non cambia se si riscrive il futuro.

    E' il controllo di causalita' fra intervalli nella sua forma stretta, e non e' il troncamento:
    troncando fra le barre corte la barra lunga incriminata resta identica, quindi un troncamento
    **non vedrebbe** il difetto. Qui si riscrive invece la seconda meta' di una barra lunga gia'
    cominciata: chi legge quella barra prima che chiuda se ne accorge, chi aspetta la chiusura no.
    """
    scosso = candele.copy()
    scosso.iloc[TAGLIO:, :4] *= 1.5

    prima = confluence.stati_dei_votanti(candele, "15m")
    dopo = confluence.stati_dei_votanti(scosso, "15m")
    for nome in prima:
        assert np.array_equal(prima[nome][:TAGLIO], dopo[nome][:TAGLIO]), f"{nome} legge il futuro"
    assert any(prima[n].any() for n in prima), "con tutti gli stati a zero il test non proverebbe niente"


def test_nessuna_decisione_passata_cambia_riscrivendo_il_futuro(candele):
    """Lo stesso controllo sul risultato intero, eventi compresi.

    Da solo **non basta**: reintroducendo il difetto questo passa e cade solo quello sopra, perche'
    un piano che anticipa sposta lo stato di ogni votante ma non necessariamente un
    ingresso proprio in quella finestra. Sta qui per coprire il resto della catena, non il
    look-ahead.
    """
    scosso = candele.copy()
    scosso.iloc[TAGLIO:, :4] *= 1.5
    limite = candele.index[TAGLIO]

    prima = [e for e in confluence.evaluate(candele, "15m").eventi if e[0] < limite]
    dopo = [e for e in confluence.evaluate(scosso, "15m").eventi if e[0] < limite]
    assert prima == dopo, "una decisione passata e' cambiata riscrivendo il futuro"
    assert prima, "senza eventi il test non proverebbe niente"


def test_gli_stati_dei_votanti_non_dipendono_dalla_griglia(candele):
    """I votanti sono **congelati**: e' il vincolo che tiene a nove il conto dei parametri liberi."""
    stati = confluence.stati_dei_votanti(candele, "15m")
    a = confluence.evaluate(candele, "15m", theta_base=0.2, emivita=3)
    b = confluence.evaluate(candele, "15m", theta_base=0.2, emivita=3, stati=stati)
    assert a.eventi == b.eventi
    for nome in stati:
        assert np.array_equal(a.voti[nome], b.voti[nome])


def test_l_ampiezza_minima_puo_impedire_ogni_ingresso(candele):
    """Un peso grande, da solo, non deve poter aprire una posizione: qui il freno si vede tirato
    a fondo -- piu' famiglie di quante ne esistano, quindi nessun ingresso possibile."""
    stati = confluence.stati_dei_votanti(candele, "15m")
    aperto = confluence.evaluate(candele, "15m", k_famiglie=1, stati=stati)
    chiuso = confluence.evaluate(candele, "15m", k_famiglie=99, stati=stati)
    assert aperto.ingressi > 0
    assert chiuso.ingressi == 0


def test_il_tetto_sui_pesi_tiene_e_la_somma_resta_uno():
    nomi = ["a", "b", "c", "d"]
    uguali = confluence._pesi(nomi, w_max=0.30)
    assert all(abs(p - 0.25) < 1e-12 for p in uguali.values()), "a pesi uguali il tetto non morde"

    sbilanciati = confluence._pesi(nomi, w_max=0.30, pesi={"a": 10, "b": 1, "c": 1, "d": 1})
    assert abs(sum(sbilanciati.values()) - 1.0) < 1e-12
    assert all(p <= 0.30 + 1e-12 for p in sbilanciati.values())
    assert abs(sbilanciati["a"] - 0.30) < 1e-12, "chi eccede va tagliato al tetto, non ridotto un po'"


def test_non_si_entra_e_si_esce_sulla_stessa_barra(candele):
    """E' quello che l'isteresi compra. Senza, il punteggio che oscilla attorno alla soglia paga
    due commissioni per niente, ripetutamente."""
    eventi = confluence.evaluate(candele, "15m", isteresi=0.10).eventi
    quando = [e[0] for e in eventi]
    assert len(quando) == len(set(quando)), "due eventi sulla stessa barra"


def test_le_barre_in_formazione_cambiano_qualcosa(candele):
    """L'ablazione deve misurare qualcosa: se accendere e spegnere le barre in formazione desse lo
    stesso risultato, il meccanismo non sarebbe collegato a niente."""
    stati = confluence.stati_dei_votanti(candele, "15m")
    viva = confluence.evaluate(candele, "15m", barre_in_formazione=True, stati=stati)
    chiusa = confluence.evaluate(candele, "15m", barre_in_formazione=False, stati=stati)
    assert viva.eventi != chiusa.eventi


def test_la_necessarieta_e_una_frazione_per_ogni_votante(candele):
    risultato = confluence.evaluate(candele, "15m")
    assert set(risultato.necessarieta) == {v.nome for v in confluence.VOTANTI}
    assert all(0.0 <= q <= 1.0 for q in risultato.necessarieta.values())


def test_spiega_dice_chi_ha_generato_il_segnale(candele):
    risultato = confluence.evaluate(candele, "15m")
    testo = risultato.spiega(risultato.eventi[0][0])
    assert "punteggio" in testo and "soglia" in testo and "famiglie" in testo
    assert any(v.nome in testo for v in confluence.VOTANTI)


def test_i_piani_sono_multipli_dell_intervallo_di_base():
    """Su barre da quindici minuti la scala e' esattamente 15m/1h/4h/1d, ma non c'e' nessun
    intervallo scritto dentro il codice: la stessa strategia gira su qualunque base."""
    minuti = 15
    assert [confluence._intervallo(minuti * f) for f in confluence.FATTORI.values()] == [
        "15m",
        "1h",
        "4h",
        "1d",
    ]
    assert confluence._intervallo(60 * 16) == "16h"


def test_senza_verso_corto_non_si_apre_mai_una_posizione_corta(candele):
    eventi = confluence.evaluate(candele, "15m", allow_short=False).eventi
    assert all(e[2] >= 0 for e in eventi)
