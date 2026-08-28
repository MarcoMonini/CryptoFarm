"""La strategia a confluenza: `trading/confluence.py`.

Il difetto che conta qui e' uno solo, e produce risultati **falsi positivi**: leggere una barra
lunga prima che sia chiusa. Non solleva niente, non si vede nel grafico, e migliora i numeri. Il
primo test e' scritto contro quello, e non e' il test ovvio: troncare la serie non lo vedrebbe,
perche' troncando fra le barre corte la barra lunga incriminata resta identica. Serve invece
**perturbare il futuro dentro una barra lunga gia' cominciata** e verificare che le decisioni
precedenti non si spostino di un capello.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import confluence
from cryptofarm.trading.indicators_extra import ExtraCache


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
    ingresso = next(quando for quando, _, obiettivo in risultato.eventi if obiettivo != 0)
    testo = risultato.spiega(ingresso)
    assert "score" in testo and "threshold" in testo and "families" in testo
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


def test_la_priorita_e_il_margine_sopra_la_soglia(candele):
    """Serve al paniere a capitale condiviso: a parita' di barra vince il segnale piu' netto."""
    risultato = confluence.evaluate(candele, "15m")
    con_priorita = risultato.eventi_con_priorita()
    assert [e[:3] for e in con_priorita] == risultato.eventi
    aperture = [e for e in con_priorita if e[2] != 0]
    assert all(e[3] >= 0 for e in aperture), "si apre solo oltre la soglia: il margine non e' negativo"
    assert all(e[3] == 0.0 for e in con_priorita if e[2] == 0), "una chiusura non compete con niente"


# -------------------------------------------------------------------------------------------------
# La pagina: chi ha generato il segnale deve **vedersi**, non essere deducibile
# -------------------------------------------------------------------------------------------------


def _figura(candele, strategia="Confluence"):
    from cryptofarm.trading.simulator import trading_analysis

    figura, _, _ = trading_analysis(
        asset="TEST",
        interval="15m",
        wallet=100.0,
        valori={},
        strategia=strategia,
        show=True,
        market_data=candele,
    )
    return figura


def test_la_pagina_mostra_la_decisione_e_i_votanti(candele):
    nomi = {traccia.name for traccia in _figura(candele).data}
    assert {"Score", "Threshold"} <= nomi, "manca il riquadro della decisione"
    assert {"Regime plane (gate)", "Structure plane"} <= nomi, "mancano i piani lunghi"
    assert sum("·" in (n or "") for n in nomi) == len(confluence.VOTANTI), "manca un votante"


def test_ogni_segnale_dice_chi_l_ha_generato(candele):
    """Senza questo si vedrebbe un triangolo e bisognerebbe crederci.

    Gli acquisti dicono chi ha votato; le vendite dicono cosa ha chiuso la posizione, che nella
    grande maggioranza dei casi e' lo stop e non un voto.
    """
    per_nome = {t.name: t for t in _figura(candele).data if t.name in ("Buy", "Sell")}
    assert set(per_nome) == {"Buy", "Sell"}, "nessun segnale: il test non proverebbe niente"
    for nome, traccia in per_nome.items():
        assert traccia.text and all(traccia.text), f"{nome} senza spiegazione"
    compra = per_nome["Buy"].text[0]
    assert compra.startswith("entry — ") and "score" in compra and "families" in compra
    assert any(v.nome in compra for v in confluence.VOTANTI)
    assert per_nome["Sell"].text[0].startswith("exit — ")


def test_con_poca_storia_la_pagina_degrada_invece_di_cadere():
    """I piani lunghi non esistono ancora: la pagina si apre lo stesso, senza segnali.

    E' la condizione in cui gira il servizio pubblico appena avviato, ed e' anche il livello da
    cui e' gia' passato un guasto che tolse il simulatore dalla produzione.
    """
    figura = _figura(_candele(giorni=1))
    assert not [t for t in figura.data if t.name in ("Buy", "Sell")]


def test_le_altre_strategie_non_perdono_i_segnali(candele):
    """Lo scompattamento in `pnl` e' cambiato per accettare un terzo elemento: le strategie che
    non lo usano devono comportarsi esattamente come prima."""
    marcatori = [t for t in _figura(candele, "Ichimoku Trend").data if t.name in ("Buy", "Sell")]
    assert marcatori and all(t.text is None or not any(t.text) for t in marcatori)


# -------------------------------------------------------------------------------------------------
# Zero operazioni non e' un risultato, e' una domanda
# -------------------------------------------------------------------------------------------------


def test_con_poca_storia_la_diagnosi_dice_che_manca_la_storia():
    """Il caso in cui cade chi apre la pagina: 240 ore, il default, sono dieci barre giornaliere.

    La media di regime ne chiede cinquanta, quindi il cancello non puo' aprirsi e non si opera mai
    -- senza che niente lo dica. E' il difetto piu' grave della prima versione: falliva in
    silenzio, e il silenzio si legge come «la strategia non ha trovato occasioni».
    """
    risultato = confluence.evaluate(_candele(giorni=10), "15m")
    assert risultato.ingressi == 0
    messaggio = risultato.perche_non_entra()
    assert "not enough history" in messaggio
    assert "10 bars" in messaggio and "50" in messaggio


def test_la_diagnosi_distingue_la_soglia_dalla_storia(candele):
    """Tre cause diverse chiedono tre rimedi diversi: caricare piu' storia, abbassare la soglia,
    abbassare l'ampiezza. Un messaggio solo per tutte non servirebbe a niente."""
    stati = confluence.stati_dei_votanti(candele, "15m")
    impossibile = confluence.evaluate(candele, "15m", theta_base=0.99, stati=stati)
    assert "never reached the threshold" in impossibile.perche_non_entra()

    stretta = confluence.evaluate(candele, "15m", k_famiglie=99, stati=stati)
    assert "families at once" in stretta.perche_non_entra()


def test_chi_opera_non_ha_niente_da_spiegare(candele):
    risultato = confluence.evaluate(candele, "15m")
    assert risultato.ingressi > 0 and risultato.perche_non_entra() == ""


@pytest.mark.parametrize(
    ("intervallo", "atteso"),
    [("15m", ""), ("30m", ""), ("1h", ""), ("1m", "too short"), ("4h", "decades"), ("1d", "decades")],
)
def test_la_scala_dei_piani_si_dichiara_fuori_misura(intervallo, atteso):
    """Il menu offre nove intervalli e la scala x1/x4/x16/x96 e' nata su quindici minuti.

    A un minuto il «regime» dura un'ora e mezza; a un giorno chiede barre da 96 giorni, cioe'
    decenni di storia. La strategia gira lo stesso -- non e' un errore, e' una scelta di chi
    guarda -- ma va detto, perche' dal menu non si vede.
    """
    avviso = confluence.scala_fuori_misura(intervallo)
    assert (atteso in avviso) if atteso else (avviso == "")


def test_i_piani_si_possono_leggere_prima_di_lanciare():
    """La pagina li mostra nella barra laterale: e' cosi' che si vede che le aggregazioni ci sono."""
    assert confluence.piani("15m") == {
        "innesco": "15m",
        "conferma": "1h",
        "struttura": "4h",
        "regime": "1d",
    }
    assert confluence.piani("1h")["regime"] == "4d"


def test_la_pagina_spiega_perche_non_ha_operato():
    from cryptofarm.trading import panels

    corta = _candele(giorni=10)
    assert "not enough history" in panels.diagnosi_confluenza(corta, panels.valori_predefiniti(), "15m")


def test_le_ore_richieste_sono_il_numero_che_manca_a_chi_apre_la_pagina():
    """Il default della pagina e' 240 ore. A quindici minuti ne servono piu' di mille."""
    assert confluence.ore_richieste("15m", 50) == 1200
    assert confluence.ore_richieste("1h", 50) == 4800


# -------------------------------------------------------------------------------------------------
# Il grafico deve essere un testimone affidabile di cio' che il motore ha fatto
# -------------------------------------------------------------------------------------------------


def test_ogni_ingresso_soddisfa_tutte_e_quattro_le_condizioni(candele):
    """L'audit che va fatto prima di credere a qualunque grafico: il motore e' coerente con se'.

    Se questo cade, l'incoerenza e' nelle regole. Se passa e il grafico sembra incoerente,
    l'incoerenza e' nel grafico -- ed e' li' che e' stata, la prima volta.
    """
    risultato = confluence.evaluate(candele, "15m")
    posizione = {quando: i for i, quando in enumerate(candele.index)}
    for quando, _, obiettivo in risultato.eventi:
        if obiettivo == 0:
            continue
        i = posizione[quando]
        assert risultato.regime[i] > 0, f"{quando}: aperto col cancello chiuso"
        assert risultato.punteggio[i] >= risultato.soglia[i], f"{quando}: aperto sotto la soglia"
        assert risultato.concordi_lungo[i] >= risultato.k_famiglie, f"{quando}: aperto senza ampiezza"


def test_ogni_uscita_ha_un_motivo_registrato(candele):
    risultato = confluence.evaluate(candele, "15m")
    uscite = [quando for quando, _, obiettivo in risultato.eventi if obiettivo == 0]
    assert uscite
    for quando in uscite:
        assert quando in risultato.motivi, f"{quando}: uscita senza motivo"
    assert set(risultato.motivi.values()) <= {
        "trailing stop",
        "regime gate shut",
        "score fell through the hysteresis band",
        "score below threshold for too long",
    }


def test_un_uscita_sullo_stop_non_elenca_i_votanti(candele):
    """Il difetto che faceva leggere «venduto mentre cinque votanti dicevano di comprare».

    E' vero e del tutto fuorviante: quella posizione l'ha chiusa il prezzo, non il voto. Quattro
    uscite su cinque sono lo stop, quindi era il caso piu' comune, non un angolo.
    """
    risultato = confluence.evaluate(candele, "15m")
    per_stop = [q for q, motivo in risultato.motivi.items() if motivo == "trailing stop"]
    assert per_stop, "senza uscite sullo stop il test non proverebbe niente"
    for quando in per_stop[:20]:
        testo = risultato.spiega(quando)
        assert testo.startswith("exit — trailing stop")
        assert not any(v.nome in testo for v in confluence.VOTANTI), testo


def test_gli_ingressi_si_distinguono_dalle_uscite_a_colpo_d_occhio(candele):
    risultato = confluence.evaluate(candele, "15m")
    for quando, _, obiettivo in risultato.eventi[:20]:
        atteso = "entry — " if obiettivo != 0 else "exit — "
        assert risultato.spiega(quando).startswith(atteso)


def test_il_cancello_non_sta_sullo_stesso_riquadro_del_punteggio():
    """Il cancello vale ±1 e il punteggio sta in ±0,5: sullo stesso asse il primo schiaccia il
    secondo in una riga piatta, e si vede «una linea ferma a 1» mentre si compra e si vende."""
    from cryptofarm.trading import panels

    decisione = panels.INDICATORI["confluenza"]
    piani = panels.INDICATORI["piani_lunghi"]
    assert decisione.pannello != piani.pannello
    assert {t.serie for t in decisione.tracce} == {"punteggio", "soglia"}
    assert {t.serie for t in piani.tracce} == {"regime", "struttura"}


def test_i_due_piani_lunghi_si_vedono_tutti_e_due(candele):
    """`struttura` e' meta' di `accordo_alto`, cioe' muove la soglia, e non era disegnata affatto."""
    figura = _figura(candele)
    nomi = {traccia.name for traccia in figura.data}
    assert "Regime plane (gate)" in nomi and "Structure plane" in nomi


def test_lo_stop_a_trailing_si_vede_sulle_candele(candele):
    """Chiude quattro operazioni su cinque: senza la linea, quelle vendite sono inspiegabili."""
    import numpy as np

    figura = _figura(candele)
    stop = [t for t in figura.data if t.name == "Trailing stop"]
    assert stop, "lo stop non e' disegnato"
    valori = np.asarray(stop[0].y, dtype=float)
    assert np.isfinite(valori).any(), "la serie dello stop e' tutta vuota"
    assert np.isnan(valori).any(), "lo stop deve essere assente quando si e' fuori dal mercato"


# -------------------------------------------------------------------------------------------------
# La soglia e l'isteresi: i due difetti di valutazione trovati provandola
# -------------------------------------------------------------------------------------------------


def test_la_soglia_si_muove_con_continuita(candele):
    """Con `np.sign` sui piani la soglia prendeva cinque valori e saltava di 0,15 per volta, contro
    un punteggio la cui ampiezza totale e' 0,91. Un salto del genere decide da solo."""
    import numpy as np

    risultato = confluence.evaluate(candele, "15m")
    salti = np.abs(np.diff(risultato.soglia))
    # La prima barra in cui il piano lungo diventa disponibile e' un gradino per forza: si passa
    # da «non c'e' dato» a un valore. Non e' quella il difetto, e contarla renderebbe il test una
    # misura della lunghezza del riscaldamento invece che della continuita'.
    partenza = int(np.flatnonzero(risultato.regime != 0)[0])
    salti = salti[partenza:]
    assert salti.max() < 0.05, f"la soglia salta di {salti.max():.3f} in una barra"
    assert (salti > 0.02).mean() < 0.001, "troppi salti grossi"
    assert len(set(np.round(risultato.soglia, 4))) > 50, "la soglia e' ancora una scala a gradini"


def test_nessuna_uscita_per_punteggio_e_causata_dal_salto_della_soglia(candele):
    """Una su quattro lo era: l'aveva decisa la soglia che si spostava, non il punteggio."""
    import numpy as np

    risultato = confluence.evaluate(candele, "15m")
    posizione = {quando: i for i, quando in enumerate(candele.index)}
    salto = np.abs(np.concatenate([[0.0], np.diff(risultato.soglia)]))
    per_punteggio = [posizione[q] for q, motivo in risultato.motivi.items() if "score" in motivo]
    assert per_punteggio
    assert not any(salto[i] > 0.02 for i in per_punteggio)


def test_non_si_apre_e_si_chiude_in_due_barre(candele):
    """Su barre da quindici minuti succedeva, e ogni volta paga due commissioni per niente."""
    posizione = {quando: i for i, quando in enumerate(candele.index)}
    risultato = confluence.evaluate(candele, "15m", barre_minime=4)
    apertura = None
    for quando, _, obiettivo in risultato.eventi:
        i = posizione[quando]
        if obiettivo != 0:
            apertura = i
        elif apertura is not None:
            durata = i - apertura
            motivo = risultato.motivi[quando]
            if "score" in motivo:
                assert durata >= 4, f"chiusa dal punteggio dopo {durata} barre"
            apertura = None


def test_il_pavimento_non_trattiene_ne_lo_stop_ne_il_cancello(candele):
    """La distinzione che conta: sono regole di rischio, non di opinione. Un pavimento che tiene
    aperta una posizione mentre lo stop e' saltato non e' pazienza, e' un difetto travestito."""
    risultato = confluence.evaluate(candele, "15m", barre_minime=500)
    rapide = [m for m in risultato.motivi.values() if m == "trailing stop"]
    assert rapide, "con un pavimento altissimo lo stop deve poter comunque chiudere"


def test_la_pazienza_taglia_la_coda_dell_isteresi(candele):
    """L'isteresi come idea e' buona, ma il punteggio decade piano e la posizione restava aperta
    per ore oltre il primo segnale di uscita."""
    import numpy as np

    posizione = {quando: i for i, quando in enumerate(candele.index)}

    def coda(**kwargs):
        risultato = confluence.evaluate(candele, "15m", **kwargs)
        apertura, ritardi = None, []
        for quando, _, obiettivo in risultato.eventi:
            i = posizione[quando]
            if obiettivo != 0:
                apertura = i
            elif apertura is not None:
                sotto = np.flatnonzero(risultato.punteggio[apertura:i] < risultato.soglia[apertura:i])
                if len(sotto):
                    ritardi.append(i - (apertura + int(sotto[0])))
                apertura = None
        return np.percentile(ritardi, 90)

    assert coda(pazienza=24) < coda(pazienza=10**9), "la pazienza non accorcia niente"


def test_la_pazienza_ha_un_motivo_suo(candele):
    """Perche' «uscito perche' il punteggio e' stato sotto troppo a lungo» e «uscito perche' e'
    caduto attraverso la banda» sono due cose diverse, e il grafico deve dirlo."""
    risultato = confluence.evaluate(candele, "15m")
    assert "score below threshold for too long" in risultato.motivi.values()


# -------------------------------------------------------------------------------------------------
# I votanti sono moduli: aggiungerne uno o toglierlo deve essere un'operazione sola
# -------------------------------------------------------------------------------------------------


def test_si_puo_scegliere_un_sottoinsieme_di_votanti(candele):
    tre = confluence.evaluate(candele, "15m", votanti=confluence.selezione("ichimoku", "flusso", "bande_innesco"))
    assert list(tre.voti) == ["ichimoku", "flusso", "bande_innesco"]
    assert set(tre.necessarieta) == {"ichimoku", "flusso", "bande_innesco"}
    assert abs(sum(tre.pesi.values()) - 1.0) < 1e-12, "i pesi si rinormalizzano sui votanti scelti"


def test_un_votante_sconosciuto_si_fa_notare():
    with pytest.raises(KeyError, match="votanti sconosciuti"):
        confluence.selezione("ichimoku", "inventato")


def test_registrare_un_votante_basta_a_farlo_entrare_ovunque(candele):
    """La prova della modularita': si registra e basta, senza toccare nessun elenco.

    Se un giorno l'aggiunta richiedesse anche una riga in `panels`, una in `config` e una nella
    griglia del banco, tre posti su quattro si disallineerebbero al primo votante distratto.
    """
    from cryptofarm.trading import panels

    def sempre_lungo(df, cache, p):
        return [(df.index[int(p["ritardo"])], float(df["Close"].iloc[int(p["ritardo"])]), 1)]

    finto = confluence.Votante(
        "prova",
        "sperimentale",
        "conferma",
        sempre_lungo,
        (confluence.Par("CONF_INNESCO", "ritardo"),),
    )
    scelti = (*confluence.selezione("ichimoku"), finto)
    risultato = confluence.evaluate(candele, "15m", votanti=scelti, k_famiglie=1)

    assert "prova" in risultato.voti and "prova" in risultato.necessarieta
    assert "sperimentale" in {v.famiglia for v in scelti}
    # E il registro e' l'unica fonte da cui la pagina ricava i suoi riquadri.
    titoli = [titolo for titolo, _ in panels.gruppi_di("Confluence")]
    assert [f"Voter · {v.nome}" for v in confluence.VOTANTI] == [t for t in titoli if t.startswith("Voter · ")]


def test_ogni_parametro_di_ogni_votante_ha_la_sua_costante_e_la_sua_etichetta():
    """Un parametro dichiarato e non configurabile comparirebbe col nome della costante, o non
    comparirebbe affatto: in tutti e due i casi e' una manopola che esiste e non si vede."""
    from cryptofarm.trading import config, panels

    for votante in confluence.VOTANTI:
        for parametro in votante.parametri:
            assert isinstance(getattr(config, parametro.config, None), config.Param), parametro.config
            assert parametro.config in panels.ETICHETTE, parametro.config
            assert parametro.config in panels.parametri_di("Confluence"), parametro.config


def test_i_parametri_dei_votanti_cambiano_davvero_il_risultato(candele):
    """Altrimenti sarebbero widget che non fanno niente, che e' peggio di non averli."""
    normale = confluence.evaluate(candele, "15m")
    diverso = confluence.evaluate(candele, "15m", parametri_votanti={"ichimoku": {"fast": 5, "slow": 13, "span": 26}})
    assert normale.eventi != diverso.eventi


def test_i_valori_misurati_vengono_dal_piano_del_votante_non_dalla_pagina():
    """Un votante di struttura a base 15m gira a 4h: il suo valore misurato e' quello di 4h.

    Prendere quello di 15m sarebbe sbagliato **in silenzio**, che e' il modo peggiore.
    """
    from cryptofarm.trading.tuned_defaults import PER_INTERVALLO

    ichimoku = confluence.REGISTRO["ichimoku"]
    a_quindici = confluence.valori_del_votante(ichimoku, "4h")
    assert a_quindici["require_cloud"] == PER_INTERVALLO["4h"]["Ichimoku Trend"].get("REQUIRE_CLOUD", 1)
    assert confluence.valori_del_votante(ichimoku, "1d")["require_cloud"] == 0


def test_gli_stati_precalcolati_con_un_override_sollevano(candele):
    """Sarebbero stati vecchi, e il risultato sbagliato non lo direbbe nessuno."""
    stati = confluence.stati_dei_votanti(candele, "15m")
    with pytest.raises(ValueError, match="vecchi"):
        confluence.evaluate(candele, "15m", stati=stati, parametri_votanti={"ichimoku": {"fast": 5}})


@pytest.mark.parametrize("quanti", [2, 3, 4, 6])
def test_i_pesi_sommano_a_uno_con_qualunque_numero_di_votanti(quanti):
    """Con tre votanti un tetto di 0,30 li cappava tutti e la somma faceva 0,90: il punteggio
    restava sistematicamente sotto la soglia, e nessuno lo diceva."""
    nomi = [f"v{i}" for i in range(quanti)]
    pesi = confluence._pesi(nomi, w_max=0.30)
    assert abs(sum(pesi.values()) - 1.0) < 1e-12, f"{quanti} votanti: somma {sum(pesi.values())}"


def test_la_diagnosi_regge_il_dizionario_parziale_della_barra_laterale():
    """La pagina passa **solo cio' che disegna**, non tutti i parametri di `config`.

    Il test precedente passa `valori_predefiniti()`, cioe' un dizionario completo che la barra
    laterale non produce mai: con quello un parametro mancante sembra presente e il `KeyError` che
    la pagina solleva davvero non si vede. Qui il dizionario si costruisce dai nomi dei widget
    numerici, che e' la forma minima con cui `confluenza_di` puo' essere chiamata.
    """
    from cryptofarm.trading import config, panels

    iniziali = panels.valori_predefiniti(config.CONFLUENCE_STRATEGY, "15m")
    dalla_barra = {nome: iniziali[nome] for _, nomi in panels.gruppi_di(config.CONFLUENCE_STRATEGY) for nome in nomi}
    assert "CONF_IN_FORMAZIONE" not in dalla_barra

    assert "not enough history" in panels.diagnosi_confluenza(_candele(giorni=10), dalla_barra, "15m")


def test_la_necessarieta_vale_quanto_la_definizione_che_la_descrive(candele):
    """Il valore, non solo la forma: la riscrittura veloce deve dare gli stessi numeri.

    `_necessarieta` costruiva le serie **intere** una volta per ogni coppia (votante, ingresso)
    per poi leggerne un elemento solo: su sette anni di barre da quindici minuti erano l'86% del
    tempo di `evaluate`, e nessuna di quelle serie serviva oltre la barra d'ingresso. Ritagliare
    prima e contare dopo da' per costruzione lo stesso risultato, ma «per costruzione» e' esatta-
    mente cio' che va verificato: qui la definizione lenta sta scritta nel test e i due numeri si
    confrontano. Se qualcuno riscrive di nuovo quel ciclo, questo test dice se ha cambiato idea.
    """
    risultato = confluence.evaluate(candele, "15m")
    voti, pesi, soglia = risultato.voti, risultato.pesi, risultato.soglia
    famiglie = {v.nome: v.famiglia for v in confluence.VOTANTI}

    # La definizione, trascritta senza furbizie: per ogni votante, la frazione di ingressi in cui
    # azzerarlo avrebbe impedito l'ingresso -- per punteggio sotto soglia o per ampiezza sotto il
    # minimo di famiglie.
    barre = np.array([risultato.indice.get_loc(q) for q, _, obiettivo in risultato.eventi if obiettivo != 0])
    assert len(barre) > 10, "servono abbastanza ingressi perche' il confronto significhi qualcosa"
    verso = np.sign(sum(pesi[n] * voti[n] for n in voti)[barre])
    verso[verso == 0] = 1
    atteso = {}
    for nome in voti:
        restanti = {n: v for n, v in voti.items() if n != nome}
        punteggio = sum(pesi[n] * restanti[n] for n in restanti)[barre]
        sotto_soglia = punteggio * verso < soglia[barre]
        ampiezza = np.array(
            [confluence._famiglie_concordi(restanti, famiglie, int(v))[b] for b, v in zip(barre, verso)]
        )
        atteso[nome] = float(np.mean(sotto_soglia | (ampiezza < risultato.k_famiglie)))

    assert risultato.necessarieta == pytest.approx(atteso)


# --- il votante a modello -------------------------------------------------------------------------


def test_il_votante_a_modello_non_vota_mai_corto(candele, monkeypatch):
    """La proprieta' che non si vede dai tipi, e che un refactoring distratto romperebbe.

    Il modello a swing prevede la prossimita' a un estremo locale e la forma misurata di quel
    segnale e' a U: entrambi i poli precedono rendimenti sopra la media
    (`.claude/docs/modello-swing.md` §5.1). Far votare `sign(previsione)` -- la lettura naturale
    di un target in [-1, 1] -- darebbe un voto **corto** proprio sulle barre che rendono di piu'.
    Il votante vota quindi +1 o 0, mai -1, e questo test e' cio' che lo tiene fermo.

    Il modello finto mette i due poli a blocchi di una cadenza, cosi' che la decisione ne veda sia
    di negativi sia di positivi, e ogni terzo blocco al centro, cosi' che ci siano anche uscite.
    """
    cadenza = confluence.signals.swing_cadenza(candele.index)

    class Poli:
        def predict(self, X):
            blocco = np.arange(len(X)) // cadenza
            return np.where(blocco % 2 == 0, -1.0, 1.0) * np.where(blocco % 3 == 2, 0.1, 0.9)

    monkeypatch.setattr(confluence.signals, "swing_model", lambda: Poli())
    eventi = confluence._modello(candele, ExtraCache(candele), {"entra": 0.5, "esci": 0.4})

    stati = {stato for _, _, stato in eventi}
    assert stati == {0, 1}, f"servono ingressi e uscite per misurare qualcosa, visti {stati}"


def test_senza_artefatto_il_votante_a_modello_tace_e_resta_fuori_dal_default(candele, monkeypatch):
    """In produzione `models/` e' vuoto, e li' la confluenza deve restare quella misurata.

    Non basta che il votante si astenga: i pesi si normalizzano sui votanti **presenti**, quindi
    un ottavo che tace sempre alzerebbe di fatto la soglia per gli altri sette. Deve proprio
    restare fuori dall'insieme di default -- pur restando nel registro, cosi' `selezione` lo
    raggiunge per misurarlo.
    """
    monkeypatch.setattr(confluence.signals, "MODELS_DIR", Path("/nessun/modello/qui"))
    for nome in ("swing_model", "rl_model"):
        getattr(confluence.signals, nome).cache_clear()
        monkeypatch.setattr(confluence.signals, nome, getattr(confluence.signals, nome).__wrapped__)

    assert confluence._modello(candele, ExtraCache(candele), {"entra": 0.5, "esci": 0.4}) == []
    nomi = [v.nome for v in confluence.votanti_predefiniti()]
    assert "modello" not in nomi
    assert len(nomi) == len(confluence.REGISTRO) - 1
    assert confluence.selezione("modello")[0].nome == "modello", "il registro lo tiene comunque"
