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

from cryptofarm.trading import confluence, panels
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
    assert {"Score", "Long threshold", "Short threshold"} <= nomi, "manca il riquadro della decisione"
    assert {"Regime plane (gate)", "Structure plane"} <= nomi, "mancano i piani lunghi"
    assert sum("·" in (n or "") for n in nomi) == len(confluence.VOTANTI), "manca un votante"


def test_il_riquadro_dei_votanti_ha_una_traccia_per_votante():
    """I nomi delle serie sono quelli del **registro**, non quelli del default.

    Il test qui sopra conta le tracce disegnate, e ne conta giuste anche se un nome e' sbagliato:
    un votante che non ha una traccia e una traccia che non ha un votante si compensano. Questo
    confronta gli insiemi, e lo fa contro `REGISTRO` invece che contro `VOTANTI` perche' il
    secondo dipende da cosa c'e' in `models/`: e' l'unico modo di verificare la traccia del
    votante a modello anche dove l'artefatto non c'e', cioe' in CI e in produzione.
    """
    dichiarate = {traccia.serie for traccia in panels.INDICATORI["votanti"].tracce}
    assert dichiarate == set(confluence.REGISTRO)


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
        # Il punteggio e' nell'orientamento dei voti (-1 e' lungo): il confronto con la soglia,
        # che e' una magnitudine, passa da `convinzione` esattamente come nel motore.
        sostegno = confluence.convinzione(risultato.punteggio[i], obiettivo)
        attiva = risultato.soglia[i] if obiettivo > 0 else risultato.soglia_corta[i]
        assert sostegno >= attiva, f"{quando}: aperto sotto la soglia del proprio verso"
        concordi = risultato.concordi_lungo if obiettivo > 0 else risultato.concordi_corto
        assert concordi[i] >= risultato.k_famiglie, f"{quando}: aperto senza ampiezza"


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
    assert {t.serie for t in decisione.tracce} == {"punteggio", "soglia", "soglia_corta"}
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
    # Il piano e' NaN finche' non si sa e non zero, quindi «disponibile» si chiede con `isfinite`:
    # con `!= 0` il NaN risponde di si' e la partenza cadrebbe sulla barra zero, includendo proprio
    # il gradino di riscaldamento che questo test esiste per escludere.
    partenza = int(np.flatnonzero(np.isfinite(risultato.regime) & (risultato.regime != 0))[0])
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
                # «Sotto la soglia» si chiede con `convinzione`: il punteggio e' sull'asse dei
                # voti, dove una posizione lunga e' negativa, e il confronto crudo sarebbe vero
                # quasi sempre.
                sostegno = confluence.convinzione(risultato.punteggio[apertura:i], +1)
                sotto = np.flatnonzero(sostegno < risultato.soglia[apertura:i])
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
    stati = risultato.stati
    famiglie = {v.nome: v.famiglia for v in confluence.VOTANTI}

    # La definizione, trascritta senza furbizie: per ogni votante, la frazione di ingressi in cui
    # azzerarlo avrebbe impedito l'ingresso -- per punteggio sotto soglia o per ampiezza sotto il
    # minimo di famiglie.
    barre = np.array([risultato.indice.get_loc(q) for q, _, obiettivo in risultato.eventi if obiettivo != 0])
    assert len(barre) > 10, "servono abbastanza ingressi perche' il confronto significhi qualcosa"
    # Il verso dell'**operazione**: il punteggio e' sull'asse dei voti, dove un consenso lungo e'
    # negativo, quindi il segno va convertito e non letto cosi' com'e'.
    verso = np.sign(confluence.convinzione(sum(pesi[n] * voti[n] for n in voti)[barre], +1))
    verso[verso == 0] = 1
    attiva = np.where(verso > 0, soglia[barre], risultato.soglia_corta[barre])
    atteso = {}
    for nome in voti:
        restanti = {n: v for n, v in voti.items() if n != nome}
        punteggio = sum(pesi[n] * restanti[n] for n in restanti)[barre]
        sotto_soglia = confluence.convinzione(punteggio, verso) < attiva
        # L'ampiezza si conta sugli **stati**: togliere un votante e' togliere la sua opinione,
        # non la coda del suo voto.
        altri_stati = {n: v for n, v in stati.items() if n != nome}
        ampiezza = np.array(
            [confluence._famiglie_concordi(altri_stati, famiglie, int(v))[b] for b, v in zip(barre, verso)]
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
    for nome in ("swing_model", "rl_model", "entry_model"):
        getattr(confluence.signals, nome).cache_clear()
        monkeypatch.setattr(confluence.signals, nome, getattr(confluence.signals, nome).__wrapped__)

    assert confluence._modello(candele, ExtraCache(candele), {"entra": 0.5, "esci": 0.4}) == []
    nomi = [v.nome for v in confluence.votanti_predefiniti()]
    assert "modello" not in nomi
    assert len(nomi) == len(confluence.REGISTRO) - 1
    assert confluence.selezione("modello")[0].nome == "modello", "il registro lo tiene comunque"


# -------------------------------------------------------------------------------------------------
# I due versi: ogni votante li sa dire tutti e due, e la soglia li tratta allo stesso modo
# -------------------------------------------------------------------------------------------------


def _candele_con_inversione(giorni: int = 300, seme: int = 0) -> pd.DataFrame:
    """Candele che salgono per meta' finestra e scendono per l'altra meta'.

    Il random walk di `_candele` ha deriva positiva e su una finestra qualunque puo' non offrire
    mai a un votante lento l'occasione di dire «corto». Qui l'occasione c'e' per costruzione in
    tutti e due i versi, che e' la condizione minima perche' «non vota mai corto» voglia dire
    qualcosa invece di «su questi dati non gli e' capitato».

    La **volatilita' resta quella di `_candele`** (`sigma = 0,4` per barra, non riscalata), e non
    e' un dettaglio: con un rumore dieci volte piu' piccolo la serie e' cosi' liscia che ichimoku
    non incrocia mai e le bande a 2,5 ATR non si toccano mai. Quei due votanti risultavano allora
    «incapaci di votare corto» perche' non votavano affatto, e il test avrebbe accusato il codice
    di un difetto del dato di prova.
    """
    n = 96 * giorni
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", name="Open time")
    rng = np.random.default_rng(seme)
    t = np.arange(n)
    passo = 100 + np.where(t < n // 2, t * 0.02, (n // 2) * 0.02 - (t - n // 2) * 0.02)
    passo = passo + np.cumsum(rng.normal(0, 0.4, n))
    return pd.DataFrame(
        {
            "Open": passo,
            "High": passo + abs(rng.normal(0, 0.5, n)),
            "Low": passo - abs(rng.normal(0, 0.5, n)),
            "Close": passo + rng.normal(0, 0.1, n),
            "Volume": rng.random(n) * 10,
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def candele_con_inversione():
    return _candele_con_inversione()


def test_ogni_votante_sa_dire_tutti_e_due_i_versi(candele_con_inversione):
    """Il difetto che questo test esiste per prendere: un votante che non puo' votare corto.

    `_bande` chiamava `atr_band_bounce` senza `allow_short`, e quella funzione e' l'unica di
    `strategies_ls` che ha `False` per default. La famiglia `bande` -- due votanti su otto --
    era percio' **strutturalmente incapace** di dire «corto»: non per prudenza e non per misura,
    per un default preso in silenzio. Non sollevava niente e nessun test lo vedeva.

    Il modello e' l'unica eccezione, e dichiarata: la forma misurata del suo segnale e' a U, il
    segno non dice il verso, quindi vota +1 o tace (`.claude/docs/modello-swing.md` §5.1).
    """
    stati = confluence.stati_dei_votanti(candele_con_inversione, "15m", votanti=confluence.selezione())
    for nome, stato in stati.items():
        if nome == "modello":
            assert not (stato < 0).any(), "il votante a modello non vota mai corto, per disegno"
            continue
        assert (stato > 0).any(), f"{nome} non vota mai lungo su candele che salgono per meta' finestra"
        assert (stato < 0).any(), f"{nome} non vota mai corto: controlla il default di `allow_short`"


def test_il_macro_sconta_la_soglia_nel_verso_dell_operazione(candele_con_inversione):
    """Il difetto: `theta_base - theta_macro * macro` abbassava la soglia per **tutti e due** i
    versi quando il macro saliva.

    Sul lungo e' il disegno. Sul corto era il suo contrario esatto: con regime e struttura a -1,
    cioe' con il quadro macro che da' ragione al corto, la soglia saliva a 0,50 -- mentre il lungo
    con macro a +1 ne chiedeva 0,20. La barra si alzava proprio dove doveva abbassarsi.
    """
    r = confluence.evaluate(candele_con_inversione, "15m", theta_base=0.35, theta_macro=0.15)
    su = (r.regime > 0.5) & (r.struttura > 0.5)
    giu = (r.regime < -0.5) & (r.struttura < -0.5)
    assert su.any() and giu.any(), "le candele devono offrire tutti e due i quadri macro"

    # Il macro favorevole sconta la soglia del **proprio** verso, e alza quella dell'altro.
    assert r.soglia[su].mean() < 0.35 < r.soglia_corta[su].mean()
    assert r.soglia_corta[giu].mean() < 0.35 < r.soglia[giu].mean()

    # E lo sconto e' lo stesso numero: i due versi sono simmetrici rispetto a `theta_base`.
    assert np.allclose(r.soglia + r.soglia_corta, 2 * 0.35)


def test_con_macro_favorevole_il_corto_non_e_piu_difficile_del_lungo(candele_con_inversione):
    """La lettura operativa del test precedente, sugli ingressi che avvengono davvero.

    Con il difetto in casa gli ingressi corti erano piu' rari di quanto il disegno volesse, e la
    causa non era il punteggio: era la soglia. Qui si chiede che, a quadro macro ugualmente
    favorevole, la barra da superare sia la stessa nei due versi.
    """
    r = confluence.evaluate(candele_con_inversione, "15m", allow_short=True)
    barre = {e[0]: e[2] for e in r.eventi if e[2] != 0}
    posizioni = r.indice.get_indexer(list(barre))
    versi = list(barre.values())
    assert -1 in versi, "senza ingressi corti questo test non misura niente"

    for i, verso in zip(posizioni, versi):
        attiva = r.soglia[i] if verso > 0 else r.soglia_corta[i]
        assert abs(r.punteggio[i]) >= attiva - 1e-12, "un ingresso deve superare la soglia del proprio verso"
        # E la soglia superata e' quella scontata dal macro, non quella dell'altro verso.
        assert attiva == pytest.approx((0.35 - 0.15 * (r.regime[i] + r.struttura[i]) / 2 * verso))


# -------------------------------------------------------------------------------------------------
# L'orientamento dei voti: -1 e' lungo, +1 e' corto
# -------------------------------------------------------------------------------------------------


def test_i_voti_di_un_consenso_lungo_vanno_tutti_verso_meno_uno(candele):
    """La domanda da cui e' partita la revisione: perche' i votanti sembrano contraddirsi.

    Una delle risposte era che sulla stessa pagina convivevano **due assi opposti**: il riquadro
    *Voters* con +1 = lungo (convenzione di posizione) e il riquadro *Swing target* con -1 = zona
    d'acquisto (`ml/labeling.swing_leg_target`). Letti insieme sembravano darsi torto mentre
    dicevano la stessa cosa. Ora i voti stanno sull'asse dell'etichetta, dichiarato da
    `VERSO_DEL_VOTO`.
    """
    risultato = confluence.evaluate(candele, "15m")
    assert confluence.VERSO_DEL_VOTO == -1

    barre = [risultato.indice.get_loc(q) for q, _, obiettivo in risultato.eventi if obiettivo > 0]
    assert len(barre) > 10, "servono abbastanza ingressi lunghi perche' il confronto significhi qualcosa"

    for i in barre:
        assert risultato.punteggio[i] < 0, "un ingresso lungo avviene su un punteggio negativo"
        # E i votanti che lo sostengono sono negativi anche loro: e' l'accordo che si deve vedere.
        sostenitori = [v[i] for v in risultato.voti.values() if abs(v[i]) > 1e-9 and v[i] < 0]
        assert sostenitori, "nessun votante sostiene un ingresso lungo"

    # `convinzione` e' l'unico posto in cui i due assi si incontrano, e va in tutte e due le
    # direzioni: un punteggio negativo sostiene il lungo, uno positivo sostiene il corto.
    assert confluence.convinzione(-0.4, +1) == pytest.approx(0.4)
    assert confluence.convinzione(-0.4, -1) == pytest.approx(-0.4)
    assert confluence.convinzione(+0.4, -1) == pytest.approx(0.4)


def test_gli_eventi_emessi_sono_quelli_pinnati(candele, candele_con_inversione):
    """Il golden del comportamento: **quali** operazioni escono, su che barra e a che prezzo.

    Nato per un vincolo piu' stretto -- dimostrare che portare i voti sull'asse `-1 = lungo` era
    una rietichettatura e non un cambio di strategia -- e quel giro lo passo' con le firme
    identiche. Resta come golden generale, e va letto per quel che pinna: gli eventi emessi sono
    nella convenzione di **posizione** (+1 = lungo), che e' quella di `pnl.simulate_positions`, di
    `portfolio` e del bot live che piazza ordini veri.

    **Quando cade, la domanda e' se il cambio era voluto.** Una rietichettatura, una correzione di
    segno o una riscrittura che non cambia la strategia non devono spostarlo di un evento: li' si
    cerca il difetto, non si rigenera. Un cambio deliberato della forma del punteggio lo sposta per
    definizione, e allora si rigenera **dopo** aver guardato il diff -- che e' cio' che e' successo
    con il pavimento e l'azzeramento del voto: 81 ingressi lunghi diventarono 85 sul primo caso e
    72 diventarono 110 sul secondo, perche' i votanti hanno smesso di ammutolire mentre erano
    convinti. Il conto dei numeri qui sotto e' quel passaggio, non una rigenerazione automatica.
    """
    import hashlib
    import json

    atteso = {
        "base_long_only": (170, 85, 0, "1f8b9b36a490eff2"),
        "inversione_long_only": (220, 110, 0, "86ca177d5df050ab"),
        "inversione_short": (494, 110, 137, "53ea55974cacdcf6"),
    }
    casi = {
        "base_long_only": (candele, {}),
        "inversione_long_only": (candele_con_inversione, {}),
        "inversione_short": (candele_con_inversione, {"allow_short": True}),
    }

    for nome, (df, kw) in casi.items():
        eventi = confluence.evaluate(df, "15m", **kw).eventi
        crudi = [(str(t), round(float(p), 6), int(o)) for t, p, o in eventi]
        n, lunghi, corti, firma = atteso[nome]
        assert len(crudi) == n, f"{nome}: il numero di eventi e' cambiato"
        assert sum(1 for e in crudi if e[2] > 0) == lunghi, f"{nome}: gli ingressi lunghi sono cambiati"
        assert sum(1 for e in crudi if e[2] < 0) == corti, f"{nome}: gli ingressi corti sono cambiati"
        # Il conteggio puo' tornare mentre un'operazione si e' spostata di barra o di prezzo.
        assert (
            hashlib.sha256(json.dumps(crudi).encode()).hexdigest()[:16] == firma
        ), f"{nome}: stesso numero di operazioni, ma almeno una e' su una barra o un prezzo diverso"


# -------------------------------------------------------------------------------------------------
# Un piano che non si sa: NaN, non zero
# -------------------------------------------------------------------------------------------------


def _finestra_corta_per_il_regime():
    """Venti giorni a 15m: il piano di regime e' 1d e la sua media ne chiede cinquanta.

    E' la finestra con cui si guarda la pagina, non un caso limite costruito: il valore iniziale
    sono 240 ore e il cancello ne chiede 1.200.
    """
    return _candele(giorni=20, seme=5)


def test_un_piano_che_non_si_sa_vale_nan_e_non_zero():
    """Il difetto: `nan_to_num(..., nan=0.0)` faceva valere zero un piano ignoto.

    Zero qui e' una bugia, e non solo sul grafico: e' il valore che significa «prezzo esattamente
    sulla sua media», cioe' **macro neutro**. Un cancello chiuso per ignoranza si leggeva percio'
    come un cancello neutro, e la pagina mostrava una strategia che sembrava poter operare mentre
    nessun ingresso era possibile.
    """
    candele = _finestra_corta_per_il_regime()
    r = confluence.evaluate(candele, "15m")

    assert np.isnan(r.regime).all(), "il piano di regime non e' noto su questa finestra: deve essere NaN"
    assert np.isfinite(r.struttura).any(), "il piano di struttura invece si sa: non deve essere NaN ovunque"

    # Il cancello non cambia comportamento -- `NaN > 0` e' False -- ma adesso lo dichiara.
    assert r.ingressi == 0
    assert "not enough history" in r.perche_non_entra()


def test_un_piano_che_non_si_sa_non_vota_nella_soglia():
    """L'altra meta' del difetto, quella che non si vedeva affatto.

    La soglia era `theta_base - theta_macro * (regime + struttura) / 2`. Con il regime ignoto a
    zero, quella media **dimezzava** il contributo del piano noto: il piano che non si sa votava,
    e votava «neutro». Ora si astiene e la media e' sui piani noti.
    """
    candele = _finestra_corta_per_il_regime()
    r = confluence.evaluate(candele, "15m", theta_base=0.35, theta_macro=0.15)

    noto = np.isfinite(r.struttura)
    # Con un piano solo noto, la soglia e' scontata da quello **per intero**.
    atteso = 0.35 - 0.15 * r.struttura[noto]
    assert r.soglia[noto] == pytest.approx(atteso), "il piano noto deve scontare la soglia per intero"

    # E la versione col difetto -- la media che conta lo zero -- dava un numero diverso.
    diluito = 0.35 - 0.15 * (0.0 + r.struttura[noto]) / 2
    assert not np.allclose(r.soglia[noto], diluito), "la soglia e' ancora diluita dal piano ignoto"

    # Dove non si sa nessuno dei due piani lo sconto e' nullo, non NaN: `theta_base` e basta.
    if (~noto).any():
        assert r.soglia[~noto] == pytest.approx(0.35)


def test_un_piano_che_non_si_sa_non_si_disegna():
    """Il riquadro vuoto e' il segnale, e va dove l'utente guarda.

    La pagina diceva gia' «not enough history», ma nella sezione *Trades*: chi guarda il grafico
    vedeva una linea verde a 0,0 e la leggeva come un cancello neutro. Stessa regola dello stop a
    trailing (`_serie_stop`): una serie che non ha niente da disegnare non entra in legenda.
    """
    candele = _finestra_corta_per_il_regime()
    serie = panels._serie_piani(candele, ExtraCache(candele), {"INTERVALLO": "15m"})

    assert "regime" not in serie, "un piano ignoto non deve comparire come una riga piatta a zero"
    assert "struttura" in serie, "il piano noto invece si disegna"

    # Su una finestra lunga abbastanza tornano tutti e due.
    lunga = _candele(giorni=120, seme=5)
    completa = panels._serie_piani(lunga, ExtraCache(lunga), {"INTERVALLO": "15m"})
    assert {"regime", "struttura"} <= set(completa)


# -------------------------------------------------------------------------------------------------
# Il voto e' opinione per recenza, non recenza soltanto
# -------------------------------------------------------------------------------------------------


def test_nessun_votante_ammutolisce_mentre_tiene_la_posizione(candele):
    """Il difetto grosso, e quello che produceva «i segnali non arrivano».

    Il voto decadeva verso zero dall'ultimo scatto, indipendentemente dal fatto che il votante
    fosse ancora convinto. Misurato su 400 giorni sintetici: `zone_regime` in posizione sul 74,5%
    delle barre e **muto sul 91,3%** di quelle, `zone_struttura` 67,1%, `bande_conferma` 84,4%.
    Con sette votanti a 1/7 e una soglia di 0,35 servivano due voti e mezzo pieni e allineati, e
    quasi mai lo erano: il collegio era quasi sempre in minoranza di se stesso.
    """
    r = confluence.evaluate(candele, "15m")
    for nome, voto in r.voti.items():
        stato = np.asarray(r.stati[nome])
        in_posizione = stato != 0
        if not in_posizione.any():
            continue
        assert (voto[in_posizione] != 0).all(), f"{nome}: muto mentre tiene la posizione"

    # E il conto che conta: quanti votanti parlano su una barra media.
    accesi = np.mean([(np.abs(v) > 0).mean() for v in r.voti.values()]) * len(r.voti)
    assert accesi > len(r.voti) / 2, f"solo {accesi:.2f} votanti su {len(r.voti)} parlano su una barra media"


def test_nessun_votante_vota_dopo_essere_uscito(candele):
    """Il difetto opposto: i voti fantasma.

    `pullback` aveva un voto acceso a posizione gia' chiusa sul 49,0% delle barre. Le bande sono
    il caso che si vede a occhio: entrano sulla banda inferiore, escono su quella **opposta**, e
    il voto +1 sopravviveva all'uscita continuando a dire «lungo» dal massimo in giu'.
    """
    r = confluence.evaluate(candele, "15m")
    for nome, voto in r.voti.items():
        fuori = np.asarray(r.stati[nome]) == 0
        assert (voto[fuori] == 0).all(), f"{nome}: vota mentre e' fuori posizione"


def test_il_voto_e_lo_stato_per_la_recenza(candele):
    """La forma, in una riga: stesso segno dello stato, forza fra il pavimento e uno."""
    from cryptofarm.trading.voters import PAVIMENTO_DEL_VOTO

    r = confluence.evaluate(candele, "15m")
    for nome, voto in r.voti.items():
        stato = np.asarray(r.stati[nome])
        dentro = stato != 0
        if not dentro.any():
            continue
        # Il voto e' sull'asse dei voti, lo stato su quello delle posizioni: `convinzione` converte.
        assert (confluence.convinzione(voto[dentro], 1) * stato[dentro] > 0).all(), f"{nome}: segno discorde"
        forza = np.abs(voto[dentro])
        assert (forza >= PAVIMENTO_DEL_VOTO - 1e-9).all(), f"{nome}: sotto il pavimento"
        assert (forza <= 1.0 + 1e-9).all(), f"{nome}: sopra uno"


def test_un_collegio_fermo_non_apre_da_solo(candele):
    """Il vincolo che sceglie il pavimento, verificato sul motore e non solo sull'aritmetica.

    A pesi a somma 1 un collegio interamente d'accordo e interamente vecchio vale esattamente
    `pavimento`. Deve restare sotto la soglia **minima raggiungibile** -- `theta_base -
    theta_macro`, perche' un macro a favore sconta la soglia -- altrimenti la confluenza apre
    perche' tutti sono dentro, e smette di decidere *quando*.
    """
    from cryptofarm.trading.voters import PAVIMENTO_DEL_VOTO

    theta_base, theta_macro = 0.35, 0.15
    assert PAVIMENTO_DEL_VOTO < theta_base - theta_macro, "un collegio fermo supererebbe la soglia piu' bassa"

    r = confluence.evaluate(candele, "15m", theta_base=theta_base, theta_macro=theta_macro)
    assert r.soglia.min() >= theta_base - theta_macro - 1e-9, "la soglia non scende sotto il minimo previsto"


def test_l_emivita_di_un_voto_ha_un_tetto():
    """Senza, il piano di regime arriva a `6 x 96 = 576` barre di base: sei giorni di emivita.

    Il tetto e' in **minuti di calendario**, non in barre, perche' e' una durata: «un giorno» deve
    voler dire un giorno tanto a 15m quanto a 1h, mentre «96 barre» vuol dire due cose diverse.
    """
    senza = {p: confluence.emivita_in_barre(6.0, p, 15, None) for p in confluence.FATTORI}
    assert senza["regime"] == 576.0, "senza tetto il regime resta fuori scala"

    con = {p: confluence.emivita_in_barre(6.0, p, 15, confluence.TETTO_EMIVITA_MINUTI) for p in confluence.FATTORI}
    assert con["regime"] == 96.0, "un giorno a base 15m sono 96 barre"
    assert con["innesco"] == senza["innesco"], "i piani corti non vengono toccati dal tetto"
    assert con["conferma"] == senza["conferma"]

    # La stessa durata su una base diversa da' un numero di barre diverso, che e' il punto.
    a_un_ora = confluence.emivita_in_barre(6.0, "regime", 60, confluence.TETTO_EMIVITA_MINUTI)
    assert a_un_ora == 24.0, "un giorno a base 1h sono 24 barre"

    # E il tetto non puo' scendere sotto una barra, che sarebbe un'emivita non rappresentabile.
    assert confluence.emivita_in_barre(6.0, "regime", 1440, 60) == 1.0


def test_l_ampiezza_si_conta_sugli_stati_non_sui_voti(candele):
    """Punto (5): una famiglia «concorde» dev'essere una famiglia che **ha una posizione**.

    Con i voti, una famiglia contava come concorde finche' la coda del suo voto era sopra epsilon,
    anche a posizione chiusa da un pezzo -- meta' delle barre, per `pullback`. Oggi il voto e' zero
    fuori posizione e i due conteggi coincidono, ma il conteggio non deve **dipendere** da quella
    coincidenza: se un domani il voto cambia forma, l'ampiezza non deve cambiare di nascosto.
    """
    r = confluence.evaluate(candele, "15m")
    famiglie = {v.nome: v.famiglia for v in confluence.VOTANTI}

    # Un voto inventato, di segno opposto allo stato e acceso ovunque, non deve spostare nulla.
    bugiardi = {n: -np.ones(len(r.indice)) for n in r.stati}
    assert np.array_equal(
        confluence._famiglie_concordi(r.stati, famiglie, +1),
        confluence._famiglie_concordi(r.stati, famiglie, +1),
    )
    dagli_stati = confluence._famiglie_concordi(r.stati, famiglie, +1)
    assert np.array_equal(dagli_stati, r.concordi_lungo), "il motore conta l'ampiezza sugli stati"
    assert not np.array_equal(
        dagli_stati, confluence._famiglie_concordi(bugiardi, famiglie, +1)
    ), "il conteggio deve leggere davvero gli stati che riceve"


# -------------------------------------------------------------------------------------------------
# La modalita' a inversione: sempre a mercato, lunga o corta
# -------------------------------------------------------------------------------------------------


def test_in_inversione_non_si_e_mai_fuori_dal_mercato(candele_con_inversione):
    """La regola che definisce la modalita': dopo il primo attraversamento la posizione e' sempre
    +1 o -1, e un evento a zero non esiste."""
    r = confluence.evaluate(candele_con_inversione, "15m", modalita="inversione", allow_short=True)
    versi = [e[2] for e in r.eventi]
    assert versi, "senza operazioni questo test non misura niente"
    assert 0 not in versi, "in inversione non si va mai a flat: si ribalta"
    # E i versi si alternano: un ribaltamento porta sempre dalla parte opposta.
    assert all(a != b for a, b in zip(versi, versi[1:])), "due eventi di fila nello stesso verso"
    assert 1 in versi and -1 in versi, "devono esserci tutte e due le gambe"


def test_in_inversione_la_soglia_e_simmetrica(candele_con_inversione):
    """Lo sconto macro resta spento, e non per semplificare.

    Su una serie con deriva il piano di regime satura: misurato, media +1,000 e deviazione
    standard 0,000. Lo sconto non modula niente, sposta soltanto in permanenza la soglia lunga a
    0,208 e quella corta a 0,492, e su 17.280 barre il punteggio bastava per un corto in **zero**.
    In un sistema che deve stare sempre a mercato quella non e' un'opinione sul macro: e' una
    gamba amputata.
    """
    r = confluence.evaluate(
        candele_con_inversione, "15m", modalita="inversione", allow_short=True, theta_base=0.4, theta_macro=0.15
    )
    assert (r.soglia == 0.4).all(), "la soglia in inversione e' costante"
    assert np.array_equal(r.soglia, r.soglia_corta), "e uguale per i due versi"

    # E i due versi si misurano davvero contro la stessa barriera.
    for quando, _, verso in r.eventi:
        i = r.indice.get_loc(quando)
        if r.motivi.get(quando) == "score crossed the threshold":
            assert confluence.convinzione(r.punteggio[i], verso) >= r.soglia[i] - 1e-12


def test_in_inversione_si_tiene_fra_i_due_attraversamenti(candele_con_inversione):
    """Il cuore del disegno: fra le due soglie **non succede niente**.

    Nessuna isteresi, nessuna pazienza, nessun pavimento di barre. Senza stop, la posizione cambia
    se e solo se il punteggio ha attraversato la soglia opposta, e la durata di un'operazione e' la
    distanza fra due attraversamenti -- non un parametro.
    """
    r = confluence.evaluate(candele_con_inversione, "15m", modalita="inversione", allow_short=True, atr_multiplier=0.0)
    barre = [r.indice.get_loc(q) for q, _, _ in r.eventi]
    assert len(barre) > 2, "servono abbastanza ribaltamenti"

    for (inizio, fine), (_, _, verso) in zip(zip(barre, barre[1:]), r.eventi):
        # Fra un ribaltamento e il successivo il punteggio non deve mai aver toccato la soglia
        # opposta: se l'avesse fatto, si sarebbe ribaltato prima.
        opposto = confluence.convinzione(r.punteggio[inizio + 1 : fine], -verso)
        assert (opposto < r.soglia[inizio + 1 : fine]).all(), "ha tenuto una posizione oltre il segnale opposto"


def test_in_inversione_lo_stop_ribalta_invece_di_chiudere(candele_con_inversione):
    """Lo stop e' una regola di rischio e resta, ma non manda a flat: gira la posizione."""
    r = confluence.evaluate(candele_con_inversione, "15m", modalita="inversione", allow_short=True, atr_multiplier=3.0)
    da_stop = [q for q, m in r.motivi.items() if m == "trailing stop reversal"]
    assert da_stop, "con tre ATR lo stop deve scattare"
    for quando in da_stop:
        verso = next(e[2] for e in r.eventi if e[0] == quando)
        assert verso != 0, "lo stop ribalta, non chiude"


def test_lo_stop_si_spegne_con_un_moltiplicatore_non_positivo(candele_con_inversione):
    """E la misura per cui va spento: con lo stop acceso e' lui a decidere, non il punteggio.

    Su queste candele, con lo stop a 3 ATR le operazioni sono 1.167 e il **97%** dei ribaltamenti
    viene dallo stop, con durata mediana 4,2 ore; senza stop sono 12 e la mediana e' 171 ore. Con
    lo stop acceso la modalita' non e' «si compra e si tiene fino al segnale opposto»: e' una
    macchina a stop con il punteggio come comparsa.
    """
    comuni = dict(modalita="inversione", allow_short=True)
    con = confluence.evaluate(candele_con_inversione, "15m", atr_multiplier=3.0, **comuni)
    senza = confluence.evaluate(candele_con_inversione, "15m", atr_multiplier=0.0, **comuni)

    assert not any(m == "trailing stop reversal" for m in senza.motivi.values()), "lo stop deve essere spento"
    assert np.isnan(senza.stop).all(), "e non deve nemmeno disegnarsi"
    assert len(senza.eventi) < len(con.eventi) / 10, "senza stop le operazioni sono un ordine di grandezza meno"

    quota_stop = sum(1 for m in con.motivi.values() if m == "trailing stop reversal") / len(con.eventi)
    assert quota_stop > 0.9, "con lo stop acceso e' lo stop a decidere quasi tutto"


def test_una_modalita_sconosciuta_si_fa_notare(candele):
    with pytest.raises(ValueError, match="modalita sconosciuta"):
        confluence.evaluate(candele, "15m", modalita="inventata")


def test_la_modalita_a_cancello_resta_il_default(candele):
    """Le misure gia' scritte nei documenti valgono per quella: non deve cambiare da sotto."""
    assert confluence.evaluate(candele, "15m").modalita == "cancello"
    esplicita = confluence.evaluate(candele, "15m", modalita="cancello")
    assert [e[:3] for e in esplicita.eventi] == [e[:3] for e in confluence.evaluate(candele, "15m").eventi]


def test_lo_switch_della_pagina_arriva_al_motore(candele):
    """Un widget che non cambia niente e' peggio di non averlo: qui si verifica il collegamento."""
    valori = panels.valori_predefiniti()
    valori["INTERVALLO"] = "15m"

    a_cancello = panels.confluenza_di(candele, {**valori, "CONF_MODALITA": "cancello"})
    a_inversione = panels.confluenza_di(candele, {**valori, "CONF_MODALITA": "inversione"})
    assert a_cancello.modalita == "cancello" and a_inversione.modalita == "inversione"
    assert [e[:3] for e in a_cancello.eventi] != [e[:3] for e in a_inversione.eventi]


def test_in_inversione_la_pagina_accende_il_verso_corto_da_se(candele):
    """Senza gamba corta una macchina che non va mai a flat sarebbe lunga per sempre: non e' una
    scelta da lasciare a una casella che in una modalita' su due non ha senso spegnere."""
    valori = panels.valori_predefiniti()
    valori["INTERVALLO"] = "15m"
    assert not valori["CONF_ALLOW_SHORT"], "in modalita' a cancello il default resta solo lunghe"

    r = panels.confluenza_di(candele, {**valori, "CONF_MODALITA": "inversione"})
    versi = [e[2] for e in r.eventi]
    assert -1 in versi, "la pagina deve accendere il verso corto in inversione"
