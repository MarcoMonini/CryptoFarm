"""Il registro di `trading/panels.py` deve corrispondere alla realta', non alle intenzioni.

Un registro scritto a mano si stacca dal codice in silenzio: basta rinominare una colonna, o
aggiungere una voce al menu senza registrarla, e la pagina mostra un indicatore vuoto o ne nasconde
uno che serve. Nessuno se ne accorge guardando, perche' una traccia mancante non e' un errore --
e' solo una linea che non c'e'.

Questi test chiudono le quattro strade per cui puo' succedere: una voce di menu senza registro, un
parametro che in `config` non esiste, una serie che nel frame non c'e', un pannello dichiarato che
poi non produce niente.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import config, indicators, panels, simulator, strategies_ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions, simulate_trading_with_commisions


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Candele sintetiche con tutte le colonne che `add_technical_indicator` produce."""
    generatore = np.random.default_rng(11)
    chiusure = 30_000 * np.exp(np.cumsum(generatore.normal(0.0003, 0.008, 600)))
    ampiezza = np.abs(generatore.normal(0, 0.004, 600)) * chiusure
    aperture = np.concatenate([[chiusure[0]], chiusure[:-1]])
    grezzo = pd.DataFrame(
        {
            "Open": aperture,
            "High": np.maximum(aperture, chiusure) + ampiezza,
            "Low": np.minimum(aperture, chiusure) - ampiezza,
            "Close": chiusure,
            "Volume": generatore.uniform(10, 900, 600),
        },
        index=pd.date_range("2024-01-01", periods=600, freq="1h", name="Open time"),
    )
    return indicators.add_technical_indicator(grezzo)


@pytest.fixture(scope="module")
def frame_laterale() -> pd.DataFrame:
    """Mercato che oscilla senza andare da nessuna parte: ADX basso.

    Serve alla strategia di ritorno alla media, che entra solo quando l'ADX dice che un trend non
    c'e'. Sulla passeggiata casuale in trend non produce operazioni, e il confronto fra i due
    motori resterebbe non verificato proprio dove serve.
    """
    generatore = np.random.default_rng(3)
    passi = np.arange(1_200)
    # Il rumore non e' decorativo: una sinusoide liscia ha ADX mediano 51, perche' per l'ADX
    # un'oscillazione regolare e' un trend fortissimo. Con il rumore scende a 22 e la strategia
    # trova le barre di intervallo che cerca.
    chiusure = 30_000 + 900 * np.sin(passi / 9.0) + generatore.normal(0, 250, passi.size)
    aperture = np.concatenate([[chiusure[0]], chiusure[:-1]])
    grezzo = pd.DataFrame(
        {
            "Open": aperture,
            "High": np.maximum(aperture, chiusure) + 60,
            "Low": np.minimum(aperture, chiusure) - 60,
            "Close": chiusure,
            "Volume": np.full(passi.size, 500.0),
        },
        index=pd.date_range("2024-01-01", periods=passi.size, freq="1h", name="Open time"),
    )
    return indicators.add_technical_indicator(grezzo)


def test_il_menu_e_il_registro_coincidono() -> None:
    """Ogni voce di `config.STRATEGIES` ha una riga nel registro, e viceversa.

    Le due liste vivono in file diversi: senza questo controllo una voce aggiunta al menu resta
    senza indicatori e senza parametri, e la pagina la mostra spoglia invece di rompersi.
    """
    menu = {voce for voce in config.STRATEGIES if voce != panels.VUOTA}
    assert menu == set(panels.STRATEGIE)


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_ogni_parametro_di_un_indicatore_esiste_in_config(chiave: str) -> None:
    for nome in panels.INDICATORI[chiave].parametri:
        assert isinstance(getattr(config, nome, None), config.Param), f"{chiave}: manca config.{nome}"


@pytest.mark.parametrize("nome", sorted(panels.STRATEGIE))
def test_ogni_strategia_riferisce_indicatori_e_parametri_esistenti(nome: str) -> None:
    strategia = panels.STRATEGIE[nome]
    for chiave in strategia.indicatori:
        assert chiave in panels.INDICATORI, f"{nome}: indicatore sconosciuto {chiave}"
    for parametro in strategia.parametri:
        assert isinstance(getattr(config, parametro, None), config.Param), f"{nome}: manca config.{parametro}"


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_le_serie_dichiarate_esistono_davvero_nel_frame(chiave: str, frame: pd.DataFrame) -> None:
    """E' il controllo che intercetta un nome di colonna sbagliato, come la vecchia `EMA200`."""
    indicatore = panels.INDICATORI[chiave]
    prodotte = indicatore.serie(frame, ExtraCache(frame), panels.valori_predefiniti())
    for traccia in indicatore.tracce:
        assert traccia.serie in prodotte, f"{chiave}: la traccia '{traccia.nome}' non ha la serie {traccia.serie}"
        assert prodotte[traccia.serie].notna().any(), f"{chiave}: la serie {traccia.serie} e' tutta vuota"


@pytest.mark.parametrize("chiave", sorted(panels.INDICATORI))
def test_un_indicatore_con_pannello_disegna_qualcosa(chiave: str) -> None:
    """Un pannello senza tracce sarebbe un riquadro vuoto che occupa spazio nel grafico."""
    indicatore = panels.INDICATORI[chiave]
    if indicatore.pannello is not None:
        assert indicatore.tracce, f"{chiave}: dichiara un pannello ma non disegna nulla"


def test_senza_strategia_si_mostra_tutto() -> None:
    mostrati = panels.indicatori_di(panels.VUOTA)
    assert set(mostrati) == set(panels.INDICATORI) - set(panels.PANORAMICA_ESCLUSI)
    assert panels.indicatori_di("Trend Zones") == ("medie_trend",)


def test_la_panoramica_non_ha_due_voci_di_legenda_uguali() -> None:
    """Due etichette identiche su linee diverse si leggono come un errore, non come ricchezza."""
    nomi = [
        traccia.nome for chiave in panels.indicatori_di(panels.VUOTA) for traccia in panels.INDICATORI[chiave].tracce
    ]
    doppi = {nome for nome in nomi if nomi.count(nome) > 1}
    assert not doppi, f"nomi ripetuti nella panoramica: {sorted(doppi)}"


def test_i_parametri_non_si_ripetono_e_includono_le_dipendenze() -> None:
    """`ATR Bands` non disegna medie, ma le sue bande dipendono da EMA Short via KAMA."""
    parametri = panels.parametri_di("ATR Bands")
    assert len(parametri) == len(set(parametri))
    assert "EMA_SHORT" in parametri
    assert "RSI_SHORT" not in parametri


def test_i_colori_degli_indicatori_non_sono_quelli_di_stato() -> None:
    """Verde e rosso dicono rialzo e ribasso: un indicatore che li usa si legge come un segnale."""
    stato = {panels.RIALZO, panels.RIBASSO}
    for chiave, indicatore in panels.INDICATORI.items():
        for traccia in indicatore.tracce:
            assert traccia.colore not in stato, f"{chiave}: '{traccia.nome}' usa un colore di stato"


# -------------------------------------------------------------------------------------------------
# L'adattatore fra i due motori
# -------------------------------------------------------------------------------------------------

NUOVE = ("Donchian Breakout", "Squeeze Breakout", "Ichimoku Trend")


@pytest.mark.parametrize("nome", NUOVE)
def test_le_strategie_nuove_nella_pagina_non_vanno_mai_corte(nome: str, frame: pd.DataFrame) -> None:
    """La pagina le chiama con `allow_short=False`: il verso corto e' misurato in perdita."""
    eventi = strategies_ls.STRATEGIES[_funzione(nome)](frame, ExtraCache(frame), allow_short=False)
    assert all(obiettivo >= 0 for _, _, obiettivo in eventi)


@pytest.mark.parametrize("nome", NUOVE)
def test_la_conversione_in_due_liste_conserva_le_operazioni(
    nome: str, frame: pd.DataFrame, frame_laterale: pd.DataFrame
) -> None:
    """Le stesse operazioni, agli stessi prezzi, dei due motori.

    Il formato a due liste accoppia per indice, quindi la conversione regge solo se gli eventi si
    alternano ingresso/uscita -- vero senza il verso corto, falso con. Il controllo e' diretto:
    stesse operazioni di `simulate_positions`, e a commissioni nulle anche lo stesso capitale.
    """
    valori = panels.valori_predefiniti()
    for candele in (frame, frame_laterale):
        cache = ExtraCache(candele)
        eventi = strategies_ls.STRATEGIES[_funzione(nome)](candele, cache, allow_short=False)
        if len(eventi) >= 4:
            break
    else:
        pytest.fail(f"{nome} non produce operazioni in nessuno dei due scenari: il confronto sarebbe vuoto")

    acquisti, uscite = panels.STRATEGIE[nome].esegui(candele, cache, valori)
    a_posizioni = simulate_positions(eventi, wallet=100, fee_percent=0.0, carry_daily_percent=0.0)
    a_segnali = simulate_trading_with_commisions(acquisti, uscite, wallet=100, fee_percent=0.0)

    assert [(o["Buy_Time"], o["Sell_Time"]) for o in a_segnali] == [
        (o["Buy_Time"], o["Sell_Time"]) for o in a_posizioni
    ]
    assert [(o["Buy_Price"], o["Sell_Price"]) for o in a_segnali] == [
        (o["Buy_Price"], o["Sell_Price"]) for o in a_posizioni
    ]
    assert a_segnali[-1]["Wallet_After"] == pytest.approx(a_posizioni[-1]["Wallet_After"], rel=1e-9)


def _funzione(voce_di_menu: str) -> str:
    """Dalla voce del menu al nome in `strategies_ls.STRATEGIES`."""
    return {
        "Donchian Breakout": "donchian_breakout",
        "Squeeze Breakout": "squeeze_breakout",
        "Trend Pullback": "trend_pullback",
        "Ichimoku Trend": "ichimoku_trend",
        "Band Reversion": "band_reversion_gated",
    }[voce_di_menu]


@pytest.mark.parametrize("nome", sorted(panels.ETICHETTE))
def test_ogni_etichetta_corrisponde_a_un_parametro(nome: str) -> None:
    assert isinstance(getattr(config, nome, None), config.Param), f"etichetta orfana: {nome}"


@pytest.mark.parametrize("strategia", [panels.VUOTA, *sorted(panels.STRATEGIE)])
def test_ogni_parametro_mostrato_ha_un_etichetta(strategia: str) -> None:
    """Un parametro senza etichetta comparirebbe con il nome della costante."""
    for nome in panels.parametri_di(strategia):
        assert nome in panels.ETICHETTE, f"{strategia}: manca l'etichetta di {nome}"


@pytest.mark.parametrize("strategia", [panels.VUOTA, *sorted(panels.STRATEGIE)])
def test_i_gruppi_coprono_i_parametri_senza_ripeterli(strategia: str) -> None:
    """Due widget con la stessa chiave sono un errore di Streamlit, non un doppione innocuo."""
    dai_gruppi = [nome for _, dentro in panels.gruppi_di(strategia) for nome in dentro]
    assert len(dai_gruppi) == len(set(dai_gruppi)), f"{strategia}: parametro ripetuto"
    assert set(dai_gruppi) == set(panels.parametri_di(strategia))


def test_ema_short_compare_una_volta_sola_anche_se_due_indicatori_la_usano() -> None:
    """`Close Buy/Sell Limits` usa bande (che dipendono da EMA Short via KAMA) e RSI."""
    gruppi = dict(panels.gruppi_di("Close Buy/Sell Limits"))
    quante = sum(dentro.count("EMA_SHORT") for dentro in gruppi.values())
    assert quante == 1


# -------------------------------------------------------------------------------------------------
# La pagina, per ogni voce del menu
# -------------------------------------------------------------------------------------------------
# `trading_analysis` non aveva test: il golden master copre i moduli in cui e' stata spezzata, non
# lei. Questo la esegue su candele sintetiche per ogni voce, grafico compreso, e intercetta la
# classe di guasto piu' probabile dopo il passaggio al registro -- una strategia che non trova un
# parametro, o una traccia che chiede una colonna che non c'e'.


@pytest.mark.parametrize("strategia", [panels.VUOTA, *sorted(panels.STRATEGIE)])
def test_la_pagina_gira_per_ogni_voce_del_menu(strategia: str, frame_laterale: pd.DataFrame) -> None:
    if strategia == "AI Model":
        pytest.skip("richiede un modello addestrato, che nell'ambiente dei test non c'e'")
    from cryptofarm.trading.simulator import trading_analysis

    grezzo = frame_laterale[["Open", "High", "Low", "Close", "Volume"]]
    figura, operazioni, _ = trading_analysis(
        asset="TEST",
        interval="1h",
        wallet=100.0,
        valori={},
        strategia=strategia,
        show=True,
        market_data=grezzo,
    )
    assert figura is not None
    assert list(operazioni.columns) if not operazioni.empty else True


@pytest.mark.parametrize("strategia", sorted(panels.STRATEGIE))
def test_il_grafico_non_mostra_indicatori_che_la_strategia_non_usa(
    strategia: str, frame_laterale: pd.DataFrame
) -> None:
    """E' la richiesta, verificata sulla figura e non a occhio.

    Non basta che le tracce giuste ci siano: devono mancare quelle degli altri indicatori. Il modo
    piu' facile di sbagliare questa pagina e' lasciare una traccia fuori dal registro, dove nessuno
    la nota perche' una linea in piu' sembra solo un grafico ricco.
    """
    if strategia == "AI Model":
        pytest.skip("richiede un modello addestrato")
    from cryptofarm.trading.simulator import trading_analysis

    figura, _, _ = trading_analysis(
        asset="TEST",
        interval="1h",
        wallet=100.0,
        valori={},
        strategia=strategia,
        show=True,
        market_data=frame_laterale[["Open", "High", "Low", "Close", "Volume"]],
    )
    disegnate = {traccia.name for traccia in figura.data}
    usati = set(panels.indicatori_di(strategia))

    for chiave, indicatore in panels.INDICATORI.items():
        if chiave in usati:
            continue
        for traccia in indicatore.tracce:
            # Un nome puo' essere condiviso da due indicatori (le bande si chiamano cosi' in due
            # famiglie): si conta come intruso solo se nessun indicatore usato lo produce.
            if any(t.nome == traccia.nome for c in usati for t in panels.INDICATORI[c].tracce):
                continue
            assert traccia.nome not in disegnate, f"{strategia}: '{traccia.nome}' viene da {chiave}, che non usa"


@pytest.mark.parametrize("strategia", [panels.VUOTA, *sorted(panels.STRATEGIE)])
def test_il_numero_di_riquadri_segue_gli_oscillatori_usati(strategia: str, frame_laterale: pd.DataFrame) -> None:
    if strategia == "AI Model":
        pytest.skip("richiede un modello addestrato")
    from cryptofarm.trading.simulator import trading_analysis

    figura, _, _ = trading_analysis(
        asset="TEST",
        interval="1h",
        wallet=100.0,
        valori={},
        strategia=strategia,
        show=True,
        market_data=frame_laterale[["Open", "High", "Low", "Close", "Volume"]],
    )
    attesi = 1 + len(panels.pannelli_di(strategia))
    assi = {traccia.yaxis or "y" for traccia in figura.data}
    assert len(assi) == attesi, f"{strategia}: {len(assi)} riquadri invece di {attesi}"


def test_gli_overlay_non_usano_l_acquamarina() -> None:
    """Sopra le candele l'acquamarina si confonde con il corpo rialzista.

    E' un vincolo che si vede solo guardando la figura renderizzata, non leggendo il codice: il
    validatore approva la coppia, perche' non sa che una delle due tinte e' un corpo pieno che
    occupa meta' del riquadro. Gli overlay restano sulle due famiglie, blu e arancio.
    """
    for chiave, indicatore in panels.INDICATORI.items():
        if indicatore.pannello is not None:
            continue
        for traccia in indicatore.tracce:
            ammessi = {*panels.FAMIGLIA_BLU, *panels.FAMIGLIA_ARANCIO}
            assert traccia.colore in ammessi, f"{chiave}: '{traccia.nome}' non e' blu ne' arancio sopra le candele"


# -------------------------------------------------------------------------------------------------
# La lingua dell'interfaccia
# -------------------------------------------------------------------------------------------------
# Il codice, i commenti e le docstring restano in italiano: e' la lingua di lavoro del progetto.
# Cio' che l'utente legge no. Senza un controllo la regola dura fino al widget successivo, perche'
# scrivere l'etichetta nella lingua in cui si sta pensando e' la cosa piu' naturale del mondo.

# Parole che in inglese non esistono: bastano a intercettare un'etichetta rimasta in italiano,
# senza pretendere di riconoscere una lingua.
SPIE = re.compile(
    r"\b(della|dello|delle|degli|nessun\w*|quando|viene|perche|soltanto|oppure|invece|questo|"
    r"questa|sono|dalla|nella|senza|solo|con|per|il|lo|gli|una|un'|non|piu')\b|[àèéìòù]",
    re.IGNORECASE,
)


def _testi_del_registro() -> list[tuple[str, str]]:
    """Tutto cio' che il registro fa arrivare sotto gli occhi di chi usa la pagina."""
    testi = [("SOGLIE", panels.SOGLIE)]
    testi += [(f"ETICHETTE[{k}]", v) for k, v in panels.ETICHETTE.items()]
    for chiave, indicatore in panels.INDICATORI.items():
        testi.append((f"{chiave}.etichetta", indicatore.etichetta))
        if indicatore.pannello:
            testi.append((f"{chiave}.pannello", indicatore.pannello))
        testi += [(f"{chiave}.{t.serie}", t.nome) for t in indicatore.tracce]
    for nome, strategia in panels.STRATEGIE.items():
        if strategia.note:
            testi.append((f"{nome}.note", strategia.note))
    return testi


@pytest.mark.parametrize("dove, testo", _testi_del_registro())
def test_il_registro_parla_inglese(dove: str, testo: str) -> None:
    trovata = SPIE.search(testo)
    assert not trovata, f"{dove}: «{testo}» sembra italiano ({trovata.group()})"


def test_i_widget_della_pagina_parlano_inglese() -> None:
    """Le stringhe passate alle chiamate `st.*` e `go.*`, prese dall'albero sintattico.

    Commenti e docstring non sono argomenti di chiamata, quindi restano fuori da soli: il
    controllo guarda solo cio' che Streamlit disegna.
    """
    sorgente = Path(inspect.getfile(simulator)).read_text()
    problemi = []
    for nodo in ast.walk(ast.parse(sorgente)):
        if not isinstance(nodo, ast.Call):
            continue
        radice = nodo.func
        while isinstance(radice, ast.Attribute):
            radice = radice.value
        # `st.*` disegna i widget, `go.*` costruisce le tracce: i nomi in legenda venivano di
        # li', e la prima versione di questo test non li guardava. Se ne sono accorti gli occhi
        # sul grafico renderizzato, non la suite.
        if not (isinstance(radice, ast.Name) and radice.id in {"st", "go"}):
            continue
        pezzi = list(nodo.args) + [k.value for k in nodo.keywords]
        for pezzo in pezzi:
            for costante in ast.walk(pezzo):
                if isinstance(costante, ast.Constant) and isinstance(costante.value, str):
                    trovata = SPIE.search(costante.value)
                    if trovata:
                        problemi.append(f"riga {costante.lineno}: «{costante.value}» ({trovata.group()})")
    assert not problemi, "stringhe non inglesi nei widget:\n  " + "\n  ".join(problemi)


def test_dentro_un_indicatore_le_linee_si_distinguono() -> None:
    """Tre serie con lo stesso colore, spessore e tratteggio non si distinguono.

    E' il difetto che si vedeva guardando il grafico: le tre EMA erano tre linee blu identiche
    salvo il tratteggio, che a schermo non si legge. Due tracce possono condividere lo stile solo
    se sono una coppia -- una banda superiore e una inferiore si leggono come inviluppo, ed e'
    giusto che siano uguali -- ma tre no.
    """
    for chiave, indicatore in panels.INDICATORI.items():
        stili = [(t.colore, t.tratteggio, t.larghezza) for t in indicatore.tracce]
        for stile in set(stili):
            quante = stili.count(stile)
            assert quante <= 2, f"{chiave}: {quante} tracce con lo stesso stile {stile}"


def test_le_rampe_ordinali_sono_monotone() -> None:
    """Chiarezza crescente e spessore crescente devono andare nello stesso verso.

    Se una famiglia ordinata avesse la linea piu' chiara anche piu' spessa in mezzo alla serie,
    i due canali direbbero cose diverse e l'ordine smetterebbe di leggersi.
    """
    for chiave in ("medie", "rsi"):
        tracce = panels.INDICATORI[chiave].tracce
        posizioni = [panels.FAMIGLIA_BLU.index(t.colore) for t in tracce]
        larghezze = [t.larghezza for t in tracce]
        assert posizioni == sorted(posizioni), f"{chiave}: la rampa di chiarezza non e' monotona"
        assert larghezze == sorted(larghezze), f"{chiave}: gli spessori non seguono la rampa"
