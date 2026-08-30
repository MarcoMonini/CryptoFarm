"""Il target di prossimita' agli estremi: forma, saturazione e confine col futuro."""

import inspect

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.labeling import TIME_WEIGHT, swing_leg_target, swing_pivots, swing_target


def _rango_centrato_a_forza_bruta(close: np.ndarray, window: int) -> np.ndarray:
    """Definizione ingenua, O(n x window): il rango medio della barra centrale nella sua finestra."""
    fuori = np.full(len(close), np.nan)
    for i in range(window, len(close) - window):
        finestra = close[i - window : i + window + 1]
        sotto = (finestra < close[i]).sum()
        pari = (finestra == close[i]).sum()
        rango = sotto + (pari + 1) / 2  # rango medio sui pari merito, come pandas
        fuori[i] = (rango - 1.0) / window - 1.0
    return fuori


def test_coincide_con_la_definizione_ingenua():
    close = np.random.default_rng(0).normal(size=400).cumsum() + 100.0
    atteso = _rango_centrato_a_forza_bruta(close, 20)
    ottenuto = swing_target(close, 20).to_numpy()
    valide = ~np.isnan(atteso)
    assert np.allclose(ottenuto[valide], atteso[valide])


def test_massimo_e_minimo_saturano():
    close = np.concatenate([np.arange(30.0), np.arange(28.0, -1.0, -1.0)])  # V rovesciata
    target = swing_target(close, 10).to_numpy()
    assert target[29] == pytest.approx(1.0)  # la punta
    close_v = -close
    assert swing_target(close_v, 10).to_numpy()[29] == pytest.approx(-1.0)


def test_dentro_una_tendenza_regolare_vale_zero():
    """La proprieta' per cui questo target esiste: la salita non e' un massimo."""
    close = np.arange(200.0)
    target = swing_target(close, 20).to_numpy()
    centro = target[50:150]
    assert np.allclose(centro, 0.0), f"in tendenza il target dovrebbe stare a 0, sta a {centro[:5]}"


def test_le_ultime_barre_non_sono_etichettabili():
    close = np.random.default_rng(1).normal(size=100).cumsum() + 50.0
    target = swing_target(close, 12)
    assert target.iloc[-12:].isna().all()
    assert target.iloc[:12].isna().all()
    assert target.iloc[12:-12].notna().all()


def test_ritardare_di_una_barra_non_basta():
    """Il target ritardato di 1 conosce ancora quasi tutto il futuro che il target di oggi conosce.

    E' la trappola che il docstring segnala: solo un ritardo > window scollega le due finestre.
    """
    close = pd.Series(np.random.default_rng(2).normal(size=3000).cumsum() + 100.0)
    target = swing_target(close, 24)
    assert target.corr(target.shift(1)) > 0.75  # misurato 0,80: sa quasi tutto del futuro
    assert abs(target.corr(target.shift(25))) < 0.4  # misurato -0,22: informazione, non copia


# -------------------------------------------------------------------------------------------------
# L'etichetta a gambe: da un estremo al successivo
# -------------------------------------------------------------------------------------------------


def _zigzag(periodo: int = 200, ampiezza: float = 10.0, barre: int = 400) -> np.ndarray:
    """Minimo a 50, massimo a 150, minimo a 250, ...: i vertici si sanno in anticipo."""
    passi = np.arange(barre)
    return 100 + ampiezza * np.sin(2 * np.pi * (passi - periodo // 2) / periodo)


def test_gli_estremi_si_alternano_sempre():
    """Due massimi di fila senza un minimo in mezzo non sono i vertici di nessuna gamba."""
    close = np.random.default_rng(5).normal(size=3000).cumsum() + 500.0
    indici, versi = swing_pivots(close, 50)
    assert len(indici) > 5
    assert (np.diff(versi) != 0).all(), "due estremi consecutivi con lo stesso verso"
    assert (np.diff(indici) > 0).all()


def test_nessun_estremo_nelle_barre_che_il_futuro_non_ha_confermato():
    """`argrelextrema` ai bordi confronta con indici ritagliati: sarebbe l'unico look-ahead vero."""
    close = np.random.default_rng(6).normal(size=800).cumsum() + 300.0
    indici, _ = swing_pivots(close, 40)
    assert indici.min() >= 40
    assert indici.max() < len(close) - 40


def test_scorre_da_meno_uno_a_piu_uno_fra_due_vertici():
    target = swing_leg_target(_zigzag(), 50).to_numpy()
    indici, versi = swing_pivots(_zigzag(), 50)
    assert list(versi[:2]) == [-1, 1]
    assert target[indici[0]] == pytest.approx(-1.0, abs=0.02)
    assert target[indici[1]] == pytest.approx(1.0, abs=0.02)
    gamba = target[indici[0] : indici[1] + 1]
    assert (np.diff(gamba) >= -1e-9).all(), "dentro la gamba il target deve salire senza tornare"


def test_il_tempo_conta_quanto_il_prezzo():
    """La differenza dal rango: a prezzo fermo il target si muove lo stesso verso l'estremo.

    Un prezzo che resta a meta' gamba mentre le barre scorrono si sta avvicinando al vertice, e
    l'etichetta lo dice. Con `peso_tempo=0` -- cioe' solo prezzo -- non lo direbbe.
    """
    close = np.concatenate(
        [
            np.linspace(110.0, 100.0, 60),  # discesa fino al minimo
            np.linspace(100.1, 110.0, 60),  # salita
            110.0 + np.arange(60) * 0.001,  # il prezzo si ferma in cima, il tempo no
            np.linspace(109.9, 95.0, 60),  # e poi scende
        ]
    )
    misto = swing_leg_target(close, 20, peso_tempo=0.5).to_numpy()
    solo_prezzo = swing_leg_target(close, 20, peso_tempo=0.0).to_numpy()
    fermo = slice(125, 175)  # il tratto in cui il prezzo non si muove piu'
    assert np.ptp(solo_prezzo[fermo]) < 0.02, "a peso zero il target sta fermo col prezzo"
    assert np.diff(misto[fermo]).min() > 0, "col tempo dentro, il target avanza anche a prezzo fermo"
    assert np.ptp(misto[fermo]) > 0.2, "e avanza di un tratto che si vede"


def test_una_gamba_dentro_il_rumore_non_satura():
    """L'ampiezza conta: la stessa forma vale +-1 se e' una gamba vera, molto meno se e' rumore."""
    generatore = np.random.default_rng(7)
    forte = swing_leg_target(_zigzag(ampiezza=10.0), 50).to_numpy()
    debole = swing_leg_target(_zigzag(ampiezza=0.3) + generatore.normal(0, 0.3, 400), 50).to_numpy()
    assert np.nanmax(np.abs(forte)) > 0.9
    assert np.nanmax(np.abs(debole)) < 0.7


def test_dentro_una_tendenza_regolare_non_ci_sono_vertici():
    """La proprieta' che l'etichetta a rango aveva, e che questa deve conservare."""
    indici, _ = swing_pivots(np.arange(500.0), 50)
    assert len(indici) == 0
    assert swing_leg_target(np.arange(500.0), 50).isna().all()


def test_lo_smoothing_temporale_di_partenza_e_quello_documentato():
    """0,7, e non un valore qualunque: e' cio' con cui `swing_model` viene addestrato.

    Il default vive in `labeling.TIME_WEIGHT` e la firma lo eredita. Se qualcuno riscrive un
    numero nella firma, i tre usi dell'etichetta -- addestramento, grafico, misura -- ricominciano
    a divergere in silenzio, che e' il difetto che questo file esiste per non far tornare.
    """
    import inspect

    assert TIME_WEIGHT == 0.7
    assert inspect.signature(swing_leg_target).parameters["peso_tempo"].default == TIME_WEIGHT


def test_la_pagina_parte_dallo_smoothing_con_cui_si_addestra():
    """`config` non importa `ml` di proposito, quindi il valore e' ricopiato: qui si tiene fermo.

    Un modello addestrato a 0,7 e un grafico disegnato a 0,5 mostrano due curve diverse chiamandole
    entrambe «l'etichetta». E' successo fino al 2026-08-30, in senso opposto: il grafico aveva lo
    smoothing e l'addestramento no.
    """
    from cryptofarm.trading import config

    assert config.SWING_TARGET_TEMPO.value == TIME_WEIGHT
    assert config.SWING_TARGET_TEMPO.minimum <= TIME_WEIGHT <= config.SWING_TARGET_TEMPO.maximum


def test_il_trainer_etichetta_con_le_gambe_e_non_col_rango():
    """La regressione da cui nasce tutto questo: il trainer imparava `swing_target`.

    Il rango centrato non ha smoothing temporale per costruzione -- e' una posizione dentro una
    finestra fissa -- quindi finche' il trainer usava quello il parametro non poteva mordere,
    per quanto lo si muovesse nella pagina.
    """
    from cryptofarm.ml import swing_trainer

    assert swing_trainer.PESO_TEMPO == TIME_WEIGHT
    sorgente = inspect.getsource(swing_trainer.campione_simbolo)
    assert 'frame["Target"] = swing_leg_target(' in sorgente
    assert 'frame["Target"] = swing_target(' not in sorgente


def test_l_embargo_copre_l_orizzonte_variabile_dell_etichetta():
    """L'etichetta guarda avanti fino all'estremo successivo, che dista piu' di `w` barre.

    Un embargo di `w` sole -- quello che bastava al rango -- lascia le ultime righe dello stima con
    un target che ha gia' letto dentro la verifica, e il numero fuori campione esce gonfio.
    """
    from cryptofarm.ml import swing_trainer

    assert swing_trainer.EMBARGO_FINESTRE >= 2
    indice = pd.date_range("2023-01-01", periods=4000, freq="5min")
    dati = pd.DataFrame({"x": np.arange(len(indice))}, index=indice)
    taglio = indice[3000]
    dentro, fuori = swing_trainer.taglia(dati, str(taglio), w=100)
    distanza = (fuori.index[0] - dentro.index[-1]) / pd.Timedelta(minutes=5)
    assert distanza >= swing_trainer.EMBARGO_FINESTRE * 100
