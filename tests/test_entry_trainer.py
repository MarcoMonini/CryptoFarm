"""Il modello d'ingresso: le regole su cui poggiano i numeri fuori campione."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from cryptofarm.ml.entry_trainer import (
    COMMISSIONE,
    controllo_casuale,
    operazioni,
    rendimento_futuro,
    separa,
)


def _passeggiata(n: int = 3_000, seme: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seme)
    return np.exp(np.cumsum(rng.normal(scale=0.001, size=n))) * 100.0


def test_the_tail_has_no_future_and_says_so():
    close = _passeggiata(500)

    avanti = rendimento_futuro(close, 150)

    # L'alternativa silenziosa sarebbe riportare avanti l'ultimo prezzo, che addestrerebbe il
    # modello su un rendimento nullo inventato proprio dove il mercato non e' ancora arrivato.
    assert np.isnan(avanti[-150:]).all()
    assert np.isfinite(avanti[:-150]).all()
    assert avanti[0] == pytest.approx(np.log(close[150] / close[0]))


def test_two_signals_inside_one_position_are_one_trade():
    close = _passeggiata()

    # Contare entrambi misurerebbe un capitale che non si ha, e gonfierebbe allo stesso modo il
    # controllo casuale, che passa da qui.
    assert len(operazioni(close, [10, 11, 12, 13], 150)) == 1
    assert len(operazioni(close, [10, 160, 320], 150)) == 3
    assert len(operazioni(close, [10, 159, 161], 150)) == 2


def test_an_entry_too_close_to_the_end_is_not_a_trade():
    close = _passeggiata(500)

    # Il suo esito non esiste ancora: includerlo con l'ultimo prezzo disponibile accorcerebbe la
    # tenuta solo sulle ultime operazioni, cioe' proprio dove il campione e' piu' sottile.
    assert operazioni(close, [len(close) - 10], 150) == []
    assert len(operazioni(close, [len(close) - 200], 150)) == 1


def test_the_fee_is_paid_on_every_trade():
    fermo = np.full(500, 100.0)

    esiti = operazioni(fermo, [0, 200], 150)

    assert len(esiti) == 2
    assert all(e == pytest.approx(-COMMISSIONE) for e in esiti)


def test_the_random_control_matches_the_model_trade_for_trade():
    close = _passeggiata(5_000)
    campioni = {"X": {"close": close, "posizioni": np.arange(0, 4_000, 12)}}

    esito = controllo_casuale(campioni, {"X": 7}, tenuta=150, estrazioni=20)

    # Pari numero di operazioni e' l'intero senso del controllo: su un mercato sceso, una regola
    # che opera meno batte il possesso passivo senza sapere niente.
    assert len(esito["medie"]) == 20
    assert np.isfinite(esito["medie"]).all()
    # Su una passeggiata senza deriva il caso rende circa meno la commissione.
    assert esito["media"] == pytest.approx(-COMMISSIONE, abs=0.01)


def test_the_estimate_window_stops_a_full_horizon_before_the_split():
    quando = pd.date_range("2022-01-01", periods=2_000, freq="5min")
    campione = {
        "quando": quando,
        "posizioni": np.arange(2_000),
        "avanti": np.zeros(2_000),
    }

    dentro, fuori = separa(campione, str(quando[1_000]), str(quando[1_500]), h=150)

    # Senza l'embargo le ultime righe dello stima hanno un rendimento futuro che cade dentro la
    # verifica, e il numero fuori campione e' gonfio senza che si veda da nessuna parte.
    assert quando[dentro[-1]] < quando[1_000] - 150 * pd.Timedelta(minutes=5)
    assert quando[fuori[0]] >= quando[1_500]


def test_the_saved_summary_is_the_whole_sample_not_the_last_symbol(tmp_path, monkeypatch):
    """I metadata sono cio' che il servizio legge: se riportano il simbolo peggiore, mente."""
    from cryptofarm.ml import entry_trainer

    def campione(symbol, since, h, passo):
        # Due simboli con deriva diversa: la media complessiva deve stare **fra** le due.
        rng = np.random.default_rng({"A": 1, "B": 2}[symbol])
        deriva = {"A": 0.0006, "B": -0.0006}[symbol]
        close = np.exp(np.cumsum(rng.normal(loc=deriva, scale=0.004, size=6_000))) * 100.0
        righe = np.arange(0, 6_000, passo)
        return {
            "X": rng.normal(size=(6_000, len(entry_trainer.SWING_COLUMNS))).astype(np.float32),
            "close": close,
            "quando": pd.date_range("2019-01-01", periods=6_000, freq="5min"),
            "avanti": entry_trainer.rendimento_futuro(close, h),
            "posizioni": righe,
        }

    monkeypatch.setattr(entry_trainer, "campione_simbolo", campione)
    monkeypatch.setattr(entry_trainer, "MODELS_DIR", tmp_path)
    quando = pd.date_range("2019-01-01", periods=6_000, freq="5min")
    args = argparse.Namespace(
        symbols=["A", "B"],
        since="2019-01-01",
        stima=str(quando[3_000]),
        oos=str(quando[3_500]),
        h=20,
        tenuta=20,
        passo=12,
        quantile=0.3,
        nome="prova",
    )

    entry_trainer.addestra(args)

    salvato = json.loads((tmp_path / "prova.json").read_text())["fuori_campione"]
    assert salvato["operazioni"] > 0
    # Il difetto che questo test guarda e' un nome riusato nel ciclo di stampa per simbolo, che
    # sostituiva la media complessiva con quella dell'ultimo simbolo stampato -- il peggiore.
    per_simbolo = salvato["per_simbolo"]
    assert len(per_simbolo) == 2
    assert min(per_simbolo.values()) < salvato["medio_netto"] < max(per_simbolo.values())
    assert salvato["simboli_in_utile"].endswith("/2")
