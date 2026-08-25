"""Il motore a due versi e le strategie che lo usano.

Tre proprieta' da tenere ferme:

1. **I conti di `simulate_positions`** -- lungo, corto, inversione diretta, commissioni,
   mantenimento e leva -- si verificano a mano su numeri scelti, perche' un errore di segno qui
   non rompe niente: produce solo risultati sbagliati con l'aria di essere giusti.
2. **Nessuna strategia guarda avanti.** Il controllo e' diretto: si tronca la serie a meta' e i
   segnali generati sulla parte iniziale devono essere identici a quelli generati sulla serie
   intera fino a quel punto. Un indicatore centrato o uno `shift` col segno sbagliato lo rompe.
3. **`allow_short=False` produce solo posizioni lunghe**, e nient'altro cambia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptofarm.trading import strategies_ls as ls
from cryptofarm.trading.indicators_extra import ExtraCache
from cryptofarm.trading.pnl import simulate_positions


@pytest.fixture(scope="module")
def candles() -> pd.DataFrame:
    generator = np.random.default_rng(7)
    steps = generator.normal(0.0002, 0.01, 3_000)
    close = 20_000 * np.exp(np.cumsum(steps))
    spread = np.abs(generator.normal(0, 0.004, 3_000)) * close
    frame = pd.DataFrame(
        {
            "Open": np.concatenate([[close[0]], close[:-1]]),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": generator.uniform(10, 500, 3_000),
        },
        index=pd.date_range("2023-01-01", periods=3_000, freq="4h", name="Open time"),
    )
    frame["High"] = frame[["High", "Open", "Close"]].max(axis=1)
    frame["Low"] = frame[["Low", "Open", "Close"]].min(axis=1)
    return frame


def test_long_and_short_are_symmetric() -> None:
    """Senza costi, un lungo da 100 a 110 e un corto da 110 a 99 rendono entrambi il 10%."""
    times = pd.date_range("2024-01-01", periods=3, freq="1D")
    operations = simulate_positions(
        [(times[0], 100, 1), (times[1], 110, -1), (times[2], 99, 0)], wallet=100, fee_percent=0, carry_daily_percent=0
    )
    assert [operation["Side"] for operation in operations] == ["long", "short"]
    assert operations[0]["Profit"] == pytest.approx(10.0)
    assert operations[1]["Profit"] == pytest.approx(11.0)  # il 10% di un capitale gia' cresciuto
    assert operations[-1]["Wallet_After"] == pytest.approx(121.0)


def test_costs_are_charged_on_both_legs_and_over_time() -> None:
    times = pd.date_range("2024-01-01", periods=2, freq="10D")
    operations = simulate_positions(
        [(times[0], 100, 1), (times[1], 100, 0)], wallet=100, fee_percent=0.05, carry_daily_percent=0.03
    )
    # Prezzo invariato: resta solo il costo. Due gambe allo 0,05% piu' dieci giorni allo 0,03%.
    assert operations[0]["Profit"] == pytest.approx(-(0.0005 * 200 + 0.0003 * 100 * 10), rel=1e-6)


def test_leverage_scales_profit_and_can_wipe_the_account() -> None:
    times = pd.date_range("2024-01-01", periods=2, freq="1D")
    events = [(times[0], 100, 1), (times[1], 110, 0)]
    single = simulate_positions(events, 100, 0, 0, leverage=1)[0]["Profit"]
    double = simulate_positions(events, 100, 0, 0, leverage=2)[0]["Profit"]
    assert double == pytest.approx(2 * single)

    liquidato = simulate_positions([(times[0], 100, 1), (times[1], 60, 0)], 100, 0, 0, leverage=3)
    assert liquidato[-1]["Wallet_After"] == 0.0


def test_events_after_liquidation_are_ignored() -> None:
    times = pd.date_range("2024-01-01", periods=4, freq="1D")
    operations = simulate_positions(
        [(times[0], 100, -1), (times[1], 250, 0), (times[2], 100, 1), (times[3], 200, 0)], 100, 0, 0
    )
    assert len(operations) == 1
    assert operations[0]["Wallet_After"] == 0.0


@pytest.mark.parametrize("name", sorted(ls.STRATEGIES))
def test_no_look_ahead(candles: pd.DataFrame, name: str) -> None:
    """I segnali sulla prima meta' non devono cambiare quando esiste anche la seconda."""
    strategy = ls.STRATEGIES[name]
    intera = strategy(candles, ExtraCache(candles))
    meta = candles.iloc[: len(candles) // 2]
    troncata = strategy(meta, ExtraCache(meta))
    limite = meta.index[-1]
    attesi = [event for event in intera if event[0] <= limite]
    assert troncata == attesi


@pytest.mark.parametrize("name", sorted(ls.STRATEGIES))
def test_long_only_never_goes_short(candles: pd.DataFrame, name: str) -> None:
    events = ls.STRATEGIES[name](candles, ExtraCache(candles), allow_short=False)
    assert all(target >= 0 for _, _, target in events)
    operations = simulate_positions(events, 100, 0.05, 0.03)
    assert all(operation["Side"] == "long" for operation in operations)


@pytest.mark.parametrize("name", ["donchian_breakout", "squeeze_breakout"])
def test_trailing_stop_ignores_the_high_of_its_own_bar(candles: pd.DataFrame, name: str) -> None:
    """Lo stop a trailing in vigore *durante* una barra non puo' dipendere dal massimo di quella
    barra.

    Dentro la barra il minimo puo' arrivare prima del massimo: assumere il contrario alza lo stop
    con un massimo non ancora avvenuto e fa uscire a un prezzo migliore di quello ottenibile.
    `test_no_look_ahead` non lo vede, perche' tronca la serie *fra* le barre e la barra che scatena
    l'uscita resta identica nelle due versioni.
    """
    strategy = ls.STRATEGIES[name]
    base = strategy(candles, ExtraCache(candles))
    uscite = [event for event in base if event[2] == 0]
    assert uscite, "lo scenario deve produrre almeno un'uscita, altrimenti il test non prova nulla"
    quando = uscite[0][0]

    alterate = candles.copy()
    alterate.iloc[alterate.index.get_loc(quando), alterate.columns.get_loc("High")] *= 1.05
    dopo = strategy(alterate, ExtraCache(alterate))

    # Il massimo di una barra non puo' cambiare nulla di cio' che e' gia' successo,
    assert [e for e in dopo if e[0] < quando] == [e for e in base if e[0] < quando]
    # ne' l'uscita che avviene su quella barra stessa.
    assert [e for e in dopo if e[0] == quando and e[2] == 0] == [e for e in base if e[0] == quando and e[2] == 0]
