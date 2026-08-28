"""Il motore a due versi e le strategie che lo usano.

Tre proprieta' da tenere ferme:

1. **I conti di `simulate_positions`** -- lungo, corto, inversione diretta, commissioni,
   mantenimento e leva -- si verificano a mano su numeri scelti, perche' un errore di segno qui
   non rompe niente: produce solo risultati sbagliati con l'aria di essere giusti.
2. **Nessuna strategia guarda avanti**, e servono due controlli distinti. Fra le barre: si tronca
   la serie a meta' e i segnali sulla parte iniziale devono essere identici a quelli sulla serie
   intera fino a quel punto -- un indicatore centrato o uno `shift` col segno sbagliato lo rompe.
   Dentro la barra la troncatura non vede niente, perche' la barra che scatena l'evento e'
   identica nelle due versioni: li' si perturba il massimo della sola barra dell'uscita e si
   pretende che l'uscita non si sposti.
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


def test_squeeze_non_entra_se_la_conferma_di_volume_non_e_calcolabile(candles: pd.DataFrame) -> None:
    """Con `confirm_volume=True` la conferma deve valere sempre, o si sta fuori.

    `obv_slope` e' NaN quando il volume della finestra e' zero. Il ramo che gestiva il NaN cadeva
    nel caso "nessuna conferma richiesta" ed entrava lo stesso: il chiamante credeva che il filtro
    fosse attivo su ogni ingresso, e su quelle barre non lo era.
    """
    senza_volume = candles.copy()
    senza_volume["Volume"] = 0.0

    eventi = ls.squeeze_breakout(senza_volume, ExtraCache(senza_volume), confirm_volume=True)
    assert [event for event in eventi if event[2] != 0] == []

    # Senza conferma richiesta lo stesso scenario opera: e' la prova che il test non passa
    # semplicemente perche' non c'e' nessun segnale da filtrare.
    liberi = ls.squeeze_breakout(senza_volume, ExtraCache(senza_volume), confirm_volume=False)
    assert [event for event in liberi if event[2] != 0] != []


# --- le due aggiunte: rimbalzo fra bande e zone di trend --------------------------------------


def test_il_rimbalzo_esce_alla_banda_opposta_non_alla_media(candles: pd.DataFrame) -> None:
    """E' la differenza che giustifica `atr_band_bounce` accanto a `band_reversion_gated`.

    Le due condividono le bande e differiscono nell'obiettivo: la media (KAMA) contro la banda
    opposta. Se l'uscita cadesse alla media, il guadagno per operazione sarebbe lo stesso e il
    votante nuovo sarebbe un doppione con un nome diverso.
    """
    cache = ExtraCache(candles)
    eventi = ls.atr_band_bounce(candles, cache, allow_short=False)
    kama = cache.kama(10)
    atr = cache.atr(14)
    indice = candles.index

    uscite_a_obiettivo = 0
    for quando, prezzo, obiettivo in eventi:
        if obiettivo != 0:
            continue
        i = indice.get_loc(quando)
        banda_alta = kama[i] + 2.5 * atr[i]
        if np.isclose(prezzo, banda_alta):
            uscite_a_obiettivo += 1
            # L'uscita a obiettivo sta sopra la media, non sopra di essa per caso.
            assert prezzo > kama[i]
    assert uscite_a_obiettivo > 0, "nessuna uscita alla banda opposta: l'obiettivo non e' quello"


def test_il_rimbalzo_senza_cancello_parla_piu_del_votante_gated(candles: pd.DataFrame) -> None:
    """Il cancello `adx < adx_max` e' la ragione per cui `reversione` quasi non parla.

    Toglierlo deve produrre **piu'** occasioni sugli stessi dati, altrimenti la differenza fra i
    due votanti non e' quella dichiarata e uno dei due e' inutile.
    """
    cache = ExtraCache(candles)
    con_cancello = ls.band_reversion_gated(candles, cache, allow_short=False)
    senza = ls.atr_band_bounce(candles, cache, allow_short=False)
    ingressi = lambda eventi: len([e for e in eventi if e[2] != 0])  # noqa: E731
    assert ingressi(senza) > ingressi(con_cancello), (ingressi(senza), ingressi(con_cancello))


def test_la_zona_di_trend_non_sta_mai_fuori(candles: pd.DataFrame) -> None:
    """Uno stato di struttura non ha un «fuori»: e' sopra o sotto, sempre.

    Un votante che torna a flat abbassa il punteggio dell'insieme esattamente come uno contrario,
    e una macrostruttura deve poter sostenere il punteggio per tutta la durata di un trend.
    """
    eventi = ls.trend_zone(candles, ExtraCache(candles), allow_short=True)
    assert eventi, "lo scenario deve produrre almeno un incrocio"
    assert all(obiettivo != 0 for _, _, obiettivo in eventi)
    # E si alterna: due segnali consecutivi non possono essere dello stesso verso.
    versi = [obiettivo for _, _, obiettivo in eventi]
    assert all(a != b for a, b in zip(versi, versi[1:])), versi


def test_la_zona_di_trend_cambia_solo_sugli_incroci(candles: pd.DataFrame) -> None:
    """Ogni evento deve cadere su una barra in cui le due medie si sono incrociate davvero."""
    cache = ExtraCache(candles)
    eventi = ls.trend_zone(candles, cache, fast=20, slow=100, allow_short=True)
    veloce, lenta = cache.ema(20), cache.ema(100)
    indice = candles.index

    # Il primo evento e' la **dichiarazione dello stato iniziale**, non un incrocio: senza,
    # `held_state` resterebbe a zero fino al primo incrocio vero e la struttura risulterebbe
    # sconosciuta mentre invece e' nota. Va esentato qui, non tolto dalla strategia.
    primo = eventi[0]
    i0 = indice.get_loc(primo[0])
    assert (veloce[i0] >= lenta[i0]) == (primo[2] > 0)

    for quando, _, obiettivo in eventi[1:]:
        i = indice.get_loc(quando)
        sopra_ora = veloce[i] >= lenta[i]
        assert sopra_ora == (obiettivo > 0), (quando, obiettivo)
        # e la barra precedente stava dall'altra parte
        assert (veloce[i - 1] >= lenta[i - 1]) != sopra_ora, f"{quando}: nessun incrocio qui"
