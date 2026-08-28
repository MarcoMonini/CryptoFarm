"""Una strategia, molti asset, **un solo capitale**.

La domanda e' diversa da quella della rotazione (`trading/rotation.py`), e va tenuta distinta. La
rotazione sceglie *quale* asset tenere e ci sta dentro sempre. Qui invece si sorveglia un paniere
con la stessa strategia su ognuno, si sta fuori finche' nessuno parla, e quando **uno** da' il
segnale ci si mette tutto il capitale.

## Perche' puo' valere la pena, e cosa costa

Il difetto piu' probabile della confluenza, dichiarato prima di misurarla, e' che operi troppo
poco perche' si possa dire se ha funzionato. Sorvegliare cinque asset invece di uno non cambia la
regola: cambia quante volte la regola trova qualcosa. E' la strada piu' diretta per portare il
campione a una dimensione in cui i numeri significano qualcosa.

Il prezzo lo si paga in due monete, e vanno tutte e due riportate:

- **le occasioni perse.** Con una posizione alla volta, ogni segnale che arriva mentre il capitale
  e' impegnato viene buttato. `Portafoglio.occasioni_perse` le conta: se sono molte piu' delle
  operazioni fatte, la scarsita' del campione non era il problema e questo non e' il rimedio;
- **la concentrazione.** Se il 90% delle operazioni sta su un asset solo, non si sta sorvegliando
  un paniere: si sta operando su quell'asset con quattro spettatori. `per_asset` lo dice.

## Le pari merito non sono un dettaglio

Su asset che si muovono insieme -- e le criptovalute lo fanno -- i segnali arrivano spesso sulla
stessa barra. Chi vince quella barra decide l'operazione, e sceglierlo per ordine alfabetico
sarebbe una decisione arbitraria travestita da dettaglio di attuazione. Gli eventi possono quindi
portare un quarto elemento, la **priorita'**: per la confluenza e' il margine del punteggio sopra
la soglia, cioe' quanto quel segnale e' netto. A parita' di priorita' decide l'ordine in cui gli
asset sono stati passati, e quello resta una scelta di chi chiama.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cryptofarm.trading.pnl import CARRY_DAILY_PERCENT


@dataclass
class Portafoglio:
    """Le operazioni fatte, piu' cio' che serve a sapere se il paniere e' servito a qualcosa."""

    operazioni: list
    occasioni_perse: int = 0
    per_asset: dict[str, int] = field(default_factory=dict)

    @property
    def capitale_finale(self) -> float:
        return float(self.operazioni[-1]["Wallet_After"]) if self.operazioni else float("nan")

    @property
    def concentrazione(self) -> float:
        """La quota dell'asset piu' operato. Sopra 0,9 il paniere e' finzione."""
        totale = sum(self.per_asset.values())
        return max(self.per_asset.values()) / totale if totale else float("nan")


def simulate_shared_capital(
    eventi_per_asset: dict[str, list],
    wallet: float = 100.0,
    fee_percent: float = 0.1,
    carry_daily_percent: float = CARRY_DAILY_PERCENT,
    leverage: float = 1.0,
) -> Portafoglio:
    """Esegue gli eventi di piu' asset su un capitale solo, una posizione alla volta.

    `eventi_per_asset` sono le liste `(timestamp, prezzo, obiettivo)` di `strategies_ls` e
    `confluence`, eventualmente con un quarto elemento di priorita' per le pari merito.

    Le convenzioni di costo sono **identiche** a `pnl.simulate_positions` -- commissione su
    entrambe le gambe calcolata sul nozionale scambiato, mantenimento proporzionale ai giorni,
    liquidazione valutata alla chiusura -- perche' i due risultati devono essere confrontabili:
    la domanda a cui questo modulo risponde e' quanto cambia passando da un asset a cinque, e se
    cambiassero anche i costi la risposta non direbbe piu' quello.

    Mentre il capitale e' impegnato su un asset, i segnali degli altri **non si accodano**: si
    perdono, e vengono contati. Accodarli vorrebbe dire entrare su un segnale vecchio, che e'
    esattamente la cosa che la memoria del segnale gia' regola con un decadimento.
    """
    stream = sorted(
        (
            (e[0], -float(e[3]) if len(e) > 3 else 0.0, indice, nome, float(e[1]), int(e[2]))
            for indice, (nome, eventi) in enumerate(eventi_per_asset.items())
            for e in eventi
        ),
        key=lambda r: r[:3],
    )

    fee = fee_percent / 100.0
    carry = carry_daily_percent / 100.0
    operazioni: list = []
    per_asset = {nome: 0 for nome in eventi_per_asset}
    perse = 0

    aperto: str | None = None
    verso = 0
    prezzo_ingresso = 0.0
    quando_ingresso = None
    nozionale = 0.0

    for quando, _, _, nome, prezzo, obiettivo in stream:
        if aperto is not None and nome != aperto:
            # Il capitale e' impegnato altrove. Un segnale di apertura qui e' un'occasione persa;
            # una chiusura su un asset su cui non siamo dentro non e' niente.
            perse += obiettivo != 0
            continue
        if aperto is None:
            if obiettivo != 0:
                aperto, verso = nome, obiettivo
                prezzo_ingresso, quando_ingresso, nozionale = prezzo, quando, wallet * leverage
            continue
        if obiettivo == verso:
            continue

        giorni = (quando - quando_ingresso).total_seconds() / 86400.0
        lordo = nozionale * verso * (prezzo - prezzo_ingresso) / prezzo_ingresso
        commissioni = fee * nozionale * (1 + prezzo / prezzo_ingresso)
        mantenimento = carry * nozionale * giorni
        profitto = lordo - commissioni - mantenimento
        wallet = max(0.0, wallet + profitto)
        operazioni.append(
            {
                "Asset": aperto,
                "Side": "long" if verso > 0 else "short",
                "Buy_Time": quando_ingresso,
                "Buy_Price": prezzo_ingresso,
                "Sell_Time": quando,
                "Sell_Price": prezzo,
                "Quantity": nozionale / prezzo_ingresso,
                "Profit": profitto,
                "Wallet_After": wallet,
            }
        )
        per_asset[aperto] += 1
        aperto, verso = None, 0
        if wallet <= 0:
            break
        if obiettivo != 0:
            # Inversione diretta sullo stesso asset: si riapre subito nel verso nuovo.
            aperto, verso = nome, obiettivo
            prezzo_ingresso, quando_ingresso, nozionale = prezzo, quando, wallet * leverage

    return Portafoglio(operazioni=operazioni, occasioni_perse=perse, per_asset=per_asset)


def simulate_slots(
    eventi_per_asset: dict[str, list],
    n_slot: int = 3,
    wallet: float = 100.0,
    fee_percent: float = 0.1,
    carry_daily_percent: float = CARRY_DAILY_PERCENT,
) -> Portafoglio:
    """Lo stesso capitale su **piu' posizioni contemporanee**, una per asset, fino a `n_slot`.

    E' la differenza che conta rispetto a `simulate_shared_capital`, che ne tiene **una** e butta
    via tutto il resto. Con `n_slot=1` i due coincidono, ed e' il controllo che lo dimostra.

    ## Perche' esiste

    Un segnale a IC 0,05 non e' eseguibile su un asset alla volta: l'errore non ha su cosa
    mediarsi. La sezione trasversale e' il posto in cui un vantaggio debole diventa pagabile
    (`ricerca-quant-ml.md` §1.5.1), e per starci dentro servono scommesse **simultanee e
    indipendenti**, non una coda di scommesse sequenziali. Sequenziale e' cio' che il paniere a
    capitale condiviso gia' fa, e le sue «occasioni perse» misurano esattamente quanto costa.

    ## Come si divide il capitale

    All'apertura la quota e' `contante / slot ancora liberi`. Non e' `capitale / n_slot`: con
    quello, dopo la prima perdita le quote successive resterebbero tarate sul capitale iniziale e
    il portafoglio andrebbe in leva senza dirlo. Cosi' invece la somma delle quote non supera mai
    il contante disponibile, che e' il vincolo vero, e a slot tutti liberi le quote sono uguali.

    Il capitale riportato in `Wallet_After` e' `contante + nozionali aperti al costo`: ignora il
    non realizzato, esattamente come `curva_capitale` e `pnl.simulate_positions`, tenuto uguale di
    proposito perche' i tre numeri vanno confrontati fra loro.
    """
    if n_slot < 1:
        raise ValueError(f"servono almeno uno slot: {n_slot}")

    stream = sorted(
        (
            (e[0], -float(e[3]) if len(e) > 3 else 0.0, indice, nome, float(e[1]), int(e[2]))
            for indice, (nome, eventi) in enumerate(eventi_per_asset.items())
            for e in eventi
        ),
        key=lambda r: r[:3],
    )

    fee = fee_percent / 100.0
    carry = carry_daily_percent / 100.0
    operazioni: list = []
    per_asset = {nome: 0 for nome in eventi_per_asset}
    perse = 0
    contante = wallet
    aperte: dict[str, tuple] = {}

    def apri(nome, prezzo, quando, verso):
        nonlocal contante
        liberi = n_slot - len(aperte)
        quota = contante / liberi
        aperte[nome] = (verso, prezzo, quando, quota)
        contante -= quota

    for quando, _, _, nome, prezzo, obiettivo in stream:
        if nome not in aperte:
            # Un'uscita su un asset su cui non siamo dentro non e' niente; un ingresso senza slot
            # liberi e' un'occasione persa, ed e' il numero che dice se `n_slot` basta.
            if obiettivo != 0:
                if len(aperte) < n_slot:
                    apri(nome, prezzo, quando, obiettivo)
                else:
                    perse += 1
            continue

        verso, prezzo_ingresso, quando_ingresso, nozionale = aperte[nome]
        if obiettivo == verso:
            continue

        giorni = (quando - quando_ingresso).total_seconds() / 86400.0
        lordo = nozionale * verso * (prezzo - prezzo_ingresso) / prezzo_ingresso
        commissioni = fee * nozionale * (1 + prezzo / prezzo_ingresso)
        mantenimento = carry * nozionale * giorni
        profitto = lordo - commissioni - mantenimento
        contante = max(0.0, contante + nozionale + profitto)
        del aperte[nome]
        operazioni.append(
            {
                "Asset": nome,
                "Side": "long" if verso > 0 else "short",
                "Buy_Time": quando_ingresso,
                "Buy_Price": prezzo_ingresso,
                "Sell_Time": quando,
                "Sell_Price": prezzo,
                "Quantity": nozionale / prezzo_ingresso,
                "Profit": profitto,
                "Wallet_After": contante + sum(q for _, _, _, q in aperte.values()),
            }
        )
        per_asset[nome] += 1
        if obiettivo != 0:
            apri(nome, prezzo, quando, obiettivo)

    return Portafoglio(operazioni=operazioni, occasioni_perse=perse, per_asset=per_asset)


def curva_capitale(operazioni: list, indice: pd.DatetimeIndex, wallet: float = 100.0) -> np.ndarray:
    """Il capitale su tutto l'indice, per poter calcolare drawdown e Sharpe con `pnl`.

    Il capitale si muove **alla chiusura di ogni operazione**, non barra per barra: fra un'entrata
    e un'uscita la curva resta piatta. E' una sottostima del drawdown vero -- una posizione che
    scende del 30% e risale non lascia traccia -- ed e' lo stesso limite che ha gia'
    `pnl.simulate_positions`, tenuto uguale di proposito perche' i due numeri vanno confrontati.
    """
    curva = pd.Series(np.nan, index=pd.DatetimeIndex(indice), dtype=float)
    curva.iloc[0] = wallet
    for operazione in operazioni:
        posizione = curva.index.searchsorted(operazione["Sell_Time"])
        if posizione < len(curva):
            curva.iloc[posizione] = operazione["Wallet_After"]
    return curva.ffill().to_numpy()


def _selfcheck() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="1d")
    solo_a = {"A": [(idx[0], 100.0, 1), (idx[2], 110.0, 0)]}
    uno = simulate_shared_capital(solo_a, fee_percent=0.0, carry_daily_percent=0.0)

    # 1. Con un asset solo il risultato e' quello di sempre: +10% su 100.
    assert len(uno.operazioni) == 1 and abs(uno.capitale_finale - 110.0) < 1e-9

    # 2. Il secondo asset che parla mentre il capitale e' impegnato viene perso, non accodato.
    due = simulate_shared_capital(
        {**solo_a, "B": [(idx[1], 50.0, 1), (idx[3], 60.0, 0)]}, fee_percent=0.0, carry_daily_percent=0.0
    )
    assert due.occasioni_perse == 1
    assert due.per_asset == {"A": 1, "B": 0}

    # 3. Appena il capitale si libera il paniere lo riusa: e' tutto il punto della cosa.
    dopo = simulate_shared_capital(
        {**solo_a, "B": [(idx[4], 50.0, 1), (idx[6], 60.0, 0)]}, fee_percent=0.0, carry_daily_percent=0.0
    )
    assert dopo.per_asset == {"A": 1, "B": 1}
    assert abs(dopo.capitale_finale - 132.0) < 1e-9  # 110 * 1,2

    # 4. Le pari merito le vince la priorita' piu' alta, non l'ordine in cui sono passati.
    pari = simulate_shared_capital(
        {
            "A": [(idx[0], 100.0, 1, 0.1), (idx[2], 90.0, 0, 0.0)],
            "B": [(idx[0], 50.0, 1, 0.9), (idx[2], 60.0, 0, 0.0)],
        },
        fee_percent=0.0,
        carry_daily_percent=0.0,
    )
    assert pari.per_asset == {"A": 0, "B": 1}, "ha vinto l'ordine invece della priorita'"

    # 5. La curva del capitale e' piatta fino alla chiusura, poi salta.
    curva = curva_capitale(uno.operazioni, idx)
    assert curva[0] == 100.0 and curva[1] == 100.0 and curva[2] == 110.0 and curva[-1] == 110.0

    print("portfolio selfcheck: 5 controlli passati")


if __name__ == "__main__":
    _selfcheck()
