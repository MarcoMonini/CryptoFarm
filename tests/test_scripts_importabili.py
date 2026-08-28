"""Ogni modulo di `scripts/` deve almeno importarsi.

Non e' un test di comportamento ed e' voluto: `scripts/meta_gate.py` e' rimasto **non importabile**
finche' non lo si e' chiamato a mano, perche' importava `MAJORS, WIDE` da `scripts.cross_section`
dopo che quei nomi erano passati a `trading/rotation.py`. La CI gira `pytest` su `scripts`, ma
nessun test importava quel modulo, quindi il guasto non aveva modo di comparire: un punto d'ingresso
documentato in `CLAUDE.md` che solleva `ImportError` alla prima riga.

Un test per modulo sarebbe stato scritto per `meta_gate` e dimenticato per il prossimo. Questo li
prende tutti, ed e' l'unica forma che non richiede di ricordarsene.
"""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import scripts

NOMI = sorted(modulo.name for modulo in pkgutil.iter_modules(scripts.__path__) if not modulo.name.startswith("_"))


def test_ci_sono_moduli_da_controllare():
    """Se la scoperta smettesse di trovarli, il test sotto passerebbe a vuoto."""
    assert len(NOMI) >= 10, NOMI


@pytest.mark.parametrize("nome", NOMI)
def test_il_modulo_si_importa(nome: str):
    importlib.import_module(f"scripts.{nome}")
