# `tests/data/`

Un file solo: **`simulator_golden.json`**, l'atteso di `../test_simulator_golden.py`.

Tiene l'output di 21 funzioni di `trading/simulator.py` e dei suoi moduli su quattro scenari di
mercato sintetici (tendenza, laterale, regimi, sbandate). Non è un file di configurazione e non si
modifica a mano: si rigenera con

```bash
SIMULATOR_GOLDEN_REGEN=1 .venv312/bin/python -m pytest tests/test_simulator_golden.py
```

**Rigenerare accetta qualunque differenza di comportamento**, anche una regressione. Il flusso
corretto è: capire perché il test fallisce, verificare a mano che la differenza sia voluta,
rigenerare, e poi **leggere il diff del JSON** controllando che contenga solo le righe attese. Un
diff più largo del previsto significa che la modifica ha toccato anche strategie che non si stavano
guardando — che è esattamente il difetto per cui questo file esiste.
