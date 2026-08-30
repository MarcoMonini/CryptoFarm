# `.claude/`

Configurazione di Claude Code per questo progetto, e la documentazione di lavoro.

| cosa | tracciato | a cosa serve |
|---|---|---|
| [`docs/`](docs/) | sì | **le decisioni di progetto e le misure che le giustificano.** Il grosso del valore sta qui |
| `settings.json` | sì | i tre marketplace di plugin, ognuno agganciato a un commit |
| `settings.local.json` | no | preferenze della macchina, non del progetto |
| `RESUME.md`, `.headroom_*` | no | scarti di sessione e di plugin. Si cancellano |

## `docs/`

L'ordine di lettura sta in [`docs/README.md`](docs/README.md). La regola che tiene insieme quei
documenti è una sola: **un risultato negativo si scrive come si scrive uno positivo**. Diverse
strade che sembrano ragionevoli a prima vista sono state chiuse misurandole, e il costo di
riaprirle per sbaglio è settimane.

Chi tocca la pipeline ML legge `docs/strategy.md` prima; chi tocca la pagina o le strategie legge
`docs/backtest-strategie.md` e `docs/ricerca-quant-ml.md`; chi riprende il lavoro a freddo legge
`docs/HANDOFF.md`.

## `settings.json`

Dichiara tre marketplace (`ponytail`, `agent-skills`, e tre plugin di
`anthropics/financial-services`) con i plugin abilitati per il progetto. Ogni marketplace è
**agganciato a un commit** (`ref`, SHA a 40 caratteri): è l'unico modo di fissare le versioni,
perché `enabledPlugins` accetta solo un booleano e la versione la dichiara il manifesto del
marketplace. Per aggiornarli si sposta il `ref` su un commit più recente, deliberatamente — non
succede da solo.

Le skill di un plugin sono disponibili **dalla sessione successiva** all'installazione, non da
quella in cui si modifica il file.
