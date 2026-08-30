# `models/` — gli artefatti addestrati

Ogni modello sta in **due file con lo stesso nome**: `nome.joblib` (il modello) e `nome.json`
(i metadata). **Nessuno dei due si traccia** — il `.gitignore` di questa cartella copre `*.joblib`,
`*.json` e `*.keras`, e tiene solo questo README. Un clone del repository trova la cartella vuota,
ed è la condizione in cui gira anche il servizio pubblico: Render non ha dischi persistenti.

Non si modificano a mano. Si rigenerano con i trainer.

## Quali artefatti, e chi li produce

| artefatto | comando | cosa prevede | documento |
|---|---|---|---|
| `entry_model_veloce` | `python -m cryptofarm.ml.entry_trainer --h 20 --quantile 0.995 --nome entry_model_veloce` | il rendimento delle prossime 20 barre da 5m | `modello-ingresso.md` |
| `entry_model` | `python -m cryptofarm.ml.entry_trainer` | lo stesso su 150 barre: fa da **cancello** al veloce | `modello-ingresso.md` |
| `rl_model` | `python -m cryptofarm.ml.rl_trainer` | la posizione, col costo dentro la ricompensa | `politica-rl.md` |
| `swing_model` | `python -m cryptofarm.ml.swing_trainer` | la prossimità agli estremi locali | `modello-swing.md` |
| `meta_model` | `python -m cryptofarm.ml.meta_trainer` | se un ingresso del primario chiude in profitto netto | `strategy.md` |
| `signal_model` | `python -m cryptofarm.ml.trainer` | quale barriera viene toccata per prima | `strategy.md` |

## Chi governa la pagina

`ml/trainer.MODEL_PRECEDENCE` in quest'ordine, e `active_model_name()` è **l'unica fonte di
verità**: decide sia quale artefatto si carica sia quale strategia di servizio lo interpreta,
quindi i due non possono divergere. **Per tornare al modello precedente si sposta altrove
l'artefatto di quello più recente** — non si tocca il codice.

Oggi in testa c'è `entry_model_veloce`, e i due artefatti d'ingresso lavorano **in coppia**: il
veloce genera le operazioni, il lento fa da cancello sulla sola barra d'ingresso. Senza il lento il
veloce opera da solo, e il netto per operazione fuori campione scende da +2,071% a +1,360%.

## I metadata non sono decorazione

`meta_parameters()` legge barriere, soglia CUSUM e parametri di esecuzione **dal `.json`
dell'artefatto** e non da costanti, perché devono essere esattamente quelli con cui il modello è
stato addestrato. Per il modello d'ingresso valgono anche soglia, tenuta e cancello: sono nei
metadata e non nei widget, perché la selettività **è** il vantaggio del modello — cambiarla non
regola una manopola, serve un'altra strategia.

## Se qui dentro trovi un artefatto di cui non esiste il trainer

Sono i resti di due famiglie **chiuse in negativo**, e si possono cancellare: la politica a tre
azioni (`policy_model`, `policy_alta`, `policy_bassa` — `strategy.md` §12-13), il modello a gambe
(`leg_model` — `modello-swing.md` §1) e i tre `.keras` dell'era precedente al gradient boosting
(`optimized_model.keras`, `trained_model.keras`, `trained_model1.keras`). Il loro codice non c'è
più: rimetterne il nome in `MODEL_PRECEDENCE` non basta a farli girare, e un test lo verifica.
