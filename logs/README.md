# `logs/`

**Non tracciata.** Uscita degli script lunghi lanciati in background, redirezionata a mano — non
c'è nessuna configurazione di logging che scriva qui: chi lancia una misura da mezz'ora ci manda
`stdout` per poterla rileggere.

Un file per corsa, col nome della misura: `swing_train.log`, `positioning_update.log`,
`audit.log`, `ai_voter.log`, `stop_ampia.log`, e così via. Nessuno di questi file è letto da
codice: sono per gli occhi. Si possono cancellare in qualunque momento.

Perché non sono in `analysis_cache/`: quelli sono risultati e si rigenerano deterministicamente,
questi sono il racconto di una corsa specifica e non si rigenerano affatto.
