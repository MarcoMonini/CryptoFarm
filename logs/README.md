# `logs/`

**Not tracked.** Output of the long scripts launched in the background, redirected by hand — there is
no logging configuration writing here: whoever starts a half-hour measurement sends `stdout` here so
it can be read back.

One file per run, named after the measurement: `swing_train.log`, `positioning_update.log`,
`audit.log`, `ai_voter.log`, `stop_ampia.log`, and so on. None of these files is read by code: they
are for the eyes. They can be deleted at any time.

Why they are not in `analysis_cache/`: those are results and regenerate deterministically, these are
the story of one specific run and do not regenerate at all.
