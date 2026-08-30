# `.claude/`

Claude Code configuration for this project, and the working documentation.

| what | tracked | what it is for |
|---|---|---|
| [`docs/`](docs/) | yes | **the project decisions and the measurements that justify them.** Most of the value is here |
| `settings.json` | yes | the three plugin marketplaces, each pinned to a commit |
| `settings.local.json` | no | machine preferences, not project ones |
| `RESUME.md`, `.headroom_*` | no | session and plugin leftovers. They get deleted |

## `docs/`

The reading order is in [`docs/README.md`](docs/README.md). There is a single rule holding those
documents together: **a negative result is written up the same way a positive one is**. Several
paths that look reasonable at first sight were closed by measuring them, and the cost of reopening
one by mistake is weeks.

Everything in `docs/` is written **in English** — the rule is at the top of `../CLAUDE.md`.

Whoever touches the ML pipeline reads `docs/strategy.md` first; whoever touches labels or training
reads `docs/labeling-strategy.md`; whoever touches the page or the strategies reads
`docs/backtest-strategie.md` and `docs/ricerca-quant-ml.md`; whoever picks the work up cold reads
`docs/HANDOFF.md`.

## `settings.json`

Declares three marketplaces (`ponytail`, `agent-skills`, and three plugins from
`anthropics/financial-services`) with the plugins enabled for the project. Every marketplace is
**pinned to a commit** (`ref`, a 40-character SHA): it is the only way to fix the versions, because
`enabledPlugins` accepts only a boolean and the version is declared by the marketplace manifest. To
update them the `ref` is moved to a more recent commit, deliberately — it does not happen by itself.

A plugin's skills are available **from the session after** installation, not from the one in which
the file is edited.
