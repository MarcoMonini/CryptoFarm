# `.github/workflows/`

A single workflow: **`ci.yml`**, on every pull request and every push to `main`. Two jobs.

## `quality` — lint, format and tests

Installs `.[app,data,dev]` on Python 3.12 and passes `ruff check`, `black --check` and `pytest` over
`src`, `tests` and `scripts`.

## `docker` — building the images

Builds `runtime`, `dev` and `web`, and verifies **four things that are not visible from the source**:

1. that the package imports inside the image and resolves the data directories to `/app/...` (i.e.
   that the `paths.py` override works, otherwise models trained in a container end up in a
   throwaway layer);
2. that the tests pass inside the image, not only on the CI machine;
3. that a build **without `--target`** does not carry TensorFlow — i.e. that `web` is still the last
   stage of the `Dockerfile`, which is how Render builds it;
4. that the container really binds to `$PORT`: it starts it with `PORT=10000` and queries
   `/_stcore/health`.

## What it does not do

**It publishes no image to a registry.** Render builds the `Dockerfile` itself on every push to
`main`; these builds exist to catch a breakage before it gets there, not to produce the artifact
that goes into production.
