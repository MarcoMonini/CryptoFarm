# `docker/`

A single file: **`healthcheck.py`**, the image's liveness probe.

It queries `http://127.0.0.1:$PORT/_stcore/health`, Streamlit's health endpoint, and exits 0 if it
answers 200. The `Dockerfile` copies it (line 64) and uses it as `HEALTHCHECK CMD` (line 81).

It lives in a file, rather than inside a one-line `HEALTHCHECK`, for two reasons: the port is known
only at runtime (`$PORT` on PaaS, 8501 locally) and the inline version would be unreadable. It uses
only the standard library, so the image does not have to install `curl`.

The rest of the container configuration is not here but in the root:
[`../Dockerfile`](../Dockerfile) (four targets: `runtime`, `dev`, `dl`, `web`),
[`../compose.yaml`](../compose.yaml) and [`../render.yaml`](../render.yaml).

**`web` is the last stage of the `Dockerfile` and must stay there**: a build without `--target` takes
the last stage, and Render has no field to choose one. Moving it means shipping the image with
TensorFlow inside to production. A new stage goes above `web`, never below — and CI builds without
`--target` precisely to notice.
