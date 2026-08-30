# `.streamlit/`

A single file: **`config.toml`**, which sets the dark theme (`[theme] base = "dark"`) and nothing
else.

It is the project's only Streamlit configuration, and it applies both locally and in a container:
the `Dockerfile` copies the folder into the image. Port, address and CORS are **not** here — they
are command-line arguments, because in production the port is known only at runtime
(`streamlit run ... --server.port ${PORT:-8501} --server.address 0.0.0.0`).

The theme matters more than it seems: the chart trace colours are chosen for contrast **on a dark
surface**, and there are three — blue, orange, aquamarine — because they are the only ones that pass
every pair of the validator, deuteranopia included. Three tests in `tests/test_panels.py` hold that
rule in place. Changing `base` to `"light"` would invalidate it without making anything fail.
