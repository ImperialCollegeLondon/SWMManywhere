# SWMManywhere web frontend

Contributed by [Zhonghao Zhang](https://github.com/Zhonghao1995).

A browser interface for SWMManywhere: draw a bounding box on a map, adjust the
package's parameters, build, inspect the synthesised network on the map, click any
element for its attributes and simulated time series, and download the model
package. Every build is an ordinary SWMManywhere `config.yml`; the page is a form
for that file, and it shows you the file it used so the run can be reproduced with
`python -m swmmanywhere --config_path=...`.

Nothing under `src/swmmanywhere` is modified. The folder is self-contained:

| Path | What it is |
|---|---|
| `src/` | The web app: React, Vite, MapLibre GL, Zustand, Tailwind (TypeScript). |
| `server/` | A small FastAPI service that runs the package for the browser. It calls only the public API (`load_config`, `swmmanywhere`, `run` from `swmmanywhere.swmmanywhere`). |
| `server/tests/` | Contract tests that use the package's bundled Bellinge fixtures; no network, no SWMM. |

A browser cannot run Python, so the service is the smallest possible bridge: it
writes the config, validates it with `load_config`, runs the synthesis in a worker
thread, runs the SWMM simulation with `run`, and serves the nodes / edges /
subcatchments as GeoJSON (outfalls identified the same way as
`swmmanywhere.utilities.plot_map`) plus per-element time series from the results.

## Run it

From the repository root, in the environment where `swmmanywhere` is installed:

```bash
pip install -e .
pip install -r frontend/server/requirements.txt
cd frontend
uvicorn server.app:app --port 8000      # API on http://localhost:8000
npm install && npm run dev              # UI on http://localhost:5174 (proxies /api)
```

For a single-origin deployment build the app once (`npm run build`); the server
then serves `frontend/dist` at `/`.

The page uses the docs' primary colour (`#00BCD4`, see `docs/custom.css`) as its
only accent and loads the Geist typeface from Google Fonts; drop the `<link>` in
`index.html` for a fully offline build, the system font stack is the fallback.

| Environment variable | Meaning | Default |
|---|---|---|
| `SWMMANYWHERE_WEB_WORKDIR` | `base_dir` for every build; downloads are reused per bounding box (SWMManywhere's `bbox_N/model_N` layout). | `<tmp>/swmmanywhere_web` |
| `SWMMANYWHERE_WEB_MAX_KM2` | Largest bounding box accepted. | `5` |
| `SWMMANYWHERE_WEB_ORIGINS` | Comma-separated CORS origins. | `*` |

Builds run one at a time: the package shares `base_dir` between runs and its logger
is process-global. The first build of a box spends a few minutes downloading
(NASADEM, Overture buildings, OpenStreetMap); rebuilding the same box with other
parameters takes about ten seconds for the documentation's Andorra example (0.6 km²),
including the SWMM run.

## What the page exposes

| Page section | Config key | Notes |
|---|---|---|
| Bounding box | `bbox` | Drawn with two clicks or typed; the area limit is enforced server-side. |
| Project | `project` | Folder name under `base_dir`. |
| Parameters | `parameter_overrides` | Rendered from the pydantic models in `swmmanywhere.parameters` (defaults, units, bounds, descriptions). Only changed values are sent. `metric_evaluation` is hidden because the page does not supply a `real` network. |
| Graph functions | `graphfcn_list` | The default list from `defs/demo_config.yml`, editable; the server validates it with `validate_graphfcn_list` via `load_config`. |
| Simulation | `run_model`, `run_settings` | `duration`, `reporting_iters`, `storevars` (the enum from `defs/schema.yml`). |

Rainfall is the package's bundled demo storm (`defs/storm.dat`), exactly what the
CLI uses when no precipitation file is given. Real-network comparison (`real:`,
metrics), `starting_graph`, and custom modules are not exposed.

The server additionally range-checks overrides with the package's own pydantic
models before building, because `swmmanywhere()` applies overrides with `setattr`
and only checks the names (issue #379).

## HTTP surface

| Method and path | Purpose |
|---|---|
| `GET /api/v1/defaults` | Parameter groups, graph functions, run settings, limits, all read from the package. |
| `POST /api/v1/tasks` | Start a build. Body: `bbox`, `project`, `parameter_overrides`, `graphfcn_list`, `run_model`, `run_settings`. Returns `task_id`. |
| `GET /api/v1/tasks/{id}` | State, stage, progress, last log lines, error. |
| `GET /api/v1/tasks/{id}/config` | The `config.yml` that was validated and run. |
| `GET /api/v1/tasks/{id}/preview` | GeoJSON (EPSG:4326) with `kind` = `junction`, `outfall`, `conduit`, `subcatchment`. |
| `GET /api/v1/tasks/{id}/timeseries?id=` | Simulated series for one SWMM object (`flooding`, `flow`, `depth`, `runoff`). |
| `GET /api/v1/tasks/{id}/result` | Zip of the model directory: `.inp`, nodes, edges, subcatchments, `config.yml`, `results.parquet`. The rain file is included and the `.inp` inside the zip refers to it by name so it opens in EPA SWMM as-is. |

## Tests

```bash
cd frontend
pytest server/tests          # 8 tests, bundled fixtures only
npm run typecheck
```

## Things found while building this

- **Overture release id.** `prepare_data._get_latest_s3_url` derives the release
  from `Path(href).parent` of the STAC catalog's child links. Those hrefs are now
  absolute URLs, so the package caches `https:/stac.overturemaps.org/<release>`
  and every buildings download fails on a non-existent S3 key. The server seeds
  `.cache/overture_release.json` with a well-formed id before each build; the
  package itself is unchanged. This wants a one-line fix upstream.
- **`affine` 3.** With `affine>=3` (where `Affine` is no longer a namedtuple),
  `pyflwdir.dem.slope`, a numba function SWMManywhere calls with a raster
  transform in `geospatial_utilities`, fails with "Cannot determine Numba type of
  <class 'affine.Affine'>" during `clip_to_catchments`. `server/requirements.txt`
  pins `affine<3`; the package's own dependencies probably want the same pin until
  `pyflwdir` supports affine 3.
- **macOS on Apple silicon.** `import pyswmm` is killed by the kernel because
  `libomp.dylib` inside the `swmm-toolkit` wheel has an invalid code signature.
  Re-signing the wheel's binaries fixes it:
  `codesign --force --sign - <site-packages>/swmm/toolkit/*.dylib <site-packages>/swmm/toolkit/*.so`.
- **Paths with spaces.** SWMM cannot read a `[RAINGAGES] FILE` path containing a
  space. The server copies the demo storm into its work directory and the
  downloaded zip references it by bare name.
- **Subcatchment slope units (question for the maintainers).** `derive_subcatchments`
  stores the mean `pyflwdir.dem.slope` gradient, which that function documents as
  m/m, and `synthetic_write` copies it unchanged into the `%Slope` column of
  `[SUBCATCHMENTS]`, which SWMM reads as a percentage. For the Andorra example the
  written values are 0.0–0.9 where 0–89 % would be expected. The map shows the
  stored value labelled m/m.
