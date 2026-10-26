"""Web service that runs SWMManywhere for the browser frontend in ../src.

It uses only the package's public Python API (``load_config``, ``swmmanywhere`` and
``run`` from ``swmmanywhere.swmmanywhere``) and never patches anything in ``src/``.
Every build writes the same ``config.yml`` a CLI user would pass to
``python -m swmmanywhere``, validates it with ``load_config`` and runs it; the
frontend is a form for that file.

Run from the ``frontend/`` directory::

    uvicorn server.app:app --port 8000

Environment:
    SWMMANYWHERE_WEB_WORKDIR   build directory (default: <tmp>/swmmanywhere_web)
    SWMMANYWHERE_WEB_MAX_KM2   largest bounding box accepted, km² (default: 5)
    SWMMANYWHERE_WEB_ORIGINS   comma-separated CORS origins (default: *)
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import threading
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from jsonschema import ValidationError as SchemaError
from pydantic import BaseModel, Field, ValidationError

import swmmanywhere
from swmmanywhere import parameters
from swmmanywhere.graph_utilities import graphfcns
from swmmanywhere.logging import logger
from swmmanywhere.swmmanywhere import load_config
from swmmanywhere.swmmanywhere import run as run_simulation
from swmmanywhere.swmmanywhere import swmmanywhere as run_synthesis
from swmmanywhere.utilities import yaml_dump, yaml_load

from .preview import element_timeseries, model_geojson

DEFS = Path(swmmanywhere.__file__).parent / "defs"
DIST = Path(__file__).resolve().parent.parent / "dist"
PROJECT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Log lines SWMManywhere emits along the way (swmmanywhere.py, preprocessing.py),
# mapped to a label and a rough percentage for the progress bar.
STAGES: list[tuple[str, str, float]] = [
    ("Running downloads", "Downloading open data", 5),
    ("downloading elevation", "Downloading elevation (NASADEM)", 6),
    ("downloading buildings", "Downloading buildings (Overture)", 9),
    ("downloading network", "Downloading streets (OpenStreetMap)", 12),
    ("downloading river", "Downloading rivers (OpenStreetMap)", 15),
    ("Iterating graph functions", "Running graph functions", 20),
    ("Saving final graph", "Writing the SWMM .inp", 82),
    ("initialised in pyswmm", "Running the SWMM simulation", 86),
    ("Model run complete", "Simulation complete", 96),
]
GRAPHFCN_DONE = re.compile(r"graphfcn: (\S+) completed")
# `FILE "<path>"` or `FILE <path>` in a [RAINGAGES] line.
RAINGAGE_FILE = re.compile(r'(FILE\s+)("([^"]+)"|(\S+))')

# Overture release cache. ``swmmanywhere.prepare_data._get_latest_s3_url`` derives
# the release id as ``Path(href).parent`` of the newest child link in the Overture
# STAC catalog. Those hrefs are absolute URLs now, so it caches
# "https:/stac.overturemaps.org/2026-08-19.0" in ./.cache/overture_release.json and
# every buildings download then fails on a non-existent S3 key. The cache is read
# before the catalog, so seeding it with a well-formed id (fresh timestamp) lets the
# unmodified package download buildings. Drop this once it is fixed upstream.
RELEASE_ID = re.compile(r"\d{4}-\d{2}-\d{2}\.\d+")
OVERTURE_CATALOG = "https://stac.overturemaps.org/catalog.json"


def _fetch_overture_catalog() -> dict:
    response = requests.get(OVERTURE_CATALOG, timeout=15)
    response.raise_for_status()
    return response.json()


def latest_overture_release(catalog: dict) -> str | None:
    """Newest release id among the catalog's child links, e.g. '2026-08-19.0'."""
    ids = []
    for link in catalog.get("links", []):
        m = (
            RELEASE_ID.search(str(link.get("href", "")))
            if link.get("rel") == "child"
            else None
        )
        if m:
            ids.append(m.group(0))
    return max(ids) if ids else None


def seed_overture_cache(
    workdir: Path, fetch_catalog: Callable[[], dict] = _fetch_overture_catalog
) -> str | None:
    """Write a well-formed Overture release id to the cache the package reads.

    Offline, the id embedded in an existing (possibly malformed) cache is reused.
    """
    cache = workdir / ".cache" / "overture_release.json"
    try:
        release = latest_overture_release(fetch_catalog())
    except Exception:  # noqa: BLE001 — offline is fine if a cache exists
        m = RELEASE_ID.search(cache.read_text()) if cache.exists() else None
        release = m.group(0) if m else None
    if release:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps({"release": release, "timestamp": datetime.now().isoformat()})
        )
    return release


class SubmitBody(BaseModel):
    """What the frontend posts; everything maps onto a SWMManywhere config key."""

    bbox: list[float] = Field(min_length=4, max_length=4)
    project: str = "swmmanywhere_web"
    parameter_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)
    graphfcn_list: list[str] | None = None
    run_model: bool = True
    run_settings: dict[str, Any] | None = None


@dataclass
class Task:
    """One build: its validated config, progress and outputs."""

    id: str
    config: dict
    config_text: str
    run_model: bool
    n_graphfcns: int
    state: str = "QUEUED"  # QUEUED | RUNNING | SUCCEEDED | FAILED
    stage: str = "Queued"
    progress_pct: float = 0.0
    log: list[str] = field(default_factory=list)
    error: str | None = None
    model_dir: Path | None = None
    results: Path | None = None
    preview: dict | None = None
    zip_path: Path | None = None


def bbox_area_km2(bbox: list[float]) -> float:
    """Approximate area of a lon/lat box, enough for the size guard."""
    minx, miny, maxx, maxy = bbox
    mid_lat = math.radians((miny + maxy) / 2)
    return (maxx - minx) * 111.32 * math.cos(mid_lat) * (maxy - miny) * 110.57


def validate_bbox(bbox: list[float], max_km2: float) -> list[float]:
    """Reject a malformed or oversized bounding box with a 422."""
    if not all(math.isfinite(v) for v in bbox):
        raise HTTPException(422, "bbox must be four finite numbers.")
    minx, miny, maxx, maxy = bbox
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise HTTPException(
            422,
            "bbox must be [min lon, min lat, max lon, max lat] in EPSG:4326 "
            "with min < max.",
        )
    area = bbox_area_km2(bbox)
    if area > max_km2:
        raise HTTPException(
            422,
            f"Bounding box is {area:.1f} km²; this server accepts at most "
            f"{max_km2:g} km².",
        )
    return bbox.copy()


def validate_overrides(overrides: dict[str, dict[str, Any]]) -> None:
    """Range-check overrides with the package's own pydantic parameter models.

    ``swmmanywhere.swmmanywhere`` applies overrides with ``setattr`` and only checks
    that the names exist (issue #379), so a web user could otherwise submit values
    outside the documented bounds without noticing.
    """
    groups = parameters.get_full_parameters()
    for category, values in overrides.items():
        if category not in groups:
            raise HTTPException(
                422,
                f"{category} is not a parameter group. "
                f"Must be one of {sorted(groups)}.",
            )
        model = groups[category]
        current = {k: v for k, v in model.model_dump().items() if v is not None}
        try:
            type(model).model_validate(current | values)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or category}: {e['msg']}"
                for e in exc.errors()
            )
            raise HTTPException(422, f"Invalid {category} override — {problems}")


def parameter_groups() -> list[dict]:
    """Parameter groups with fields, defaults and bounds, from the package."""
    groups = []
    for name, model in parameters.get_full_parameters().items():
        fields = [
            {
                "name": key,
                "type": prop.get("type", "unknown"),
                "default": prop.get("default"),
                "unit": prop.get("unit"),
                "description": " ".join((prop.get("description") or "").split()),
                "minimum": prop.get("minimum"),
                "maximum": prop.get("maximum"),
                "exclusiveMaximum": prop.get("exclusiveMaximum"),
            }
            for key, prop in model.model_json_schema()["properties"].items()
        ]
        groups.append(
            {"name": name, "doc": (type(model).__doc__ or "").strip(), "fields": fields}
        )
    return groups


def package_zip(task: Task) -> Path:
    """Zip the model directory; the rain file is added and referenced by bare name.

    SWMManywhere writes the absolute path of the rain file into ``[RAINGAGES]``, which
    only resolves on the machine that built the model. The copy inside the zip points
    at the file next to it so the download opens and runs in EPA SWMM as-is.
    """
    assert task.model_dir is not None
    zip_path = task.model_dir / "model_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(task.model_dir.iterdir()):
            if not f.is_file() or f == zip_path:
                continue
            if f.suffix != ".inp":
                z.write(f, f.name)
                continue
            lines = []
            for line in f.read_text().splitlines():
                m = RAINGAGE_FILE.search(line)
                if m:
                    path = Path(m.group(3) or m.group(4))
                    if path.is_absolute() and path.exists():
                        z.write(path, path.name)
                        line = (
                            line[: m.start()] + f'FILE "{path.name}"' + line[m.end() :]
                        )
                lines.append(line)
            z.writestr(f.name, "\n".join(lines) + "\n")
    return zip_path


def create_app(
    *,
    synthesis: Callable[[dict], tuple[Path, Any]] = run_synthesis,
    simulate: Callable[..., Any] = run_simulation,
    prepare: Callable[[Path], Any] = seed_overture_cache,
    workdir: Path | str | None = None,
    run_inline: bool = False,
) -> FastAPI:
    """Build the FastAPI app.

    ``synthesis``, ``simulate`` and ``prepare`` are injectable so tests run without
    downloads or SWMM; ``run_inline`` runs tasks synchronously in the request.
    """
    workdir = Path(
        workdir
        or os.environ.get("SWMMANYWHERE_WEB_WORKDIR")
        or Path(tempfile.gettempdir()) / "swmmanywhere_web"
    ).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # The package's bundled demo storm, copied out so the path never contains spaces
    # (SWMM cannot read a [RAINGAGES] FILE path with spaces) and so a future upload has
    # an obvious place to go. Same file the CLI falls back to.
    precipitation = workdir / "precipitation.dat"
    if not precipitation.exists():
        shutil.copyfile(DEFS / "storm.dat", precipitation)
    max_km2 = float(os.environ.get("SWMMANYWHERE_WEB_MAX_KM2", "5"))
    schema = yaml_load((DEFS / "schema.yml").read_text())
    demo = load_config(validation=False)  # package defaults (defs/demo_config.yml)

    app = FastAPI(title="SWMManywhere web")
    origins = [
        o.strip()
        for o in os.environ.get("SWMMANYWHERE_WEB_ORIGINS", "*").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )

    tasks: dict[str, Task] = {}
    lock = threading.Lock()
    # One build at a time: SWMManywhere shares the base_dir between builds (it reuses
    # downloads per bbox and numbers models), and loguru is process-global.
    executor = ThreadPoolExecutor(max_workers=1)

    def on_log(task: Task, message: str) -> None:
        line = message.rstrip()
        if not line:
            return
        with lock:
            task.log.append(line)
            del task.log[:-300]
            for needle, label, pct in STAGES:
                if needle in line:
                    task.stage, task.progress_pct = label, pct
            m = GRAPHFCN_DONE.search(line)
            if m:
                done = sum(1 for entry in task.log if GRAPHFCN_DONE.search(entry))
                task.stage = f"Graph function {done}/{task.n_graphfcns}: {m.group(1)}"
                task.progress_pct = 20 + 60 * done / max(task.n_graphfcns, 1)

    def run_task(task: Task) -> None:
        # SWMManywhere writes temporary DEM tiles and the WhiteboxTools download into
        # the current directory; keep those in the workdir.
        os.chdir(workdir)
        task.state, task.stage = "RUNNING", "Starting"
        handler = logger.add(
            lambda message: on_log(task, str(message)),
            filter=lambda record: True,
            format="{message}",
        )
        try:
            prepare(workdir)
            inp, _ = synthesis(task.config)
            if inp.suffix != ".inp":
                raise RuntimeError(
                    "SWMManywhere derived no pipes for this bounding box (the graph "
                    "has no edges). Try a larger or more built-up area."
                )
            task.model_dir = inp.parent
            (task.model_dir / "config.yml").write_text(task.config_text)
            if task.run_model:
                task.stage, task.progress_pct = "Running the SWMM simulation", 85
                results = simulate(inp, **task.config["run_settings"])
                task.results = task.model_dir / "results.parquet"
                results.to_parquet(task.results)
            task.stage, task.progress_pct, task.state = "Complete", 100, "SUCCEEDED"
        except Exception as exc:  # noqa: BLE001 — any failure must surface in the UI
            task.state, task.stage = "FAILED", "Failed"
            task.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()  # full trace in the server log
        finally:
            logger.remove(handler)

    def get_task(task_id: str) -> Task:
        task = tasks.get(task_id)
        if task is None:
            raise HTTPException(404, "Unknown task.")
        return task

    @app.get("/api/v1/healthz")
    def healthz():
        return {
            "status": "ok",
            "swmmanywhere": importlib.metadata.version("swmmanywhere"),
        }

    @app.get("/api/v1/defaults")
    def defaults():
        return {
            "version": importlib.metadata.version("swmmanywhere"),
            "parameters": parameter_groups(),
            "graphfcn_list": demo["graphfcn_list"],
            "graphfcns": {
                name: (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
                for name, fn in sorted(graphfcns.items())
            },
            "run_settings": demo["run_settings"],
            "storevars": schema["properties"]["run_settings"]["properties"][
                "storevars"
            ]["items"]["enum"],
            "max_area_km2": max_km2,
        }

    @app.post("/api/v1/tasks", status_code=202)
    def submit(body: SubmitBody):
        bbox = validate_bbox(body.bbox, max_km2)
        if not PROJECT_RE.match(body.project):
            raise HTTPException(
                422, "project must be 1–64 letters, digits, '-' or '_'."
            )
        validate_overrides(body.parameter_overrides)

        # The config a CLI user would write, then SWMManywhere's own validation.
        config: dict[str, Any] = {
            "base_dir": str(workdir),
            "project": body.project,
            "bbox": bbox,
            "run_model": body.run_model,
            "run_settings": {**demo["run_settings"], **(body.run_settings or {})},
            "address_overrides": {"precipitation": str(precipitation)},
        }
        overrides = {k: v for k, v in body.parameter_overrides.items() if v}
        if overrides:
            config["parameter_overrides"] = overrides
        if body.graphfcn_list is not None:
            config["graphfcn_list"] = body.graphfcn_list
        config_text = yaml_dump(config)

        task_id = uuid.uuid4().hex[:12]
        task_dir = workdir / "tasks" / task_id
        task_dir.mkdir(parents=True)
        config_path = task_dir / "config.yml"
        config_path.write_text(config_text)
        try:
            loaded = load_config(config_path)
        except SchemaError as exc:
            raise HTTPException(
                422, f"SWMManywhere rejected the configuration: {exc.message}"
            )
        except (ValueError, TypeError, AssertionError, FileNotFoundError) as exc:
            raise HTTPException(422, f"SWMManywhere rejected the configuration: {exc}")
        # The simulation is run here (see run_task) so its results can be served.
        loaded["run_model"] = False

        task = Task(
            id=task_id,
            config=loaded,
            config_text=config_text,
            run_model=body.run_model,
            n_graphfcns=len(loaded.get("graphfcn_list") or demo["graphfcn_list"]),
        )
        with lock:
            tasks[task_id] = task
        if run_inline:
            run_task(task)
        else:
            executor.submit(run_task, task)
        return {"task_id": task_id, "status": task.state}

    @app.get("/api/v1/tasks/{task_id}")
    def status(task_id: str):
        task = get_task(task_id)
        with lock:
            return {
                "state": task.state,
                "stage": task.stage,
                "progress_pct": task.progress_pct,
                "log": task.log[-8:],
                "error": task.error,
            }

    @app.get("/api/v1/tasks/{task_id}/config")
    def config_yaml(task_id: str):
        return PlainTextResponse(get_task(task_id).config_text, media_type="text/yaml")

    @app.get("/api/v1/tasks/{task_id}/preview")
    def preview(task_id: str):
        task = get_task(task_id)
        if task.state != "SUCCEEDED" or task.model_dir is None:
            raise HTTPException(409, "Preview not ready.")
        if task.preview is None:
            task.preview = model_geojson(task.model_dir)
        return task.preview

    @app.get("/api/v1/tasks/{task_id}/result")
    def result(task_id: str):
        task = get_task(task_id)
        if task.state != "SUCCEEDED" or task.model_dir is None:
            raise HTTPException(409, "Result not ready.")
        if task.zip_path is None:
            task.zip_path = package_zip(task)
        return FileResponse(
            str(task.zip_path),
            media_type="application/zip",
            filename="swmmanywhere_model.zip",
        )

    @app.get("/api/v1/tasks/{task_id}/timeseries")
    def timeseries(task_id: str, id: str):
        task = get_task(task_id)
        if task.state != "SUCCEEDED":
            raise HTTPException(409, "Results not ready.")
        if task.results is None:
            raise HTTPException(
                409, "This build did not run the model (run_model was off)."
            )
        series = element_timeseries(task.results, id)
        if not series:
            raise HTTPException(404, f"No simulation results stored for {id}.")
        return series

    # Serve the built frontend (npm run build) from the same origin when present.
    if DIST.is_dir():
        app.mount("/", StaticFiles(directory=str(DIST), html=True), name="frontend")

    return app


app = create_app()
