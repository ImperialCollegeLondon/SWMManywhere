"""Contract tests for frontend/server using SWMManywhere's bundled Bellinge fixtures.

No network and no SWMM run: synthesis and simulation are injected fakes that write
the package's own test data (src/swmmanywhere/defs/bellinge_small_*) into the model
directory. Config validation, GeoJSON conversion, packaging and time-series
extraction are the real code.
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import swmmanywhere
from server.app import create_app, seed_overture_cache
from swmmanywhere.utilities import load_graph, save_graph_to_features

DEFS = Path(swmmanywhere.__file__).parent / "defs"
CATALOG = {
    "links": [
        {"rel": "self", "href": "https://stac.overturemaps.org/catalog.json"},
        {
            "rel": "child",
            "title": "Latest Overture Release",
            "href": "https://stac.overturemaps.org/2026-08-19.0/catalog.json",
        },
        {
            "rel": "child",
            "title": "2026-07-22.0 Overture Release",
            "href": "https://stac.overturemaps.org/2026-07-22.0/catalog.json",
        },
    ]
}
BBOX = [10.290, 55.318, 10.300, 55.324]  # small; the fake synthesis ignores it
NODE = "G72F820"  # a node of bellinge_small_graph.json


def fake_synthesis(config: dict):
    """Write the Bellinge fixtures where SWMManywhere would put a model."""
    model_dir = Path(config["base_dir"]) / config["project"] / "bbox_1" / "model_1"
    model_dir.mkdir(parents=True, exist_ok=True)
    graph = load_graph(DEFS / "bellinge_small_graph.json")
    save_graph_to_features(
        graph,
        model_dir / "nodes.geojson",
        model_dir / "edges.geojson",
        graph.graph["crs"],
    )
    shutil.copy(
        DEFS / "bellinge_small_subcatchments.geojson",
        model_dir / "subcatchments.geojson",
    )
    rain = config["address_overrides"]["precipitation"]
    inp = model_dir / "model_1.inp"
    inp.write_text(
        "[TITLE]\nfake\n\n[RAINGAGES]\n;;Name Format Interval SCF Source\n"
        f'1 INTENSITY 0:15 1.0 FILE "{rain}" 1 MM\n\n[JUNCTIONS]\n'
    )
    return inp, None


def fake_simulate(
    model: Path, reporting_iters: int, duration: int, storevars: list[str]
):
    """Return a short flooding series for one node instead of running SWMM."""
    t0 = datetime(2000, 1, 1)
    rows = [
        {
            "date": t0 + timedelta(minutes=15 * i),
            "value": float(i),
            "variable": "flooding",
            "id": NODE,
        }
        for i in range(8)
    ]
    return pd.DataFrame(rows)


def no_edges_synthesis(config: dict):
    """Mimic SWMManywhere returning the graph file when no pipes were derived."""
    model_dir = Path(config["base_dir"]) / config["project"] / "bbox_1" / "model_1"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "graph.parquet", None


@pytest.fixture
def client(tmp_path):
    """A test client whose builds run inline on the Bellinge fixtures."""
    app = create_app(
        synthesis=fake_synthesis,
        simulate=fake_simulate,
        prepare=lambda _: None,
        workdir=tmp_path,
        run_inline=True,
    )
    return TestClient(app)


def test_overture_cache_is_seeded_with_a_well_formed_release_id(tmp_path):
    """The cache the package reads gets a plain release id, online or offline."""
    cache = tmp_path / ".cache" / "overture_release.json"
    assert (
        seed_overture_cache(tmp_path, fetch_catalog=lambda: CATALOG) == "2026-08-19.0"
    )
    assert json.loads(cache.read_text())["release"] == "2026-08-19.0"
    # Offline: the id embedded in the package's malformed cache is reused.
    cache.write_text(
        json.dumps(
            {
                "release": "https:/stac.overturemaps.org/2026-07-22.0",
                "timestamp": "2026-09-01T00:00:00",
            }
        )
    )

    def offline():
        raise ConnectionError("no network")

    assert seed_overture_cache(tmp_path, fetch_catalog=offline) == "2026-07-22.0"
    assert json.loads(cache.read_text())["release"] == "2026-07-22.0"


def test_defaults_come_from_the_package(client):
    """Parameter schema, graphfcn list and run settings are read from the package."""
    d = client.get("/api/v1/defaults").json()
    groups = {g["name"]: g for g in d["parameters"]}
    assert {
        "subcatchment_derivation",
        "outfall_derivation",
        "topology_derivation",
        "hydraulic_design",
    } <= set(groups)
    lane_width = next(
        f
        for f in groups["subcatchment_derivation"]["fields"]
        if f["name"] == "lane_width"
    )
    assert (
        lane_width["default"] == 3.5
        and lane_width["unit"] == "m"
        and lane_width["minimum"] == 2.0
    )
    assert (
        d["graphfcn_list"][0] == "assign_id"
        and d["graphfcn_list"][-2] == "fix_geometries"
    )
    assert "pipe_by_pipe" in d["graphfcns"]
    assert "flooding" in d["storevars"] and d["run_settings"]["duration"] == 86400


@pytest.mark.parametrize(
    "bbox, fragment",
    [([1.0, 2.0, 0.5, 3.0], "min < max"), ([0.0, 0.0, 1.0, 1.0], "km²")],
)
def test_bad_bbox_is_rejected(client, bbox, fragment):
    """Reversed corners and oversized boxes are refused before any build starts."""
    r = client.post("/api/v1/tasks", json={"bbox": bbox})
    assert r.status_code == 422 and fragment in r.json()["detail"]


def test_parameter_overrides_are_range_checked(client):
    """Out-of-range values and unknown groups are refused with the field named."""
    r = client.post(
        "/api/v1/tasks",
        json={"bbox": BBOX, "parameter_overrides": {"hydraulic_design": {"max_fr": 5}}},
    )
    assert r.status_code == 422 and "max_fr" in r.json()["detail"]
    r = client.post(
        "/api/v1/tasks", json={"bbox": BBOX, "parameter_overrides": {"nope": {"x": 1}}}
    )
    assert r.status_code == 422 and "nope" in r.json()["detail"]


def test_unknown_graphfcn_is_rejected_by_swmmanywhere(client):
    """The package's own graphfcn validation runs on the submitted list."""
    r = client.post(
        "/api/v1/tasks", json={"bbox": BBOX, "graphfcn_list": ["assign_id", "teleport"]}
    )
    assert r.status_code == 422 and "teleport" in r.json()["detail"]


def test_build_preview_results_and_package(client, tmp_path):
    """A build yields status, config, GeoJSON preview, time series and a zip."""
    r = client.post(
        "/api/v1/tasks",
        json={
            "bbox": BBOX,
            "project": "bellinge",
            "parameter_overrides": {"outfall_derivation": {"method": "withtopo"}},
            "run_settings": {
                "duration": 3600,
                "reporting_iters": 10,
                "storevars": ["flooding"],
            },
        },
    )
    assert r.status_code == 202, r.text
    task_id = r.json()["task_id"]

    status = client.get(f"/api/v1/tasks/{task_id}").json()
    assert status["state"] == "SUCCEEDED", status
    assert status["progress_pct"] == 100

    cfg = client.get(f"/api/v1/tasks/{task_id}/config").text
    assert "bbox:" in cfg and "withtopo" in cfg and "duration: 3600" in cfg
    assert (tmp_path / "bellinge" / "bbox_1" / "model_1" / "config.yml").exists()

    fc = client.get(f"/api/v1/tasks/{task_id}/preview").json()
    kinds = {f["properties"]["kind"] for f in fc["features"]}
    assert kinds == {"junction", "outfall", "conduit", "subcatchment"}
    outfalls = [f for f in fc["features"] if f["properties"]["kind"] == "outfall"]
    assert len(outfalls) >= 1
    lon, lat = outfalls[0]["geometry"]["coordinates"]
    assert 10 < lon < 11 and 55 < lat < 56  # reprojected from EPSG:32632 to lon/lat
    sub = next(f for f in fc["features"] if f["properties"]["kind"] == "subcatchment")
    assert sub["properties"]["id"].endswith("-sub") and sub["properties"]["outlet"]
    node = next(f for f in fc["features"] if f["properties"]["id"] == NODE)
    assert node["properties"]["surface_elevation"] == pytest.approx(26.129, abs=1e-3)

    series = client.get(
        f"/api/v1/tasks/{task_id}/timeseries", params={"id": NODE}
    ).json()
    assert list(series) == ["flooding"] and series["flooding"]["values"][-1] == 7.0
    assert (
        client.get(
            f"/api/v1/tasks/{task_id}/timeseries", params={"id": "nowhere"}
        ).status_code
        == 404
    )

    z = zipfile.ZipFile(
        io.BytesIO(client.get(f"/api/v1/tasks/{task_id}/result").content)
    )
    names = set(z.namelist())
    assert {
        "model_1.inp",
        "nodes.geojson",
        "edges.geojson",
        "config.yml",
        "results.parquet",
        "precipitation.dat",
    } <= names
    assert 'FILE "precipitation.dat"' in z.read("model_1.inp").decode()


def test_no_edges_fails_with_a_clear_message(tmp_path):
    """A graph without edges ends as FAILED with an explanation, not a traceback."""
    app = create_app(
        synthesis=no_edges_synthesis,
        simulate=fake_simulate,
        prepare=lambda _: None,
        workdir=tmp_path,
        run_inline=True,
    )
    client = TestClient(app)
    task_id = client.post(
        "/api/v1/tasks", json={"bbox": BBOX, "run_model": False}
    ).json()["task_id"]
    status = client.get(f"/api/v1/tasks/{task_id}").json()
    assert status["state"] == "FAILED" and "no pipes" in status["error"]
    assert client.get(f"/api/v1/tasks/{task_id}/preview").status_code == 409
