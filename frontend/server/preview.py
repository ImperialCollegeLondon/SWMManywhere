"""Turn a SWMManywhere model directory into GeoJSON and per-element time series.

Reads the same nodes / edges / subcatchments files ``swmmanywhere.utilities.plot_map``
reads and identifies outfalls the same way (``plot_basic``), so the map matches what the
package's own folium plot would show.
"""

from __future__ import annotations

import math
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import mapping

from swmmanywhere.utilities import read_df


def _first(model_dir: Path, stem: str) -> Path | None:
    return next(iter(sorted(model_dir.glob(f"{stem}.*"))), None)


def _clean(value: Any) -> Any:
    """JSON-safe scalar: numpy -> python, NaN/inf -> None, anything odd -> str."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    with suppress(TypeError, ValueError):
        if pd.isna(value):
            return None
    return str(value)


def _features(df: pd.DataFrame, kind: str, **extra: pd.Series) -> list[dict]:
    """One GeoJSON feature per row, every non-geometry column as a property."""
    columns = [c for c in df.columns if c != "geometry"]
    features = []
    for i, (_, row) in enumerate(df.iterrows()):
        props = {c: _clean(row[c]) for c in columns}
        props["kind"] = kind
        for name, series in extra.items():
            props[name] = _clean(series.iloc[i])
        features.append(
            {"type": "Feature", "geometry": mapping(row.geometry), "properties": props}
        )
    return features


def model_geojson(model_dir: Path) -> dict:
    """GeoJSON (EPSG:4326) of a synthesised model with a ``kind`` per feature.

    kinds: ``junction``, ``outfall``, ``conduit``, ``subcatchment``. Feature ``id``
    is the SWMM object name so it matches ``results.parquet`` (subcatchments are
    named ``<node id>-sub`` in the .inp, see ``post_processing.synthetic_write``).
    """
    nodes_fid, edges_fid = _first(model_dir, "nodes"), _first(model_dir, "edges")
    if nodes_fid is None or edges_fid is None:
        raise FileNotFoundError("No nodes or edges found in model directory.")
    nodes = read_df(nodes_fid).to_crs(4326)
    edges = read_df(edges_fid).to_crs(4326)

    node_ids = nodes["id"].astype(str)
    # Same rule as swmmanywhere.utilities.plot_basic.
    if "outfall" in nodes.columns:
        is_outfall = node_ids == nodes["outfall"].astype(str)
    else:
        is_outfall = ~node_ids.isin(edges["u"].astype(str))

    features: list[dict] = []
    subs_fid = _first(model_dir, "subcatchments")
    if subs_fid is not None:
        subs = read_df(subs_fid).to_crs(4326)
        outlet = subs["id"].astype(str)
        subs = subs.assign(id=outlet + "-sub")
        features += _features(subs, "subcatchment", outlet=outlet)

    edges = edges.assign(
        id=edges["id"].astype(str), u=edges["u"].astype(str), v=edges["v"].astype(str)
    )
    features += _features(edges, "conduit")
    nodes = nodes.assign(id=node_ids)
    features += _features(nodes.loc[~is_outfall], "junction")
    features += _features(nodes.loc[is_outfall], "outfall")
    return {"type": "FeatureCollection", "features": features}


def element_timeseries(
    results_path: Path, element_id: str
) -> dict[str, dict[str, list]]:
    """``{variable: {dates: [...iso...], values: [...]}}`` for one SWMM object."""
    df = pd.read_parquet(results_path)
    sel = df[df["id"].astype(str) == element_id]
    out: dict[str, dict[str, list]] = {}
    for variable, grp in sel.groupby("variable"):
        grp = grp.sort_values("date")
        out[str(variable)] = {
            "dates": [pd.Timestamp(d).isoformat() for d in grp["date"]],
            "values": [float(v) for v in grp["value"]],
        }
    return out
