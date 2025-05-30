"""Utilities for YAML save/load.

Author: cheginit
"""

from __future__ import annotations

import base64
import functools
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import folium
import geopandas as gpd
import pandas as pd
import yaml
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    SafeDumper = yaml.SafeDumper
    from yaml.nodes import Node
else:
    Node = Any
    SafeDumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)

yaml_load = functools.partial(
    yaml.load, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)
)


class PathDumper(SafeDumper):
    """A dumper that can represent pathlib.Path objects as strings."""

    def represent_data(self, data: Any) -> Node:
        """Represent data."""
        if isinstance(data, Path):
            return self.represent_scalar("tag:yaml.org,2002:str", str(data))
        return super().represent_data(data)


def yaml_dump(o: Any, stream: Any = None, **kwargs: Any) -> str:
    """Dump YAML.

    Notes:
    -----
    When python/mypy#1484 is solved, this can be ``functools.partial``
    """
    return yaml.dump(
        o,
        Dumper=PathDumper,
        stream=stream,
        default_flow_style=False,
        indent=2,
        sort_keys=False,
        **kwargs,
    )


def plot_basic(nodes_path: Path, edges_path: Path):
    """Create a basic map with nodes and edges.

    Args:
        nodes_path (Path): The path to the nodes file.
        edges_path (Path): The path to the edges file.

    Returns:
        folium.Map: The folium map.
    """
    # Load and inspect results
    nodes = gpd.read_file(nodes_path)
    edges = gpd.read_file(edges_path)

    # Convert to EPSG 4326 for plotting
    nodes = nodes.to_crs(4326)
    edges = edges.to_crs(4326)

    # Identify outfalls
    if "outfall" in nodes.columns:
        outfall = nodes.id == nodes.outfall
    else:
        outfall = ~nodes.id.isin(edges.u.astype(str))

    # Plot on map
    m = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()], zoom_start=16
    )
    folium.GeoJson(edges, color="black", weight=1).add_to(m)
    folium.GeoJson(
        nodes.loc[~outfall],
        marker=folium.CircleMarker(
            radius=3,  # Radius in metres
            weight=0,  # outline weight
            fill_color="black",
            fill_opacity=1,
        ),
    ).add_to(m)

    folium.GeoJson(
        nodes.loc[outfall],
        marker=folium.CircleMarker(
            radius=3,  # Radius in metres
            weight=0,  # outline weight
            fill_color="red",
            fill_opacity=1,
        ),
    ).add_to(m)

    # Display the map
    return m


def plot_clickable(nodes_path: Path, edges_path: Path, results_path: Path):
    """Create a clickable map with nodes, edges and results.

    Args:
        nodes_path (Path): The path to the nodes file.
        edges_path (Path): The path to the edges file.
        results_path (Path): The path to the results file.

    Returns:
        folium.Map: The folium map.
    """
    # Load and inspect results
    nodes = gpd.read_file(nodes_path)
    edges = gpd.read_file(edges_path)
    df = pd.read_parquet(results_path)
    df.id = df.id.astype(str)
    floods = df.loc[df.variable == "flooding"].groupby("id")
    flows = df.loc[df.variable == "flow"].groupby("id")

    if "outfall" not in nodes.columns:
        nodes["outfall"] = None
        nodes.loc[~nodes.id.isin(edges.u.astype(str)), "outfall"] = nodes.id

    # Convert to EPSG 4326 for plotting
    nodes = nodes.to_crs(4326).set_index("id")
    edges = edges.to_crs(4326).set_index("id")

    # Create map
    m = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()], zoom_start=16
    )

    # Add edges
    for edge, row in edges.iterrows():
        # Create a plot for each edge
        grp = flows.get_group(str(edge))
        f, ax = plt.subplots(figsize=(4, 3))
        grp.set_index("date").value.plot(ylabel="flow (l/s)", title=edge, ax=ax)
        img = BytesIO()
        f.tight_layout()
        f.savefig(img, format="png", dpi=94)
        plt.close(f)

        # Convert plot to base64
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_base64}">'

        # Add edge to map
        folium.PolyLine(
            [[c[1], c[0]] for c in row.geometry.coords],
            color="black",
            weight=2,
            popup=folium.Popup(img_html),
        ).add_to(m)

    # Add nodes
    for node, row in nodes.iterrows():
        grp = floods.get_group(str(node))
        f, ax = plt.subplots(figsize=(4, 3))
        grp.set_index("date").value.plot(ylabel="flooding (l)", title=node, ax=ax)
        img = BytesIO()
        f.tight_layout()
        f.savefig(img, format="png", dpi=94)
        plt.close(f)
        img.seek(0)
        img_base64 = base64.b64encode(img.read()).decode()
        img_html = f'<img src="data:image/png;base64,{img_base64}">'
        if row.get("outfall") == node:
            color = "red"
        else:
            color = "black"
        folium.CircleMarker(
            [nodes.loc[node].geometry.y, nodes.loc[node].geometry.x],
            color=color,
            radius=3,
            weight=0,
            fill_color=color,
            fill_opacity=1,
            popup=folium.Popup(img_html),
        ).add_to(m)

    return m


def plot_map(model_dir: Path):
    """Create a map from a model directory.

    Args:
        model_dir (Path): The directory containing the model files.

    Returns:
        folium.Map: The folium map.
    """
    nodes_fids = list(model_dir.glob("nodes.*"))
    if not any(nodes_fids):
        raise FileNotFoundError("No nodes or edges found in model directory.")
    nodes = nodes_fids[0]

    edges_fids = list(model_dir.glob("edges.*"))
    if not any(edges_fids):
        raise FileNotFoundError("No edges found in model directory.")
    edges = edges_fids[0]

    results_fids = list(model_dir.glob("*results.*"))

    if results_fids:
        results = results_fids[0]
        m = plot_clickable(nodes, edges, results)
    else:
        m = plot_basic(nodes, edges)
    return m
