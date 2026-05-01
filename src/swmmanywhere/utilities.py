"""Utilities for YAML save/load, file I/O, and graph operations.

Author: cheginit
"""

from __future__ import annotations

import base64
import functools
import json
import warnings
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import folium
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
import yaml
from matplotlib import pyplot as plt
from shapely import geometry as sgeom

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
    nodes = read_df(nodes_path)
    edges = read_df(edges_path)

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
    nodes = read_df(nodes_path)
    edges = read_df(edges_path)
    df = read_df(results_path)
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


def read_df(fid: Path) -> pd.DataFrame | gpd.GeoDataFrame:
    """Read a DataFrame from a file.

    Read a DataFrame or GeoDataFrame from a file. The file type is determined by the
    file extension.

    Args:
        fid (Path): Path to the file.

    Returns:
        DataFrame or GeoDataFrame: The loaded DataFrame.
    """
    if fid.suffix == ".geoparquet":
        return gpd.read_parquet(fid)
    elif fid.suffix == ".parquet":
        return pd.read_parquet(fid)
    elif fid.suffix == ".geojson":
        return gpd.read_file(fid)
    elif fid.suffix == ".json":
        return pd.read_json(fid)
    else:
        raise ValueError(f"Unsupported file extension: {fid.suffix}")


def write_df(df: pd.DataFrame | gpd.GeoDataFrame, fid: Path):
    """Write a DataFrame to a file.

    Write a DataFrame to a file. The file type is determined by the file
    extension.

    Args:
        df (DataFrame): DataFrame to write to a file.
        fid (Path): Path to the file.
    """
    if fid.suffix in (".geoparquet", ".parquet"):
        df.to_parquet(fid)
    elif fid.suffix in (".geojson", ".json"):
        if isinstance(df, gpd.GeoDataFrame):
            df.to_file(fid, driver="GeoJSON")
        else:
            df.to_json(fid)
    else:
        raise ValueError(f"Unsupported file extension: {fid.suffix}")


def _serialize_line_string(obj):
    """Serialize a LineString to WKT format for JSON."""
    if isinstance(obj, shapely.LineString):
        return obj.wkt
    return obj


def save_graph(G: nx.Graph, fid: Path) -> None:
    """Save a graph to a file.

    Currently only supports JSON format.

    Args:
        G (nx.Graph): A graph
        fid (Path): The path to the file
    """
    # Save as JSON format (original implementation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        json_data = nx.node_link_data(G, edges="links")

    with fid.open("w") as file:
        json.dump(json_data, file, default=_serialize_line_string)


def load_graph(fid: Path) -> nx.Graph:
    """Load a graph from a file saved with save_graph.

    Supports both JSON and parquet formats. For parquet format, expects
    nodes and edges files with a metadata JSON file for graph-level attributes.

    Args:
        fid (Path): The path to the file

    Returns:
        G (nx.Graph): A graph
    """
    # Load from JSON format (original implementation)
    json_data = json.loads(fid.read_text())
    G = nx.node_link_graph(json_data, directed=True, edges="links")
    for u, v, data in G.edges(data=True):
        if "geometry" in data:
            geometry_coords = data["geometry"]
            line_string = shapely.LineString(shapely.wkt.loads(geometry_coords))
            data["geometry"] = line_string
        else:
            # Use a straight line between the nodes
            data["geometry"] = shapely.LineString(
                [
                    (G.nodes[u]["x"], G.nodes[u]["y"]),
                    (G.nodes[v]["x"], G.nodes[v]["y"]),
                ]
            )
    return G


def _make_json_serializable(value):
    """Convert a value to a JSON-serializable type.

    Args:
        value: The value to convert.

    Returns:
        A JSON-serializable value.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_make_json_serializable(v) for v in value]
    elif isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (set, frozenset)):
        return list(value)
    elif isinstance(value, dict):
        return {k: _make_json_serializable(v) for k, v in value.items()}
    elif hasattr(value, "__dict__"):
        return str(value)

    return value


def nodes_to_features(G: nx.Graph):
    """Convert a graph to a GeoJSON node feature collection.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        list: A list of GeoJSON features.
    """
    features = []
    for node, data in G.nodes(data=True):
        # Make properties JSON-serializable
        properties = {"id": node}
        for key, value in data.items():
            properties[key] = _make_json_serializable(value)
        feature = {
            "type": "Feature",
            "geometry": sgeom.mapping(sgeom.Point(data["x"], data["y"])),
            "properties": properties,
        }
        features.append(feature)
    return features


def edges_to_features(G: nx.Graph):
    """Convert a graph to a GeoJSON edge feature collection.

    Args:
        G (nx.Graph): The input graph.

    Returns:
        list: A list of GeoJSON features.
    """
    features = []
    for u, v, data in G.edges(data=True):
        data_copy = data.copy()
        if "geometry" not in data_copy:
            geom = None
        else:
            geom = sgeom.mapping(data_copy["geometry"])
            del data_copy["geometry"]
        # Make properties JSON-serializable
        properties = {"u": u, "v": v}
        for key, value in data_copy.items():
            properties[key] = _make_json_serializable(value)
        feature = {
            "type": "Feature",
            "geometry": geom,
            "properties": properties,
        }
        features.append(feature)
    return features


def _cast_to_str(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cast list-like columns to strings for parquet compatibility.

    Args:
        df (gpd.GeoDataFrame): The GeoDataFrame to process.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with list-like columns cast to strings.
    """
    for col in df.columns:
        # check if any value is list-like or string
        has_iter = (
            df[col]
            .apply(lambda x: isinstance(x, (list, tuple, set, np.ndarray, str)))
            .any()
        )

        if has_iter:
            df[col] = df[col].astype(str)

    return df


def save_graph_to_features(graph: nx.Graph, fid_nodes: Path, fid_edges: Path, crs: str):
    """Write a graph to GeoJSON or GeoParquet feature files.

    Args:
        graph (nx.Graph): The input graph.
        fid_nodes (Path): The filepath to save the nodes file.
        fid_edges (Path): The filepath to save the edges file.
        crs (str): The CRS of the graph.
    """
    graph = graph.copy()
    nodes = nodes_to_features(graph)
    edges = edges_to_features(graph)

    nodes = gpd.GeoDataFrame(
        [x["properties"] for x in nodes],
        geometry=[sgeom.Point(x["geometry"]["coordinates"]) for x in nodes],
        crs=crs,
    )
    edges = gpd.GeoDataFrame(
        [x["properties"] for x in edges],
        geometry=[sgeom.LineString(x["geometry"]["coordinates"]) for x in edges],
        crs=crs,
    )

    if fid_nodes.suffix == ".geoparquet":
        _cast_to_str(nodes).to_parquet(fid_nodes)
        _cast_to_str(edges).to_parquet(fid_edges)
    elif fid_nodes.suffix == ".geojson":
        nodes.to_file(fid_nodes, driver="GeoJSON")
        edges.to_file(fid_edges, driver="GeoJSON")
    else:
        raise ValueError(f"Unsupported file type: {fid_nodes.suffix}")


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
