"""Post processing module for SWMManywhere.

A module containing functions to format and write processed data into SWMM .inp
files.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from swmmanywhere.filepaths import FilePaths
from swmmanywhere.logging import logger


def synthetic_write(addresses: FilePaths):
    """Load synthetic data and write to SWMM input file.

    Loads nodes, edges and subcatchments from synthetic data, assumes that
    these are all located in `addresses`. Fills in appropriate default values
    for many SWMM parameters. More parameters are available to edit (see
    defs/swmm_conversion.yml). Identifies outfalls and automatically ensures
    that they have only one link to them (as is required by SWMM). Formats
    (with format_to_swmm_dict) and writes (with data_dict_to_inp) the data to
    a SWMM input (.inp) file.

    Args:
        addresses (FilePaths): A dictionary of file paths.
    """
    nodes = gpd.read_file(addresses.model_paths.nodes)
    edges = gpd.read_file(addresses.model_paths.edges)

    if addresses.model_paths.subcatchments.suffix == ".geoparquet":
        subs = gpd.read_parquet(addresses.model_paths.subcatchments)
    else:
        subs = gpd.read_file(addresses.model_paths.subcatchments)
    subs = subs.loc[subs.id.isin(nodes.id)]

    # Extract SWMM relevant data
    edges = edges[["id", "u", "v", "diameter", "length"]]
    nodes = nodes[["id", "x", "y", "chamber_floor_elevation", "surface_elevation"]]
    subs = subs[["id", "geometry", "area", "slope", "width", "rc"]]

    # Nodes
    nodes["id"] = nodes["id"].astype(str)
    nodes["max_depth"] = nodes.surface_elevation - nodes.chamber_floor_elevation
    nodes["surcharge_depth"] = 0
    nodes["flooded_area"] = 100  # TODO arbitrary... not sure how to calc this
    nodes["manhole_area"] = 0.5

    # Subs
    subs["id"] = subs["id"].astype(str)
    subs["subcatchment"] = subs["id"] + "-sub"
    subs["rain_gage"] = 1  # TODO revise when revising storms
    subs["area"] /= 10000  # convert to ha

    # Edges
    edges["u"] = edges["u"].astype(str)
    edges["v"] = edges["v"].astype(str)
    edges["roughness"] = 0.01
    edges["capacity"] = 1e10  # capacity in swmm is a hard limit

    # Outfalls (create new nodes that link to the stores connected tothem
    outfalls = nodes.loc[~nodes.id.isin(edges.u)].copy()
    outfalls["id"] = outfalls["id"] + "_outfall"

    # Reduce elevation to ensure flow
    outfalls["chamber_floor_elevation"] -= 5
    outfalls["x"] -= 50
    outfalls["y"] -= 50

    # Link stores to outfalls
    new_edges = pd.DataFrame(columns=edges.columns, index=range(len(outfalls)))
    new_edges["u"] = outfalls["id"].str.replace("_outfall", "").values
    new_edges["v"] = outfalls["id"].values
    new_edges["id"] = [f"{u}-{v}" for u, v in zip(new_edges["u"], new_edges["v"])]
    new_edges["diameter"] = 15  # TODO .. big pipe to enable all outfall...
    new_edges["length"] = (50**2 + 50**2) ** 0.5
    new_edges["roughness"] = 0.01
    new_edges["capacity"] = 1e10

    # Append new edges
    edges = pd.concat([edges, new_edges], ignore_index=True)

    # Create event
    # TODO will need some updating if multiple rain gages
    # TODO automatically match units to storm.csv?
    event = {
        "name": "1",
        "unit": "mm",
        "interval": "00:05",  # hh:mm
        "fid": str(addresses.bbox_paths.precipitation),
    }

    # Locate raingage(s) on the map
    symbol = {
        "x": nodes.x.min(),
        "y": nodes.y.min(),
        "name": "1",  # matches event name(s)
    }

    # Template SWMM input file
    existing_input_file = Path(__file__).parent / "defs" / "basic_drainage_all_bits.inp"

    # Format to dict
    data_dict = format_to_swmm_dict(nodes, outfalls, edges, subs, event, symbol)

    # Write new input file
    data_dict_to_inp(data_dict, existing_input_file, addresses.model_paths.inp)


def overwrite_section(data: np.ndarray, section: str, fid: Path):
    """Overwrite a section of a SWMM .inp file with new data.

    Args:
        data (np.ndarray): Data array to be written to the SWMM .inp file.
        section (str): Section of the SWMM .inp file to be overwritten.
        fid (Path): File path to the SWMM .inp file.

    Example:
        data = np.array([
                ['1', '1', '1', '1.166', '100', '500', '0.5', '0', 'empty'],
                ['2', '1', '1', '1.1', '100', '500', '0.5', '0', 'empty'],
                ['3', '1', '1', '2', '100', '400', '0.5', '0', 'empty']])
        fid = 'my_pre_existing_swmm_input_file.inp'
        section = '[SUBCATCHMENTS]'
        overwrite_section(data, section, fid)
    """
    # Read the existing SWMM .inp file
    with fid.open("r") as infile:
        lines = infile.readlines()

    # Create a flag to indicate whether we are within the target section
    within_target_section = False

    # Iterate through the lines and make modifications as needed
    with fid.open("w") as outfile:
        for ix, line in enumerate(lines):
            if line.strip() != section and re.search(r"\[.*?\]", line):
                within_target_section = False

            if line.strip() != section and not within_target_section:
                outfile.write(line)  # Write lines outside the target section

            if line.strip() != section:
                continue

            within_target_section = True
            outfile.write(line)  # Write the start section header

            # Write headers
            i = 1
            while lines[ix + i][0] == ";":
                outfile.write(lines[ix + i])  # Write column headers
                i += 1

            example_line = lines[ix + i]
            # print('example_line {1}: {0}'.format(
            #     example_line.replace('\n', ''), section))
            # print('note - this line must have at least as many column')
            # print('entries as all other rows in this section\n')
            pattern = r"(\s+)"

            # Find all matches of the pattern in the input line
            matches = re.findall(pattern, example_line)

            # Calculate the space counts by taking the length of each match
            space_counts = [
                len(x) + len(y) for x, y in zip(matches, example_line.split())
            ]
            if not space_counts:
                if data.shape[0] != 0:
                    logger.warning("no template for data?")
                continue

            space_counts[-1] -= 1
            new_text = ""
            if data is None:
                continue

            for i, row in enumerate(data):
                if section == "[CONTROLS]":
                    new_text = row
                    continue

                formatted_row = []
                for x, y in zip(row, space_counts):
                    formatted_value = "{0:<{1}}".format(x, max(y, len(str(x)) + 1))
                    formatted_row.append(formatted_value)
                new_line = "{0}\n".format("".join(formatted_row))
                new_text += new_line

            outfile.write(new_text)  # Write the new content
            outfile.write("\n")


def change_flow_routing(
    routing_method: Literal["KINWAVE", "DYNWAVE", "STEADY"], file_path: Path
) -> None:
    """Replace the flow routing method in a SWMM inp file with a new method, in-place.

    Args:
        file_path : Path
            Path to the SWMM inp file to be modified.
        routing_method : {"KINWAVE", "DYNWAVE", "STEADY"}
            The new flow routing method to be used. Available options are:
            ``KINWAVE``, ``DYNWAVE``, or ``STEADY``.
    """
    if routing_method.upper() not in ("KINWAVE", "DYNWAVE", "STEADY"):
        raise ValueError(
            "routing_method must be one of 'KINWAVE', 'DYNWAVE', or 'STEADY'."
        )
    updated_contents = re.sub(
        r"^FLOW_ROUTING\s+.*$",
        f"FLOW_ROUTING {routing_method.upper()}",
        file_path.read_text(),
        flags=re.MULTILINE,
    )
    file_path.write_text(updated_contents)


def data_dict_to_inp(
    data_dict: dict[str, np.ndarray],
    base_input_file: Path,
    new_input_file: Path,
    routing: Literal["KINWAVE", "DYNWAVE", "STEADY"] = "DYNWAVE",
):
    """Write a SWMM .inp file from a dictionary of data arrays.

    Args:
        data_dict (dict[str, np.ndarray]): Dictionary of data arrays. Where
            each key is a SWMM section and each value is a numpy array of
            data to be written to that section. The existing section is
            overwritten
        base_input_file (Path): File path to the example/template .inp file.
        new_input_file (Path): File path to the new SWMM .inp file.
        routing (str, optional): Flow routing method (KINWAVE, DYNWAVE,
            STEADY). Defaults to "DYNWAVE".
    """
    shutil.copy2(base_input_file, new_input_file)

    # Write the inp file
    for key, data in data_dict.items():
        start_section = "[{0}]".format(key)

        overwrite_section(data, start_section, new_input_file)

    # Set the flow routing
    change_flow_routing(routing, new_input_file)


def explode_polygon(row: pd.Series):
    """Explode a polygon into a DataFrame of coordinates.

    Args:
        row (pd.Series): A row of a GeoDataFrame containing a polygon.

    Example:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Polygon
        >>> df = pd.Series({'subcatchment' : '1',
        ...                 'geometry' : Polygon([(0,0), (1,0),
        ...                                       (1,1), (0,1)])})
        >>> explode_polygon(df)
             x    y subcatchment
        0  0.0  0.0            1
        1  1.0  0.0            1
        2  1.0  1.0            1
        3  0.0  1.0            1
        4  0.0  0.0            1
    """
    # Get the vertices of the polygon
    vertices = list(row["geometry"].exterior.coords)

    # Create a new DataFrame for this row
    df = pd.DataFrame(columns=["x", "y"], data=vertices)
    df["subcatchment"] = row["subcatchment"]
    return df


def format_to_swmm_dict(
    nodes: pd.DataFrame,
    outfalls: pd.DataFrame,
    conduits: pd.DataFrame,
    subs: gpd.GeoDataFrame,
    event: dict[str, Any],
    symbol: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Format data to a dictionary of data arrays with columns matching SWMM.

    These data are the parameters of all assets that are written to the SWMM
    input file. More parameters are available to edit (see
    defs/swmm_conversion.yml).

    Args:
        nodes (pd.DataFrame): GeoDataFrame of nodes. With at least columns:
            'id', 'x', 'y', 'max_depth', 'chamber_floor_elevation',
            'manhole_area'.
        outfalls (pd.DataFrame): GeoDataFrame of outfalls. With at least
            columns: 'id', 'chamber_floor_elevation'.
        conduits (pd.DataFrame): GeoDataFrame of conduits. With at least
            columns: 'id', 'u', 'v', 'length', 'roughness', 'shape_swmm',
            'diameter'.
        subs (gpd.GeoDataFrame): GeoDataFrame of subcatchments. With at least
            columns: 'subcatchment', 'rain_gage', 'id', 'area', 'rc', 'width',
            'geometry',
        event (dict): Dict describing storm event. With at least
            keys: 'name', 'unit', 'interval', 'fid'.
        symbol (dict): Dict with coordinates of rain gage. With at least keys:
            'x', 'y', 'name'.

    Example:
        >>> import os
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point, Polygon
        >>> nodes = gpd.GeoDataFrame({'id' : ['node1', 'node2'],
        ...                        'x' : [0, 1],
        ...                        'y' : [0, 1],
        ...                        'max_depth' : [1, 1],
        ...                        'chamber_floor_elevation' : [1, 1],
        ...                        'manhole_area' : [1,1]
        ...                        })
        >>> outfalls = gpd.GeoDataFrame({'id' : ['outfall3'],
        ...                                'chamber_floor_elevation' : [1],
        ...                                'x' : [0],
        ...                                'y' : [0]})
        >>> conduits = gpd.GeoDataFrame({'id' : ['link1','link2'],
        ...                                'u' : ['node1','node2'],
        ...                                'v' : ['node2','outfall3'],
        ...                                'length' : [1,1],
        ...                                'roughness' : [1,1],
        ...                                'shape_swmm' : ['CIRCULAR','CIRCULAR'],
        ...                                'diameter' : [1,1],
        ...                                'capacity' : [0.1,0.1]
        ...                                })
        >>> subs = gpd.GeoDataFrame({'subcatchment' : ['sub1'],
        ...                                'rain_gage' : ['1'],
        ...                                'id' : ['node1'],
        ...                                'area' : [1],
        ...                                'rc' : [1],
        ...                                'width' : [1],
        ...                                'slope' : [0.001],
        ...                                'geometry' : [Polygon([(0.0,0.0),
        ...                                                       (1.0,0.0),
        ...                                                       (1.0,1.0),
        ...                                                       (0.0,1.0)])]})
        >>> rain_fid = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        ...                '..',
        ...                    'swmmanywhere',
        ...                    'defs',
        ...                    'storm.dat')
        >>> event = {'name' : '1',
        ...                'unit' : 'mm',
        ...                'interval' : 1,
        ...                'fid' : rain_fid}
        >>> symbol = {'x' : 0,
        ...            'y' : 0,
        ...            'name' : 'name'}
        >>> data_dict = format_to_swmm_dict(nodes,
        ...                                    outfalls,
        ...                                    conduits,
        ...                                    subs,
        ...                                    event,
        ...                                    symbol)
    """
    # Get the directory of the current module
    current_dir = Path(__file__).parent

    # TODO use 'load_yaml_from_defs'
    # Create the path to iso_converter.yml
    iso_path = current_dir / "defs" / "swmm_conversion.yml"

    # Load conversion mapping from YAML file
    with iso_path.open("r") as file:
        conversion_dict = yaml.safe_load(file)

    ## Create nodes, coordinates and map dimensions
    dims = {
        "x1": nodes.x.min(),
        "y1": nodes.y.min(),
        "x2": nodes.x.max(),
        "y2": nodes.y.max(),
    }
    dims = {x: str(y) + " " for x, y in dims.items()}

    map_dimensions = pd.Series(dims).reset_index().set_index("index").T
    polygons = subs[["subcatchment", "geometry"]].copy()

    # Format dicts to DataFrames
    event = pd.Series(event).reset_index().set_index("index").T
    symbol = pd.Series(symbol).reset_index().set_index("index").T

    # Apply the function to each row
    polygons = pd.concat(polygons.apply(explode_polygon, axis=1).tolist())

    ## Specify sections
    shps = {
        "SUBCATCHMENTS": subs,
        "CONDUITS": conduits,
        "OUTFALLS": outfalls,
        "STORAGE": nodes,
        "XSECTIONS": conduits,
        "SUBAREAS": subs,
        "INFILTRATION": subs,
        "COORDINATES": pd.concat([nodes, outfalls], axis=0),
        "MAP": map_dimensions,
        "Polygons": polygons,
        "PUMPS": None,
        "ORIFICES": None,
        "WEIRS": None,
        "OUTLETS": None,
        "JUNCTIONS": None,
        "RAINGAGES": event,
        "SYMBOLS": symbol,
    }

    # Fill backslash columns and store data in data_dict in the correct order
    import numpy.typing as npt

    def _fill_backslash_columns(
        shp: pd.DataFrame | None, key: str
    ) -> npt.ArrayLike | None:
        if shp is None:
            return None

        # Extract SWMM order and default values
        columns = conversion_dict[key]["iwcolumns"]
        numeric_cols = shp.select_dtypes(include=[np.number]).columns
        shp[numeric_cols] = shp[numeric_cols].fillna(0)

        # Find columns with a default specified
        cols_default = [c[1:] for c in columns if c.startswith("/")]

        # Fill columns with defaults
        shp[["/" + c for c in cols_default]] = np.array(cols_default).T
        return shp[columns].values

    data_dict = {key: _fill_backslash_columns(shp, key) for key, shp in shps.items()}
    return data_dict
