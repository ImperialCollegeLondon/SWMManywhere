"""Prepare data module for SWMManywhere.

A module to download data needed for SWMManywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import cdsapi
import networkx as nx
import osmnx as ox
import pandas as pd
import planetary_computer
import pystac_client
import requests
import rioxarray
import rioxarray.merge as rxr_merge
import xarray as xr
from geopy.geocoders import Nominatim
from pyarrow import RecordBatchReader
from pyarrow.compute import field
from pyarrow.dataset import dataset
from pyarrow.fs import S3FileSystem
from pyarrow.parquet import ParquetWriter

from swmmanywhere.logging import logger
from swmmanywhere.utilities import yaml_load


def get_country(x: float, y: float) -> dict[int, str]:
    """Identify which country a point (x,y) is in.

    Return the two and three letter ISO code for coordinates x (longitude)
    and y (latitude)

    Args:
        x (float): Longitude.
        y (float): Latitude.

    Returns:
        dict: A dictionary containing the two and three letter ISO codes.
            Example: {2: 'GB', 3: 'GBR'}
    """
    # Get the directory of the current module
    current_dir = Path(__file__).parent

    # TODO use 'load_yaml_from_defs'
    # Create the path to iso_converter.yml
    iso_path = current_dir / "defs" / "iso_converter.yml"

    # Initialize geolocator
    geolocator = Nominatim(user_agent="get_iso")

    # Load ISO code mapping from YAML file
    data = yaml_load(iso_path.read_text())

    # Get country ISO code from coordinates
    location = geolocator.reverse(f"{y}, {x}", exactly_one=True)
    iso_country_code = location.raw.get("address", {}).get("country_code").upper()

    # Return a dictionary with the two and three letter ISO codes
    return {2: iso_country_code, 3: data.get(iso_country_code, "")}


def _record_batch_reader(bbox: tuple[float, float, float, float]) -> RecordBatchReader:
    """Get a pyarrow batch reader this for bounding box and s3 path."""
    s3_region = "us-west-2"
    version = "2024-07-22.0"
    path = f"overturemaps-{s3_region}/release/{version}/theme=buildings/type=building/"
    xmin, ymin, xmax, ymax = bbox
    ds_filter = (
        (field("bbox", "xmin") < xmax)
        & (field("bbox", "xmax") > xmin)
        & (field("bbox", "ymin") < ymax)
        & (field("bbox", "ymax") > ymin)
    )

    ds = dataset(path, filesystem=S3FileSystem(anonymous=True, region=s3_region))
    batches = ds.to_batches(filter=ds_filter)
    non_empty_batches = (b for b in batches if b.num_rows > 0)

    geoarrow_schema = ds.schema.set(
        ds.schema.get_field_index("geometry"),
        ds.schema.field("geometry").with_metadata(
            {b"ARROW:extension:name": b"geoarrow.wkb"}
        ),
    )
    return RecordBatchReader.from_batches(geoarrow_schema, non_empty_batches)


def download_buildings_bbox(
    file_address: Path, bbox: tuple[float, float, float, float]
) -> None:
    """Retrieve building data in bbox from Overture Maps to file.

    This function is based on
    `overturemaps-py <https://github.com/OvertureMaps/overturemaps-py>`__.

    Args:
        bbox (tuple): Bounding box coordinates (xmin, ymin, xmax, ymax)
        file_address (Path): File address to save the downloaded data.
    """
    reader = _record_batch_reader(bbox)
    with ParquetWriter(file_address, reader.schema) as writer:
        for batch in reader:
            writer.write_batch(batch)


def download_buildings(file_address: Path, x: float, y: float) -> int:
    """Download buildings data based on coordinates and save to a file.

    Args:
        file_address (Path): File address to save the downloaded data.
        x (float): Longitude.
        y (float): Latitude.

    Returns:
        status_code (int): Response status code
    """
    # Get three letter ISO code
    iso_country_code = get_country(x, y)[3]

    # Construct API URL
    api_url = f"https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso={iso_country_code}/{iso_country_code}.parquet"

    # Download data
    response = requests.get(api_url)
    if response.status_code == 200:
        # Save data to the specified file address
        with file_address.open("wb") as file:
            file.write(response.content)
        logger.info(f"Data downloaded and saved to {file_address}")
    else:
        logger.error(f"Error downloading data. Status code: {response.status_code}")
    return response.status_code


def download_street(
    bbox: tuple[float, float, float, float], network_type="drive"
) -> nx.MultiDiGraph:
    """Get street network within a bounding box using OSMNX.

    [CREDIT: Taher Cheghini busn_estimator package]

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.
        network_type (str, optional): Type of street network. Defaults to 'drive'.

    Returns:
        nx.MultiDiGraph: Street network with type drive and
            ``truncate_by_edge set`` to True.
    """
    west, south, east, north = bbox
    bbox = (north, south, east, west)  # not sure why osmnx uses this order
    graph = ox.graph_from_bbox(
        bbox=bbox, network_type=network_type, truncate_by_edge=True
    )
    return cast("nx.MultiDiGraph", graph)


def download_river(bbox: tuple[float, float, float, float]) -> nx.MultiDiGraph:
    """Get water network within a bounding box using OSMNX.

    Args:
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.

    Returns:
        nx.MultiDiGraph: River network with type waterway and
            ``truncate_by_edge set`` to True.
    """
    west, south, east, north = bbox
    try:
        graph = ox.graph_from_bbox(
            bbox=(north, south, east, west),
            truncate_by_edge=True,
            custom_filter='["waterway"]',
            retain_all=True,
        )
    except ValueError as e:
        if str(e) == "Found no graph nodes within the requested polygon":
            logger.warning("No water network found within the bounding box.")
            return nx.MultiDiGraph()
        else:
            raise  # Re-raise the exception

    return cast("nx.MultiDiGraph", graph)


def download_elevation(fid: Path, bbox: tuple[float, float, float, float]) -> None:
    """Download NASADEM elevation data from Microsoft Planetary computer.

    Downloads elevation data in GeoTIFF format from Microsoft Planetary computer
      based on the specified bounding box.

    Args:
        fid (Path): File path to save the downloaded elevation data.
        bbox (tuple[float, float, float, float]): Bounding box as tuple in form
            of (west, south, east, north) at EPSG:4326.

    Example:
        ```
        bbox = (-120, 35, -118, 37)  # Example bounding box coordinates
        download_elevation('elevation_data.tif',
                            bbox)
        ```

    Author:
        cheginit
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    search = catalog.search(
        collections=["nasadem"],
        bbox=bbox,
    )
    signed_asset = (
        planetary_computer.sign(item.assets["elevation"]).href
        for item in search.items()
    )
    dem = rxr_merge.merge_arrays(
        [rioxarray.open_rasterio(href).squeeze(drop=True) for href in signed_asset]
    )
    dem = dem.rio.clip_box(*bbox)
    dem.rio.to_raster(fid)


def download_precipitation(
    bbox: tuple[float, float, float, float],
    start_date: str = "2015-01-01",
    end_date: str = "2015-01-05",
    username: str = "<your_username>",
    api_key: str = "<your_api_key>",
) -> pd.DataFrame:
    """Download precipitation data within bbox from ERA5.

    Register for an account and API key at: https://cds.climate.copernicus.eu.
    Produces hourly gridded data (31km grid) in CRS EPSG:4326.
    More information at: https://github.com/ecmwf/cdsapi.

    Args:
        bbox (tuple): Bounding box coordinates in the format
            (minx, miny, maxx, maxy).
        start_date (str, optional): Start date. Defaults to '2015-01-01'.
        end_date (str, optional): End date. Defaults to '2015-01-05'.
        username (str, optional): CDS api username.
            Defaults to '<your_username>'.
        api_key (str, optional): CDS api key. Defaults to '<your_api_key>'.

    Returns:
        df (DataFrame): DataFrame containing downloaded data.
    """
    # Notes for docstring:
    # looks like api will give nearest point if not in bbox

    # Define the request parameters
    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": "total_precipitation",
        "date": "/".join([start_date, end_date]),
        "time": "/".join([f"{str(x).zfill(2)}:00" for x in range(24)]),
        "area": bbox,
    }

    c = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api/v2", key=f"{username}:{api_key}"
    )
    # Get data
    c.retrieve("reanalysis-era5-single-levels", request, "download.nc")

    with xr.open_dataset("download.nc") as data:
        # Convert the xarray Dataset to a pandas DataFrame
        df = data.to_dataframe()
        df["unit"] = "m/hr"

    # Delete nc file
    Path("download.nc").unlink()

    return df
