# -*- coding: utf-8 -*-
"""Test the prepare_data module.

By default downloads themselves are mocked, but these can be enabled with the
following test command:

pytest -m downloads
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pytest
import rasterio
import yaml
from geopy.geocoders import Nominatim

from swmmanywhere import prepare_data as downloaders


# Test get_country
@pytest.mark.downloads
def test_get_uk_download():
    """Check a UK point."""
    # Coordinates for London, UK
    x = -0.1276
    y = 51.5074

    result = downloaders.get_country(x, y)

    assert result[2] == "GB"
    assert result[3] == "GBR"


@pytest.mark.downloads
def test_get_us_download():
    """Check a US point."""
    x = -113.43318
    y = 33.81869

    result = downloaders.get_country(x, y)

    assert result[2] == "US"
    assert result[3] == "USA"


@pytest.mark.downloads
def test_building_downloader_download():
    """Check buildings are downloaded."""
    # Coordinates for small country (VAT)
    x = 7.41839
    y = 43.73205
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.parquet"
        # Download
        response = downloaders.download_buildings(temp_fid, x, y)
        # Check response
        assert response == 200

        # Check file exists
        assert temp_fid.exists(), "Buildings data file not found after download."

        # Load data
        gdf = gpd.read_parquet(temp_fid)

        # Make sure has some rows
        assert gdf.shape[0] > 0


@pytest.mark.downloads
def test_building_bbox_downloader_download():
    """Check buildings are downloaded."""
    # Coordinates for small country (VAT)
    bbox = (-0.17929, 51.49638, -0.17383, 51.49846)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.parquet"
        # Download
        downloaders.download_buildings_bbox(temp_fid, bbox)

        # Check file exists
        assert temp_fid.exists(), "Buildings data file not found after download."

        # Load data
        gdf = gpd.read_parquet(temp_fid)

        # Make sure has some rows
        assert gdf.shape[0] > 0


@pytest.mark.downloads
def test_street_downloader_download():
    """Check streets are downloaded and a specific point in the graph."""
    bbox = (-0.17929, 51.49638, -0.17383, 51.49846)
    G = downloaders.download_street(bbox)

    # Not sure if they they are likely to change the osmid
    assert 26389449 in G.nodes


@pytest.mark.downloads
def test_river_downloader_download():
    """Check rivers are downloaded and a specific point in the graph."""
    bbox = (0.0402, 51.55759, 0.09825591114207548, 51.6205)
    G = downloaders.download_river(bbox)

    # Not sure if they they are likely to change the osmid
    assert 21473922 in G.nodes


@pytest.mark.downloads
def test_elevation_downloader_download():
    """Check elevation downloads, writes, contains data, and a known elevation."""
    bbox = (-0.17929, 51.49638, -0.17383, 51.49846)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.tif"

        # Download
        downloaders.download_elevation(temp_fid, bbox)

        # Check response
        assert temp_fid.exists(), "Elevation data file not found after download."

        # Load data
        with rasterio.open(temp_fid) as src:
            data = src.read(1)  # Reading the first band as an example

        # Make sure it has some values
        assert data.size > 0, "Elevation data should have some values."

        # Test some property of data (not sure if they may change this
        # data)
        assert data.max().max() > 25, "Elevation data should be higher."


@pytest.fixture
def setup_mocks():
    """Set up get_country mock for the tests."""
    # Mock for geolocator.reverse
    mock_location = mock.Mock()
    mock_location.raw = {"address": {"country_code": "gb"}}

    # Mock Nominatim
    nominatim_patch = mock.patch.object(
        Nominatim, "reverse", return_value=mock_location
    )
    # Mock yaml.safe_load
    yaml_patch = mock.patch.object(yaml, "safe_load", return_value={"GB": "GBR"})

    with nominatim_patch, yaml_patch:
        yield


def test_get_uk(setup_mocks):
    """Check a UK point."""
    # Coordinates for London, UK
    x = -0.1276
    y = 51.5074

    # Call get_country
    result = downloaders.get_country(x, y)

    assert result[2] == "GB"
    assert result[3] == "GBR"


def test_building_downloader(setup_mocks):
    """Check buildings are downloaded."""
    # Coordinates
    x = -0.1276
    y = 51.5074

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.parquet"
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"features": []}'
        with mock.patch("requests.get", return_value=mock_response) as mock_get:
            # Call your function that uses requests.get
            response = downloaders.download_buildings(temp_fid, x, y)

            # Assert that requests.get was called with the right arguments
            mock_get.assert_called_once_with(
                "https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso=GBR/GBR.parquet"
            )

        # Check response
        assert response == 200


def test_street_downloader():
    """Check streets are downloaded and a specific point in the graph."""
    bbox = (-0.17929, 51.49638, -0.17383, 51.49846)

    mock_graph = nx.MultiDiGraph()
    # Mock ox.graph_from_bbox
    with mock.patch.object(ox, "graph_from_bbox", return_value=mock_graph):
        # Call download_street
        G = downloaders.download_street(bbox)
        assert G == mock_graph


def test_river_downloader():
    """Check rivers are downloaded and a specific point in the graph."""
    bbox = (0.0402, 51.55759, 0.09825591114207548, 51.6205)

    mock_graph = nx.MultiDiGraph()
    mock_graph.add_node(1)
    # Mock ox.graph_from_bbox
    with mock.patch.object(
        ox, "graph_from_bbox", return_value=mock_graph
    ) as mock_from_bbox:
        # Call download_street
        G = downloaders.download_river(bbox)
        assert G == mock_graph

        mock_from_bbox.side_effect = ValueError(
            "Found no graph nodes within the requested polygon"
        )
        G = downloaders.download_river(bbox)
        assert G.size() == 0


def test_download_elevation():
    """Check elevation downloads, writes, contains data, and a known elevation."""
    bbox = (-0.17929, 51.49638, -0.17383, 51.49846)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / "temp.tif"

        # Mock the external dependencies
        module_base = "swmmanywhere.prepare_data."
        with (
            mock.patch(f"{module_base}pystac_client.Client.open") as mock_open,
            mock.patch(f"{module_base}planetary_computer.sign_inplace"),
            mock.patch(f"{module_base}planetary_computer.sign") as mock_sign,
            mock.patch(f"{module_base}rioxarray.open_rasterio") as mock_open_rasterio,
            mock.patch(f"{module_base}rxr_merge.merge_arrays") as mock_merge_arrays,
        ):
            # Mock the behavior of the catalog search and items
            mock_catalog = mock.MagicMock()
            mock_open.return_value = mock_catalog
            mock_search = mock.MagicMock()
            mock_catalog.search.return_value = mock_search
            mock_items = [mock.MagicMock(), mock.MagicMock()]
            for item in mock_items:
                item.assets = {"elevation": mock.MagicMock()}
            mock_search.items.return_value = mock_items

            # Mock the signed URLs
            mock_sign.side_effect = lambda x: mock.MagicMock(href=f"signed_{x}")

            # Mock the raster data
            mock_raster = mock.MagicMock()
            mock_open_rasterio.return_value = mock_raster

            # Mock the merged array
            mock_merged_array = mock.MagicMock()
            mock_merge_arrays.return_value = mock_merged_array

            # Mock the `rio` attribute on the merged array
            mock_merged_array.rio = mock.MagicMock()
            mock_merged_array.rio.clip_box.return_value = mock_merged_array
            mock_merged_array.rio.to_raster.return_value = None

            # Call the function
            downloaders.download_elevation(temp_fid, bbox)

            # Assertions
            mock_open.assert_called_once_with(
                "https://planetarycomputer.microsoft.com/api/stac/v1",
                modifier=mock.ANY,
            )
            mock_catalog.search.assert_called_once_with(
                collections=["nasadem"],
                bbox=bbox,
            )
            assert len(mock_items) == 2
            assert mock_sign.call_count == len(mock_items)
            mock_open_rasterio.assert_called()
            mock_merge_arrays.assert_called_once()
            mock_merged_array.rio.clip_box.assert_called_once_with(*bbox)
            mock_merged_array.rio.to_raster.assert_called_once_with(temp_fid)
