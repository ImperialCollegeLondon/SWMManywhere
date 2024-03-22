# -*- coding: utf-8 -*-
"""Created on Tue Oct 18 10:35:51 2022.

@author: Barney
"""

import io
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

    assert result[2] == 'GB'
    assert result[3] == 'GBR'

@pytest.mark.downloads
def test_get_us_download():
    """Check a US point."""
    x = -113.43318
    y = 33.81869

    result = downloaders.get_country(x, y)

    assert result[2] == 'US'
    assert result[3] == 'USA'

@pytest.mark.downloads
def test_building_downloader_download():
    """Check buildings are downloaded."""
    # Coordinates for small country (VAT)
    x = 7.41839
    y = 43.73205
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / 'temp.parquet'
        # Download
        response = downloaders.download_buildings(temp_fid, x,y)
        # Check response
        assert response == 200

        # Check file exists
        assert temp_fid.exists(), "Buildings data file not found after download."
        
        # Load data
        gdf = gpd.read_parquet(temp_fid)

        # Make sure has some rows
        assert gdf.shape[0] > 0

@pytest.mark.downloads
def test_street_downloader_download():
    """Check streets are downloaded and a specific point in the graph."""
    bbox = (-0.17929,51.49638, -0.17383,51.49846)
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
    # Please do not reuse api_key
    test_api_key = 'b206e65629ac0e53d599e43438560d28' 

    bbox = (-0.17929,51.49638, -0.17383,51.49846)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / 'temp.tif'
        
        # Download
        response = downloaders.download_elevation(temp_fid, bbox, test_api_key)

        # Check response
        assert response == 200
        
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

def test_get_uk():
    """Check a UK point."""
    # Coordinates for London, UK
    x = -0.1276
    y = 51.5074
      
    # Create a mock response for geolocator.reverse
    mock_location = mock.Mock()
    mock_location.raw = {'address': {'country_code': 'gb'}}

    # Mock Nominatim
    with mock.patch.object(Nominatim, 'reverse', return_value=mock_location):
        # Mock yaml.safe_load
        with mock.patch.object(yaml, 'safe_load', return_value={'GB': 'GBR'}):
            # Call get_country
            result = downloaders.get_country(x, y)
    
    assert result[2] == 'GB'
    assert result[3] == 'GBR'

def test_building_downloader():
    """Check buildings are downloaded."""
    # Coordinates for small country (VAT)
    x = 7.41839
    y = 43.73205
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / 'temp.parquet'
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"features": []}'
        with mock.patch('requests.get', 
                        return_value=mock_response) as mock_get:
            # Call your function that uses requests.get
            response = downloaders.download_buildings(temp_fid, x, y)

            # Assert that requests.get was called with the right arguments
            mock_get.assert_called_once_with('https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso=MCO/MCO.parquet')
     
        # Check response
        assert response == 200

def test_street_downloader():
    """Check streets are downloaded and a specific point in the graph."""
    bbox = (-0.17929,51.49638, -0.17383,51.49846)

    mock_graph = nx.MultiDiGraph()
    # Mock ox.graph_from_bbox
    with mock.patch.object(ox, 'graph_from_bbox', return_value=mock_graph):
        # Call download_street
        G = downloaders.download_street(bbox)
        assert G == mock_graph

def test_river_downloader():
    """Check rivers are downloaded and a specific point in the graph."""
    bbox = (0.0402, 51.55759, 0.09825591114207548, 51.6205)

    mock_graph = nx.MultiDiGraph()
    # Mock ox.graph_from_bbox
    with mock.patch.object(ox, 'graph_from_bbox', return_value=mock_graph):
        # Call download_street
        G = downloaders.download_river(bbox)
        assert G == mock_graph

def test_elevation_downloader():
    """Check elevation downloads, writes, contains data, and a known elevation."""
    # Please do not reuse api_key
    test_api_key = 'b206e65629ac0e53d599e43438560d28' 

    bbox = (-0.17929,51.49638, -0.17383,51.49846)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_fid = Path(temp_dir) / 'temp.tif'
        
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.raw = io.BytesIO(b'25')
        with mock.patch('requests.get', 
                        return_value=mock_response) as mock_get:
            # Call your function that uses requests.get
            response = downloaders.download_elevation(temp_fid, 
                                                        bbox, 
                                                        test_api_key)
            # Assert that requests.get was called with the right arguments
            assert 'https://portal.opentopography.org/API/globaldem?demtype=NASADEM&south=51.49638&north=51.49846&west=-0.17929&east=-0.17383&outputFormat=GTiff&API_Key' in mock_get.call_args[0][0] # noqa: E501

        # Check response
        assert response == 200
        
        # Check response
        assert temp_fid.exists(), "Elevation data file not found after download."